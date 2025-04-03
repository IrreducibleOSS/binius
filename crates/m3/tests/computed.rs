// Copyright 2025 Irreducible Inc.

use binius_core::{fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackScalar, Field};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_m3::builder::{
	Col, ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment, B1, B128, B64,
};
use bumpalo::Bump;
use bytemuck::Pod;

const VALUES_PER_ROW: usize = 32;
const N_ROWS: usize = 8;
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 30;

pub struct MyTable {
	id: TableId,
	committed_1: Col<B128, VALUES_PER_ROW>,
	committed_2: Col<B128, VALUES_PER_ROW>,
	_zeros_1: [Col<B64, 16>; 10],
	_zeros_2: [Col<B64, 64>; 10],
	computed: Col<B128, VALUES_PER_ROW>,
}

impl MyTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("table_1");
		let committed_1 = table.add_committed::<B128, VALUES_PER_ROW>("committed_2");
		let zeros_1 = table.add_committed_multiple::<B64, 16, 10>("col_zeros_1");
		let committed_2 = table.add_committed::<B128, VALUES_PER_ROW>("committed_2");
		let zeros_2 = table.add_committed_multiple::<B64, 64, 10>("col_zeros_2");
		let expr = (committed_1 + committed_2) * committed_1 * B128::from(10) + B128::ONE;
		let computed = table.add_computed("computed", expr);
		for col in &zeros_1 {
			table.assert_zero("zeros_1", (*col).into());
		}
		for col in &zeros_2 {
			table.assert_zero("zeros_2", (*col).into());
		}
		Self {
			id: table.id(),
			committed_1,
			committed_2,
			computed,
			_zeros_1: zeros_1,
			_zeros_2: zeros_2,
		}
	}
}

impl<U> TableFiller<U> for MyTable
where
	U: Pod + PackScalar<B1>,
{
	type Event = (u128, u128);

	fn id(&self) -> TableId {
		self.id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &'a mut TableWitnessSegment<U>,
	) -> Result<(), anyhow::Error> {
		let mut committed_1 = witness.get_mut_as(self.committed_1)?;
		let mut committed_2 = witness.get_mut_as(self.committed_2)?;
		let mut computed = witness.get_mut_as(self.computed)?;

		for (i, &(com1, com2)) in rows.enumerate() {
			for j in 0..VALUES_PER_ROW {
				committed_1[i * VALUES_PER_ROW + j] = com1;
				committed_2[i * VALUES_PER_ROW + j] = com2;
				computed[i * VALUES_PER_ROW + j] =
					(B128::from(com1) + B128::from(com2)) * B128::from(com1) * B128::from(10)
						+ B128::ONE;
			}
		}
		Ok(())
	}
}

#[test]
fn test_m3_computed_col() {
	let allocator = Bump::new();
	let mut cs = ConstraintSystem::<B128>::new();
	let table = MyTable::new(&mut cs);
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![N_ROWS],
	};
	let mut witness = cs
		.build_witness::<OptimalUnderlier128b>(&allocator, &statement)
		.unwrap();
	witness
		.fill_table_sequential(
			&table,
			&(0..N_ROWS as u128)
				.map(|i| (i, i + 10_u128))
				.collect::<Vec<_>>(),
		)
		.unwrap();

	let constraint_system = cs.compile(&statement).unwrap();
	let witness = witness.into_multilinear_extension_index();

	binius_core::constraint_system::validate::validate_witness(
		&constraint_system,
		&statement.boundaries,
		&witness,
	)
	.unwrap();

	let proof = binius_core::constraint_system::prove::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&constraint_system,
		LOG_INV_RATE,
		SECURITY_BITS,
		&statement.boundaries,
		witness,
		&binius_hal::make_portable_backend(),
	)
	.unwrap();

	binius_core::constraint_system::verify::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&constraint_system, LOG_INV_RATE, SECURITY_BITS, &statement.boundaries, proof)
	.unwrap();
}
