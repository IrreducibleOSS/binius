// Copyright 2025 Irreducible Inc.

use binius_compute::cpu::alloc::CpuComputeAllocator;
use binius_field::{
	Field, PackedExtension, PackedFieldIndexable, arch::OptimalUnderlier128b,
	as_packed_field::PackedType,
};
use binius_m3::builder::{
	B128, Col, ConstraintSystem, TableFiller, TableId, TableWitnessSegment, WitnessIndex,
	test_utils::validate_system_witness,
};

const VALUES_PER_ROW: usize = 32;
const N_ROWS: usize = 8;

pub struct MyTable {
	id: TableId,
	committed_1: Col<B128, VALUES_PER_ROW>,
	committed_2: Col<B128, VALUES_PER_ROW>,
	computed: Col<B128, VALUES_PER_ROW>,
}

impl MyTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("table_1");
		let committed_1 = table.add_committed::<B128, VALUES_PER_ROW>("committed_2");
		let committed_2 = table.add_committed::<B128, VALUES_PER_ROW>("committed_2");
		let expr = (committed_1 + committed_2) * committed_1 * B128::from(10) + B128::ONE;
		let computed = table.add_computed("computed", expr.clone());

		// Test that the computed column equals the composite evaluation over the table.
		table.assert_zero("computed = expr", expr - computed);

		Self {
			id: table.id(),
			committed_1,
			committed_2,
			computed,
		}
	}
}

impl<P> TableFiller<P> for MyTable
where
	P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B128>,
{
	type Event = (u128, u128);

	fn id(&self) -> TableId {
		self.id
	}

	fn fill(
		&self,
		rows: &[Self::Event],
		witness: &mut TableWitnessSegment<P>,
	) -> Result<(), anyhow::Error> {
		let mut committed_1 = witness.get_mut_as(self.committed_1)?;
		let mut committed_2 = witness.get_mut_as(self.committed_2)?;
		let mut computed = witness.get_mut_as(self.computed)?;

		for (i, &(com1, com2)) in rows.iter().enumerate() {
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
	let mut allocator = CpuComputeAllocator::new(1 << 12);
	let allocator = allocator.into_bump_allocator();
	let mut cs = ConstraintSystem::<B128>::new();
	let table = MyTable::new(&mut cs);

	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
	witness
		.fill_table_sequential(
			&table,
			&(0..N_ROWS as u128)
				.map(|i| (i, i + 10_u128))
				.collect::<Vec<_>>(),
		)
		.unwrap();

	validate_system_witness::<OptimalUnderlier128b>(&cs, witness, vec![]);
}
