// Copyright 2025 Irreducible Inc.

use binius_field::{
	PackedExtension, PackedFieldIndexable, arch::OptimalUnderlier128b, as_packed_field::PackedType,
};
use binius_m3::builder::{
	B16, B128, Col, ConstraintSystem, TableFiller, TableId, TableWitnessSegment, WitnessIndex,
	test_utils::validate_system_witness,
};
use bumpalo::Bump;

const VALUES_PER_ROW: usize = 32;
const N_ROWS: usize = 8;

pub struct MyTable {
	id: TableId,
	committed_b128: Col<B128, VALUES_PER_ROW>,
	squared_b128: Col<B128, VALUES_PER_ROW>,
	committed_b16: Col<B16, VALUES_PER_ROW>,
	squared_b16: Col<B16, VALUES_PER_ROW>,
}

impl MyTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("table_1");
		let committed_b128 = table.add_committed::<B128, VALUES_PER_ROW>("committed b128");
		let squared_b128 = table.add_squared("squared b128", committed_b128);

		let committed_b16 = table.add_committed::<B16, VALUES_PER_ROW>("committed b16");
		let squared_b16 = table.add_squared("squared b16", committed_b16);

		table.assert_zero(
			"squared_b128 = committed_b128^2",
			squared_b128 - committed_b128 * committed_b128,
		);
		table.assert_zero(
			"squared_b16 = committed_b16^2",
			squared_b16 - committed_b16 * committed_b16,
		);

		Self {
			id: table.id(),
			committed_b128,
			squared_b128,
			committed_b16,
			squared_b16,
		}
	}
}

impl<P> TableFiller<P> for MyTable
where
	P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B128> + PackedExtension<B16>,
{
	type Event = u128;

	fn id(&self) -> TableId {
		self.id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &'a mut TableWitnessSegment<P>,
	) -> Result<(), anyhow::Error> {
		let mut committed_b128 = witness.get_mut_as(self.committed_b128)?;
		let mut squared_b128 = witness.get_mut_as(self.squared_b128)?;

		let mut committed_b16 = witness.get_mut_as(self.committed_b16)?;
		let mut squared_b16 = witness.get_mut_as(self.squared_b16)?;

		for (i, &com) in rows.enumerate() {
			for j in 0..VALUES_PER_ROW {
				committed_b128[i * VALUES_PER_ROW + j] = com;
				squared_b128[i * VALUES_PER_ROW + j] = B128::from(com) * B128::from(com);

				committed_b16[i * VALUES_PER_ROW + j] = com as u16;
				squared_b16[i * VALUES_PER_ROW + j] = B16::from(com as u16) * B16::from(com as u16);
			}
		}
		Ok(())
	}
}

#[test]
fn test_squared() {
	let allocator = Bump::new();
	let mut cs = ConstraintSystem::<B128>::new();
	let table = MyTable::new(&mut cs);

	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
	witness
		.fill_table_sequential(&table, &(0..N_ROWS as u128).collect::<Vec<_>>())
		.unwrap();

	validate_system_witness::<OptimalUnderlier128b>(&cs, witness, vec![]);
}
