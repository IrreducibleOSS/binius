// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_core::oracle::ShiftVariant;
use binius_field::{arch::OptimalUnderlier, underlier::UnderlierType, TowerField};

use super::constraint_system::{ChannelId, Col, ConstraintSystem, Table};
use crate::{
	types::*,
	u32::{U32Add, U32AddFlags},
	witness::{TableFiller, TableWitnessIndex, WitnessIndex},
};

pub struct EvensTable {
	in_bits: Col<B1, 5>,
	out_bits: Col<B1, 5>,
	pub in_val: Col<B32>,
	pub out_val: Col<B32>,
}

impl<F: TowerField> EvensTable {
	pub fn new(cs: &mut ConstraintSystem<F>, seq_chan: ChannelId) -> Self {
		let mut table = cs.add_table("evens");

		let in_bits = table.add_committed::<B1, 5>("in_bits");

		// Logical right shift is division by 2
		let out_bits =
			table.add_shifted::<B1, 5>("out_bits", in_bits, 5, 1, ShiftVariant::LogicalRight);

		let in_val = table.add_packed::<B32, 0, _, _>("in", in_bits);
		let out_val = table.add_packed::<B32, 0, _, _>("out", out_bits);

		table.pull(seq_chan, [in_val]);
		table.push(seq_chan, [out_val]);

		Self {
			in_bits,
			out_bits,
			in_val,
			out_val,
		}
	}
}

impl<U: UnderlierType> TableFiller<U> for EvensTable {
	type Row = u32;
	type Error = anyhow::Error;

	fn fill(&self, rows: &[u32], witness: TableWitnessIndex<U>) -> Result<(), anyhow::Error> {
		let in_val = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.in_val));
		let out_val = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.out_val));

		for (i, row) in rows.iter().enumerate() {
			in_val[i] = row.val;
			out_val[i] = row.val / 2;
		}
	}
}

pub struct OddsTable {
	in_bits: Col<B1, 5>,
	in_dbl: Col<B1, 5>,
	carry_bit: Col<B1, 5>,
	add_in_x3: U32Add,
	pub in_val: Col<B32>,
	pub out_val: Col<B32>,
}

impl<F: TowerField> OddsTable {
	pub fn new(cs: &mut ConstraintSystem<F>, seq_chan: ChannelId) -> Self {
		let mut table = cs.add_table("odds");

		let in_bits = table.add_committed::<B1, 5>("in_bits");

		// Input times 2
		let in_dbl = table.add_shifted::<B1, 5>("in_dbl", in_bits, 5, 1, ShiftVariant::LogicalLeft);

		// TODO: Figure out how to add repeating constants (repeating transparents). Basically a
		// multilinear extension of some constant vector, repeating for the number of rows.
		// This shouldn't actually be committed. It should be the carry bit, repeated for each row.
		let carry_bit = table.add_commited::<B1, 5>("carry_bit");

		// Input times 3 + 1
		// TODO: This needs
		let add_in_x3 = U32Add::new(
			&mut table,
			in_dbl,
			in_bits,
			U32AddFlags {
				carry_in_bit: Some(carry_bit),
				..U32AddFlags::default()
			},
		);

		let out_bits = table.add_committed::<B1, 5>("out_bits");

		table.add_packed::<B32, 0, _, _>("in", in_bits);
		table.add_packed::<B32, 0, _, _>("out", add_in_x3.zout);

		Self {
			in_bits,
			in_dbl,
			carry_bit,
			out_bits,

			in_val: u32add_gadget.zout,
			out_val: table.get_col("out"),
		}
	}
}

impl<U: UnderlierType> TableFiller<U> for OddsTable {
	type Row = u32;
	type Error = anyhow::Error;

	fn fill(&self, rows: &[u32], mut witness: TableWitnessIndex<U>) -> Result<(), anyhow::Error> {
		let in_val = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.in_val));
		let in_dbl = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.in_dbl));
		let carry_bit = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.carry_bit));

		for (i, row) in rows.iter().enumerate() {
			in_val[i] = row.val;
			(in_dbl[i], _) = row.val.overflowing_shl(1);
			carry_bit[i] = 1u32;
		}

		self.add_in_x3.populate(&mut Witness);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constraint_system::Instance;

	#[test]
	fn test_collatz() {
		let mut cs_builder = ConstraintSystemBuilder::new();
		let sequence_chan = cs_builder.add_channel("sequence");
		let evens_table = EvensTable::new(&mut cs_builder, sequence_chan);
		let odds_table = OddsTable::new(&mut cs_builder, sequence_chan);
		let cs = cs_builder.build().unwrap();

		let statement = make_instance(3999);
		let compiled_cs = cs.compile(statement).unwrap();

		let mut allocator = bumpalo::Bump::new();

		// This should really just accept padded table sizes.
		let witness = WitnessIndex::new(cs, instance.table_sizes(), &mut allocator).unwrap();

		let table_witness = witness.table(evens_table.id);

		fill_table_sequential();
		evens_table.populate(&mut table_witness);

		evens_table
	}
}
