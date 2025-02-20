// Copyright 2025 Irreducible Inc.

use binius_core::oracle::ShiftVariant;
use binius_field::{underlier::UnderlierType, TowerField};
use bytemuck::must_cast_slice_mut;

use super::{
	builder::{Col, ConstraintSystemBuilder},
	constraint_system::ChannelId,
};
use crate::{
	collatz_high_level::{EvensEvent, OddsEvent},
	constraint_system::TableId,
	types::*,
	u32::{U32Add, U32AddFlags},
	witness::{TableFiller, TableWitnessIndex},
};

#[derive(Debug)]
pub struct EvensTable {
	pub id: TableId,
	in_bits: Col<B1, 5>,
	out_bits: Col<B1, 5>,
	pub in_val: Col<B32>,
	pub out_val: Col<B32>,
}

impl<F: TowerField> EvensTable {
	pub fn new(cs: &mut ConstraintSystemBuilder<F>, seq_chan: ChannelId) -> Self {
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
			id: table.id(),
			in_bits,
			out_bits,
			in_val,
			out_val,
		}
	}
}

impl<U: UnderlierType> TableFiller<U> for EvensTable {
	type Event = EvensEvent;
	type Error = anyhow::Error;

	fn id(&self) -> TableId {
		self.id
	}

	fn fill(
		&self,
		rows: &[Self::Event],
		witness: TableWitnessIndex<U>,
	) -> Result<(), anyhow::Error> {
		let in_val = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.in_val));
		let out_val = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.out_val));

		for (i, event) in rows.iter().enumerate() {
			in_val[i] = event.val;
			out_val[i] = event.val / 2;
		}
	}
}

/// Table of transitions for odd numbers in the Collatz sequence.
#[derive(Debug)]
pub struct OddsTable {
	id: TableId,
	in_bits: Col<B1, 5>,
	in_dbl: Col<B1, 5>,
	carry_bit: Col<B1, 5>,
	add_in_x3: U32Add,
	pub in_val: Col<B32>,
	pub out_val: Col<B32>,
}

impl<F: TowerField> OddsTable {
	pub fn new(cs: &mut ConstraintSystemBuilder<F>, seq_chan: ChannelId) -> Self {
		let mut table = cs.add_table("odds");

		let in_bits = table.add_committed::<B1, 5>("in_bits");

		// Input times 2
		let in_dbl = table.add_shifted::<B1, 5>("in_dbl", in_bits, 5, 1, ShiftVariant::LogicalLeft);

		// TODO: Figure out how to add repeating constants (repeating transparents). Basically a
		// multilinear extension of some constant vector, repeating for the number of rows.
		// This shouldn't actually be committed. It should be the carry bit, repeated for each row.
		let carry_bit = table.add_commited::<B1, 5>("carry_bit");

		// Input times 3 + 1
		let add_in_x3 = U32Add::new(
			&mut table,
			in_dbl,
			in_bits,
			U32AddFlags {
				carry_in_bit: Some(carry_bit),
				..U32AddFlags::default()
			},
		);

		let in_val = table.add_packed::<B32, 0, _, _>("in", in_bits);
		let out_val = table.add_packed::<B32, 0, _, _>("out", add_in_x3.zout);

		table.pull(seq_chan, [in_val]);
		table.push(seq_chan, [out_val]);

		Self {
			id: table.id(),
			in_bits,
			in_dbl,
			carry_bit,
			add_in_x3,
			in_val,
			out_val: table.get_col("out"),
		}
	}
}

impl<U: UnderlierType> TableFiller<U> for OddsTable {
	type Event = OddsEvent;
	type Error = anyhow::Error;

	fn id(&self) -> TableId {
		self.id
	}

	fn fill(
		&self,
		rows: &[Self::Event],
		mut witness: TableWitnessIndex<U>,
	) -> Result<(), anyhow::Error> {
		let in_val = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.in_val));
		let in_dbl = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.in_dbl));
		let carry_bit = must_cast_slice_mut::<u32, _>(&mut *witness.get_mut(self.carry_bit));

		for (i, event) in rows.iter().enumerate() {
			in_val[i] = event.val;
			(in_dbl[i], _) = event.val.overflowing_shl(1);
			carry_bit[i] = 1u32;
		}

		self.add_in_x3.populate(&mut witness);
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::channel::{Boundary, FlushDirection};
	use bumpalo::Bump;

	use super::*;
	use crate::{
		builder::ConstraintSystemBuilder, collatz_high_level::CollatzTrace,
		constraint_system::Instance, witness::fill_table_sequential,
	};

	#[test]
	fn test_collatz() {
		let mut cs_builder = ConstraintSystemBuilder::new();
		let sequence_chan = cs_builder.add_channel("sequence");
		let evens_table = EvensTable::new(&mut cs_builder, sequence_chan);
		let odds_table = OddsTable::new(&mut cs_builder, sequence_chan);
		let cs = cs_builder.build();

		let trace = CollatzTrace::generate(3999);
		// TODO: Refactor boundary creation
		let instance = Instance {
			boundaries: vec![
				Boundary {
					values: vec![B128::new(3999)],
					channel_id: sequence_chan,
					direction: FlushDirection::Push,
					multiplicity: 1,
				},
				Boundary {
					values: vec![B128::new(1)],
					channel_id: sequence_chan,
					direction: FlushDirection::Pull,
					multiplicity: 1,
				},
			],
			table_sizes: vec![trace.evens.len(), trace.odds.len()],
		};

		let allocator = Bump::new();
		let witness = cs.build_witness(&allocator, &instance).unwrap();

		// TODO: Maybe we can consolidate these calls too onto WitnessIndex
		let evens_witness = witness.get_table(evens_table.id()).unwrap();
		fill_table_sequential(&evens_table, &trace.evens, evens_witness).unwrap();

		let odds_witness = witness.get_table(odds_table.id());
		fill_table_sequential(&odds_table, &trace.odds, odds_witness).unwrap();

		let compiled_cs = cs.compile(instance).unwrap();

		// TODO: Convert the WitnessIndex into MultilinearExtensionIndex

		// prove/verify with compiled_cs and witness
	}
}
