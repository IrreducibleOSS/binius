// Copyright 2025 Irreducible Inc.

use binius_core::{constraint_system::channel::ChannelId, oracle::ShiftVariant};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
};
use bytemuck::Pod;

use crate::{
	builder::{
		column::Col,
		constraint_system::ConstraintSystem,
		table::TableId,
		types::{B1, B32},
		witness::{TableFiller, TableWitnessIndexSegment},
	},
	collatz_high_level::{EvensEvent, OddsEvent},
	u32::{U32Add, U32AddFlags},
};

#[derive(Debug)]
pub struct EvensTable {
	pub id: TableId,
	pub in_bits: Col<B1, 5>,
	pub out_bits: Col<B1, 5>,
	pub in_val: Col<B32>,
	pub out_val: Col<B32>,
}

impl EvensTable {
	pub fn new(cs: &mut ConstraintSystem, seq_chan: ChannelId) -> Self {
		let table = cs.add_table("evens");

		let in_bits = table.add_committed::<B1, 5>("in_bits");

		// Logical right shift is division by 2
		let out_bits =
			table.add_shifted::<B1, 5>("out_bits", in_bits, 5, 1, ShiftVariant::LogicalRight);

		let in_val = table.add_packed::<_, 5, B32, 0>("in", in_bits);
		let out_val = table.add_packed::<_, 5, B32, 0>("out", out_bits);

		table.pull_one(seq_chan, in_val);
		table.push_one(seq_chan, out_val);

		Self {
			id: table.id(),
			in_bits,
			out_bits,
			in_val,
			out_val,
		}
	}
}

impl<U> TableFiller<U> for EvensTable
where
	U: Pod + PackScalar<B32>,
{
	type Event = EvensEvent;

	fn id(&self) -> TableId {
		self.id
	}

	fn fill(
		&self,
		rows: &[Self::Event],
		witness: &mut TableWitnessIndexSegment<U>,
	) -> Result<(), anyhow::Error> {
		let mut in_val = witness.get_mut_as(self.in_val)?;
		let mut out_val = witness.get_mut_as(self.out_val)?;

		for (i, event) in rows.iter().enumerate() {
			in_val[i] = event.val;
			out_val[i] = event.val / 2;
		}

		Ok(())
	}
}

/// Table of transitions for odd numbers in the Collatz sequence.
#[derive(Debug)]
pub struct OddsTable {
	id: TableId,
	pub in_bits: Col<B1, 5>,
	in_dbl: Col<B1, 5>,
	carry_bit: Col<B1, 5>,
	add_in_x3: U32Add,
	pub in_val: Col<B32>,
	pub out_val: Col<B32>,
}

impl OddsTable {
	pub fn new(cs: &mut ConstraintSystem, seq_chan: ChannelId) -> Self {
		let table = cs.add_table("odds");

		let in_bits = table.add_committed::<B1, 5>("in_bits");

		// Input times 2
		let in_dbl = table.add_shifted::<B1, 5>("in_dbl", in_bits, 5, 1, ShiftVariant::LogicalLeft);

		// TODO: Figure out how to add repeating constants (repeating transparents). Basically a
		// multilinear extension of some constant vector, repeating for the number of rows.
		// This shouldn't actually be committed. It should be the carry bit, repeated for each row.
		let carry_bit = table.add_committed::<B1, 5>("carry_bit");

		// Input times 3 + 1
		let add_in_x3 = U32Add::new(
			table,
			in_dbl,
			in_bits,
			U32AddFlags {
				carry_in_bit: Some(carry_bit),
				..U32AddFlags::default()
			},
		);

		let in_val = table.add_packed::<_, 5, B32, 0>("in", in_bits);
		let out_val = table.add_packed::<_, 5, B32, 0>("out", add_in_x3.zout);

		table.pull_one(seq_chan, in_val);
		table.push_one(seq_chan, out_val);

		Self {
			id: table.id(),
			in_bits,
			in_dbl,
			carry_bit,
			add_in_x3,
			in_val,
			out_val,
		}
	}
}

impl<U: UnderlierType> TableFiller<U> for OddsTable
where
	U: Pod + PackScalar<B1> + PackScalar<B32>,
	PackedType<U, B1>: Pod,
	PackedType<U, B32>: Pod,
{
	type Event = OddsEvent;

	fn id(&self) -> TableId {
		self.id
	}

	fn fill(
		&self,
		rows: &[Self::Event],
		witness: &mut TableWitnessIndexSegment<U>,
	) -> Result<(), anyhow::Error> {
		{
			let mut in_val = witness.get_mut_as(self.in_val)?;
			let mut in_dbl = witness.get_mut_as(self.in_dbl)?;
			let mut carry_bit = witness.get_mut_as(self.carry_bit)?;

			for (i, event) in rows.iter().enumerate() {
				in_val[i] = event.val;
				(in_dbl[i], _) = event.val.overflowing_shl(1);
				carry_bit[i] = 1u32;
			}
		}
		self.add_in_x3.populate(witness)?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::channel::{Boundary, FlushDirection};
	use binius_field::arch::OptimalUnderlier128b;
	use bumpalo::Bump;

	use super::*;
	use crate::{
		builder::{constraint_system::ConstraintSystem, statement::Statement, types::B128},
		collatz_high_level::CollatzTrace,
	};

	#[test]
	fn test_collatz() {
		let mut cs = ConstraintSystem::new();
		let sequence_chan = cs.add_channel("sequence");
		let evens_table = EvensTable::new(&mut cs, sequence_chan);
		let odds_table = OddsTable::new(&mut cs, sequence_chan);

		let trace = CollatzTrace::generate(3999);
		// TODO: Refactor boundary creation
		let instance = Statement {
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
		let mut witness = cs
			.build_witness::<OptimalUnderlier128b>(&allocator, &instance)
			.unwrap();

		witness
			.fill_table_sequential(&evens_table, &trace.evens)
			.unwrap();
		witness
			.fill_table_sequential(&odds_table, &trace.odds)
			.unwrap();

		let compiled_cs = cs.compile(&instance).unwrap();
		let witness = witness.into_multilinear_extension_index::<B128>();

		binius_core::constraint_system::validate::validate_witness(
			&compiled_cs,
			&instance.boundaries,
			&witness,
		)
		.unwrap();

		// prove/verify with compiled_cs and witness
	}
}
