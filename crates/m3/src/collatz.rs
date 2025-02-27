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
	pub even: Col<B1, 5>,
	pub half: Col<B1, 5>,
	pub even_packed: Col<B32>,
	pub half_packed: Col<B32>,
}

impl EvensTable {
	pub fn new(cs: &mut ConstraintSystem, seq_chan: ChannelId) -> Self {
		let table = cs.add_table("evens");

		let even = table.add_committed::<B1, 5>("even");

		// Logical right shift is division by 2
		let half = table.add_shifted::<B1, 5>("half", even, 5, 1, ShiftVariant::LogicalRight);

		let even_packed = table.add_packed::<_, 5, B32, 0>("even_packed", even);
		let half_packed = table.add_packed::<_, 5, B32, 0>("half_packed", half);

		table.pull_one(seq_chan, even_packed);
		table.push_one(seq_chan, half_packed);

		Self {
			id: table.id(),
			even,
			half,
			even_packed,
			half_packed,
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
		let mut even = witness.get_mut_as(self.even)?;
		let mut half = witness.get_mut_as(self.half)?;
		let mut even_packed = witness.get_mut_as(self.even_packed)?;
		let mut half_packed = witness.get_mut_as(self.half_packed)?;

		for (i, event) in rows.iter().enumerate() {
			even[i] = event.val;
			half[i] = event.val >> 1;
			even_packed[i] = event.val;
			half_packed[i] = event.val >> 1;
		}

		Ok(())
	}
}

/// Table of transitions for odd numbers in the Collatz sequence.
#[derive(Debug)]
pub struct OddsTable {
	id: TableId,
	pub odd: Col<B1, 5>,
	double: Col<B1, 5>,
	carry_bit: Col<B1, 5>,
	triple_plus_one: U32Add,
	pub odd_packed: Col<B32>,
	pub triple_plus_one_packed: Col<B32>,
}

impl OddsTable {
	pub fn new(cs: &mut ConstraintSystem, seq_chan: ChannelId) -> Self {
		let table = cs.add_table("odds");

		let odd = table.add_committed::<B1, 5>("odd_bits");

		// Input times 2
		let double =
			table.add_shifted::<B1, 5>("double_bits", odd, 5, 1, ShiftVariant::LogicalLeft);

		// TODO: Figure out how to add repeating constants (repeating transparents). Basically a
		// multilinear extension of some constant vector, repeating for the number of rows.
		// This shouldn't actually be committed. It should be the carry bit, repeated for each row.
		let carry_bit = table.add_committed::<B1, 5>("carry_bit");

		// Input times 3 + 1
		let triple_plus_one = U32Add::new(
			table,
			double,
			odd,
			U32AddFlags {
				carry_in_bit: Some(carry_bit),
				..U32AddFlags::default()
			},
		);

		let odd_packed = table.add_packed::<_, 5, B32, 0>("odd_packed", odd);
		let triple_plus_one_packed =
			table.add_packed::<_, 5, B32, 0>("triple_plus_one_packed", triple_plus_one.zout);

		table.pull_one(seq_chan, odd_packed);
		table.push_one(seq_chan, triple_plus_one_packed);

		Self {
			id: table.id(),
			odd,
			double,
			carry_bit,
			triple_plus_one,
			odd_packed,
			triple_plus_one_packed,
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
			let mut odd_packed = witness.get_mut_as(self.odd_packed)?;
			let mut triple_plus_one_packed = witness.get_mut_as(self.triple_plus_one_packed)?;

			let mut odd = witness.get_mut_as(self.odd)?;
			let mut double = witness.get_mut_as(self.double)?;
			let mut carry_bit = witness.get_mut_as(self.carry_bit)?;

			for (i, event) in rows.iter().enumerate() {
				odd_packed[i] = event.val;
				triple_plus_one_packed[i] = 3 * event.val + 1;

				odd[i] = event.val;
				double[i] = event.val << 1;
				carry_bit[i] = 1u32;
			}
		}
		self.triple_plus_one.populate(witness)?;
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
		let collatz_orbit = cs.add_channel("collatz_orbit");
		let evens_table = EvensTable::new(&mut cs, collatz_orbit);
		let odds_table = OddsTable::new(&mut cs, collatz_orbit);

		let trace = CollatzTrace::generate(3999);
		// TODO: Refactor boundary creation
		let statement = Statement {
			boundaries: vec![
				Boundary {
					values: vec![B128::new(3999)],
					channel_id: collatz_orbit,
					direction: FlushDirection::Push,
					multiplicity: 1,
				},
				Boundary {
					values: vec![B128::new(1)],
					channel_id: collatz_orbit,
					direction: FlushDirection::Pull,
					multiplicity: 1,
				},
			],
			table_sizes: vec![trace.evens.len(), trace.odds.len()],
		};
		let allocator = Bump::new();
		let mut witness = cs
			.build_witness::<OptimalUnderlier128b>(&allocator, &statement)
			.unwrap();

		witness
			.fill_table_sequential(&evens_table, &trace.evens)
			.unwrap();
		witness
			.fill_table_sequential(&odds_table, &trace.odds)
			.unwrap();

		let compiled_cs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index::<B128>(&statement);

		binius_core::constraint_system::validate::validate_witness(
			&compiled_cs,
			&statement.boundaries,
			&witness,
		)
		.unwrap();

		// prove/verify with compiled_cs and witness
	}
}
