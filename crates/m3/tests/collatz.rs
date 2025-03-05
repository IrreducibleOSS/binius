// Copyright 2025 Irreducible Inc.

//! Example of a Collatz M3 arithmetization.
//!
//! See [Collatz] M3 example documentation for further information.
//!
//! [Collatz]: <https://www.binius.xyz/basics/arithmetization/m3/collatz>

mod model {
	use binius_m3::emulate::Channel;

	#[derive(Debug, Default)]
	pub struct CollatzTrace {
		pub evens: Vec<EvensEvent>,
		pub odds: Vec<OddsEvent>,
	}

	impl CollatzTrace {
		pub fn generate(initial_val: u32) -> Self {
			assert_ne!(initial_val, 0);

			let mut trace = CollatzTrace::default();
			let mut val = initial_val;

			while val != 1 {
				if val % 2 == 0 {
					trace.evens.push(EvensEvent { val });
					val /= 2;
				} else {
					trace.odds.push(OddsEvent { val });
					val = 3 * val + 1;
				}
			}

			trace
		}

		pub fn validate(&self, initial_val: u32) {
			let mut sequence_chan = Channel::default();

			// Boundaries
			sequence_chan.push(initial_val);
			sequence_chan.pull(1);

			// Events
			for event in &self.evens {
				event.fire(&mut sequence_chan);
			}
			for event in &self.odds {
				event.fire(&mut sequence_chan);
			}

			assert!(sequence_chan.is_balanced());
		}
	}

	#[derive(Debug, Default, Clone)]
	pub struct EvensEvent {
		pub val: u32,
	}

	impl EvensEvent {
		fn fire(&self, sequence_chan: &mut Channel<u32>) {
			assert_eq!(self.val % 2, 0);

			sequence_chan.pull(self.val);
			sequence_chan.push(self.val / 2);
		}
	}

	#[derive(Debug, Default, Clone)]
	pub struct OddsEvent {
		pub val: u32,
	}

	impl OddsEvent {
		pub fn fire(&self, sequence_chan: &mut Channel<u32>) {
			assert_eq!(self.val % 2, 1);
			let next_val = self
				.val
				.checked_mul(3)
				.and_then(|val| val.checked_add(1))
				.unwrap();

			sequence_chan.pull(self.val);
			sequence_chan.push(next_val);
		}
	}

	#[test]
	fn test_collatz_high_level_validation() {
		use crate::model::CollatzTrace;

		let initial_val = 3999;
		let trace = CollatzTrace::generate(initial_val);
		trace.validate(initial_val);
	}
}

mod arithmetization {
	use binius_core::{
		constraint_system::channel::{Boundary, ChannelId, FlushDirection},
		oracle::ShiftVariant,
	};
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackScalar, underlier::UnderlierType,
	};
	use binius_m3::{
		builder::{
			Col, ConstraintSystem, Statement, TableFiller, TableId, TableWitnessIndexSegment, B1,
			B128, B32,
		},
		gadgets::u32::{U32Add, U32AddFlags},
	};
	use bumpalo::Bump;
	use bytemuck::Pod;

	use super::model;

	/// Table of transitions for even numbers in the Collatz sequence.
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
			let mut table = cs.add_table("evens");

			let even = table.add_committed::<B1, 5>("even");

			// TODO: Check that the bottom bit is 0. We can do this with a selected derived column and
			// an assert_zero constraint.

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
		type Event = model::EvensEvent;

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
			let mut table = cs.add_table("odds");

			let odd = table.add_committed::<B1, 5>("odd_bits");

			// TODO: Check that the bottom bit is 1. We can do this with a selected derived column and
			// an assert_zero constraint.

			// Input times 2
			let double =
				table.add_shifted::<B1, 5>("double_bits", odd, 5, 1, ShiftVariant::LogicalLeft);

			// TODO: Figure out how to add repeating constants (repeating transparents). Basically a
			// multilinear extension of some constant vector, repeating for the number of rows.
			// This shouldn't actually be committed. It should be the carry bit, repeated for each row.
			let carry_bit = table.add_committed::<B1, 5>("carry_bit");

			// Input times 3 + 1
			let triple_plus_one = U32Add::new(
				&mut table.with_namespace("triple_plus_one"),
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
	{
		type Event = model::OddsEvent;

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

	#[test]
	fn test_collatz() {
		use model::CollatzTrace;

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
	}
}
