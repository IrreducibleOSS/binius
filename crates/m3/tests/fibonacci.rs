// Copyright 2025 Irreducible Inc.

//! Example of a Fibonacci M3 arithmetization.
mod model {
	use binius_m3::emulate::Channel;

	#[derive(Debug, Default)]
	pub struct FibonacciTrace {
		pub rows: Vec<FibEvent>,
	}

	impl FibonacciTrace {
		pub fn generate(start: (u32, u32), n: usize) -> Self {
			let mut trace = FibonacciTrace::default();
			let (mut f0, mut f1) = start;
			let mut f2 = f0 + f1;
			trace.rows.push(FibEvent { f0, f1, f2 });

			for _ in 0..n {
				f0 = f1;
				f1 = f2;
				f2 = f0 + f1;
				trace.rows.push(FibEvent { f0, f1, f2 });
			}
			trace
		}

		pub fn validate(&self, start: (u32, u32), end: (u32, u32)) {
			let mut sequence_chan = Channel::default();
			sequence_chan.push(start);
			sequence_chan.pull(end);
			for event in self.rows.iter() {
				event.fire(&mut sequence_chan);
			}
			assert!(sequence_chan.is_balanced());
		}
	}

	#[derive(Debug, Default, Clone)]
	pub struct FibEvent {
		pub f0: u32,
		pub f1: u32,
		pub f2: u32,
	}

	impl FibEvent {
		pub fn fire(&self, sequence_chan: &mut Channel<(u32, u32)>) {
			assert_eq!(self.f0 + self.f1, self.f2);
			sequence_chan.pull((self.f0, self.f1));
			sequence_chan.push((self.f1, self.f2));
		}
	}

	#[test]
	fn test_fibonacci_high_level_validation() {
		use crate::model::FibonacciTrace;

		let start = (0, 1);
		let end = (165580141, 267914296);
		let trace = FibonacciTrace::generate(start, 40);
		trace.validate(start, end);
	}
}

mod arithmetization {
	use binius_core::constraint_system::channel::ChannelId;
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackScalar};
	use binius_m3::{
		builder::{
			upcast_col, Boundary, Col, ConstraintSystem, FlushDirection, Statement, TableFiller,
			TableId, TableWitnessIndexSegment, B1, B128, B32,
		},
		gadgets::u32::{U32Add, U32AddFlags},
	};
	use bumpalo::Bump;
	use bytemuck::Pod;

	use crate::model::{self, FibonacciTrace};

	pub struct FibonacciTable {
		pub id: TableId,
		pub f0: Col<B32>,
		pub f1: Col<B32>,
		pub f2: Col<B32>,
		pub f0_bits: Col<B1, 32>,
		pub f1_bits: Col<B1, 32>,
		pub f2_bits: U32Add,
	}

	impl FibonacciTable {
		pub fn new(cs: &mut ConstraintSystem, fibonacci_pairs: ChannelId) -> Self {
			let mut table = cs.add_table("fibonacci");
			let f0_bits = table.add_committed("f0_bits");
			let f1_bits = table.add_committed("f1_bits");
			let f2_bits = U32Add::new(
				&mut table.with_namespace("f2_bits"),
				f0_bits,
				f1_bits,
				U32AddFlags::default(),
			);

			let f0 = table.add_packed("f0", f0_bits);
			let f1 = table.add_packed("f1", f1_bits);
			let f2 = table.add_packed("f2", f2_bits.zout);

			table.pull(fibonacci_pairs, [upcast_col(f0), upcast_col(f1)]);
			table.push(fibonacci_pairs, [upcast_col(f1), upcast_col(f2)]);

			Self {
				id: table.id(),
				f0,
				f1,
				f2,
				f0_bits,
				f1_bits,
				f2_bits,
			}
		}
	}

	impl<U> TableFiller<U> for FibonacciTable
	where
		U: Pod + PackScalar<B1>,
	{
		type Event = model::FibEvent;

		fn id(&self) -> binius_m3::builder::TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event>,
			witness: &'a mut TableWitnessIndexSegment<U>,
		) -> anyhow::Result<()> {
			{
				let mut f0 = witness.get_mut_as(self.f0)?;
				let mut f1 = witness.get_mut_as(self.f1)?;
				let mut f2 = witness.get_mut_as(self.f2)?;
				let mut f0_bits = witness.get_mut_as(self.f0_bits)?;
				let mut f1_bits = witness.get_mut_as(self.f1_bits)?;

				for (i, event) in rows.enumerate() {
					f0_bits[i] = event.f0;
					f1_bits[i] = event.f1;
					f0[i] = event.f0;
					f1[i] = event.f1;
					f2[i] = event.f2;
				}
			}
			self.f2_bits.populate(witness)?;
			Ok(())
		}
	}

	#[test]
	fn test_fibonacci() {
		let mut cs = ConstraintSystem::new();
		let fibonacci_pairs = cs.add_channel("fibonacci_pairs");
		let fibonacci_table = FibonacciTable::new(&mut cs, fibonacci_pairs);
		let trace = FibonacciTrace::generate((0, 1), 40);
		let statement = Statement {
			boundaries: vec![
				Boundary {
					values: vec![B128::new(0), B128::new(1)],
					channel_id: fibonacci_pairs,
					direction: FlushDirection::Push,
					multiplicity: 1,
				},
				Boundary {
					values: vec![B128::new(165580141), B128::new(267914296)],
					channel_id: fibonacci_pairs,
					direction: FlushDirection::Pull,
					multiplicity: 1,
				},
			],
			table_sizes: vec![trace.rows.len()],
		};
		let allocator = Bump::new();
		let mut witness = cs
			.build_witness::<OptimalUnderlier128b>(&allocator, &statement)
			.unwrap();

		witness
			.fill_table_sequential(&fibonacci_table, &trace.rows)
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
