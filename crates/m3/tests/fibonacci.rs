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
	use binius_core::{
		constraint_system::channel::ChannelId,
		fiat_shamir::HasherChallenger,
		tower::{CanonicalOptimalPackedTowerFamily, CanonicalTowerFamily},
	};
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, PackedExtension,
		PackedFieldIndexable,
	};
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
	use binius_m3::{
		builder::{
			Boundary, Col, ConstraintSystem, FlushDirection, Statement, TableBuilder, TableFiller,
			TableId, TableWitnessSegment, WitnessIndex, B1, B128, B32,
		},
		gadgets::u32::{U32Add, U32AddFlags},
	};
	use bumpalo::Bump;

	use crate::model::{self, FibonacciTrace};

	pub struct FibonacciTable {
		pub id: TableId,
		pub _f0: Col<B32>,
		pub _f1: Col<B32>,
		pub _f2: Col<B32>,
		pub f0_bits: Col<B1, 32>,
		pub f1_bits: Col<B1, 32>,
		pub f2_bits: U32Add,
	}

	impl FibonacciTable {
		pub fn new(cs: &mut ConstraintSystem, fibonacci_pairs: ChannelId) -> Self {
			let mut table = cs.add_table("fibonacci");
			Self::with_table_builder(&mut table, fibonacci_pairs)
		}

		pub fn with_table_builder(table: &mut TableBuilder, fibonacci_pairs: ChannelId) -> Self {
			let f0_bits = table.add_committed("f0_bits");
			let f1_bits = table.add_committed("f1_bits");
			let f2_bits = U32Add::new(
				&mut table.with_namespace("f2_bits"),
				f0_bits,
				f1_bits,
				U32AddFlags {
					expose_final_carry: true,
					..U32AddFlags::default()
				},
			);
			let final_carry = f2_bits.final_carry.expect("expose_final_carry is true");

			table.assert_zero("carry out", final_carry.into());

			let f0 = table.add_packed("f0", f0_bits);
			let f1 = table.add_packed("f1", f1_bits);
			let f2 = table.add_packed("f2", f2_bits.zout);

			table.pull(fibonacci_pairs, [f0, f1]);
			table.push(fibonacci_pairs, [f1, f2]);

			Self {
				id: table.id(),
				_f0: f0,
				_f1: f1,
				_f2: f2,
				f0_bits,
				f1_bits,
				f2_bits,
			}
		}
	}

	impl<P> TableFiller<P> for FibonacciTable
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		type Event = model::FibEvent;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event>,
			witness: &'a mut TableWitnessSegment<P>,
		) -> anyhow::Result<()> {
			{
				let mut f0_bits = witness.get_mut_as(self.f0_bits)?;
				let mut f1_bits = witness.get_mut_as(self.f1_bits)?;

				for (i, event) in rows.enumerate() {
					f0_bits[i] = event.f0;
					f1_bits[i] = event.f1;
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
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		witness
			.fill_table_sequential(&fibonacci_table, &trace.rows)
			.unwrap();

		let compiled_cs = cs
			.compile::<CanonicalOptimalPackedTowerFamily>(&statement)
			.unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&compiled_cs,
			&statement.boundaries,
			&witness,
		)
		.unwrap();
	}

	fn compile_validate_prove_verify(
		cs: &ConstraintSystem,
		statement: &Statement,
		witness: WitnessIndex<PackedType<OptimalUnderlier128b, B128>>,
	) {
		let compiled_cs = cs
			.compile::<CanonicalOptimalPackedTowerFamily>(statement)
			.unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&compiled_cs,
			&statement.boundaries,
			&witness,
		)
		.unwrap();

		const LOG_INV_RATE: usize = 1;
		const SECURITY_BITS: usize = 100;

		let proof = binius_core::constraint_system::prove::<
			OptimalUnderlier128b,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(
			&compiled_cs,
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
		>(&compiled_cs, LOG_INV_RATE, SECURITY_BITS, &statement.boundaries, proof)
		.unwrap();
	}

	#[test]
	fn test_fibonacci_prove_verify_small_table() {
		let mut cs = ConstraintSystem::new();
		let fibonacci_pairs = cs.add_channel("fibonacci_pairs");
		let fibonacci_table = FibonacciTable::new(&mut cs, fibonacci_pairs);
		let trace = FibonacciTrace::generate((0, 1), 1);
		let statement = Statement {
			boundaries: vec![
				Boundary {
					values: vec![B128::new(0), B128::new(1)],
					channel_id: fibonacci_pairs,
					direction: FlushDirection::Push,
					multiplicity: 1,
				},
				Boundary {
					values: vec![B128::new(1), B128::new(2)],
					channel_id: fibonacci_pairs,
					direction: FlushDirection::Pull,
					multiplicity: 1,
				},
			],
			table_sizes: vec![trace.rows.len()],
		};
		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		witness
			.fill_table_sequential(&fibonacci_table, &trace.rows)
			.unwrap();

		compile_validate_prove_verify(&cs, &statement, witness);
	}

	#[test]
	fn test_fibonacci_prove_verify_po2_sized() {
		let mut cs = ConstraintSystem::new();
		let fibonacci_pairs = cs.add_channel("fibonacci_pairs");
		let mut fib_table_builder = cs.add_table("fibonacci");
		fib_table_builder.require_power_of_two_size();
		let fibonacci_table =
			FibonacciTable::with_table_builder(&mut fib_table_builder, fibonacci_pairs);
		let trace = FibonacciTrace::generate((0, 1), 31);

		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		witness
			.fill_table_sequential(&fibonacci_table, &trace.rows)
			.unwrap();

		let statement = Statement {
			boundaries: vec![
				Boundary {
					values: vec![B128::new(0), B128::new(1)],
					channel_id: fibonacci_pairs,
					direction: FlushDirection::Push,
					multiplicity: 1,
				},
				Boundary {
					values: vec![B128::new(2178309), B128::new(3524578)],
					channel_id: fibonacci_pairs,
					direction: FlushDirection::Pull,
					multiplicity: 1,
				},
			],
			table_sizes: witness.table_sizes(),
		};
		compile_validate_prove_verify(&cs, &statement, witness);
	}
}
