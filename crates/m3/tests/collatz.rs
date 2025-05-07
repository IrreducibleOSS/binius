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

			sequence_chan.assert_balanced();
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
		fiat_shamir::HasherChallenger,
		oracle::ShiftVariant,
	};
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, tower::CanonicalTowerFamily,
		underlier::SmallU, Field, PackedExtension, PackedField, PackedFieldIndexable,
		PackedSubfield,
	};
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
	use binius_m3::{
		builder::{
			Col, ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment,
			WitnessIndex, B1, B128, B32,
		},
		gadgets::{
			add::{U32Add, U32AddFlags},
			mul::MulSS32,
		},
	};
	use bumpalo::Bump;

	use super::model;

	/// Table of transitions for even numbers in the Collatz sequence.
	#[derive(Debug)]
	pub struct EvensTable {
		id: TableId,
		even: Col<B1, 32>,
		even_lsb: Col<B1>,
		half: Col<B1, 32>,
		_even_packed: Col<B32>,
		_half_packed: Col<B32>,
	}

	impl EvensTable {
		pub fn new(cs: &mut ConstraintSystem, seq_chan: ChannelId) -> Self {
			let mut table = cs.add_table("evens");

			let even = table.add_committed::<B1, 32>("even");
			let even_lsb = table.add_selected("even_lsb", even, 0);

			table.assert_zero("even_lsb is 0", even_lsb.into());

			// Logical right shift is division by 2
			let half = table.add_shifted::<B1, 32>("half", even, 5, 1, ShiftVariant::LogicalRight);

			let even_packed = table.add_packed::<_, 32, B32, 1>("even_packed", even);
			let half_packed = table.add_packed::<_, 32, B32, 1>("half_packed", half);

			table.pull(seq_chan, [even_packed]);
			table.push(seq_chan, [half_packed]);

			Self {
				id: table.id(),
				even,
				even_lsb,
				half,
				_even_packed: even_packed,
				_half_packed: half_packed,
			}
		}
	}

	impl<P> TableFiller<P> for EvensTable
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		type Event = model::EvensEvent;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event>,
			witness: &'a mut TableWitnessSegment<P>,
		) -> Result<(), anyhow::Error> {
			let mut even = witness.get_mut_as(self.even)?;
			let mut even_lsb = witness.get_mut(self.even_lsb)?;
			let mut half = witness.get_mut_as(self.half)?;

			for (i, event) in rows.enumerate() {
				even[i] = event.val;
				half[i] = event.val >> 1;
			}

			even_lsb.fill(<PackedSubfield<P, B1>>::zero());

			Ok(())
		}
	}

	/// Table of transitions for odd numbers in the Collatz sequence.
	#[derive(Debug)]
	pub struct OddsTable {
		id: TableId,
		odd: Col<B1, 32>,
		odd_lsb: Col<B1>,
		double: Col<B1, 32>,
		carry_bit: Col<B1, 32>,
		triple_plus_one: U32Add,
		_odd_packed: Col<B32>,
		_triple_plus_one_packed: Col<B32>,
	}

	impl OddsTable {
		pub fn new(cs: &mut ConstraintSystem, seq_chan: ChannelId) -> Self {
			let mut table = cs.add_table("odds");

			let odd = table.add_committed::<B1, 32>("odd_bits");
			let odd_lsb = table.add_selected("odd_lsb", odd, 0);

			table.assert_zero("odd_lsb is 1", odd_lsb - B1::ONE);

			// Input times 2
			let double =
				table.add_shifted::<B1, 32>("double_bits", odd, 5, 1, ShiftVariant::LogicalLeft);

			let carry_bit = table.add_constant("carry_bit", decomposed_u32_bits(1));

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

			let odd_packed = table.add_packed::<_, 32, B32, 1>("odd_packed", odd);
			let triple_plus_one_packed =
				table.add_packed::<_, 32, B32, 1>("triple_plus_one_packed", triple_plus_one.zout);

			table.pull(seq_chan, [odd_packed]);
			table.push(seq_chan, [triple_plus_one_packed]);

			MulSS32::new(&mut table);

			Self {
				id: table.id(),
				odd,
				odd_lsb,
				double,
				carry_bit,
				triple_plus_one,
				_odd_packed: odd_packed,
				_triple_plus_one_packed: triple_plus_one_packed,
			}
		}
	}

	fn decomposed_u32_bits(bits: u32) -> [B1; 32] {
		std::array::from_fn(|i| B1::new(SmallU::new(((bits >> i) & 1) as u8)))
	}

	impl<P> TableFiller<P> for OddsTable
	where
		P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1>,
	{
		type Event = model::OddsEvent;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&self,
			rows: impl Iterator<Item = &'a Self::Event>,
			witness: &'a mut TableWitnessSegment<P>,
		) -> Result<(), anyhow::Error> {
			{
				let mut odd = witness.get_mut_as(self.odd)?;
				let mut odd_lsb = witness.get_mut(self.odd_lsb)?;
				let mut double = witness.get_mut_as(self.double)?;
				let mut carry_bit = witness.get_mut_as(self.carry_bit)?;

				for (i, event) in rows.enumerate() {
					odd[i] = event.val;
					double[i] = event.val << 1;
					carry_bit[i] = 1u32;
				}

				odd_lsb.fill(<PackedSubfield<P, B1>>::one());
			}
			self.triple_plus_one.populate(witness)?;
			Ok(())
		}
	}

	fn compile_validate_prove_verify(
		cs: &ConstraintSystem,
		statement: &Statement,
		witness: WitnessIndex<PackedType<OptimalUnderlier128b, B128>>,
	) {
		let compiled_cs = cs.compile(statement).unwrap();
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
	fn test_collatz() {
		use model::CollatzTrace;

		let mut cs = ConstraintSystem::new();
		let collatz_orbit = cs.add_channel("collatz_orbit");
		let evens_table = EvensTable::new(&mut cs, collatz_orbit);
		let odds_table = OddsTable::new(&mut cs, collatz_orbit);

		let initial_val = 3999;
		let trace = CollatzTrace::generate(initial_val);

		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		witness
			.fill_table_sequential(&evens_table, &trace.evens)
			.unwrap();
		witness
			.fill_table_sequential(&odds_table, &trace.odds)
			.unwrap();

		let boundaries = vec![
			Boundary {
				values: vec![B128::new(initial_val.into())],
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
		];
		let statement = Statement {
			boundaries,
			table_sizes: vec![trace.evens.len(), trace.odds.len()],
		};
		compile_validate_prove_verify(&cs, &statement, witness);
	}
}
