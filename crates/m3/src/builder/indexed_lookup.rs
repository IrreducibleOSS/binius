// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_core::constraint_system::channel::{Boundary, ChannelId, FlushDirection};
use binius_field::{Field, PackedExtension, PackedField, TowerField};

use super::{
	constraint_system::ConstraintSystem, error::Error, witness::WitnessIndex, B1, B128, B16, B32,
	B64, B8,
};

pub trait IndexedLookup<F: TowerField> {
	/// Binary logarithm of the number of table entries.
	///
	/// This must be at most 32.
	fn log_size(&self) -> usize;

	fn entry_to_index(&self, entry: &[F]) -> usize;
}

pub fn tally<P>(
	cs: &ConstraintSystem<B128>,
	witness: &mut WitnessIndex<P>,
	boundaries: &[Boundary<B128>],
	chan: ChannelId,
	indexed_lookup: &impl IndexedLookup<B128>,
) -> Result<Vec<u32>, Error>
where
	P: PackedField<Scalar = B128>
		+ PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B16>
		+ PackedExtension<B32>
		+ PackedExtension<B64>
		+ PackedExtension<B128>,
{
	let mut counts = vec![0; 1 << indexed_lookup.log_size()];

	// Tally counts from the tables
	for table in &cs.tables {
		for partition in table.partitions.values() {
			for flush in &partition.flushes {
				if flush.channel_id == chan && flush.direction == FlushDirection::Pull {
					if let Some(table_index) = witness.get_table(table.id()) {
						let segment = table_index.full_segment();
						let cols = flush
							.column_indices
							.iter()
							.map(|&col_index| segment.get_dyn(col_index))
							.collect::<Result<Vec<_>, _>>()?;
						let mut elems = vec![B128::ZERO; cols.len()];
						for i in 0..segment.size() {
							for (elem, col) in iter::zip(&mut elems, &cols) {
								*elem = col.get(i);
							}
							let index = indexed_lookup.entry_to_index(&elems);
							counts[index] += 1;
						}
					}
				}
			}
		}
	}

	// Add in counts from boundaries
	for boundary in boundaries {
		if boundary.channel_id == chan && boundary.direction == FlushDirection::Pull {
			let index = indexed_lookup.entry_to_index(&boundary.values);
			counts[index] += 1;
		}
	}

	Ok(counts)
}

#[cfg(test)]
mod tests {
	use std::{iter::repeat_with, sync::atomic::AtomicU32};

	use binius_field::{
		arch::OptimalUnderlier128b,
		as_packed_field::PackedType,
		ext_basis,
		packed::{get_packed_slice, set_packed_slice},
		PackedFieldIndexable,
	};
	use binius_math::MultilinearPoly;
	use bumpalo::Bump;
	use rand::prelude::StdRng;

	use super::*;
	use crate::{
		builder::{
			test_utils::{validate_system_witness, ClosureFiller},
			upcast_col, Col, TableBuilder, TableFiller, TableId, TableWitnessSegment,
		},
		gadgets::lookup::LookupProducer,
	};

	fn merge_incr_cols(
		table: &mut TableBuilder,
		input: Col<B8>,
		carry_in: Col<B1>,
		output: Col<B8>,
		carry_out: Col<B1>,
	) -> Col<B32> {
		let beta_1 = ext_basis::<B32, B8>(1);
		let beta_2_0 = ext_basis::<B32, B8>(2);
		let beta_2_1 = beta_2_0 * ext_basis::<B8, B1>(1);
		table.add_computed(
			"merged",
			upcast_col(input)
				+ upcast_col(output) * beta_1
				+ upcast_col(carry_in) * beta_2_0
				+ upcast_col(carry_out) * beta_2_1,
		)
	}

	fn merge_incr_vals(input: u8, carry_out: bool, output: u8, carry_in: bool) -> u32 {
		((carry_out as u32) << 17)
			| ((carry_in as u32) << 16)
			| ((output as u32) << 8)
			| input as u32
	}

	struct Incr {
		pub input: Col<B8>,
		pub carry_in: Col<B1>,
		pub output: Col<B8>,
		pub carry_out: Col<B1>,
		pub merged: Col<B32>,
	}

	impl Incr {
		fn new(
			table: &mut TableBuilder,
			lookup_chan: ChannelId,
			input: Col<B8>,
			carry_in: Col<B1>,
		) -> Self {
			let output = table.add_committed::<B8, 1>("output");
			let carry_out = table.add_committed::<B1, 1>("carry_out");
			let merged = merge_incr_cols(table, input, carry_in, output, carry_out);

			table.pull(lookup_chan, [merged]);

			Self {
				input,
				carry_in,
				output,
				carry_out,
				merged,
			}
		}

		fn populate<P>(&self, witness: &mut TableWitnessSegment<P>) -> anyhow::Result<()>
		where
			P: PackedFieldIndexable<Scalar = B128>
				+ PackedExtension<B1>
				+ PackedExtension<B8>
				+ PackedExtension<B32>,
		{
			let input = witness.get_as::<u8, _, 1>(self.input)?;
			let carry_in = witness.get(self.carry_in)?;
			let mut output = witness.get_mut_as::<u8, _, 1>(self.output)?;
			let mut carry_out = witness.get_mut(self.carry_out)?;
			let mut merged = witness.get_mut_as::<u32, _, 1>(self.merged)?;

			for i in 0..witness.size() {
				let input_i = input[i];
				let carry_in_bit = bool::from(get_packed_slice(&carry_in, i).val());

				let (output_i, carry_out_bit) = input_i.overflowing_add(carry_in_bit.into());
				output[i] = output_i;
				set_packed_slice(&mut carry_out, i, B1::from(carry_out_bit));
				merged[i] = ((carry_out_bit as u32) << 17)
					| ((carry_in_bit as u32) << 16)
					| ((output_i as u32) << 8)
					| input_i as u32;
				//
				// let index = ((carry_in_bit as usize) << 8) | input_i as usize;
				// lookup_counts[index].fetch_add(1, Ordering::AcqRel);
			}

			Ok(())
		}
	}

	struct IncrLooker {
		pub input: Col<B8>,
		pub carry_in: Col<B1>,
		incr: Incr,
	}

	impl IncrLooker {
		fn new(table: &mut TableBuilder, lookup_chan: ChannelId) -> Self {
			let input = table.add_committed::<B8, 1>("input");
			let carry_in = table.add_committed::<B1, 1>("carry_in");
			let incr = Incr::new(table, lookup_chan, input, carry_in);

			Self {
				input,
				carry_in,
				incr,
			}
		}

		fn populate<'a, P>(
			&self,
			witness: &mut TableWitnessSegment<P>,
			events: impl Iterator<Item = &'a (u8, bool)>,
		) -> anyhow::Result<()>
		where
			P: PackedFieldIndexable<Scalar: TowerField> + PackedExtension<B1> + PackedExtension<B8>,
		{
			{
				let mut input = witness.get_mut_as::<u8, _, 1>(self.input)?;
				let mut carry_in = witness.get_mut(self.carry_in)?;

				for (i, &(input_i, carry_in_bit)) in events.enumerate() {
					input[i] = input_i;
					set_packed_slice(&mut carry_in, i, B1::from(carry_in_bit));
				}
			}

			self.incr.populate(witness)?;
			Ok(())
		}
	}

	struct IncrLookerFiller<'a> {
		table_id: usize,
		incr_looker: &'a IncrLooker,
		lookup_counts: &'a [AtomicU32],
	}

	impl<P> TableFiller<P> for IncrLookerFiller<'_>
	where
		P: PackedField<Scalar: TowerField> + PackedExtension<B1>,
	{
		type Event = (u8, bool);

		fn id(&self) -> TableId {
			self.table_id
		}

		fn fill<'a>(
			&'a self,
			events: impl Iterator<Item = &'a Self::Event>,
			witness: &'a mut TableWitnessSegment<P>,
		) -> anyhow::Result<()> {
			self.incr_looker
				.populate(witness, events, self.lookup_counts)
		}
	}

	struct IncrLookup {
		table_id: TableId,
		merged: Col<B32>,
		lookup_producer: LookupProducer,
	}

	impl IncrLookup {
		fn new(table: &mut TableBuilder, chan: ChannelId, n_multiplicity_bits: usize) -> Self {
			let merged = table.add_committed::<B32, 1>("merged");
			let lookup_producer = LookupProducer::new(table, chan, &[merged], n_multiplicity_bits);
			Self {
				table_id: table.id(),
				merged,
				lookup_producer,
			}
		}
	}

	impl TableFiller for IncrLookup {
		type Event = (u32, u32);

		fn id(&self) -> TableId {
			self.table_id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event> + Clone,
			witness: &'a mut TableWitnessSegment,
		) -> anyhow::Result<()> {
			{
				let mut merged = witness.get_mut_as::<u32, _, 1>(self.merged)?;
				for (merged_i, &(index, _)) in iter::zip(&mut *merged, rows.clone()) {
					let input_i = index % (1 << 8);
					let carry_in_bit = (index >> 8) & 1 == 1;
					let (output_i, carry_out_bit) = input_i.overflowing_add(carry_in_bit.into());
					*merged_i = ((carry_out_bit as u32) << 17)
						| ((carry_in_bit as u32) << 16)
						| (output_i << 8) | input_i;
				}
			}
			self.lookup_producer
				.populate(witness, rows.map(|&(_i, count)| count))?;
			Ok(())
		}
	}

	/// Unit test for a fixed lookup table, which requires counting lookups during witness
	/// generation of the looker tables.
	#[test]
	fn test_fixed_lookup_producer() {
		let mut cs = ConstraintSystem::new();
		let incr_lookup_chan = cs.add_channel("incr lookup");

		let incr_table_log_len = 9;
		let n_multiplicity_bits = 8;

		let mut incr_table = cs.add_table("increment");
		incr_table.require_fixed_size(incr_table_log_len);
		let incr_lookup = IncrLookup::new(&mut incr_table, incr_lookup_chan, n_multiplicity_bits);
		let incr_table_id = incr_table.id();

		let mut looker_1 = cs.add_table("looker 1");
		let looker_1_id = looker_1.id();
		let incr_1 = IncrLooker::new(&mut looker_1, incr_lookup_chan);

		let mut looker_2 = cs.add_table("looker 2");
		let looker_2_id = looker_2.id();
		let incr_2 = IncrLooker::new(&mut looker_2, incr_lookup_chan);

		let looker_1_size = 55;
		let looker_2_size = 66;

		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

		let mut rng = StdRng::seed_from_u64(0);
		let inputs_1 = repeat_with(|| {
			let input = rng.gen::<u8>();
			let carry_in_bit = rng.gen_bool(0.5);
			(input, carry_in_bit)
		})
		.take(looker_1_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_1_id, |inputs, segment| {
					incr_1.populate(segment, inputs)
				}),
				&inputs_1,
			)
			.unwrap();

		let inputs_2 = repeat_with(|| {
			let input = rng.gen::<u8>();
			let carry_in_bit = rng.gen_bool(0.5);
			(input, carry_in_bit)
		})
		.take(looker_2_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_2_id, |inputs, segment| {
					incr_2.populate(segment, inputs)
				}),
				&inputs_2,
			)
			.unwrap();

		let boundary_reads = vec![
			merge_incr_vals(111, false, 111, false),
			merge_incr_vals(111, true, 112, false),
			merge_incr_vals(255, false, 255, false),
			merge_incr_vals(255, true, 0, true),
		];
		let boundaries = boundary_reads
			.into_iter()
			.map(|val| Boundary {
				values: vec![B32::new(val).into()],
				direction: FlushDirection::Pull,
				channel_id: incr_lookup_chan,
				multiplicity: 1,
			})
			.collect::<Vec<_>>();

		let counts = tally(&cs, &witness, &boundaries, incr_lookup_chan, &incr_lookup);

		// witness.fill_table_sequential(incr_table_id).unwrap();
		// let mut segment = table_witness.fill
		// incr_lookup.fill(counts.iter(), &mut segment).unwrap();

		validate_system_witness(cs, witness, boundaries);
	}
}
