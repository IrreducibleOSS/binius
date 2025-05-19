// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_core::constraint_system::channel::{Boundary, ChannelId, FlushDirection};
use binius_field::{Field, PackedExtension, PackedField, TowerField};

use super::{
	B1, B8, B16, B32, B64, B128, constraint_system::ConstraintSystem, error::Error,
	witness::WitnessIndex,
};

/// Indexed lookup tables are fixed-size tables where every entry is easily determined by its
/// index.
///
/// Indexed lookup tables cover a large and useful class of tables, such as lookup tables for
/// bitwise operations, addition of small integer values, multiplication, etc. The entry encodes
/// an input and an output, where the index encodes the input. For example, a bitwise AND table
/// would have 2 8-bit input values and one 8-bit output value. The index encodes the input by
/// concatenating the 8-bit inputs into a 16-bit unsigned integer.
///
/// This trait helps to count the number of times a table, which is already filled, reads from a
/// lookup table. See the documentation for [`tally`] for more information.
pub trait IndexedLookup<F: TowerField> {
	/// Binary logarithm of the number of table entries.
	fn log_size(&self) -> usize;

	/// Encode a table entry as a table index.
	fn entry_to_index(&self, entry: &[F]) -> usize;

	/// Decode a table index to an entry.
	fn index_to_entry(&self, index: usize, entry: &mut [F]);
}

/// Determine the read counts of each entry in an indexed lookup table.
///
/// Before a lookup table witness can be filled, the number of times each entry is read must be
/// known. Reads from indexed lookup tables are a special case where the counts are difficult to
/// track during emulation, because the use of the lookup tables is an arithmetization detail. For
/// example, emulation of the system model should not need to know whether integer additions within
/// a table are constraint using zero constraints or a lookup table for the limbs. In most cases of
/// practical interest, the lookup table is indexed.
///
/// The method to tally counts is to scan all tables in the constraint system and boundaries
/// values, and identify those that pull from the lookup table's channel. Then we iterate over the
/// values read from the table and count all the indices.
///
/// ## Returns
///
/// A vector of counts, whose length is equal to `1 << indexed_lookup.log_size()`.
pub fn tally<P>(
	cs: &ConstraintSystem<B128>,
	// TODO: This doesn't actually need mutable access. But must of the WitnessIndex methods only
	// allow mutable access.
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
		if let Some(table_index) = witness.get_table(table.id()) {
			for partition in table.partitions.values() {
				for flush in &partition.flushes {
					if flush.channel_id == chan && flush.direction == FlushDirection::Pull {
						let table_size = table_index.size();
						// TODO: This should be parallelized, which is pretty tricky.
						let segment = table_index.full_segment();
						let cols = flush
							.columns
							.iter()
							.map(|&col_index| segment.get_dyn(col_index))
							.collect::<Result<Vec<_>, _>>()?;

						if !flush.selectors.is_empty() {
							// TODO: check flush selectors
							todo!("tally does not support selected table reads yet");
						}

						let mut elems = vec![B128::ZERO; cols.len()];
						// It's important that this is only the table size, not the full segment
						// size. The entries after the table size are not flushed.
						for i in 0..table_size {
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
	use std::{cmp::Reverse, iter::repeat_with, slice};

	use binius_field::{
		PackedFieldIndexable,
		arch::OptimalUnderlier,
		ext_basis,
		packed::{get_packed_slice, set_packed_slice},
	};
	use bumpalo::Bump;
	use itertools::Itertools;
	use rand::prelude::{Rng, SeedableRng, StdRng};

	use super::*;
	use crate::{
		builder::{
			Col, TableBuilder, TableFiller, TableId, TableWitnessSegment,
			test_utils::{ClosureFiller, validate_system_witness},
			upcast_col,
		},
		gadgets::lookup::LookupProducer,
	};

	/// Unit test for a fixed lookup table, which requires counting lookups during witness
	/// generation of the looker tables.
	#[test]
	fn test_fixed_lookup_producer() {
		let mut cs = ConstraintSystem::new();
		let incr_lookup_chan = cs.add_channel("incr lookup");
		let incr_lookup_perm_chan = cs.add_channel("incr lookup permutation");

		let n_multiplicity_bits = 8;

		let mut incr_table = cs.add_table("increment");
		let incr_lookup = IncrLookup::new(
			&mut incr_table,
			incr_lookup_chan,
			incr_lookup_perm_chan,
			n_multiplicity_bits,
		);

		let mut looker_1 = cs.add_table("looker 1");
		let looker_1_id = looker_1.id();
		let incr_1 = IncrLooker::new(&mut looker_1, incr_lookup_chan);

		let mut looker_2 = cs.add_table("looker 2");
		let looker_2_id = looker_2.id();
		let incr_2 = IncrLooker::new(&mut looker_2, incr_lookup_chan);

		let looker_1_size = 5;
		let looker_2_size = 6;

		let allocator = Bump::new();
		let mut witness = WitnessIndex::new(&cs, &allocator);

		let mut rng = StdRng::seed_from_u64(0);
		let inputs_1 = repeat_with(|| {
			let input = rng.r#gen::<u8>();
			let carry_in_bit = rng.gen_bool(0.5);
			(input, carry_in_bit)
		})
		.take(looker_1_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_1_id, |inputs, segment| {
					incr_1.populate(segment, inputs.iter().copied())
				}),
				&inputs_1,
			)
			.unwrap();

		let inputs_2 = repeat_with(|| {
			let input = rng.r#gen::<u8>();
			let carry_in_bit = rng.gen_bool(0.5);
			(input, carry_in_bit)
		})
		.take(looker_2_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_2_id, |inputs, segment| {
					incr_2.populate(segment, inputs.iter().copied())
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

		// Tally the lookup counts from the looker tables
		let counts =
			tally(&cs, &mut witness, &boundaries, incr_lookup_chan, &IncrIndexedLookup).unwrap();

		// Fill the lookup table with the sorted counts
		let sorted_counts = counts
			.into_iter()
			.enumerate()
			.sorted_by_key(|(_, count)| Reverse(*count))
			.collect::<Vec<_>>();

		witness
			.fill_table_sequential(&incr_lookup, &sorted_counts)
			.unwrap();

		validate_system_witness::<OptimalUnderlier>(&cs, witness, boundaries);
	}

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

	fn merge_incr_vals(input: u8, carry_in: bool, output: u8, carry_out: bool) -> u32 {
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
			P: PackedFieldIndexable<Scalar = B128>
				+ PackedExtension<B1>
				+ PackedExtension<B8>
				+ PackedExtension<B32>,
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

	struct IncrLookup {
		table_id: TableId,
		entries_ordered: Col<B32>,
		entries_sorted: Col<B32>,
		lookup_producer: LookupProducer,
	}

	impl IncrLookup {
		fn new(
			table: &mut TableBuilder,
			chan: ChannelId,
			permutation_chan: ChannelId,
			n_multiplicity_bits: usize,
		) -> Self {
			table.require_fixed_size(IncrIndexedLookup.log_size());
			// TODO: Create the arithmetic circuit for this and define it as a fixed column.
			let entries_ordered = table.add_committed::<B32, 1>("entries_ordered");
			let entries_sorted = table.add_committed::<B32, 1>("entries_sorted");

			// Use flush to check that entries_sorted is a permutation of entries_ordered.
			table.push(permutation_chan, [entries_ordered]);
			table.pull(permutation_chan, [entries_sorted]);

			let lookup_producer =
				LookupProducer::new(table, chan, &[entries_sorted], n_multiplicity_bits);
			Self {
				table_id: table.id(),
				entries_ordered,
				entries_sorted,
				lookup_producer,
			}
		}
	}

	// TODO: It seems very possible to make a generic table filler for indexed lookup tables.
	impl TableFiller for IncrLookup {
		// Tuple of index and count
		type Event = (usize, u32);

		fn id(&self) -> TableId {
			self.table_id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event> + Clone,
			witness: &'a mut TableWitnessSegment,
		) -> anyhow::Result<()> {
			// Fill the entries_ordered column
			{
				let mut col_data = witness.get_scalars_mut(self.entries_ordered)?;
				let start_index = witness.index() << witness.log_size();
				for (i, col_data_i) in col_data.iter_mut().enumerate() {
					let mut entry_128b = B128::default();
					IncrIndexedLookup
						.index_to_entry(start_index + i, slice::from_mut(&mut entry_128b));
					*col_data_i =
						B32::try_from(entry_128b).expect("guaranteed by IncrIndexedLookup");
				}
			}

			// Fill the entries_sorted column
			{
				let mut entries_sorted = witness.get_scalars_mut(self.entries_sorted)?;
				for (merged_i, &(index, _)) in iter::zip(&mut *entries_sorted, rows.clone()) {
					let mut entry_128b = B128::default();
					IncrIndexedLookup.index_to_entry(index, slice::from_mut(&mut entry_128b));
					*merged_i = B32::try_from(entry_128b).expect("guaranteed by IncrIndexedLookup");
				}
			}

			self.lookup_producer
				.populate(witness, rows.map(|&(_i, count)| count))?;
			Ok(())
		}
	}

	struct IncrIndexedLookup;

	impl IndexedLookup<B128> for IncrIndexedLookup {
		fn log_size(&self) -> usize {
			// Input is an 8-bit value plus 1-bit carry-in
			8 + 1
		}

		fn entry_to_index(&self, entry: &[B128]) -> usize {
			debug_assert_eq!(entry.len(), 1);
			let merged_val = entry[0].val() as u32;
			let input = merged_val & 0xFF;
			let carry_in_bit = (merged_val >> 16) & 1 == 1;
			(carry_in_bit as usize) << 8 | input as usize
		}

		fn index_to_entry(&self, index: usize, entry: &mut [B128]) {
			debug_assert_eq!(entry.len(), 1);
			let input = (index % (1 << 8)) as u8;
			let carry_in_bit = (index >> 8) & 1 == 1;
			let (output, carry_out_bit) = input.overflowing_add(carry_in_bit.into());
			let entry_u32 = merge_incr_vals(input, carry_in_bit, output, carry_out_bit);
			entry[0] = B32::new(entry_u32).into();
		}
	}
}
