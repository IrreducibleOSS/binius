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
						// It's important that this is only the unpacked table size(rows * values
						// per row in the partition), not the full segment size. The entries
						// after the table size are not flushed.
						for i in 0..table_size * partition.values_per_row {
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
