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
	P: PackedField<Scalar: TowerField>
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
	use super::*;
}
