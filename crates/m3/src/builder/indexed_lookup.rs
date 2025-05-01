// Copyright 2025 Irreducible Inc.

use std::sync::atomic::AtomicU32;

use binius_core::constraint_system::channel::{Boundary, ChannelId, FlushDirection};
use binius_field::{PackedField, TowerField};

use crate::builder::{ConstraintSystem, WitnessIndex, B128};

pub trait IndexedLookup<F: TowerField> {
	/// Binary logarithm of the number of table entries.
	///
	/// This must be at most 32.
	fn log_size(&self) -> usize;

	fn entry_to_index(&self, entry: &[F]) -> u32;
}

pub fn tally<P>(
	cs: &ConstraintSystem<B128>,
	witness: &mut WitnessIndex<P>,
	boundaries: &[Boundary<B128>],
	chan: ChannelId,
	indexed_lookup: &impl IndexedLookup<B128>,
) -> Vec<u32>
where
	P: PackedField<Scalar = B128>,
{
	for table in cs.tables {
		for partition in table.partitions.values() {
			for flush in partition.flushes {
				if flush.channel_id == chan && flush.direction == FlushDirection::Pull {
					if let Some(table_index) = witness.get_table(table.id()) {
						let segment = table_index.full_segment();
						let cols = flush.column_indices.iter().map(|&col_index| {});
					}
				}
			}
		}
	}
	Vec::new()
}
