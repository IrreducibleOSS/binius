// Copyright 2025 Irreducible Inc.

use std::sync::atomic::AtomicU32;

use binius_core::constraint_system::channel::{Boundary, ChannelId};

use crate::builder::{ConstraintSystem, WitnessIndex};

pub trait LookupTable {
	type Entry: Clone;

	/// Binary logarithm of the number of table entries.
	///
	/// This must be at most 32.
	fn log_size(&self) -> usize;

	fn entry_to_index(&self, entry: Self::Entry) -> u32;
}

pub fn tally<U, F>(
	cs: &ConstraintSystem<F>,
	witness: &WitnessIndex<U, F>,
	statement: &[Boundary<F>],
	chan: ChannelId,
) -> Vec<AtomicU32> {
	// Identify all tables with pulls from the channel.
}
