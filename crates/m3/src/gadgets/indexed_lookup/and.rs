// Copyright 2025 Irreducible Inc.

use std::{iter, slice};

use binius_core::constraint_system::channel::ChannelId;
use binius_field::{PackedExtension, PackedField, PackedFieldIndexable, PackedSubfield, ext_basis};
use binius_math::{ArithCircuit, ArithExpr};

use crate::{
	builder::{
		B1, B8, B32, B128, Col, Expr, IndexedLookup, TableBuilder, TableFiller, TableId,
		TableWitnessSegment, column::upcast_col, table,
	},
	gadgets::lookup::LookupProducer,
};

/// A gadget that computes the logical AND of two boolean columns using JOLT style
/// lookups.

pub struct AndLookup {
	/// The table ID
	pub table_id: TableId,
	entries_ordered: Col<B32>,
	entries_sorted: Col<B32>,
	lookup_producer: LookupProducer,
}

pub struct And {
	in_a: Col<B8>,
	in_b: Col<B8>,
	output: Col<B8>,
	merged: Col<B32>,
}
impl And {
	fn new(table: &mut TableBuilder, lookup_chan: ChannelId, in_a: Col<B8>, in_b: Col<B8>) -> Self {
		let output = table.add_committed::<B8, 1>("output");
		let merged = merge_and_columns(table, in_a, in_b, output);
		table.pull(lookup_chan, [merged]);
		Self {
			in_a,
			in_b,
			output,
			merged,
		}
	}
}

pub fn merge_and_columns(
	table: &mut TableBuilder,
	in_a: Col<B8>,
	in_b: Col<B8>,
	output: Col<B8>,
) -> Col<B32> {
	table.add_computed(
		"merged",
		upcast_col(in_a)
			+ upcast_col(in_b) * ext_basis::<B32, B8>(1)
			+ upcast_col(output) * ext_basis::<B32, B8>(2),
	)
}

pub fn merge_and_vals(in_a: u8, in_b: u8, output: u8) -> u32 {
	(in_a as u32) | ((in_b as u32) << 8) | ((output as u32) << 16)
}
/// Returns an arithmetic expression that represents the AND operation.
pub fn and_circuit() -> ArithCircuit<B128> {
	// The circuit is a lookup table for the and operation, which takes 2 8-bit inputs and
	// returns a field element which is the result of the bitwise andconcatenated with the inputs.
	let mut circuit = ArithExpr::zero();
	for i in 0..8 {
		circuit += ArithExpr::Var(i) * ArithExpr::Var(i + 8) * ArithExpr::Const(B32::new(1 << i));
		circuit += ArithExpr::Var(i) * ArithExpr::Const(B32::new(1 << (i + 8)));
		circuit += ArithExpr::Var(i + 8) * ArithExpr::Const(B32::new(1 << (i + 16)));
	}
	ArithCircuit::<B32>::from(circuit)
		.try_convert_field()
		.expect("And circuit should convert to B128")
}

impl AndLookup {
	pub fn new(
		table: &mut TableBuilder,
		chan: ChannelId,
		permutation_chan: ChannelId,
		n_multiplicity_bits: usize,
	) -> Self {
		table.require_fixed_size(AndIndexedLookup.log_size());

		// The entries_ordered column is the one that is filled with the lookup table entries.
		let entries_ordered = table.add_fixed("incr_lookup", and_circuit());
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

struct AndIndexedLookup;

impl IndexedLookup<B128> for AndIndexedLookup {
	fn log_size(&self) -> usize {
		16
	}

	fn entry_to_index(&self, entry: &[B128]) -> usize {
		debug_assert_eq!(entry.len(), 1, "AndLookup entry must be a single B128 field");
		let merged_val = entry[0].val() as u32;
		(merged_val & 0xFFFF) as usize
	}

	fn index_to_entry(&self, index: usize, entry: &mut [B128]) {
		debug_assert_eq!(entry.len(), 1, "AndLookup entry must be a single B128 field");
		let in_a = index & 0xFF;
		let in_b = (index >> 8) & 0xFF;
		let output = in_a & in_b;
		let merged = merge_and_vals(in_a as u8, in_b as u8, output as u8);
		entry[0] = B128::from(merged as u128);
	}
}

impl TableFiller for AndLookup {
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
				AndIndexedLookup.index_to_entry(start_index + i, slice::from_mut(&mut entry_128b));
				*col_data_i = B32::try_from(entry_128b).expect("guaranteed by AndIndexedLookup");
			}
		}

		// Fill the entries_sorted column
		{
			let mut entries_sorted = witness.get_scalars_mut(self.entries_sorted)?;
			for (merged_i, &(index, _)) in iter::zip(&mut *entries_sorted, rows.clone()) {
				let mut entry_128b = B128::default();
				AndIndexedLookup.index_to_entry(index, slice::from_mut(&mut entry_128b));
				*merged_i = B32::try_from(entry_128b).expect("guaranteed by AndIndexedLookup");
			}
		}

		self.lookup_producer
			.populate(witness, rows.map(|&(_i, count)| count))?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {

	use super::*;
	use crate::builder::ConstraintSystem;

	#[test]
	fn test_and_lookup() {
		let mut cs: ConstraintSystem<B128> = ConstraintSystem::new();
		let lookup_chan = cs.add_channel("lookup");
		let mut table = cs.add_table("and_lookup");
		let and_lookup = AndLookup::new(&mut table, lookup_chan, lookup_chan, 0);
	}
}
