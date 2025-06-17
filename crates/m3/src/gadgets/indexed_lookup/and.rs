// Copyright 2025 Irreducible Inc.

/// This module provides gadgets for performing indexed lookup operations for the bitwise AND
/// of two 8-bit values, using lookup tables. It includes types and functions for constructing,
/// populating, and testing AND lookup tables and their associated circuits.
use std::{iter, slice};

use binius_core::constraint_system::channel::ChannelId;
use binius_field::ext_basis;
use binius_math::{ArithCircuit, ArithExpr};

use crate::{
	builder::{
		B8, B32, B128, Col, IndexedLookup, TableBuilder, TableFiller, TableId, TableWitnessSegment,
		column::upcast_col,
	},
	gadgets::lookup::LookupProducer,
};

/// A gadget that computes the logical AND of two boolean columns using a lookup table.
///
/// This struct holds columns for an 8-bit AND operation where:
/// - `entries_ordered` is the fixed column containing all possible AND table entries
/// - `entries_sorted` is a committed column for sorted entries
/// - `lookup_producer` manages lookup multiplicities and constraints
pub struct BitAndLookup {
	/// The table ID
	pub table_id: TableId,
	entries_ordered: Col<B32>,
	entries_sorted: Col<B32>,
	lookup_producer: LookupProducer,
}

pub struct BitAnd<const V: usize = 1> {
	/// Input column A (8 bits)
	in_a: Col<B8, V>,
	/// Input column B (8 bits)
	in_b: Col<B8, V>,
	/// Output column (8 bits), result of in_a & in_b
	pub output: Col<B8, V>,
	/// Merged column for lookup (32 bits)
	merged: Col<B32, V>,
}
/// Constructs a new bitwise-AND gadget, registering the necessary columns in the table.
impl<const V: usize> BitAnd<V> {
	///
	/// # Arguments
	/// * `table` - The table builder to register columns with.
	/// * `lookup_chan` - The channel for lookup operations.
	/// * `in_a` - The first input column (8 bits).
	/// * `in_b` - The second input column (8 bits).
	///
	/// # Returns
	/// An `And` struct with all columns set up.
	pub fn new(
		table: &mut TableBuilder,
		lookup_chan: ChannelId,
		in_a: Col<B8, V>,
		in_b: Col<B8, V>,
	) -> Self {
		let output = table.add_committed::<B8, V>("output");
		let merged = merge_and_columns(table, in_a, in_b, output);
		table.read(lookup_chan, [merged]);
		Self {
			in_a,
			in_b,
			output,
			merged,
		}
	}

	/// Populates the witness segment for this AND operation.
	///
	/// # Arguments
	/// * `witness` - The witness segment to populate.
	///
	/// # Returns
	/// `Ok(())` if successful, or an error otherwise.
	pub fn populate(&self, witness: &mut TableWitnessSegment) -> anyhow::Result<()> {
		let in_a_col = witness.get_scalars(self.in_a)?;
		let in_b_col = witness.get_scalars(self.in_b)?;
		let mut output_col: std::cell::RefMut<'_, [B8]> = witness.get_scalars_mut(self.output)?;
		let mut merged_col: std::cell::RefMut<'_, [B32]> = witness.get_scalars_mut(self.merged)?;

		for i in 0..witness.size() {
			let in_a = in_a_col[i].val();
			let in_b = in_b_col[i].val();
			let output = in_a & in_b;
			output_col[i] = output.into();

			// Merge the values into a single u32
			merged_col[i] = merge_bitand_vals(in_a, in_b, output).into();
		}
		Ok(())
	}
}

/// Merges the input and output columns into a single B32 column for lookup.
pub fn merge_and_columns<const V: usize>(
	table: &mut TableBuilder,
	in_a: Col<B8, V>,
	in_b: Col<B8, V>,
	output: Col<B8, V>,
) -> Col<B32, V> {
	table.add_computed(
		"merged",
		upcast_col(in_a)
			+ upcast_col(in_b) * ext_basis::<B32, B8>(1)
			+ upcast_col(output) * ext_basis::<B32, B8>(2),
	)
}

/// Merges the input and output values into a single u32 for lookup.
pub fn merge_bitand_vals(in_a: u8, in_b: u8, output: u8) -> u32 {
	(in_a as u32) | ((in_b as u32) << 8) | ((output as u32) << 16)
}

/// Returns an arithmetic expression that represents the bitwise-AND operation as a lookup circuit.
/// The circuit encodes input A, input B, and output into a single value.
pub fn bitand_circuit() -> ArithCircuit<B128> {
	// The circuit is a lookup table for the and operation, which takes 2 8-bit inputs and
	// returns a field element which is the result of the bitwise andconcatenated with the inputs.
	let mut circuit = ArithExpr::zero();
	for i in 0..8 {
		circuit += ArithExpr::Var(i) * ArithExpr::Const(B32::new(1 << i));
		circuit += ArithExpr::Var(i + 8) * ArithExpr::Const(B32::new(1 << (i + 8)));
		circuit +=
			ArithExpr::Var(i) * ArithExpr::Var(i + 8) * ArithExpr::Const(B32::new(1 << (i + 16)));
	}
	ArithCircuit::<B32>::from(circuit)
		.try_convert_field()
		.expect("And circuit should convert to B128")
}

impl BitAndLookup {
	/// Constructs a new AND lookup table.
	///
	/// # Arguments
	/// * `table` - The table builder.
	/// * `chan` - The lookup channel.
	/// * `permutation_chan` - The channel for permutation checks.
	/// * `n_multiplicity_bits` - Number of bits for multiplicity.
	pub fn new(
		table: &mut TableBuilder,
		chan: ChannelId,
		permutation_chan: ChannelId,
		n_multiplicity_bits: usize,
	) -> Self {
		table.require_fixed_size(BitAndIndexedLookup.log_size());

		// The entries_ordered column is the one that is filled with the lookup table entries.
		let entries_ordered = table.add_fixed("bitand_lookup", bitand_circuit());
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

/// The 2 columns that are the inputs to the AND operation, and the gadget exposes an output column
/// that corresponds to the bitwise AND of the two inputs.
pub struct BitAndLooker {
	/// Input column A (8 bits)
	pub in_a: Col<B8>,
	/// Input column B (8 bits)
	pub in_b: Col<B8>,
	/// Internal AND gadget
	and: BitAnd,
}

impl BitAndLooker {
	/// Constructs a new AND looker, registering columns in the table.
	pub fn new(table: &mut TableBuilder, lookup_chan: ChannelId) -> Self {
		let in_a = table.add_committed::<B8, 1>("in_a");
		let in_b = table.add_committed::<B8, 1>("in_b");
		// Create the And gadget which will compute the AND of in_a and in_b
		let and = BitAnd::new(table, lookup_chan, in_a, in_b);
		Self { in_a, in_b, and }
	}

	/// Populates the witness segment for a sequence of (in_a, in_b) events.
	pub fn populate<'a>(
		&self,
		witness: &'a mut TableWitnessSegment,
		inputs: impl Iterator<Item = &'a (u8, u8)> + Clone,
	) -> anyhow::Result<()> {
		{
			let mut in_a_col: std::cell::RefMut<'_, [u8]> = witness.get_mut_as(self.in_a)?;
			let mut in_b_col: std::cell::RefMut<'_, [u8]> = witness.get_mut_as(self.in_b)?;

			for (i, &(in_a, in_b)) in inputs.enumerate() {
				in_a_col[i] = in_a;
				in_b_col[i] = in_b;
			}
		}

		self.and.populate(witness)?;
		Ok(())
	}
}

/// Internal struct for indexed lookup logic for AND operations.
pub struct BitAndIndexedLookup;

impl IndexedLookup<B128> for BitAndIndexedLookup {
	/// Returns the log2 size of the table (16 for 8 bits + 8 bits).
	fn log_size(&self) -> usize {
		16
	}

	/// Converts a table entry to its index.
	fn entry_to_index(&self, entry: &[B128]) -> usize {
		debug_assert_eq!(entry.len(), 1, "AndLookup entry must be a single B128 field");
		let merged_val = entry[0].val() as u32;
		(merged_val & 0xFFFF) as usize
	}

	/// Converts an index to a table entry.
	fn index_to_entry(&self, index: usize, entry: &mut [B128]) {
		debug_assert_eq!(entry.len(), 1, "AndLookup entry must be a single B128 field");
		let in_a = index & 0xFF;
		let in_b = (index >> 8) & 0xFF;
		let output = in_a & in_b;
		let merged = merge_bitand_vals(in_a as u8, in_b as u8, output as u8);
		entry[0] = B128::from(merged as u128);
	}
}

/// Implements filling for the AND lookup table.
impl TableFiller for BitAndLookup {
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
				BitAndIndexedLookup
					.index_to_entry(start_index + i, slice::from_mut(&mut entry_128b));
				*col_data_i = B32::try_from(entry_128b).expect("guaranteed by BitAndIndexedLookup");
			}
		}

		// Fill the entries_sorted column
		{
			let mut entries_sorted = witness.get_scalars_mut(self.entries_sorted)?;
			for (merged_i, &(index, _)) in iter::zip(&mut *entries_sorted, rows.clone()) {
				let mut entry_128b = B128::default();
				BitAndIndexedLookup.index_to_entry(index, slice::from_mut(&mut entry_128b));
				*merged_i = B32::try_from(entry_128b).expect("guaranteed by BitAndIndexedLookup");
			}
		}

		self.lookup_producer
			.populate(witness, rows.map(|&(_i, count)| count))?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	//! Tests for the AND indexed lookup gadgets.

	use std::{cmp::Reverse, iter::repeat_with};

	use binius_compute::cpu::alloc::CpuComputeAllocator;
	use binius_core::constraint_system::channel::{Boundary, FlushDirection};
	use binius_field::arch::OptimalUnderlier;
	use itertools::Itertools;
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::builder::{
		ConstraintSystem, WitnessIndex, tally,
		test_utils::{ClosureFiller, validate_system_witness},
	};

	#[test]
	fn test_and_lookup() {
		let mut cs: ConstraintSystem<B128> = ConstraintSystem::new();
		let lookup_chan = cs.add_channel("lookup");
		let permutation_chan = cs.add_channel("permutation");
		let mut and_table = cs.add_table("bitand_lookup");
		let n_multiplicity_bits = 8;

		let bitand_lookup =
			BitAndLookup::new(&mut and_table, lookup_chan, permutation_chan, n_multiplicity_bits);
		let mut and_looker = cs.add_table("bitand_looker");

		let bitand_1 = BitAndLooker::new(&mut and_looker, lookup_chan);

		let looker_1_size = 5;
		let looker_id = and_looker.id();

		let mut allocator = CpuComputeAllocator::new(1 << 16);
		let allocator = allocator.into_bump_allocator();
		let mut witness = WitnessIndex::new(&cs, &allocator);

		let mut rng = StdRng::seed_from_u64(0);
		let inputs_1 = repeat_with(|| {
			let in_a = rng.random::<u8>();
			let in_b = rng.random::<u8>();
			(in_a, in_b)
		})
		.take(looker_1_size)
		.collect::<Vec<_>>();

		witness
			.fill_table_parallel(
				&ClosureFiller::new(looker_id, |inputs, segment| {
					bitand_1.populate(segment, inputs.iter().copied())
				}),
				&inputs_1,
			)
			.unwrap();

		let boundary_reads = (0..5)
			.map(|_| {
				let in_a = rng.random::<u8>();
				let in_b = rng.random::<u8>();
				merge_bitand_vals(in_a, in_b, in_a & in_b)
			})
			.collect::<Vec<_>>();

		let boundaries = boundary_reads
			.into_iter()
			.map(|val| Boundary {
				values: vec![B32::new(val).into()],
				direction: FlushDirection::Pull,
				channel_id: lookup_chan,
				multiplicity: 1,
			})
			.collect::<Vec<_>>();

		// Tally the lookup counts from the looker tables
		let counts =
			tally(&cs, &mut witness, &boundaries, lookup_chan, &BitAndIndexedLookup).unwrap();

		// Fill the lookup table with the sorted counts
		let sorted_counts = counts
			.into_iter()
			.enumerate()
			.sorted_by_key(|(_, count)| Reverse(*count))
			.collect::<Vec<_>>();

		witness
			.fill_table_parallel(&bitand_lookup, &sorted_counts)
			.unwrap();

		validate_system_witness::<OptimalUnderlier>(&cs, witness, boundaries);
	}
}
