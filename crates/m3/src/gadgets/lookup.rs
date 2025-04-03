// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_core::constraint_system::channel::ChannelId;
use binius_field::{ExtensionField, PackedExtension, PackedField, PackedSubfield, TowerField};
use itertools::Itertools;

use crate::builder::{Col, FlushOpts, TableBuilder, TableWitnessSegment, B1, B128};

/// A lookup producer gadget is used to create a lookup table.
///
/// The lookup producer pushes the value columns to a channel with prover-chosen multiplicities.
/// This allows consumers of the channel can read any value in the table an arbitrary number of
/// times. Table values are given as tuples of column entries.
#[derive(Debug)]
pub struct LookupProducer {
	multiplicity_bits: Vec<Col<B1>>,
}

impl LookupProducer {
	pub fn new<FSub>(
		table: &mut TableBuilder,
		chan: ChannelId,
		value_cols: &[Col<FSub>],
		n_multiplicity_bits: usize,
	) -> Self
	where
		B128: ExtensionField<FSub>,
		FSub: TowerField,
	{
		let multiplicity_bits = (0..n_multiplicity_bits)
			.map(|i| table.add_committed::<B1, 1>(format!("multiplicity_bits[{i}]")))
			.collect::<Vec<_>>();

		for (i, &multiplicity_col) in multiplicity_bits.iter().enumerate() {
			table.push_with_opts(
				chan,
				value_cols.iter().copied(),
				FlushOpts {
					multiplicity: 1 << i,
					selector: Some(multiplicity_col),
				},
			);
		}

		Self { multiplicity_bits }
	}

	/// Populate the multiplicity witness columns.
	///
	/// ## Pre-condition
	///
	/// * Multiplicities must be sorted in descending order.
	pub fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		counts: impl Iterator<Item = u32> + Clone,
	) -> Result<(), anyhow::Error>
	where
		P: PackedExtension<B1>,
		P::Scalar: TowerField,
	{
		// TODO: Optimize the gadget for bit-transposing u32s
		for (j, &multiplicity_col) in self.multiplicity_bits.iter().enumerate().take(32) {
			let mut multiplicity_col = index.get_mut(multiplicity_col)?;
			for (packed, counts) in multiplicity_col
				.iter_mut()
				.zip(&counts.clone().chunks(<PackedSubfield<P, B1>>::WIDTH))
			{
				for (i, count) in counts.enumerate() {
					packed.set(i, B1::from((count >> j) & 1 == 1))
				}
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use std::{cmp::Reverse, iter, iter::repeat_with};

	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
	use bumpalo::Bump;
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::*;
	use crate::builder::{test_utils::ClosureFiller, ConstraintSystem, Statement};

	// Test configurations
	enum MultiplicityConfig {
		Partial,  // Only use a subset of values, some will have zero multiplicity
		Complete, // Ensure all values are used at least once
		None,     // No values are used (all have zero multiplicity)
	}

	// Test utility function
	fn run_lookup_test(looker_first: bool, multiplicity: MultiplicityConfig, seed: u64) {
		let mut cs = ConstraintSystem::new();
		let chan = cs.add_channel("values");

		// Table IDs and columns
		let (lookup_table_id, looker_1_id, looker_2_id);
		let (values_col, looker_1_vals, looker_2_vals);
		let lookup_producer;

		// Create tables in the specified order
		if looker_first {
			// Create looker 1 first
			let mut looker_1 = cs.add_table("looker 1");
			looker_1_id = looker_1.id();
			looker_1_vals = looker_1.add_committed::<B128, 1>("values");
			looker_1.pull(chan, [looker_1_vals]);
			drop(looker_1);

			// Create lookup table second
			let mut lookup_table = cs.add_table("lookup");
			lookup_table_id = lookup_table.id();
			values_col = lookup_table.add_committed::<B128, 1>("values");
			lookup_producer = LookupProducer::new(&mut lookup_table, chan, &[values_col], 8);
			drop(lookup_table);

			// Create looker 2
			let mut looker_2 = cs.add_table("looker 2");
			looker_2_id = looker_2.id();
			looker_2_vals = looker_2.add_committed::<B128, 1>("values");
			looker_2.pull(chan, [looker_2_vals]);
			drop(looker_2);
		} else {
			// Create lookup table first
			let mut lookup_table = cs.add_table("lookup");
			lookup_table_id = lookup_table.id();
			values_col = lookup_table.add_committed::<B128, 1>("values");
			lookup_producer = LookupProducer::new(&mut lookup_table, chan, &[values_col], 8);
			drop(lookup_table);

			// Create looker 1
			let mut looker_1 = cs.add_table("looker 1");
			looker_1_id = looker_1.id();
			looker_1_vals = looker_1.add_committed::<B128, 1>("values");
			looker_1.pull(chan, [looker_1_vals]);
			drop(looker_1);

			// Create looker 2
			let mut looker_2 = cs.add_table("looker 2");
			looker_2_id = looker_2.id();
			looker_2_vals = looker_2.add_committed::<B128, 1>("values");
			looker_2.pull(chan, [looker_2_vals]);
			drop(looker_2);
		}

		// Use consistent table sizes across test cases
		let lookup_table_size = 40;
		let looker_1_size = 50;
		let looker_2_size = 60;

		// Generate random values for the lookup table
		let mut rng = StdRng::seed_from_u64(seed);
		let values = repeat_with(|| B128::random(&mut rng))
			.take(lookup_table_size)
			.collect::<Vec<_>>();

		// Initialize counts based on test configuration
		let mut counts = vec![0u32; lookup_table_size];

		// Choose lookup range based on test configuration
		let lookup_range = match multiplicity {
			MultiplicityConfig::Partial => lookup_table_size / 2, // Only use first half
			MultiplicityConfig::Complete => lookup_table_size,
			MultiplicityConfig::None => 0, // Use no entries
		};

		// Generate inputs for lookers
		let (inputs_1, inputs_2) = if matches!(multiplicity, MultiplicityConfig::None) {
			// For AllZero case, use empty vectors - no lookups performed
			(Vec::new(), Vec::new())
		} else {
			// For other cases, generate inputs from the lookup table

			// Generate inputs for looker 1
			let inputs_1 = if matches!(multiplicity, MultiplicityConfig::Complete) {
				// For AllNonZero, ensure each value is used at least once
				let mut result = values.clone(); // Start with one of each

				// Add additional random values to reach the desired size
				let extra = repeat_with(|| {
					let index = rng.gen_range(0..lookup_range);
					counts[index] += 1;
					values[index]
				})
				.take(looker_1_size - lookup_table_size);

				result.extend(extra);
				result
			} else {
				// For SomeZero, just pick random values from the range
				repeat_with(|| {
					let index = rng.gen_range(0..lookup_range);
					counts[index] += 1;
					values[index]
				})
				.take(looker_1_size)
				.collect()
			};

			// Generate inputs for looker 2
			let inputs_2 = repeat_with(|| {
				let index = rng.gen_range(0..lookup_range);
				counts[index] += 1;
				values[index]
			})
			.take(looker_2_size)
			.collect::<Vec<_>>();

			(inputs_1, inputs_2)
		};

		// Sort values by their counts in descending order
		let values_and_counts = iter::zip(values, counts)
			.sorted_unstable_by_key(|&(_val, count)| Reverse(count))
			.collect::<Vec<_>>();

		// Table order in statement depends on table creation order
		let table_sizes = if looker_first {
			vec![looker_1_size, lookup_table_size, looker_2_size]
		} else {
			vec![lookup_table_size, looker_1_size, looker_2_size]
		};

		let statement = Statement {
			boundaries: vec![],
			table_sizes,
		};

		let allocator = Bump::new();
		let mut witness = cs
			.build_witness::<PackedType<OptimalUnderlier128b, B128>>(&allocator, &statement)
			.unwrap();

		// Fill the lookup table
		witness
			.fill_table_sequential(
				&ClosureFiller::new(lookup_table_id, |values_and_counts, witness| {
					{
						let mut values_col = witness.get_scalars_mut(values_col)?;
						for (dst, (val, _)) in iter::zip(&mut *values_col, values_and_counts) {
							*dst = *val;
						}
					}
					lookup_producer
						.populate(witness, values_and_counts.iter().map(|(_, count)| *count))?;
					Ok(())
				}),
				&values_and_counts,
			)
			.unwrap();

		// Fill looker tables if we have inputs
		if !inputs_1.is_empty() {
			witness
				.fill_table_sequential(
					&ClosureFiller::new(looker_1_id, |inputs, witness| {
						let mut vals = witness.get_scalars_mut(looker_1_vals)?;
						for (i, dst) in vals.iter_mut().enumerate() {
							if i < inputs.len() {
								*dst = *inputs[i];
							}
						}
						Ok(())
					}),
					&inputs_1,
				)
				.unwrap();
		}

		if !inputs_2.is_empty() {
			witness
				.fill_table_sequential(
					&ClosureFiller::new(looker_2_id, |inputs, witness| {
						let mut vals = witness.get_scalars_mut(looker_2_vals)?;
						for (i, dst) in vals.iter_mut().enumerate() {
							if i < inputs.len() {
								*dst = *inputs[i];
							}
						}
						Ok(())
					}),
					&inputs_2,
				)
				.unwrap();
		}

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_lookup_producer_some_zero() {
		// Some values have zero multiplicity
		run_lookup_test(
			false, // lookup table first (normal order)
			MultiplicityConfig::Partial,
			0,
		);
	}

	#[test]
	fn test_lookup_producer_all_nonzero() {
		// All values have non-zero multiplicity
		run_lookup_test(
			false, // lookup table first (normal order)
			MultiplicityConfig::Complete,
			1,
		);
	}

	#[test]
	fn test_lookup_producer_all_zero() {
		// All values have zero multiplicity - extreme corner case
		run_lookup_test(
			false, // lookup table first (normal order)
			MultiplicityConfig::None,
			3,
		);
	}

	#[test]
	fn test_lookup_producer_different_ordering() {
		// Different table creation order - looker first
		run_lookup_test(
			true, // looker first
			MultiplicityConfig::Partial,
			2,
		);
	}
}
