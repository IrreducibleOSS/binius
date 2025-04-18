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
	/// * Multiplicities must be sorted in ascending order.
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

	use binius_core::tower::CanonicalOptimalPackedTowerFamily;
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
	use bumpalo::Bump;
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::*;
	use crate::builder::{test_utils::ClosureFiller, ConstraintSystem, Statement, WitnessIndex};

	#[test]
	fn test_basic_lookup_producer() {
		let mut cs = ConstraintSystem::new();
		let chan = cs.add_channel("values");

		let mut lookup_table = cs.add_table("lookup");
		lookup_table.require_power_of_two_size();
		let lookup_table_id = lookup_table.id();
		let values_col = lookup_table.add_committed::<B128, 1>("values");
		let lookup_producer = LookupProducer::new(&mut lookup_table, chan, &[values_col], 8);

		let mut looker_1 = cs.add_table("looker 1");
		let looker_1_id = looker_1.id();
		let looker_1_vals = looker_1.add_committed::<B128, 1>("values");
		looker_1.pull(chan, [looker_1_vals]);

		let mut looker_2 = cs.add_table("looker 2");
		let looker_2_id = looker_2.id();
		let looker_2_vals = looker_2.add_committed::<B128, 1>("values");
		looker_2.pull(chan, [looker_2_vals]);

		let lookup_table_size = 64;
		let mut rng = StdRng::seed_from_u64(0);
		let values = repeat_with(|| B128::random(&mut rng))
			.take(lookup_table_size)
			.collect::<Vec<_>>();

		let mut counts = vec![0u32; lookup_table_size];

		let looker_1_size = 56;
		let inputs_1 = repeat_with(|| {
			let index = rng.gen_range(0..lookup_table_size);
			counts[index] += 1;
			values[index]
		})
		.take(looker_1_size)
		.collect::<Vec<_>>();

		let looker_2_size = 67;
		let inputs_2 = repeat_with(|| {
			let index = rng.gen_range(0..lookup_table_size);
			counts[index] += 1;
			values[index]
		})
		.take(looker_2_size)
		.collect::<Vec<_>>();

		let values_and_counts = iter::zip(values, counts)
			.sorted_unstable_by_key(|&(_val, count)| Reverse(count))
			.collect::<Vec<_>>();

		let allocator = Bump::new();
		let mut witness =
			WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

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

		// Fill looker tables
		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_1_id, |inputs_1, witness| {
					let mut looker_1_vals = witness.get_scalars_mut(looker_1_vals)?;
					for (dst, src) in iter::zip(&mut *looker_1_vals, inputs_1) {
						*dst = **src;
					}
					Ok(())
				}),
				&inputs_1,
			)
			.unwrap();

		witness
			.fill_table_sequential(
				&ClosureFiller::new(looker_2_id, |inputs_2, witness| {
					let mut looker_2_vals = witness.get_scalars_mut(looker_2_vals)?;
					for (dst, src) in iter::zip(&mut *looker_2_vals, inputs_2) {
						*dst = **src;
					}
					Ok(())
				}),
				&inputs_2,
			)
			.unwrap();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: witness.table_sizes(),
		};
		let ccs = cs
			.compile::<CanonicalOptimalPackedTowerFamily>(&statement)
			.unwrap();
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}
}
