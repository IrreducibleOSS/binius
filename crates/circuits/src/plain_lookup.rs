// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::Reverse, fmt::Debug, hash::Hash};

use anyhow::{ensure, Result};
use binius_core::{
	constraint_system::channel::{FlushDirection, OracleOrConst},
	oracle::OracleId,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::set_packed_slice,
	BinaryField1b, ExtensionField, Field, PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;
use itertools::izip;

use crate::builder::{
	types::{F, U},
	ConstraintSystemBuilder,
};

/// A gadget validating the lookup relation between:
/// * `lookups_u` - the set of "looked up" tables
/// * `lookup_t` - the lookup table
///
/// Both looked up and lookup tables are defined by tuples of column oracles, where each column has the same `FTable` type.
/// Primary reason to support this behaviour is to be able to lookup into tables "wider" than the largest 128-bit field.
///
/// Looked up tables are assumed to have `n_lookups` values each, whereas `lookup_t` is considered to be always full.
///
/// The prover needs to provide multiplicities of the lookup table elements; the helper [`count_multiplicities`] method
/// does that using a hash map of counters, but in some cases it should be possible to compute this table more efficiently
/// or in an indirect way.
///
/// # How this Works
/// We create two channel for this lookup - a multiplicity channel and a permutation channel.
/// We let the prover push all values in `lookups_u`, that is all values to be looked up, into the multiplicity channel.
/// We also must pull valid table values (i.e. values that appear in `lookup_t`) from this channel if the channel is to balance.
/// By ensuring that only valid table values get pulled from the channel, and observing the channel to balance, we ensure
/// that only valid table values get pushed (by the prover) into the channel. Therefore our construction is sound.
///
/// In order for the construction to be complete, allowing an honest prover to pass, we must pull each
/// table value from the multiplicity channel with exactly the same multiplicity (duplicate count) that the prover pushed that table
/// value into the channel. To do so, we allow the prover to commit information on the multiplicity of each table value.
///
/// The prover counts the multiplicity of each table value, and creates a bit column for each of the LOG_MAX_MULTIPLICITY bits in
/// the bit-decomposition of the multiplicities.
/// Then we flush the table values LOG_MAX_MULTIPLICITY times, each time using a different bit column as the 'selector' oracle to select
/// which values in the table actually get pushed into the channel flushed. When flushing the table with the i'th bit column as the selector,
///we flush with multiplicity 1 << i.
///
/// The reason for using _two_ channels is a prover-side optimization - instead of counting multiplicities on the original `lookup_t`, we
/// commit a permuted version of that with non-decreasing multiplicities. This enforces nonzero scalars prefixes on the committed multiplicity
/// bits columns, which can be used to optimize the flush and GKR reduction sumchecks. In order to constrain this committed lookup to be a
/// permutation of lookup_t we do a push/pull on the permutation channel.
pub fn plain_lookup<FTable, const LOG_MAX_MULTIPLICITY: usize>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	n_lookups: &[usize],
	lookups_u: &[impl AsRef<[OracleId]>],
	lookup_t: impl AsRef<[OracleId]>,
	multiplicities: Option<impl AsRef<[usize]>>,
) -> Result<()>
where
	U: PackScalar<FTable> + Pod,
	F: ExtensionField<FTable>,
	FTable: TowerField,
	PackedType<U, FTable>: PackedFieldIndexable,
{
	ensure!(n_lookups.len() == lookups_u.len(), "n_vars and lookups_u must be of the same length");
	ensure!(
		lookups_u
			.iter()
			.all(|oracles| oracles.as_ref().len() == lookup_t.as_ref().len()),
		"looked up and lookup tables must have the same number of oracles"
	);

	let lookups_u_count_sum = n_lookups.iter().sum::<usize>();
	ensure!(lookups_u_count_sum < 1 << LOG_MAX_MULTIPLICITY, "LOG_MAX_MULTIPLICITY too small");

	builder.push_namespace(name);

	let t_log_rows = builder.log_rows(lookup_t.as_ref().iter().copied())?;
	let bits = builder.add_committed_multiple::<LOG_MAX_MULTIPLICITY>(
		"multiplicity_bits",
		t_log_rows,
		BinaryField1b::TOWER_LEVEL,
	);

	let permuted_lookup_t = (0..lookup_t.as_ref().len())
		.map(|i| {
			builder.add_committed(format!("permuted_t_{}", i), t_log_rows, FTable::TOWER_LEVEL)
		})
		.collect::<Vec<_>>();

	if let Some(witness) = builder.witness() {
		let mut indexed_multiplicities = multiplicities
			.expect("multiplicities should be supplied when proving")
			.as_ref()
			.iter()
			.copied()
			.enumerate()
			.collect::<Vec<_>>();

		let multiplicities_sum = indexed_multiplicities
			.iter()
			.map(|&(_, multiplicity)| multiplicity)
			.sum::<usize>();
		ensure!(multiplicities_sum == lookups_u_count_sum, "Multiplicities do not add up.");

		indexed_multiplicities.sort_by_key(|&(_, multiplicity)| Reverse(multiplicity));

		for (i, bit) in bits.into_iter().enumerate() {
			let nonzero_scalars_prefix =
				indexed_multiplicities.partition_point(|&(_, count)| count >= 1 << i);

			let mut column = witness.new_column_with_nonzero_scalars_prefix::<BinaryField1b>(
				bit,
				nonzero_scalars_prefix,
			);

			let packed = column.packed();

			for (j, &(_, multiplicity)) in indexed_multiplicities.iter().enumerate() {
				if (1 << i) & multiplicity != 0 {
					set_packed_slice(packed, j, BinaryField1b::ONE);
				}
			}
		}

		for (&permuted, &original) in izip!(&permuted_lookup_t, lookup_t.as_ref()) {
			let original_slice =
				PackedType::<U, FTable>::unpack_scalars(witness.get::<FTable>(original)?.packed());

			let mut permuted_column = witness.new_column::<FTable>(permuted);
			let permuted_slice =
				PackedType::<U, FTable>::unpack_scalars_mut(permuted_column.packed());

			for (&(index, _), permuted) in izip!(&indexed_multiplicities, permuted_slice) {
				*permuted = original_slice[index];
			}
		}
	}

	let permutation_channel = builder.add_channel();
	let multiplicity_channel = builder.add_channel();

	builder.send(
		permutation_channel,
		1 << t_log_rows,
		permuted_lookup_t
			.iter()
			.copied()
			.map(OracleOrConst::Oracle),
	)?;
	builder.receive(
		permutation_channel,
		1 << t_log_rows,
		lookup_t
			.as_ref()
			.iter()
			.copied()
			.map(OracleOrConst::Oracle),
	)?;

	for (lookup_u, &count) in izip!(lookups_u, n_lookups) {
		builder.send(
			multiplicity_channel,
			count,
			lookup_u
				.as_ref()
				.iter()
				.copied()
				.map(OracleOrConst::Oracle),
		)?;
	}

	for (i, bit) in bits.into_iter().enumerate() {
		builder.flush_custom(
			FlushDirection::Pull,
			multiplicity_channel,
			bit,
			permuted_lookup_t
				.iter()
				.copied()
				.map(OracleOrConst::Oracle),
			1 << i,
		)?
	}

	builder.pop_namespace();

	Ok(())
}

#[cfg(test)]
pub mod test_plain_lookup {
	use binius_field::BinaryField32b;
	use binius_maybe_rayon::prelude::*;

	use super::*;
	use crate::transparent;

	const fn into_lookup_claim(x: u8, y: u8, z: u16) -> u32 {
		((z as u32) << 16) | ((y as u32) << 8) | (x as u32)
	}

	fn generate_u8_mul_table() -> Vec<u32> {
		let mut result = Vec::with_capacity(1 << 16);
		for x in 0..=255u8 {
			for y in 0..=255u8 {
				let product = x as u16 * y as u16;
				result.push(into_lookup_claim(x, y, product));
			}
		}
		result
	}

	fn generate_random_u8_mul_claims(vals: &mut [u32]) {
		use rand::Rng;
		vals.par_iter_mut().for_each(|val| {
			let mut rng = rand::thread_rng();
			let x = rng.gen_range(0..=255u8);
			let y = rng.gen_range(0..=255u8);
			let product = x as u16 * y as u16;
			*val = into_lookup_claim(x, y, product);
		});
	}

	pub fn test_u8_mul_lookup<const LOG_MAX_MULTIPLICITY: usize>(
		builder: &mut ConstraintSystemBuilder,
		log_lookup_count: usize,
	) -> Result<(), anyhow::Error> {
		let table_values = generate_u8_mul_table();
		let table = transparent::make_transparent(
			builder,
			"u8_mul_table",
			bytemuck::cast_slice::<_, BinaryField32b>(&table_values),
		)?;

		let lookup_values =
			builder.add_committed("lookup_values", log_lookup_count, BinaryField32b::TOWER_LEVEL);

		let lookup_values_count = 1 << log_lookup_count;

		let multiplicities = if let Some(witness) = builder.witness() {
			let mut lookup_values_col = witness.new_column::<BinaryField32b>(lookup_values);
			let mut_slice = lookup_values_col.as_mut_slice::<u32>();
			generate_random_u8_mul_claims(&mut mut_slice[0..lookup_values_count]);
			Some(count_multiplicities(&table_values, mut_slice, true).unwrap())
		} else {
			None
		};

		plain_lookup::<BinaryField32b, LOG_MAX_MULTIPLICITY>(
			builder,
			"u8_mul_lookup",
			&[1 << log_lookup_count],
			&[[lookup_values]],
			&[table],
			multiplicities,
		)?;

		Ok(())
	}
}

pub fn count_multiplicities<T>(
	table: &[T],
	values: &[T],
	check_inclusion: bool,
) -> Result<Vec<usize>, anyhow::Error>
where
	T: Eq + Hash + Debug,
{
	use std::collections::{HashMap, HashSet};

	if check_inclusion {
		let table_set: HashSet<_> = table.iter().collect();
		if let Some(invalid_value) = values.iter().find(|value| !table_set.contains(value)) {
			return Err(anyhow::anyhow!("value {:?} not in table", invalid_value));
		}
	}

	let counts: HashMap<_, usize> =
		values
			.iter()
			.fold(HashMap::with_capacity(table.len()), |mut acc, value| {
				*acc.entry(value).or_insert(0) += 1;
				acc
			});

	let multiplicities = table
		.iter()
		.map(|item| counts.get(item).copied().unwrap_or(0))
		.collect();

	Ok(multiplicities)
}

#[cfg(test)]
mod count_multiplicity_tests {
	use super::*;

	#[test]
	fn test_basic_functionality() {
		let table = vec![1, 2, 3, 4];
		let values = vec![1, 2, 2, 3, 3, 3];
		let result = count_multiplicities(&table, &values, true).unwrap();
		assert_eq!(result, vec![1, 2, 3, 0]);
	}

	#[test]
	fn test_empty_values() {
		let table = vec![1, 2, 3];
		let values: Vec<i32> = vec![];
		let result = count_multiplicities(&table, &values, true).unwrap();
		assert_eq!(result, vec![0, 0, 0]);
	}

	#[test]
	fn test_empty_table() {
		let table: Vec<i32> = vec![];
		let values = vec![1, 2, 3];
		let result = count_multiplicities(&table, &values, false).unwrap();
		assert_eq!(result, vec![]);
	}

	#[test]
	fn test_value_not_in_table() {
		let table = vec![1, 2, 3];
		let values = vec![1, 4, 2];
		let result = count_multiplicities(&table, &values, true);
		assert!(result.is_err());
		assert_eq!(result.unwrap_err().to_string(), "value 4 not in table");
	}

	#[test]
	fn test_duplicates_in_table() {
		let table = vec![1, 1, 2, 3];
		let values = vec![1, 2, 2, 3, 3, 3];
		let result = count_multiplicities(&table, &values, true).unwrap();
		assert_eq!(result, vec![1, 1, 2, 3]);
	}

	#[test]
	fn test_non_integer_values() {
		let table = vec!["a", "b", "c"];
		let values = vec!["a", "b", "b", "c", "c", "c"];
		let result = count_multiplicities(&table, &values, true).unwrap();
		assert_eq!(result, vec![1, 2, 3]);
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
	use binius_hal::make_portable_backend;
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};

	use super::test_plain_lookup;
	use crate::builder::ConstraintSystemBuilder;

	#[test]
	fn test_plain_u8_mul_lookup() {
		const MAX_LOG_MULTIPLICITY: usize = 20;
		let log_lookup_count = 19;

		let log_inv_rate = 1;
		let security_bits = 20;

		let proof = {
			let allocator = bumpalo::Bump::new();
			let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

			test_plain_lookup::test_u8_mul_lookup::<MAX_LOG_MULTIPLICITY>(
				&mut builder,
				log_lookup_count,
			)
			.unwrap();

			let witness = builder.take_witness().unwrap();
			let constraint_system = builder.build().unwrap();
			// validating witness with `validate_witness` is too slow for large transparents like the `table`

			let backend = make_portable_backend();

			binius_core::constraint_system::prove::<
				crate::builder::types::U,
				CanonicalTowerFamily,
				Groestl256,
				Groestl256ByteCompression,
				HasherChallenger<Groestl256>,
				_,
			>(&constraint_system, log_inv_rate, security_bits, &[], witness, &backend)
			.unwrap()
		};

		// verify
		{
			let mut builder = ConstraintSystemBuilder::new();

			test_plain_lookup::test_u8_mul_lookup::<MAX_LOG_MULTIPLICITY>(
				&mut builder,
				log_lookup_count,
			)
			.unwrap();

			let constraint_system = builder.build().unwrap();

			binius_core::constraint_system::verify::<
				crate::builder::types::U,
				CanonicalTowerFamily,
				Groestl256,
				Groestl256ByteCompression,
				HasherChallenger<Groestl256>,
			>(&constraint_system, log_inv_rate, security_bits, &[], proof)
			.unwrap();
		}
	}
}
