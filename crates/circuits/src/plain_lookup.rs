// Copyright 2024-2025 Irreducible Inc.

use binius_core::{constraint_system::channel::FlushDirection, oracle::OracleId};
use binius_field::{
	as_packed_field::PackScalar, packed::set_packed_slice, BinaryField1b, ExtensionField, Field,
	TowerField,
};
use bytemuck::Pod;

use crate::builder::{
	types::{F, U},
	ConstraintSystemBuilder,
};

/// Checks values in `lookup_values` to be in `table`.
///
/// # Introduction
/// This is a gadget for performing a "lookup", wherein a set of values are claimed by the prover to be a subset of a set of values known to the verifier.
/// We call the set of values known to the verifier as the "table", and we call the set of values held by the prover as the "lookup values."
/// We represent these sets using oracles `table` and `lookup_values` as lists of values.
/// This gadget performs the lookup by verifying that every value in the oracle `lookup_vales` appears somewhere in the oracle `table`.
///
/// # Parameters
/// - `builder`: a mutable reference to the `ConstraintSystemBuilder`.
/// - `table`: an oracle holding the table of valid lookup values.
/// - `lookup_values`: an oracle holding the values to be looked up.
/// - `lookup_values_count`: only the first `lookup_values_count` values in `lookup_values` will be looked up.
///
/// # How this Works
/// We create a single channel for this lookup.
/// We let the prover push all values in `lookup_values`, that is all values to be looked up, into the channel.
/// We also must pull valid table values (i.e. values that appear in `table`) from the channel if the channel is to balance.
/// By ensuring that only valid table values get pulled from the channel, and observing the channel to balance, we ensure that only valid table values get pushed (by the prover) into the channel.
/// Therefore our construction is sound.
/// In order for the construction to be complete, allowing an honest prover to pass, we must pull each table value from the channel with exactly the same multiplicity (duplicate count) that the prover pushed that table value into the channel.
/// To do so, we allow the prover to commit information on the multiplicity of each table value.
///
/// The prover counts the multiplicity of each table value, and creates a bit column for each of the LOG_MAX_MULTIPLICITY bits in the bit-decomposition of the multiplicities.
/// Then we flush the table values LOG_MAX_MULTIPLICITY times, each time using a different bit column as the 'selector' oracle to select which values in the table actually get pushed into the channel flushed. When flushing the table with the i'th bit column as the selector, we flush with multiplicity 1 << i.
///
pub fn plain_lookup<FS, const LOG_MAX_MULTIPLICITY: usize>(
	builder: &mut ConstraintSystemBuilder,
	table: OracleId,
	lookup_values: OracleId,
	lookup_values_count: usize,
) -> Result<(), anyhow::Error>
where
	U: PackScalar<FS> + Pod,
	F: ExtensionField<FS>,
	FS: TowerField + Pod,
{
	let n_vars = builder.log_rows([table])?;

	let channel = builder.add_channel();

	builder.send(channel, lookup_values_count, [lookup_values])?;

	let mut multiplicities = None;
	// have prover compute and fill the multiplicities
	if let Some(witness) = builder.witness() {
		let table_slice = witness.get::<FS>(table)?.as_slice::<FS>();
		let values_slice = witness.get::<FS>(lookup_values)?.as_slice::<FS>();

		multiplicities = Some(count_multiplicities(
			&table_slice[0..1 << n_vars],
			&values_slice[0..lookup_values_count],
			false,
		)?);
	}

	let bits: [OracleId; LOG_MAX_MULTIPLICITY] = get_bits(builder, table, multiplicities).unwrap();
	bits.into_iter().enumerate().try_for_each(|(i, bit)| {
		builder.flush_custom(FlushDirection::Pull, channel, bit, [table], 1 << i)
	})?;

	Ok(())
}

// the `i`'th returned bit column holds the `i`'th multiplicity bit.
fn get_bits<FS, const LOG_MAX_MULTIPLICITY: usize>(
	builder: &mut ConstraintSystemBuilder,
	table: OracleId,
	multiplicities: Option<Vec<usize>>,
) -> Result<[OracleId; LOG_MAX_MULTIPLICITY], anyhow::Error>
where
	U: PackScalar<FS>,
	F: ExtensionField<FS>,
	FS: TowerField + Pod,
{
	let n_vars = builder.log_rows([table])?;

	let bits: [OracleId; LOG_MAX_MULTIPLICITY] = builder
		.add_committed_multiple::<LOG_MAX_MULTIPLICITY>("bits", n_vars, BinaryField1b::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let multiplicities =
			multiplicities.ok_or_else(|| anyhow::anyhow!("multiplicities empty for prover"))?;
		debug_assert_eq!(1 << n_vars, multiplicities.len());

		// check all multiplicities are in range
		if multiplicities
			.iter()
			.any(|&multiplicity| multiplicity >= 1 << LOG_MAX_MULTIPLICITY)
		{
			return Err(anyhow::anyhow!(
				"one or more multiplicities exceed `1 << LOG_MAX_MULTIPLICITY`"
			));
		}

		// create the columns for the bits
		let mut bit_cols = bits.map(|bit| witness.new_column::<BinaryField1b>(bit));
		let mut packed_bit_cols = bit_cols.each_mut().map(|bit_col| bit_col.packed());

		multiplicities
			.iter()
			.enumerate()
			.for_each(|(i, multiplicity)| {
				(0..LOG_MAX_MULTIPLICITY).for_each(|j| {
					let bit_set = multiplicity & (1 << j) != 0;
					set_packed_slice(
						packed_bit_cols[j],
						i,
						match bit_set {
							true => BinaryField1b::ONE,
							false => BinaryField1b::ZERO,
						},
					);
				})
			});
	}

	Ok(bits)
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

		if let Some(witness) = builder.witness() {
			let mut lookup_values_col = witness.new_column::<BinaryField32b>(lookup_values);
			let mut_slice = lookup_values_col.as_mut_slice::<u32>();
			generate_random_u8_mul_claims(&mut mut_slice[0..lookup_values_count]);
		}

		plain_lookup::<BinaryField32b, LOG_MAX_MULTIPLICITY>(
			builder,
			table,
			lookup_values,
			lookup_values_count,
		)?;

		Ok(())
	}
}

fn count_multiplicities<T: Eq + std::hash::Hash + Clone + std::fmt::Debug>(
	table: &[T],
	values: &[T],
	check_inclusion: bool,
) -> Result<Vec<usize>, anyhow::Error> {
	use std::collections::{HashMap, HashSet};

	if check_inclusion {
		let table_set: HashSet<_> = table.iter().cloned().collect();
		if let Some(invalid_value) = values.iter().find(|value| !table_set.contains(value)) {
			return Err(anyhow::anyhow!("value {:?} not in table", invalid_value));
		}
	}

	let counts: HashMap<_, usize> = values.iter().fold(HashMap::new(), |mut acc, value| {
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
	use binius_hash::compress::Groestl256ByteCompression;
	use binius_math::DefaultEvaluationDomainFactory;
	use groestl_crypto::Groestl256;

	use super::test_plain_lookup;
	use crate::builder::ConstraintSystemBuilder;

	#[test]
	fn test_plain_u8_mul_lookup() {
		const MAX_LOG_MULTIPLICITY: usize = 18;
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

			let domain_factory = DefaultEvaluationDomainFactory::default();
			let backend = make_portable_backend();

			binius_core::constraint_system::prove::<
				crate::builder::types::U,
				CanonicalTowerFamily,
				_,
				Groestl256,
				Groestl256ByteCompression,
				HasherChallenger<Groestl256>,
				_,
			>(
				&constraint_system,
				log_inv_rate,
				security_bits,
				&[],
				witness,
				&domain_factory,
				&backend,
			)
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
