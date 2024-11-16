// Copyright 2024 Irreducible Inc.

use super::builder::ConstraintSystemBuilder;
use anyhow::Result;
use binius_core::oracle::{OracleId, ProjectionVariant};
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField128b, BinaryField1b,
	BinaryField64b, ExtensionField, Field, TowerField,
};
use bytemuck::Pod;
use std::array;

type B1 = BinaryField1b;
type B64 = BinaryField64b;
type B128 = BinaryField128b;

//const N_ROUNDS_PER_ROW: usize = 3;
//const N_ROUNDS_PER_PERM: usize = 24;
const LOG_ROWS_PER_PERM: usize = 3;

pub struct KeccakfColumns {
	pub input_state: [OracleId; 25],
	pub output_state: [OracleId; 25],
}

pub fn keccakf<U, FBase>(
	builder: &mut ConstraintSystemBuilder<U, B128, FBase>,
	log_n_permutations: usize,
) -> Result<KeccakfColumns>
where
	U: UnderlierType + Pod + PackScalar<B128> + PackScalar<FBase> + PackScalar<B1>,
	B128: ExtensionField<FBase>,
	FBase: TowerField,
{
	let state_in_bits: [OracleId; 25 * 64] = builder.add_committed_multiple(
		"state_in_bits",
		log_n_permutations + LOG_ROWS_PER_PERM,
		B1::TOWER_LEVEL,
	);
	let state_out_bits: [OracleId; 25 * 64] = builder.add_committed_multiple(
		"state_out_bits",
		log_n_permutations + LOG_ROWS_PER_PERM,
		B1::TOWER_LEVEL,
	);

	let state_in_words: [OracleId; 25] = array::try_from_fn(|xy| {
		// Pack state bits horizontally into 64-bit words
		let state_word_for_round = pack_bits_into_64b(
			builder,
			"state_in_for_round",
			log_n_permutations + LOG_ROWS_PER_PERM,
			&state_in_bits[xy * 64..(xy + 1) * 64],
		)?;

		// Select the first of every 8 rows, where 8 rows constitutes a permutation
		// TODO: add helper method for hypercube index projection
		let state_word = builder.add_projected(
			"state_in",
			state_word_for_round,
			vec![B128::ZERO; 3],
			ProjectionVariant::FirstVars,
		)?;
		Ok::<_, anyhow::Error>(state_word)
	})?;
	let state_out_words: [OracleId; 25] = array::try_from_fn(|xy| {
		// Pack state bits horizontally into 64-bit words
		let state_word_for_round = pack_bits_into_64b(
			builder,
			"state_out_for_round",
			log_n_permutations + LOG_ROWS_PER_PERM,
			&state_out_bits[xy * 64..(xy + 1) * 64],
		)?;

		// Select the last of every 8 rows, where 8 rows constitutes a permutation
		// TODO: add helper method for hypercube index projection
		let state_word = builder.add_projected(
			"state_out",
			state_word_for_round,
			vec![B128::ONE; 3],
			ProjectionVariant::FirstVars,
		)?;
		Ok::<_, anyhow::Error>(state_word)
	})?;

	let _c: [OracleId; 5 * 64] = builder.add_committed_multiple(
		"c",
		log_n_permutations + LOG_ROWS_PER_PERM,
		B1::TOWER_LEVEL,
	);

	Ok(KeccakfColumns {
		input_state: state_in_words,
		output_state: state_out_words,
	})
}

// Preconditions:
// - bit_cols is a slice with length 64
fn pack_bits_into_64b<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	n_vars: usize,
	bit_cols: &[OracleId],
) -> Result<OracleId>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase> + PackScalar<B1>,
	F: TowerField + ExtensionField<B64> + ExtensionField<FBase>,
	FBase: TowerField,
{
	assert_eq!(bit_cols.len(), 64);
	let col = builder.add_linear_combination(
		name,
		n_vars,
		(0..64).map(|i| {
			let beta =
				<B64 as ExtensionField<B1>>::basis(i).expect("i is less than extension degree");
			(bit_cols[i], beta.into())
		}),
	)?;
	Ok(col)
}
