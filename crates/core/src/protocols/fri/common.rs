// Copyright 2024 Ulvetanna Inc.

use crate::{
	linear_code::LinearCode, merkle_tree::VectorCommitScheme, protocols::fri::Error,
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{
	util::inner_product_unchecked, BinaryField, ExtensionField, PackedFieldIndexable,
};
use binius_math::extrapolate_line_scalar;
use binius_ntt::AdditiveNTT;
use binius_utils::bail;
use itertools::Itertools;
use rayon::prelude::*;
use std::iter;

/// Calculate fold of `values` at `index` with `r` random coefficient.
///
/// See [DP24], Def. 3.6.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
fn fold_pair<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	round: usize,
	index: usize,
	values: (F, F),
	r: F,
) -> F
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	// Perform inverse additive NTT butterfly
	let t = rs_code.get_ntt().get_subspace_eval(round, index);
	let (mut u, mut v) = values;
	v += u;
	u += v * t;
	extrapolate_line_scalar(u, v, r)
}

/// Calculate FRI fold of `values` at a `chunk_index` with random folding challenges.
///
/// REQUIRES:
/// - `folding_challenges` is not empty.
/// - `values.len() == 1 << folding_challenges.len()`.
/// - `scratch_buffer.len() == values.len()`.
/// - `start_round + folding_challenges.len() - 1 < rs_code.log_dim()`.
///
/// NB: This method is on a hot path and does not perform any allocations or
/// precondition checks.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
pub fn fold_chunk<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	start_round: usize,
	chunk_index: usize,
	values: &[F],
	folding_challenges: &[F],
	scratch_buffer: &mut [F],
) -> F
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	// Preconditions
	debug_assert!(!folding_challenges.is_empty());
	debug_assert!(start_round + folding_challenges.len() <= rs_code.log_dim());
	debug_assert_eq!(values.len(), 1 << folding_challenges.len());
	debug_assert!(scratch_buffer.len() >= values.len());

	scratch_buffer[..values.len()].copy_from_slice(values);
	// Fold the chunk with the folding challenges one by one
	for n_challenges_processed in 0..folding_challenges.len() {
		let n_remaining_challenges = folding_challenges.len() - n_challenges_processed;
		let scratch_buffer_len = values.len() >> n_challenges_processed;
		let new_scratch_buffer_len = scratch_buffer_len >> 1;
		let round = start_round + n_challenges_processed;
		let r = folding_challenges[n_challenges_processed];

		// Fold the (2i) and (2i+1)th cells of the scratch buffer in-place into the i-th cell
		(0..new_scratch_buffer_len).for_each(|index_offset| {
			let index = (chunk_index << (n_remaining_challenges - 1)) + index_offset;
			let values =
				(scratch_buffer[index_offset << 1], scratch_buffer[(index_offset << 1) + 1]);
			scratch_buffer[index_offset] = fold_pair(rs_code, round, index, values, r)
		});
	}

	scratch_buffer[0]
}

/// Calculate the fold of an interleaved chunk of values with random folding challenges.
///
/// The elements in the `values` vector are the interleaved cosets of a batch of codewords at the
/// index `coset_index`. That is, the layout of elements in the values slice is
///
/// ```text
/// [a0, b0, c0, d0, a1, b1, c1, d1, ...]
/// ```
///
/// where `a0, a1, ...` form a coset of a codeword `a`, `b0, b1, ...` form a coset of a codeword
/// `b`, and similarly for `c` and `d`.
///
/// The fold operation first folds the adjacent symbols in the slice using regular multilinear
/// tensor folding for the symbols from different cosets and FRI folding for the cosets themselves
/// using the remaining challenges.
//
/// NB: This method is on a hot path and does not perform any allocations or
/// precondition checks.
///
/// See [DP24], Def. 3.6 and Lemma 3.9 for more details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[inline]
pub fn fold_interleaved_chunk<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	log_batch_size: usize,
	chunk_index: usize,
	values: &[F],
	tensor: &[F],
	fold_challenges: &[F],
	scratch_buffer: &mut [F],
) -> F
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	// Preconditions
	debug_assert!(fold_challenges.len() <= rs_code.log_dim());
	debug_assert_eq!(values.len(), 1 << (log_batch_size + fold_challenges.len()));
	debug_assert_eq!(tensor.len(), 1 << log_batch_size);
	debug_assert!(scratch_buffer.len() >= 2 * (values.len() >> log_batch_size));

	// There are two types of mixing we do in this loop. Buffer 1 is populated with the
	// folding of symbols from the interleaved codewords into a single codeword. These
	// values are mixed as a regular tensor product combination. Buffer 2 is then
	// populated with `fold_chunk`, which folds a coset of a codeword using the FRI
	// folding algorithm.
	let (buffer1, buffer2) = scratch_buffer.split_at_mut(1 << fold_challenges.len());

	for (interleave_chunk, val) in values.chunks(1 << log_batch_size).zip(buffer1.iter_mut()) {
		*val = inner_product_unchecked(interleave_chunk.iter().copied(), tensor.iter().copied());
	}

	if fold_challenges.is_empty() {
		buffer1[0]
	} else {
		fold_chunk(rs_code, 0, chunk_index, buffer1, fold_challenges, buffer2)
	}
}

pub fn validate_common_fri_arguments<F, FA, VCS>(
	committed_rs_code: &ReedSolomonCode<FA>,
	committed_codeword_vcs: &VCS,
	round_vcss: &[VCS],
) -> Result<(), Error>
where
	F: BinaryField,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	if committed_rs_code.len() != committed_codeword_vcs.vector_len() {
		bail!(Error::InvalidArgs(
			"Reed–Solomon code length must match codeword vector commitment length".to_string(),
		));
	}

	// check that base two log of each round_vcs vector_length is greater than
	// the code's log_inv_rate and less than log_len.
	debug_assert!(committed_rs_code.log_dim() >= 1);
	let upper_bound = 1 << committed_rs_code.log_len();
	let lower_bound = 1 << (committed_rs_code.log_inv_rate() + 1);
	if round_vcss.iter().any(|vcs| {
		let len = vcs.vector_len();
		len < lower_bound || len > upper_bound
	}) {
		bail!(Error::RoundVCSLengthsOutOfRange);
	}

	// check that each round_vcs has power of two vector_length
	if round_vcss
		.iter()
		.any(|vcs| !vcs.vector_len().is_power_of_two())
	{
		bail!(Error::RoundVCSLengthsNotPowerOfTwo);
	}

	// check that round_vcss vector is sorted in strictly descending order by vector_length
	if round_vcss
		.windows(2)
		.any(|w| w[0].vector_len() <= w[1].vector_len())
	{
		bail!(Error::RoundVCSLengthsNotDescending);
	}
	Ok(())
}

pub fn fold_codeword<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	codeword: &[F],
	// Round is the number of total folding challenges received so far.
	round: usize,
	folding_challenges: &[F],
) -> Vec<F>
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	// Preconditions
	assert_eq!(codeword.len() % (1 << folding_challenges.len()), 0);
	assert!(round >= folding_challenges.len());
	assert!(round <= rs_code.log_dim());

	if folding_challenges.is_empty() {
		return codeword.to_vec();
	}

	let start_round = round - folding_challenges.len();
	let chunk_size = 1 << folding_challenges.len();

	// For each chunk of size `2^chunk_size` in the codeword, fold it with the folding challenges
	codeword
		.par_chunks(chunk_size)
		.enumerate()
		.map_init(
			|| vec![F::default(); chunk_size],
			|scratch_buffer, (chunk_index, chunk)| {
				fold_chunk(
					rs_code,
					start_round,
					chunk_index,
					chunk,
					folding_challenges,
					scratch_buffer,
				)
			},
		)
		.collect()
}

/// Calculates the folding arities between all rounds of the folding phase when the prover sends an
/// oracle or codeword message.
///
/// The vector returned has length at least 1, representing the number of fold rounds before the
/// prover sends the final oracle. All entries must be non-zero except for the last one.
pub fn calculate_fold_arities(
	log_code_len: usize,
	log_final_len: usize,
	log_commit_lens: impl IntoIterator<Item = usize>,
	log_batch_size: usize,
) -> Result<Vec<usize>, Error> {
	let oracle_log_lengths = log_commit_lens.into_iter().chain(iter::once(log_final_len));
	let oracle_rounds = oracle_log_lengths
		.map(|log_folded_len| {
			let round_diff = log_code_len
				.checked_sub(log_folded_len)
				.ok_or_else(|| Error::RoundVCSLengthsNotDescending)?;
			Ok(log_batch_size + round_diff)
		})
		.collect::<Result<Vec<_>, Error>>()?;

	let fold_arities = iter::once(0)
		.chain(oracle_rounds)
		.tuple_windows()
		.map(|(round, next_round)| next_round - round)
		.collect::<Vec<_>>();

	// Check that all entries are non-zero except for the last one.
	for &arity in &fold_arities[..fold_arities.len() - 1] {
		if arity == 0 {
			return Err(Error::RoundVCSLengthsNotDescending);
		}
	}

	Ok(fold_arities)
}

/// A proof for a single FRI consistency query.
pub type QueryProof<F, VCSProof> = Vec<QueryRoundProof<F, VCSProof>>;

/// The type of the termination round codeword in the FRI protocol.
pub type TerminateCodeword<F> = Vec<F>;

/// The values and vector commitment opening proofs for a coset.
#[derive(Debug, Clone)]
pub struct QueryRoundProof<F, VCSProof> {
	/// Values of the committed vector at the queried coset.
	pub values: Vec<F>,
	/// Vector commitment opening proof for the coset.
	pub vcs_proof: VCSProof,
}

/// Calculates the number of test queries required to achieve a target security level.
///
/// Throws [`Error::ParameterError`] if the security level is unattainable given the code
/// parameters.
pub fn calculate_n_test_queries<F, PS>(
	security_bits: usize,
	code: &ReedSolomonCode<PS>,
) -> Result<usize, Error>
where
	F: BinaryField + ExtensionField<PS::Scalar>,
	PS: PackedFieldIndexable<Scalar: BinaryField>,
{
	let per_query_err = 0.5 * (1f64 + 2.0f64.powi(-(code.log_inv_rate() as i32)));
	let mut n_queries = (-(security_bits as f64) / per_query_err.log2()).ceil() as usize;
	for _ in 0..10 {
		if calculate_error_bound::<F, _>(code, n_queries) >= security_bits {
			return Ok(n_queries);
		}
		n_queries += 1;
	}
	Err(Error::ParameterError)
}

fn calculate_error_bound<F, PS>(code: &ReedSolomonCode<PS>, n_queries: usize) -> usize
where
	F: BinaryField + ExtensionField<PS::Scalar>,
	PS: PackedFieldIndexable<Scalar: BinaryField>,
{
	let field_size = 2.0_f64.powi(F::N_BITS as i32);
	// ℓ' / |T_{τ}|
	let sumcheck_err = code.log_dim() as f64 / field_size;
	// 2^{ℓ' + R} / |T_{τ}|
	let folding_err = code.len() as f64 / field_size;
	let per_query_err = 0.5 * (1.0 + 2.0f64.powi(-(code.log_inv_rate() as i32)));
	let query_err = per_query_err.powi(n_queries as i32);
	let total_err = sumcheck_err + folding_err + query_err;
	-total_err.log2() as usize
}

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use binius_field::{BinaryField128b, BinaryField32b};
	use binius_ntt::NTTOptions;

	#[test]
	fn test_calculate_n_test_queries() {
		let security_bits = 96;
		let rs_code = ReedSolomonCode::new(28, 1, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries::<BinaryField128b, BinaryField32b>(security_bits, &rs_code)
				.unwrap();
		assert_eq!(n_test_queries, 232);

		let rs_code = ReedSolomonCode::new(28, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries::<BinaryField128b, BinaryField32b>(security_bits, &rs_code)
				.unwrap();
		assert_eq!(n_test_queries, 143);
	}

	#[test]
	fn test_calculate_n_test_queries_unsatisfiable() {
		let security_bits = 128;
		let rs_code = ReedSolomonCode::new(28, 1, NTTOptions::default()).unwrap();
		assert_matches!(
			calculate_n_test_queries::<BinaryField128b, BinaryField32b>(security_bits, &rs_code),
			Err(Error::ParameterError)
		);
	}
}
