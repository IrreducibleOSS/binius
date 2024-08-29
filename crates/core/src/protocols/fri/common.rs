// Copyright 2024 Ulvetanna Inc.

use crate::{
	linear_code::LinearCode, merkle_tree::VectorCommitScheme, protocols::fri::Error,
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField, PackedFieldIndexable};
use binius_math::extrapolate_line_scalar;
use binius_ntt::AdditiveNTT;
use binius_utils::bail;
use itertools::Itertools;
use p3_util::log2_strict_usize;

/// Calculate fold of `values` at `index` with `r` random coefficient.
///
/// See [DP24], Def. 3.6.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
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

/// Calculate fold of `values` at a `chunk_index` with random folding challenges.
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
	let final_round = start_round + folding_challenges.len() - 1;
	debug_assert!(final_round < rs_code.log_dim());
	debug_assert!(values.len() == 1 << folding_challenges.len());
	debug_assert_eq!(scratch_buffer.len(), values.len());

	scratch_buffer.copy_from_slice(values);
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

fn validate_common_fri_arguments<F, FA, VCS>(
	committed_rs_code: &ReedSolomonCode<FA>,
	final_rs_code: &ReedSolomonCode<F>,
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

	// This is a tricky case to handle well.
	// TODO: Change interfaces to support equality, which implies supporting parameters
	// where the final codeword is the originally committed codeword.
	if committed_rs_code.log_dim() <= final_rs_code.log_dim() {
		bail!(Error::MessageDimensionIsTooSmall);
	}

	// check that base two log of each round_vcs vector_length is greater than
	// the code's log_inv_rate and less than log_len.
	debug_assert!(committed_rs_code.log_dim() >= 1);
	let upper_bound = 1 << (committed_rs_code.log_len() - 1);
	let lower_bound = 1 << (committed_rs_code.log_inv_rate() + final_rs_code.log_dim() + 1);
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

/// Calculates the fold_rounds where folded codewords are committed by the FRIFolder.
/// Also validates consistency of round vector commitment schemes with a Reed-Solomon code for FRI.
///
/// The validation checks that:
/// - The committed Reed-Solomon code length matches the committed codeword vector commitment length.
/// - The committed Reed-Solomon code has dimension greater than the final Reed-Solomon code.
/// - The vector lengths of the round vector commitment schemes (vcss) are powers of two.
/// - The vector lengths of the round vcss are strictly decreasing.
/// - The vector lengths of the round vcss are in the range (2^(log_inv_rate + final_msg_dim), 2^log_len).
pub fn calculate_fold_commit_rounds<F, FA, VCS>(
	committed_rs_code: &ReedSolomonCode<FA>,
	final_rs_code: &ReedSolomonCode<F>,
	committed_codeword_vcs: &VCS,
	round_vcss: &[VCS],
) -> Result<Vec<usize>, Error>
where
	F: BinaryField,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	validate_common_fri_arguments(
		committed_rs_code,
		final_rs_code,
		committed_codeword_vcs,
		round_vcss,
	)?;

	let log_len = committed_rs_code.log_len();
	let commit_rounds = round_vcss
		.iter()
		.map(|vcs| log_len - 1 - log2_strict_usize(vcs.vector_len()));
	Ok(commit_rounds.collect())
}

/// Calculates the start rounds of each fold chunk call made by the FRIFolder.
///
/// REQUIRES:
/// - fold_commit_rounds is the output of calculate_fold_commit_rounds.
pub fn calculate_fold_chunk_start_rounds(fold_commit_rounds: &[usize]) -> Vec<usize> {
	let mut fold_chunk_start_rounds = vec![0; fold_commit_rounds.len() + 1];
	fold_chunk_start_rounds
		.iter_mut()
		.skip(1)
		.zip(fold_commit_rounds.iter())
		.for_each(|(fold_chunk_start_round, fold_commit_round)| {
			*fold_chunk_start_round = fold_commit_round + 1;
		});
	fold_chunk_start_rounds
}

/// Calculates the arity of each fold chunk call made by the FRIFolder.
///
/// REQUIRES:
/// - `fold_chunk_start_rounds` is the output of `calculate_fold_chunk_start_rounds`.
pub fn calculate_folding_arities(
	total_fold_rounds: usize,
	fold_chunk_start_rounds: &[usize],
) -> Vec<usize> {
	fold_chunk_start_rounds
		.iter()
		.copied()
		.chain(std::iter::once(total_fold_rounds))
		.tuple_windows()
		.map(|(prev_start_round, next_start_round)| next_start_round - prev_start_round)
		.collect()
}

/// A proof for a single FRI consistency query.
pub type QueryProof<F, VCSProof> = Vec<QueryRoundProof<F, VCSProof>>;

/// The type of the final message in the FRI protocol.
pub type FinalMessage<F> = Vec<F>;

/// The type of the final codeword in the FRI protocol.
pub type FinalCodeword<F> = Vec<F>;

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
