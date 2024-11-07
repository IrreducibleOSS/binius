// Copyright 2024 Irreducible Inc.

use crate::{
	linear_code::LinearCode, merkle_tree::VectorCommitScheme, protocols::fri::Error,
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{util::inner_product_unchecked, BinaryField, ExtensionField, PackedField};
use binius_math::extrapolate_line_scalar;
use binius_ntt::AdditiveNTT;
use binius_utils::bail;
use getset::{CopyGetters, Getters};
use itertools::Itertools;
use std::{iter, marker::PhantomData};

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

	// Fold the chunk with the folding challenges one by one
	for n_challenges_processed in 0..folding_challenges.len() {
		let n_remaining_challenges = folding_challenges.len() - n_challenges_processed;
		let scratch_buffer_len = values.len() >> n_challenges_processed;
		let new_scratch_buffer_len = scratch_buffer_len >> 1;
		let round = start_round + n_challenges_processed;
		let r = folding_challenges[n_challenges_processed];
		let index_start = chunk_index << (n_remaining_challenges - 1);

		// Fold the (2i) and (2i+1)th cells of the scratch buffer in-place into the i-th cell
		if n_challenges_processed > 0 {
			(0..new_scratch_buffer_len).for_each(|index_offset| {
				let values =
					(scratch_buffer[index_offset << 1], scratch_buffer[(index_offset << 1) + 1]);
				scratch_buffer[index_offset] =
					fold_pair(rs_code, round, index_start + index_offset, values, r)
			});
		} else {
			// For the first round, we read values directly from the `values` slice.
			(0..new_scratch_buffer_len).for_each(|index_offset| {
				let values = (values[index_offset << 1], values[(index_offset << 1) + 1]);
				scratch_buffer[index_offset] =
					fold_pair(rs_code, round, index_start + index_offset, values, r)
			});
		}
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

/// Parameters for an FRI interleaved code proximity protocol.
#[derive(Debug, Getters, CopyGetters)]
pub struct FRIParams<F, FA, VCS>
where
	F: BinaryField,
	FA: BinaryField,
{
	/// The Reed-Solomon code the verifier is testing proximity to.
	#[getset(get = "pub")]
	rs_code: ReedSolomonCode<FA>,
	/// Vector commitment scheme for the codeword oracle.
	#[getset(get = "pub")]
	codeword_vcs: VCS,
	/// The base-2 logarithm of the batch size of the interleaved code.
	#[getset(get_copy = "pub")]
	log_batch_size: usize,
	/// Vector commitment scheme for the round oracles.
	round_vcss: Vec<VCS>,
	/// The reduction arities between each query round.
	fold_arities: Vec<usize>,
	/// The number oracle consistency queries required during the query phase.
	#[getset(get_copy = "pub")]
	n_test_queries: usize,
	_marker: PhantomData<F>,
}

impl<F, FA, VCS> FRIParams<F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	pub fn new(
		rs_code: ReedSolomonCode<FA>,
		log_batch_size: usize,
		codeword_vcs: VCS,
		round_vcss: Vec<VCS>,
		n_test_queries: usize,
	) -> Result<Self, Error> {
		// check that each round_vcs has power of two vector_length
		let vcss = iter::once(&codeword_vcs).chain(round_vcss.iter());
		if vcss.clone().any(|vcs| !vcs.vector_len().is_power_of_two()) {
			bail!(Error::RoundVCSLengthsNotPowerOfTwo);
		}

		if rs_code.len() < codeword_vcs.vector_len() {
			bail!(Error::InvalidArgs(
				"Reed–Solomon code length must be at least the vector commitment length"
					.to_string(),
			));
		}

		// check that the last FRI oracle has the same length as the fully-folded, dimension-1
		// codeword.
		let last_vcs = round_vcss.last().unwrap_or(&codeword_vcs);
		if last_vcs.vector_len() != rs_code.inv_rate() {
			bail!(Error::RoundVCSLengthsOutOfRange);
		}

		let fold_arities = calculate_fold_arities(
			rs_code.log_len(),
			log_batch_size,
			codeword_vcs.vector_len().ilog2() as usize,
			round_vcss
				.iter()
				.map(|vcs| vcs.vector_len().ilog2() as usize),
		)?;

		Ok(Self {
			rs_code,
			codeword_vcs,
			log_batch_size,
			round_vcss,
			fold_arities,
			n_test_queries,
			_marker: PhantomData,
		})
	}

	pub fn n_fold_rounds(&self) -> usize {
		self.rs_code.log_dim() + self.log_batch_size
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.round_vcss.len()
	}

	/// Number of bits in the query indices sampled during the query phase.
	pub fn index_bits(&self) -> usize {
		self.codeword_vcs.vector_len().ilog2() as usize
	}

	/// Number of folding challenges the verifier sends after receiving the last oracle.
	pub fn n_final_challenges(&self) -> usize {
		self.n_fold_rounds() - self.fold_arities.iter().sum::<usize>()
	}

	/// The vector commitment schemes for each of the FRI round oracles.
	pub fn round_vcss(&self) -> &[VCS] {
		&self.round_vcss
	}

	/// The reduction arities between each oracle sent to the verifier.
	pub fn fold_arities(&self) -> &[usize] {
		&self.fold_arities
	}
}

/// Calculates the folding arities between all rounds of the folding phase when the prover sends an
/// oracle.
fn calculate_fold_arities(
	log_code_len: usize,
	log_batch_size: usize,
	log_codeword_commit_len: usize,
	log_round_commit_lens: impl IntoIterator<Item = usize>,
) -> Result<Vec<usize>, Error> {
	let first_fold_arity = log_code_len + log_batch_size - log_codeword_commit_len;
	let round_fold_arities = iter::once(log_codeword_commit_len)
		.chain(log_round_commit_lens)
		.tuple_windows()
		.map(|(log_len, next_log_len)| {
			if log_len <= next_log_len {
				return Err(Error::RoundVCSLengthsNotDescending);
			}
			Ok(log_len - next_log_len)
		});

	let mut fold_arities = iter::once(Ok(first_fold_arity))
		.chain(round_fold_arities)
		.collect::<Result<Vec<_>, _>>()?;
	let _ = fold_arities.pop();
	Ok(fold_arities)
}

/// A proof for a single FRI consistency query.
pub type QueryProof<F, VCSProof> = Vec<QueryRoundProof<F, VCSProof>>;

/// The type of the termination round codeword in the FRI protocol.
pub type TerminateCodeword<F> = Vec<F>;

#[derive(Debug, Clone)]
pub struct FRIProof<F, VCSProof> {
	pub terminate_codeword: TerminateCodeword<F>,
	pub proofs: Vec<QueryProof<F, VCSProof>>,
}

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
	PS: PackedField<Scalar: BinaryField>,
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
	PS: PackedField<Scalar: BinaryField>,
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
