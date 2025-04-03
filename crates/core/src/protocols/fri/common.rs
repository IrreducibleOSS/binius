// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{
	packed::len_packed_slice, util::inner_product_unchecked, BinaryField, ExtensionField,
	PackedField,
};
use binius_math::extrapolate_line_scalar;
use binius_ntt::AdditiveNTT;
use binius_utils::bail;
use getset::{CopyGetters, Getters};

use crate::{
	merkle_tree::MerkleTreeScheme, protocols::fri::Error,
	reed_solomon::reed_solomon::ReedSolomonCode,
};

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
pub fn fold_interleaved_chunk<F, FS, P>(
	rs_code: &ReedSolomonCode<FS>,
	log_batch_size: usize,
	chunk_index: usize,
	values: &[P],
	tensor: &[F],
	fold_challenges: &[F],
	scratch_buffer: &mut [F],
) -> F
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
	P: PackedField<Scalar = F>,
{
	// Preconditions
	debug_assert!(fold_challenges.len() <= rs_code.log_dim());
	debug_assert_eq!(len_packed_slice(values), 1 << (log_batch_size + fold_challenges.len()));
	debug_assert_eq!(tensor.len(), 1 << log_batch_size);
	debug_assert!(scratch_buffer.len() >= 2 * (values.len() >> log_batch_size));

	// TODO: handle the opposite case
	assert!(log_batch_size >= P::LOG_WIDTH);

	// There are two types of mixing we do in this loop. Buffer 1 is populated with the
	// folding of symbols from the interleaved codewords into a single codeword. These
	// values are mixed as a regular tensor product combination. Buffer 2 is then
	// populated with `fold_chunk`, which folds a coset of a codeword using the FRI
	// folding algorithm.
	let (buffer1, buffer2) = scratch_buffer.split_at_mut(1 << fold_challenges.len());

	for (interleave_chunk, val) in values
		.chunks(1 << (log_batch_size - P::LOG_WIDTH))
		.zip(buffer1.iter_mut())
	{
		*val = inner_product_unchecked(
			PackedField::iter_slice(interleave_chunk),
			tensor.iter().copied(),
		);
	}

	if fold_challenges.is_empty() {
		buffer1[0]
	} else {
		fold_chunk(rs_code, 0, chunk_index, buffer1, fold_challenges, buffer2)
	}
}

/// Parameters for an FRI interleaved code proximity protocol.
#[derive(Debug, Getters, CopyGetters)]
pub struct FRIParams<F, FA>
where
	F: BinaryField,
	FA: BinaryField,
{
	/// The Reed-Solomon code the verifier is testing proximity to.
	#[getset(get = "pub")]
	rs_code: ReedSolomonCode<FA>,
	/// Vector commitment scheme for the codeword oracle.
	#[getset(get_copy = "pub")]
	log_batch_size: usize,
	/// The reduction arities between each oracle sent to the verifier.
	fold_arities: Vec<usize>,
	/// The number oracle consistency queries required during the query phase.
	#[getset(get_copy = "pub")]
	n_test_queries: usize,
	_marker: PhantomData<F>,
}

impl<F, FA> FRIParams<F, FA>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
{
	pub fn new(
		rs_code: ReedSolomonCode<FA>,
		log_batch_size: usize,
		fold_arities: Vec<usize>,
		n_test_queries: usize,
	) -> Result<Self, Error> {
		if fold_arities.iter().sum::<usize>() >= rs_code.log_dim() + log_batch_size {
			bail!(Error::InvalidFoldAritySequence)
		}

		Ok(Self {
			rs_code,
			log_batch_size,
			fold_arities,
			n_test_queries,
			_marker: PhantomData,
		})
	}

	pub const fn n_fold_rounds(&self) -> usize {
		self.rs_code.log_dim() + self.log_batch_size
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.fold_arities.len()
	}

	/// Number of bits in the query indices sampled during the query phase.
	pub fn index_bits(&self) -> usize {
		self.fold_arities
			.first()
			.map(|arity| self.log_len() - arity)
			// If there is no folding, there are no random queries either
			.unwrap_or(0)
	}

	/// Number of folding challenges the verifier sends after receiving the last oracle.
	pub fn n_final_challenges(&self) -> usize {
		self.n_fold_rounds() - self.fold_arities.iter().sum::<usize>()
	}

	/// The reduction arities between each oracle sent to the verifier.
	pub fn fold_arities(&self) -> &[usize] {
		&self.fold_arities
	}

	/// The binary logarithm of the length of the initial oracle.
	pub fn log_len(&self) -> usize {
		self.rs_code().log_len() + self.log_batch_size()
	}
}

/// This layer allows minimizing the proof size.
pub fn vcs_optimal_layers_depths_iter<'a, F, FA, VCS>(
	fri_params: &'a FRIParams<F, FA>,
	vcs: &'a VCS,
) -> impl Iterator<Item = usize> + 'a
where
	VCS: MerkleTreeScheme<F>,
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
{
	fri_params
		.fold_arities()
		.iter()
		.scan(fri_params.log_len(), |log_n_cosets, arity| {
			*log_n_cosets -= arity;
			Some(vcs.optimal_verify_layer(fri_params.n_test_queries(), *log_n_cosets))
		})
}

/// The type of the termination round codeword in the FRI protocol.
pub type TerminateCodeword<F> = Vec<F>;

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
	let field_size = 2.0_f64.powi(F::N_BITS as i32);
	let sumcheck_err = (2 * code.log_dim()) as f64 / field_size;
	// 2 ⋅ ℓ' / |T_{τ}|
	let folding_err = code.len() as f64 / field_size;
	// 2^{ℓ' + R} / |T_{τ}|
	let per_query_err = 0.5 * (1f64 + 2.0f64.powi(-(code.log_inv_rate() as i32)));
	let allowed_query_err = 2.0_f64.powi(-(security_bits as i32)) - sumcheck_err - folding_err;
	if allowed_query_err <= 0.0 {
		return Err(Error::ParameterError);
	}
	let n_queries = allowed_query_err.log(per_query_err).ceil() as usize;
	Ok(n_queries)
}

/// Heuristic for estimating the optimal FRI folding arity that minimizes proof size.
///
/// `log_block_length` is the binary logarithm of the  block length of the Reed–Solomon code.
pub fn estimate_optimal_arity(
	log_block_length: usize,
	digest_size: usize,
	field_size: usize,
) -> usize {
	(1..=log_block_length)
		.map(|arity| {
			(
				// for given arity, return a tuple (arity, estimate of query_proof_size).
				// this estimate is basd on the following approximation of a single query_proof_size, where $\vartheta$ is the arity:
				// $\big((n-\vartheta) + (n-2\vartheta) + \ldots\big)\text{digest_size} + \frac{n-\vartheta}{\vartheta}2^{\vartheta}\text{field_size}.$
				arity,
				((log_block_length) / 2 * digest_size + (1 << arity) * field_size)
					* (log_block_length - arity)
					/ arity,
			)
		})
		// now scan and terminate the iterator when query_proof_size increases.
		.scan(None, |old: &mut Option<(usize, usize)>, new| {
			let should_continue = !matches!(*old, Some(ref old) if new.1 > old.1);
			*old = Some(new);
			should_continue.then_some(new)
		})
		.last()
		.map(|(arity, _)| arity)
		.unwrap_or(1)
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{BinaryField128b, BinaryField32b};
	use binius_ntt::NTTOptions;

	use super::*;

	#[test]
	fn test_calculate_n_test_queries() {
		let security_bits = 96;
		let rs_code = ReedSolomonCode::new(28, 1, &NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries::<BinaryField128b, BinaryField32b>(security_bits, &rs_code)
				.unwrap();
		assert_eq!(n_test_queries, 232);

		let rs_code = ReedSolomonCode::new(28, 2, &NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries::<BinaryField128b, BinaryField32b>(security_bits, &rs_code)
				.unwrap();
		assert_eq!(n_test_queries, 143);
	}

	#[test]
	fn test_calculate_n_test_queries_unsatisfiable() {
		let security_bits = 128;
		let rs_code = ReedSolomonCode::new(28, 1, &NTTOptions::default()).unwrap();
		assert_matches!(
			calculate_n_test_queries::<BinaryField128b, BinaryField32b>(security_bits, &rs_code),
			Err(Error::ParameterError)
		);
	}

	#[test]
	fn test_estimate_optimal_arity() {
		let field_size = 128;
		for log_block_length in 22..35 {
			let digest_size = 256;
			assert_eq!(estimate_optimal_arity(log_block_length, digest_size, field_size), 4);
		}

		for log_block_length in 22..28 {
			let digest_size = 1024;
			assert_eq!(estimate_optimal_arity(log_block_length, digest_size, field_size), 6);
		}
	}
}
