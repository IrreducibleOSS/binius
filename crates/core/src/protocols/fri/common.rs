// Copyright 2024 Ulvetanna Inc.

use crate::{polynomial::extrapolate_line, reed_solomon::reed_solomon::ReedSolomonCode};
use binius_field::{BinaryField, ExtensionField};
use binius_ntt::AdditiveNTT;

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
	extrapolate_line(u, v, r)
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

/// A proof for a single FRI consistency query.
pub type QueryProof<F, VCSProof> = Vec<QueryRoundProof<F, VCSProof>>;

/// The values and vector commitment opening proofs for a coset.
#[derive(Debug, Clone)]
pub struct QueryRoundProof<F, VCSProof> {
	/// Values of the committed vector at the queried coset.
	pub values: Vec<F>,
	/// Vector commitment opening proof for the coset.
	pub vcs_proof: VCSProof,
}
