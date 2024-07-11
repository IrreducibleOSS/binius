// Copyright 2024 Ulvetanna Inc.

use crate::{polynomial::extrapolate_line, reed_solomon::reed_solomon::ReedSolomonCode};
use binius_field::{BinaryField, ExtensionField};
use binius_ntt::AdditiveNTT;

/// Calculate fold of `values` at `index` with `r` random coefficient.
///
/// See [DP24], Def. 3.6.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn fold_pair<F, FS>(
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
