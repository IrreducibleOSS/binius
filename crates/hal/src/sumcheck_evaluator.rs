// Copyright 2024 Ulvetanna Inc.

use binius_field::{Field, PackedField};
use std::ops::Range;

/// Evaluations of a polynomial at a set of evaluation points.
#[derive(Debug, Clone)]
pub struct RoundEvals<F: Field>(pub Vec<F>);

pub trait SumcheckEvaluator<PBase: PackedField, P: PackedField> {
	/// The range of eval point indices over which composition evaluation and summation should happen.
	/// Returned range must equal the result of `n_round_evals()` in length.
	fn eval_point_indices(&self) -> Range<usize>;

	/// Compute composition evals over a subcube.
	///
	/// `sparse_batch_query` should contain multilinears evals over a subcube represented
	/// by `subcube_vars` and `subcube_index`.
	///
	/// Returns a packed sum (which may be spread across scalars).
	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		sparse_batch_query: &[&[PBase]],
	) -> P;
}
