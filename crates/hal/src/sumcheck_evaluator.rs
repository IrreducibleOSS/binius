// Copyright 2024-2025 Irreducible Inc.

use std::ops::Range;

use binius_field::{Field, PackedField};

/// Evaluations of a polynomial at a set of evaluation points.
#[derive(Debug, Clone)]
pub struct RoundEvals<F: Field>(pub Vec<F>);

pub trait SumcheckEvaluator<P: PackedField, Composition> {
	/// The range of eval point indices over which composition evaluation and summation should happen.
	/// Returned range must equal the result of `n_round_evals()` in length.
	fn eval_point_indices(&self) -> Range<usize>;

	/// Compute composition evals over a subcube.
	///
	/// `batch_query` should contain multilinears evals over a subcube represented
	/// by `subcube_vars` and `subcube_index`.
	///
	/// See doc comments to [EvaluationDomain](binius_math::EvaluationDomain) for the intuition
	/// behind `is_infinity_point`.
	///
	/// Returns a packed sum (which may be spread across scalars).
	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		is_infinity_point: bool,
		batch_query: &[&[P]],
	) -> P;

	/// Returns the composition evaluated by this object.
	fn composition(&self) -> &Composition;

	/// In case of zerocheck returns eq_ind that the results should be folded with.
	/// In case of sumcheck returns None.
	fn eq_ind_partial_eval(&self) -> Option<&[P]>;
}
