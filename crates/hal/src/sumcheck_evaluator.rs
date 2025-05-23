// Copyright 2024-2025 Irreducible Inc.

use std::ops::Range;

use binius_field::{Field, PackedField};
use binius_math::RowsBatchRef;

/// Evaluations of a polynomial at a set of evaluation points.
#[derive(Debug, Clone)]
pub struct RoundEvals<F: Field>(pub Vec<F>);

pub trait SumcheckEvaluator<P: PackedField, Composition> {
	/// The range of eval point indices over which composition evaluation and summation should
	/// happen. Returned range must equal the result of `n_round_evals()` in length.
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
		batch_query: &RowsBatchRef<P>,
	) -> P;

	/// Compute sum of evals over the suffix where the composite is guaranteed to evaluate to a
	/// constant.
	///
	/// It is assumed that all required inputs are known at the evaluator creation time, as
	/// `const_eval_suffix` is determined dynamically by the sumcheck round calculator and may be
	/// _smaller_ than the return value of the method with the same name.
	///
	/// See doc comments to [EvaluationDomain](binius_math::EvaluationDomain) for the intuition
	/// behind `is_infinity_point`.
	fn process_constant_eval_suffix(
		&self,
		_const_eval_suffix: usize,
		_is_infinity_point: bool,
	) -> P::Scalar {
		P::Scalar::ZERO
	}

	/// Returns the composition evaluated by this object.
	fn composition(&self) -> &Composition;

	/// In case of zerocheck returns eq_ind that the results should be folded with.
	/// In case of sumcheck returns None.
	fn eq_ind_partial_eval(&self) -> Option<&[P]>;

	/// Trace suffix where the composite is guaranteed to evaluate to a constant. The non-constant
	/// prefix would get processed via `process_subcube_at_eval_point`, whereas the remainder gets
	/// handled via `process_constant_eval_suffix`. Due to the fact that sumcheck operates over
	/// whole subcubes the `const_eval_suffix` passed to `process_constant_eval_suffix` may be
	/// _smaller_ that the return value of this method.
	fn const_eval_suffix(&self) -> usize {
		0
	}
}
