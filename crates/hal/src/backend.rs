// Copyright 2024-2025 Irreducible Inc.

use std::{
	fmt::Debug,
	ops::{Deref, DerefMut},
};

use binius_field::{Field, PackedExtension, PackedField};
use binius_math::{
	CompositionPoly, EvaluationOrder, MultilinearExtension, MultilinearPoly, MultilinearQuery,
	MultilinearQueryRef,
};
use binius_maybe_rayon::iter::FromParallelIterator;
use tracing::instrument;

use crate::{Error, RoundEvals, SumcheckEvaluator, SumcheckMultilinear};

/// HAL-managed memory containing the result of its operations.
pub trait HalSlice<P: Debug + Send + Sync>:
	Deref<Target = [P]>
	+ DerefMut<Target = [P]>
	+ Debug
	+ FromIterator<P>
	+ FromParallelIterator<P>
	+ Send
	+ Sync
	+ 'static
{
}

impl<P: Send + Sync + Debug + 'static> HalSlice<P> for Vec<P> {}

pub struct RoundEvalsOnPrefix<F: Field> {
	pub eval_prefix: usize,
	pub round_evals: RoundEvals<F>,
}

/// An abstraction to interface with acceleration hardware to perform computation intensive operations.
pub trait ComputationBackend: Send + Sync + Debug {
	type Vec<P: Send + Sync + Debug + 'static>: HalSlice<P>;

	/// Creates `Self::Vec<P>` from the given `Vec<P>`.
	fn to_hal_slice<P: Debug + Send + Sync>(v: Vec<P>) -> Self::Vec<P>;

	/// Computes tensor product expansion.
	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<Self::Vec<P>, Error>;

	/// Calculate the accumulated evaluations for an arbitrary round of zerocheck.
	fn sumcheck_compute_round_evals<FDomain, P, M, Evaluator, Composition>(
		&self,
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		tensor_query: Option<MultilinearQueryRef<P>>,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		nontrivial_evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvalsOnPrefix<P::Scalar>>, Error>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<P, Composition> + Sync,
		Composition: CompositionPoly<P>;

	/// Sumcheck round
	fn sumcheck_fold_multilinears<P, M>(
		&self,
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		multilinears: &mut [SumcheckMultilinear<P, M>],
		challenge: P::Scalar,
		tensor_query: Option<MultilinearQueryRef<P>>,
	) -> Result<bool, Error>
	where
		P: PackedField,
		M: MultilinearPoly<P> + Send + Sync;

	/// Partially evaluate the polynomial with assignment to the high-indexed variables.
	fn evaluate_partial_high<P: PackedField>(
		&self,
		multilinear: &impl MultilinearPoly<P>,
		query_expansion: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error>;
}

/// Makes it unnecessary to clone backends.
/// Can't use `auto_impl` because of the complex associated type.
impl<'a, T: 'a + ComputationBackend> ComputationBackend for &'a T
where
	&'a T: Debug + Sync + Send,
{
	type Vec<P: Send + Sync + Debug + 'static> = T::Vec<P>;

	fn to_hal_slice<P: Debug + Send + Sync>(v: Vec<P>) -> Self::Vec<P> {
		T::to_hal_slice(v)
	}

	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<Self::Vec<P>, Error> {
		T::tensor_product_full_query(self, query)
	}

	fn sumcheck_compute_round_evals<FDomain, P, M, Evaluator, Composition>(
		&self,
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		tensor_query: Option<MultilinearQueryRef<P>>,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		nontrivial_evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvalsOnPrefix<P::Scalar>>, Error>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		T::sumcheck_compute_round_evals(
			self,
			evaluation_order,
			n_vars,
			tensor_query,
			multilinears,
			evaluators,
			nontrivial_evaluation_points,
		)
	}

	fn sumcheck_fold_multilinears<P, M>(
		&self,
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		multilinears: &mut [SumcheckMultilinear<P, M>],
		challenge: P::Scalar,
		tensor_query: Option<MultilinearQueryRef<P>>,
	) -> Result<bool, Error>
	where
		P: PackedField,
		M: MultilinearPoly<P> + Send + Sync,
	{
		T::sumcheck_fold_multilinears(
			self,
			evaluation_order,
			n_vars,
			multilinears,
			challenge,
			tensor_query,
		)
	}

	fn evaluate_partial_high<P: PackedField>(
		&self,
		multilinear: &impl MultilinearPoly<P>,
		query_expansion: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error> {
		T::evaluate_partial_high(self, multilinear, query_expansion)
	}
}

pub trait ComputationBackendExt: ComputationBackend {
	/// Constructs a `MultilinearQuery` by performing tensor product expansion on the given `query`.
	#[instrument(skip_all, level = "trace")]
	fn multilinear_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<MultilinearQuery<P, Self::Vec<P>>, Error> {
		let tensor_product = self.tensor_product_full_query(query)?;
		Ok(MultilinearQuery::with_expansion(query.len(), tensor_product)?)
	}
}

impl<Backend> ComputationBackendExt for Backend where Backend: ComputationBackend {}
