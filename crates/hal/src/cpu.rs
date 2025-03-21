// Copyright 2024-2025 Irreducible Inc.

use std::fmt::Debug;

use binius_field::{Field, PackedExtension, PackedField};
use binius_math::{
	eq_ind_partial_eval, CompositionPoly, EvaluationOrder, MultilinearExtension, MultilinearPoly,
	MultilinearQueryRef,
};
use tracing::instrument;

use crate::{
	sumcheck_folding::fold_multilinears, sumcheck_round_calculation::calculate_round_evals,
	ComputationBackend, Error, SumcheckComputeRoundEvalsOutput, SumcheckEvaluator,
	SumcheckMultilinear,
};

/// Implementation of ComputationBackend for the default Backend that uses the CPU for all computations.
#[derive(Clone, Debug)]
pub struct CpuBackend;

pub const fn make_portable_backend() -> CpuBackend {
	CpuBackend
}

impl ComputationBackend for CpuBackend {
	type Vec<P: Send + Sync + Debug + 'static> = Vec<P>;

	fn to_hal_slice<P: Debug + Send + Sync + 'static>(v: Vec<P>) -> Self::Vec<P> {
		v
	}

	#[instrument(skip_all, level = "trace")]
	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<Self::Vec<P>, Error> {
		Ok(eq_ind_partial_eval(query))
	}

	fn sumcheck_compute_round_evals<FDomain, P, M, Evaluator, Composition>(
		&self,
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		eval_prefix: Option<usize>,
		tensor_query: Option<MultilinearQueryRef<P>>,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		nontrivial_evaluation_points: &[FDomain],
	) -> Result<SumcheckComputeRoundEvalsOutput<P::Scalar>, Error>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		calculate_round_evals(
			evaluation_order,
			n_vars,
			eval_prefix,
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
		fold_multilinears(evaluation_order, n_vars, multilinears, challenge, tensor_query)
	}

	#[instrument(skip_all, name = "CpuBackend::evaluate_partial_high")]
	fn evaluate_partial_high<P: PackedField>(
		&self,
		multilinear: &impl MultilinearPoly<P>,
		query_expansion: MultilinearQueryRef<P>,
	) -> Result<MultilinearExtension<P>, Error> {
		Ok(multilinear.evaluate_partial_high(query_expansion)?)
	}
}
