// Copyright 2024-2025 Irreducible Inc.

use std::fmt::Debug;

use binius_field::{Field, PackedExtension, PackedField};
use binius_math::{
	eq_ind_partial_eval, CompositionPolyOS, MultilinearExtension, MultilinearPoly,
	MultilinearQueryRef,
};
use tracing::instrument;

use crate::{
	sumcheck_round_calculator::calculate_round_evals, ComputationBackend, Error, RoundEvals,
	SumcheckEvaluator, SumcheckMultilinear,
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
		n_vars: usize,
		tensor_query: Option<MultilinearQueryRef<P>>,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvals<P::Scalar>>, Error>
	where
		FDomain: Field,
		P: PackedExtension<FDomain>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<P, Composition> + Sync,
		Composition: CompositionPolyOS<P>,
	{
		calculate_round_evals(n_vars, tensor_query, multilinears, evaluators, evaluation_points)
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
