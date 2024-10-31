// Copyright 2024 Irreducible Inc.

use crate::{Error, RoundEvals, SumcheckEvaluator, SumcheckMultilinear};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, RepackedExtension};
use binius_math::{CompositionPoly, MultilinearPoly, MultilinearQuery, MultilinearQueryRef};
use rayon::iter::FromParallelIterator;
use std::{
	fmt::Debug,
	ops::{Deref, DerefMut},
};
use tracing::instrument;

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

	/// Calculate the accumulated evaluations for the first round of zerocheck.
	fn sumcheck_compute_first_round_evals<FDomain, FBase, F, PBase, P, M, Evaluator, Composition>(
		&self,
		n_vars: usize,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvals<P::Scalar>>, Error>
	where
		FDomain: Field,
		FBase: ExtensionField<FDomain>,
		F: Field + ExtensionField<FDomain> + ExtensionField<FBase>,
		PBase: PackedField<Scalar = FBase> + PackedExtension<FDomain>,
		P: PackedField<Scalar = F> + PackedExtension<FDomain> + RepackedExtension<PBase>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<PBase, P, Composition> + Sync,
		Composition: CompositionPoly<P>;

	/// Calculate the accumulated evaluations for an arbitrary round of zerocheck.
	fn sumcheck_compute_later_round_evals<FDomain, F, P, M, Evaluator, Composition>(
		&self,
		n_vars: usize,
		tensor_query: Option<MultilinearQueryRef<P>>,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvals<P::Scalar>>, Error>
	where
		FDomain: Field,
		F: Field + ExtensionField<FDomain>,
		P: PackedField<Scalar = F> + PackedExtension<FDomain>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<P, P, Composition> + Sync,
		Composition: CompositionPoly<P>;
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

	fn sumcheck_compute_first_round_evals<FDomain, FBase, F, PBase, P, M, Evaluator, Composition>(
		&self,
		n_vars: usize,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvals<P::Scalar>>, Error>
	where
		FDomain: Field,
		FBase: ExtensionField<FDomain>,
		F: Field + ExtensionField<FDomain> + ExtensionField<FBase>,
		PBase: PackedField<Scalar = FBase> + PackedExtension<FDomain>,
		P: PackedField<Scalar = F> + PackedExtension<FDomain> + RepackedExtension<PBase>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<PBase, P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		T::sumcheck_compute_first_round_evals(
			self,
			n_vars,
			multilinears,
			evaluators,
			evaluation_points,
		)
	}

	fn sumcheck_compute_later_round_evals<FDomain, F, P, M, Evaluator, Composition>(
		&self,
		n_vars: usize,
		tensor_query: Option<MultilinearQueryRef<P>>,
		multilinears: &[SumcheckMultilinear<P, M>],
		evaluators: &[Evaluator],
		evaluation_points: &[FDomain],
	) -> Result<Vec<RoundEvals<P::Scalar>>, Error>
	where
		FDomain: Field,
		F: Field + ExtensionField<FDomain>,
		P: PackedField<Scalar = F> + PackedExtension<FDomain>,
		M: MultilinearPoly<P> + Send + Sync,
		Evaluator: SumcheckEvaluator<P, P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		T::sumcheck_compute_later_round_evals(
			self,
			n_vars,
			tensor_query,
			multilinears,
			evaluators,
			evaluation_points,
		)
	}
}

pub trait ComputationBackendExt: ComputationBackend {
	/// Constructs a `MultilinearQuery` by performing tensor product expansion on the given `query`.
	#[instrument(skip_all, level = "debug")]
	fn multilinear_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<MultilinearQuery<P, Self::Vec<P>>, Error> {
		let tensor_product = self.tensor_product_full_query(query)?;
		Ok(MultilinearQuery::with_expansion(query.len(), tensor_product)?)
	}
}

impl<Backend> ComputationBackendExt for Backend where Backend: ComputationBackend {}
