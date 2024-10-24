// Copyright 2024 Ulvetanna Inc.

use crate::{
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
	Error, MultilinearPoly, MultilinearQueryRef, RoundEvals, SumcheckEvaluator,
	SumcheckMultilinear,
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, RepackedExtension};
use binius_math::CompositionPoly;
use rayon::iter::FromParallelIterator;
use std::{
	fmt::Debug,
	ops::{Deref, DerefMut},
};

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

	/// Computes round coefficients for zerocheck.
	/// `cpu_handler` is a callback to handle the CpuBackend computation.
	/// It's a leaky abstraction, but zerocheck is too complex to refactor for a clean abstraction separation just yet.
	fn zerocheck_compute_round_coeffs<F, PW, FDomain>(
		&self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW, FDomain>,
		cpu_handler: &mut dyn ZerocheckCpuBackendHelper<F, PW, FDomain>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		F: Field,
		PW: PackedField + PackedExtension<FDomain>,
		PW::Scalar: From<F> + Into<F> + ExtensionField<FDomain>,
		FDomain: Field;

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

	fn zerocheck_compute_round_coeffs<F, PW, FDomain>(
		&self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW, FDomain>,
		cpu_handler: &mut dyn ZerocheckCpuBackendHelper<F, PW, FDomain>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		F: Field,
		PW: PackedField + PackedExtension<FDomain>,
		PW::Scalar: From<F> + Into<F> + ExtensionField<FDomain>,
		FDomain: Field,
	{
		T::zerocheck_compute_round_coeffs(self, params, input, cpu_handler)
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
