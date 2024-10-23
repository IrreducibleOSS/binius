// Copyright 2024 Ulvetanna Inc.

use crate::{
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
	Error,
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
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
}

/// Makes it unnecessary to clone backends.
/// Can't use `auto_impl` because of a complex associated type.
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
}
