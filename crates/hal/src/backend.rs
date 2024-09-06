// Copyright 2024 Ulvetanna Inc.

use crate::{
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
	Error,
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
use std::{
	fmt::Debug,
	ops::{Deref, DerefMut},
};

/// HAL-managed memory containing the result of its operations.
pub trait HalSlice<P: Debug + Send + Sync>:
	Deref<Target = [P]> + DerefMut<Target = [P]> + Send + Sync + Debug { }

impl<P: Send + Sync + Debug> HalSlice<P> for Vec<P> {}

/// An abstraction to interface with acceleration hardware to perform computation intensive operations.
pub trait ComputationBackend: Clone + Send + Sync + Debug + 'static {
	type Vec<P: Send + Sync + Debug>: HalSlice<P>;

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
