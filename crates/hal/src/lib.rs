// Copyright 2024 Ulvetanna Inc.

mod cpu;
mod eq_ind_reducer;
pub mod immutable_slice;
mod linerate;
mod tensor_product;
mod utils;
pub mod zerocheck;

use crate::{
	cpu::CpuBackend,
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
};
pub use crate::{immutable_slice::VecOrImmutableSlice, linerate::make_linerate_backend};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
use std::fmt::Debug;

/// Create the default backend that will use the CPU for all computations.
pub fn make_backend() -> impl ComputationBackend {
	make_linerate_backend()
}

/// An abstraction to interface with acceleration hardware to perform computation intensive operations.
pub trait ComputationBackend: Clone + Send + Sync + Debug {
	/// Computes tensor product expansion.
	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<VecOrImmutableSlice<P>, Error>;

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

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
	#[error("{0}")]
	MathError(#[from] binius_math::Error),
	#[error("{0}")]
	CpuHandlerError(String),
	#[error("{0}")]
	LinerateTensorProductError(linerate_binius_tensor_product::Error),
	#[error("{0}")]
	LinerateSumcheckError(linerate_binius_sumcheck::Error),
	#[error("No available compiled code matches this zerocheck round")]
	UnavailableCompiledCode,
	#[error("Cannot build the trace matrix")]
	MissingData,
	#[error("Polynomial error")]
	PolynomialError,
}
