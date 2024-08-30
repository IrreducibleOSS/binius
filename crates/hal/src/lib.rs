// Copyright 2024 Ulvetanna Inc.

mod cpu;
mod error;
mod utils;
pub mod zerocheck;

use crate::{
	cpu::CpuBackend,
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
};
use binius_field::{Field, PackedField};
pub use error::*;
use std::fmt::Debug;

/// Create the default backend that will use the CPU for all computations.
pub fn make_backend() -> CpuBackend {
	CpuBackend
}

/// An abstraction to interface with acceleration hardware to perform computation intensive operations.
pub trait ComputationBackend: Clone + Send + Sync + Debug {
	/// Computes tensor product expansion.
	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<Vec<P>, Error>;

	/// Computes round coefficients for zerocheck.
	/// `cpu_handler` is a callback to handle the CpuBackend computation.
	/// It's a leaky abstraction, but zerocheck is too complex to refactor for a clean abstraction separation just yet.
	fn zerocheck_compute_round_coeffs<F, PW>(
		&self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW>,
		cpu_handler: &mut dyn ZerocheckCpuBackendHelper<F, PW>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		F: Field,
		PW: PackedField,
		PW::Scalar: From<F> + Into<F>;
}
