// Copyright 2024 Ulvetanna Inc.

use crate::{
	immutable_slice::VecOrImmutableSlice,
	utils::tensor_product,
	zerocheck::{ZerocheckCpuBackendHelper, ZerocheckRoundInput, ZerocheckRoundParameters},
	ComputationBackend, Error,
};
use binius_field::{Field, PackedField};
use tracing::instrument;

/// Implementation of ComputationBackend for the default Backend that uses the CPU for all computations.
#[derive(Clone, Debug)]
pub struct CpuBackend;

impl ComputationBackend for CpuBackend {
	#[instrument(skip_all)]
	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<VecOrImmutableSlice<P>, Error> {
		Ok(VecOrImmutableSlice::V(tensor_product(query)?))
	}

	#[instrument(skip_all)]
	fn zerocheck_compute_round_coeffs<F, PW, FDomain>(
		&self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW, FDomain>,
		handler: &mut dyn ZerocheckCpuBackendHelper<F, PW, FDomain>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		F: Field,
		PW: PackedField,
		PW::Scalar: From<F> + Into<F>,
		FDomain: Field,
	{
		// Zerocheck involves too much complicated logic, and instead of moving that logic here, callback back to the zerocheck protocols crate.kj
		handler.handle_zerocheck_round(params, input)
	}
}
