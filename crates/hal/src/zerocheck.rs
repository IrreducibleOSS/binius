// Copyright 2024 Ulvetanna Inc.

use tracing::instrument;
use binius_field::{Field, PackedField};
use binius_math::EvaluationDomain;
use crate::Error;

/// Describes the shape of the zerocheck computation.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundParameters {
	pub round: usize,
	pub n_vars: usize,
	pub cols: usize,
	pub degree: usize,
	pub small_field_width: Option<usize>,
}

/// Represents input data of the computation round.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundInput<'a, F, PW, FDomain>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
	FDomain: Field,
{
	pub zc_challenges: &'a [F],
	pub eq_ind: &'a [PW],
	pub query: Option<&'a [PW]>,
	pub current_round_sum: F,
	pub mixing_challenge: F,
	pub domain: &'a EvaluationDomain<FDomain>,
	pub underlier_data: Option<Vec<Option<Vec<u8>>>>,
}

/// A callback interface to handle the zerocheck computation on the CPU.
pub trait ZerocheckCpuBackendHelper<F, PW, FDomain>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
	FDomain: Field,
{
	fn handle_zerocheck_round(
		&mut self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW, FDomain>,
	) -> Result<Vec<PW::Scalar>, Error>;

	#[instrument(skip_all)]
	fn remove_smaller_domain_optimization(&mut self);
}