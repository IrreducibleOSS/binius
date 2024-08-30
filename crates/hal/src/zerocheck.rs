// Copyright 2024 Ulvetanna Inc.

use crate::Error;
use binius_field::{Field, PackedField};

/// Describes the shape of the zerocheck computation.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundParameters {
	pub round: usize,
	pub n_vars: usize,
	pub cols: usize,
	pub degree: usize,
}

/// Represents input data of the computation round.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundInput<'a, F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
{
	pub zc_challenges: &'a [F],
	pub eq_ind: &'a [PW],
	pub query: Option<&'a [PW]>,
	pub current_round_sum: F,
}

/// A callback interface to handle the zerocheck computation on the CPU.
pub trait ZerocheckCpuBackendHelper<F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
{
	fn handle_zerocheck_round(
		&mut self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW>,
	) -> Result<Vec<PW::Scalar>, Error>;
}
