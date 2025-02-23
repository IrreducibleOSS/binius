// Copyright 2024-2025 Irreducible Inc.

use binius_field::Field;

use crate::protocols::gkr_gpa::LayerClaim;

#[derive(Clone)]
pub struct ExponentiationClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
	pub exponent_bit_width: usize,
	pub n_vars: usize,
	pub with_dynamic_generator: bool,
}

impl<F: Field> From<ExponentiationClaim<F>> for LayerClaim<F> {
	fn from(value: ExponentiationClaim<F>) -> Self {
		Self {
			eval: value.eval,
			eval_point: value.eval_point,
		}
	}
}

pub struct GeneratorExponentReductionOutput<F: Field> {
	pub eval_claims_on_exponent_bit_columns: Vec<Vec<LayerClaim<F>>>,
}
