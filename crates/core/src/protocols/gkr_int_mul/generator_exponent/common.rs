// Copyright 2024-2025 Irreducible Inc.

use binius_field::Field;

use crate::protocols::gkr_gpa::LayerClaim;

#[derive(Clone)]
pub struct GeneratorExponentClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
	pub exponent_bit_width: usize,
	pub n_vars: usize,
	pub with_dynamic_generator: bool,
}

impl<F: Field> GeneratorExponentClaim<F> {
	pub const fn layer_n_multilinears(&self, layer_no: usize) -> usize {
		if self.with_dynamic_generator && !self.is_last_layer(layer_no) {
			3
		} else if !self.with_dynamic_generator && self.is_last_layer(layer_no) {
			0
		} else {
			2
		}
	}

	pub const fn current_layer_exponent_bit_no(&self, layer_no: usize) -> usize {
		if self.with_dynamic_generator {
			layer_no
		} else {
			self.exponent_bit_width - 1 - layer_no
		}
	}

	pub const fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width - 1 - layer_no == 0
	}
}

impl<F: Field> From<GeneratorExponentClaim<F>> for LayerClaim<F> {
	fn from(value: GeneratorExponentClaim<F>) -> Self {
		Self {
			eval: value.eval,
			eval_point: value.eval_point,
		}
	}
}

pub struct GeneratorExponentReductionOutput<F: Field> {
	pub eval_claims_on_exponent_bit_columns: Vec<Vec<LayerClaim<F>>>,
}
