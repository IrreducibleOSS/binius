// Copyright 2024-2025 Irreducible Inc.

use binius_field::Field;

use crate::protocols::gkr_gpa::LayerClaim;

pub struct GeneratorExponentReductionOutput<F: Field, const EXPONENT_BIT_WIDTH: usize> {
	pub eval_claims_on_exponent_bit_columns: [LayerClaim<F>; EXPONENT_BIT_WIDTH],
}
