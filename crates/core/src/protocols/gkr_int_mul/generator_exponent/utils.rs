// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField};

pub fn first_layer_inverse<FGenerator, F>(input: F) -> F
where
	FGenerator: BinaryField,
	F: BinaryField,
	F: ExtensionField<FGenerator>,
{
	let generator_upcasted = F::from(FGenerator::MULTIPLICATIVE_GENERATOR);
	(input - F::ONE) * (generator_upcasted - F::ONE).invert_or_zero()
}
