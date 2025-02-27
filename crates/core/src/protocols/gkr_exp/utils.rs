// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField};

pub fn first_layer_inverse<FBase, F>(input: F) -> F
where
	FBase: BinaryField,
	F: BinaryField + ExtensionField<FBase>,
{
	let generator_upcasted = F::from(FBase::MULTIPLICATIVE_GENERATOR);
	(input - F::ONE) * (generator_upcasted - F::ONE).invert_or_zero()
}
