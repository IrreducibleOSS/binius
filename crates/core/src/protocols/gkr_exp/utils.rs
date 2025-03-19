// Copyright 2025 Irreducible Inc.

use binius_field::BinaryField;

pub fn first_layer_inverse<F>(input: F, base: F) -> F
where
	F: BinaryField,
{
	(input - F::ONE) * (base - F::ONE).invert_or_zero()
}
