// Copyright 2024 Ulvetanna Inc.

use crate::field::{ExtensionField, Field};

/// Computes the inner product of two vectors without checking that the lengths are equal
pub fn inner_product_unchecked<F, FE>(a: impl Iterator<Item = FE>, b: impl Iterator<Item = F>) -> FE
where
	F: Field,
	FE: ExtensionField<F>,
{
	a.zip(b).map(|(a_i, b_i)| a_i * b_i).sum::<FE>()
}
