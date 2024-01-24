// Copyright 2024 Ulvetanna Inc.

use crate::field::{get_packed_slice, ExtensionField, Field, PackedField};
use rayon::prelude::*;

/// Computes the inner product of two vectors without checking that the lengths are equal
pub fn inner_product_unchecked<F, FE>(a: impl Iterator<Item = FE>, b: impl Iterator<Item = F>) -> FE
where
	F: Field,
	FE: ExtensionField<F>,
{
	a.zip(b).map(|(a_i, b_i)| a_i * b_i).sum::<FE>()
}

pub fn inner_product_par<FX, PX, PY>(xs: &[PX], ys: &[PY]) -> FX
where
	PX: PackedField<Scalar = FX>,
	PY: PackedField,
	FX: ExtensionField<PY::Scalar>,
{
	debug_assert_eq!(
		PX::WIDTH * xs.len(),
		PY::WIDTH * ys.len(),
		"Both arguments must contain the same number of field elements"
	);
	(0..PX::WIDTH * xs.len())
		.into_par_iter()
		.map(|i| get_packed_slice(xs, i) * get_packed_slice(ys, i))
		.sum()
}

/// Evaluation of the 2-variate multilinear which indicates the condition x == y
#[inline(always)]
pub fn eq<F: Field>(x: F, y: F) -> F {
	x * y + (F::ONE - x) * (F::ONE - y)
}
