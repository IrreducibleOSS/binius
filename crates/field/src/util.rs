// Copyright 2024 Ulvetanna Inc.

use crate::{packed::get_packed_slice_unchecked, ExtensionField, Field, PackedField};
use binius_utils::checked_arithmetics::checked_int_div;
use rayon::prelude::*;
use std::iter;

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
	assert_eq!(
		PX::WIDTH * xs.len(),
		PY::WIDTH * ys.len(),
		"Both arguments must contain the same number of field elements"
	);

	let calc_product_by_ys = |x_offset, ys: &[PY]| {
		let mut result = FX::ZERO;
		let xs = &xs[x_offset..];

		for (j, y) in ys.iter().enumerate() {
			for (k, y) in y.iter().enumerate() {
				result += unsafe { get_packed_slice_unchecked(xs, j * PY::WIDTH + k) } * y
			}
		}

		result
	};

	// These magic numbers were chosen experimentally to have a reasonable performance
	// for the calls with small number of elements.
	// For different field sizes, the numbers may need to be adjusted.
	const CHUNK_SIZE: usize = 64;
	if ys.len() < 16 * CHUNK_SIZE {
		calc_product_by_ys(0, ys)
	} else {
		// According to benchmark results iterating by chunks here is more efficient than using `par_iter` with `min_length` directly.
		ys.par_chunks(CHUNK_SIZE)
			.enumerate()
			.map(|(i, ys)| {
				let offset = i * checked_int_div(CHUNK_SIZE * PY::WIDTH, PX::WIDTH);
				calc_product_by_ys(offset, ys)
			})
			.sum()
	}
}

/// Evaluation of the 2-variate multilinear which indicates the condition x == y
#[inline(always)]
pub fn eq<F: Field>(x: F, y: F) -> F {
	x * y + (F::ONE - x) * (F::ONE - y)
}

/// Iterate the powers of a given value, beginning with 1 (the 0'th power).
pub fn powers<F: Field>(val: F) -> impl Iterator<Item = F> {
	iter::successors(Some(F::ONE), move |&power| Some(power * val))
}
