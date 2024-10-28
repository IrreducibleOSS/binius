// Copyright 2024 Irreducible Inc.

use binius_field::PackedField;

/// Construct a packed field element from a function that returns scalar values by index with the
/// given offset in packed elements. E.g. if `offset` is 2, and `WIDTH` is 4, `f(9)` will be used
/// to set the scalar at index 1 in the packed element.
#[inline]
pub fn packed_from_fn_with_offset<P: PackedField>(
	offset: usize,
	mut f: impl FnMut(usize) -> P::Scalar,
) -> P {
	P::from_fn(|i| f(i + offset * P::WIDTH))
}
