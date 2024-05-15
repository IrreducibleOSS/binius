// Copyright 2024 Ulvetanna Inc.

use super::packed_scaled::packed_scaled_field;
use crate::arch::packed_polyval_128::PackedBinaryPolyval1x128b;

// 256 bit value that just contains two 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryPolyval2x128b = [PackedBinaryPolyval1x128b; 2]);

impl From<PackedBinaryPolyval2x128b> for [u128; 2] {
	#[inline]
	#[allow(clippy::useless_conversion)]
	fn from(value: PackedBinaryPolyval2x128b) -> Self {
		[
			value.0[0].to_underlier().into(),
			value.0[1].to_underlier().into(),
		]
	}
}
