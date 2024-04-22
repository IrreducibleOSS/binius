// Copyright 2024 Ulvetanna Inc.

use super::packed_scaled::{impl_scaled_512_bit_conversion_from_u128_array, packed_scaled_field};
use crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b;

// 512 bit value that just contains four 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryPolyval4x128b = [PackedBinaryPolyval2x128b; 2]);

impl_scaled_512_bit_conversion_from_u128_array!(
	PackedBinaryPolyval4x128b,
	PackedBinaryPolyval2x128b
);
