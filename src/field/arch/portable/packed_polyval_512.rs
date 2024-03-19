// Copyright 2024 Ulvetanna Inc.

use crate::field::BinaryField128bPolyval;

use super::packed_scaled::packed_scaled_field;

packed_scaled_field!(PackedBinaryPolyval4x128b = [BinaryField128bPolyval; 4]);

/// 512 bit value that just contains four 128-bit integers.
/// Is used for portable implementation of 512-bit packed fields.
impl From<[u128; 4]> for PackedBinaryPolyval4x128b {
	fn from(value: [u128; 4]) -> Self {
		Self(value.map(Into::into))
	}
}
