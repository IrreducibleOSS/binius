// Copyright 2024 Ulvetanna Inc.

use crate::field::BinaryField128bPolyval;

use super::packed_scaled::packed_scaled_field;

packed_scaled_field!(PackedBinaryPolyval2x128b = [BinaryField128bPolyval; 2]);

/// 256 bit value that just contains two 128-bit integers.
/// Is used for portable implementation of 512-bit packed fields.
impl From<[u128; 2]> for PackedBinaryPolyval2x128b {
	fn from(value: [u128; 2]) -> Self {
		Self(value.map(Into::into))
	}
}
