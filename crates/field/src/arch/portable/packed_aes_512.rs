// Copyright 2024 Ulvetanna Inc.

use super::packed_scaled::{impl_scaled_512_bit_conversion_from_u128_array, packed_scaled_field};
use crate::arch::packed_aes_256::*;

packed_scaled_field!(PackedAESBinaryField64x8b = [PackedAESBinaryField32x8b; 2]);
packed_scaled_field!(PackedAESBinaryField32x16b = [PackedAESBinaryField16x16b; 2]);
packed_scaled_field!(PackedAESBinaryField16x32b = [PackedAESBinaryField8x32b; 2]);
packed_scaled_field!(PackedAESBinaryField8x64b = [PackedAESBinaryField4x64b; 2]);
packed_scaled_field!(PackedAESBinaryField4x128b = [PackedAESBinaryField2x128b; 2]);

impl_scaled_512_bit_conversion_from_u128_array!(
	PackedAESBinaryField64x8b,
	PackedAESBinaryField32x8b
);
impl_scaled_512_bit_conversion_from_u128_array!(
	PackedAESBinaryField32x16b,
	PackedAESBinaryField16x16b
);
impl_scaled_512_bit_conversion_from_u128_array!(
	PackedAESBinaryField16x32b,
	PackedAESBinaryField8x32b
);
impl_scaled_512_bit_conversion_from_u128_array!(
	PackedAESBinaryField8x64b,
	PackedAESBinaryField4x64b
);
impl_scaled_512_bit_conversion_from_u128_array!(
	PackedAESBinaryField4x128b,
	PackedAESBinaryField2x128b
);
