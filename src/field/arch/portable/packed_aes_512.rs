// Copyright 2024 Ulvetanna Inc.

use super::packed_scaled::packed_scaled_field;

use crate::field::{
	PackedAESBinaryField16x8b, PackedAESBinaryField1x128b, PackedAESBinaryField2x64b,
	PackedAESBinaryField4x32b, PackedAESBinaryField8x16b,
};

packed_scaled_field!(PackedAESBinaryField64x8b = [PackedAESBinaryField16x8b; 4]);
packed_scaled_field!(PackedAESBinaryField32x16b = [PackedAESBinaryField8x16b; 4]);
packed_scaled_field!(PackedAESBinaryField16x32b = [PackedAESBinaryField4x32b; 4]);
packed_scaled_field!(PackedAESBinaryField8x64b = [PackedAESBinaryField2x64b; 4]);
packed_scaled_field!(PackedAESBinaryField4x128b = [PackedAESBinaryField1x128b; 4]);
