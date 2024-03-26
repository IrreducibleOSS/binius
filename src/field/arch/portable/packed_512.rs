// Copyright 2024 Ulvetanna Inc.

use super::packed_scaled::packed_scaled_field;

use crate::field::{
	PackedBinaryField128x1b, PackedBinaryField16x8b, PackedBinaryField1x128b,
	PackedBinaryField2x64b, PackedBinaryField32x4b, PackedBinaryField4x32b, PackedBinaryField64x2b,
	PackedBinaryField8x16b,
};

packed_scaled_field!(PackedBinaryField512x1b = [PackedBinaryField128x1b; 4]);
packed_scaled_field!(PackedBinaryField256x2b = [PackedBinaryField64x2b; 4]);
packed_scaled_field!(PackedBinaryField128x4b = [PackedBinaryField32x4b; 4]);
packed_scaled_field!(PackedBinaryField64x8b = [PackedBinaryField16x8b; 4]);
packed_scaled_field!(PackedBinaryField32x16b = [PackedBinaryField8x16b; 4]);
packed_scaled_field!(PackedBinaryField16x32b = [PackedBinaryField4x32b; 4]);
packed_scaled_field!(PackedBinaryField8x64b = [PackedBinaryField2x64b; 4]);
packed_scaled_field!(PackedBinaryField4x128b = [PackedBinaryField1x128b; 4]);
