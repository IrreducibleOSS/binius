// Copyright 2024 Ulvetanna Inc.

use crate::BinaryField128bPolyval;

use super::packed_scaled::packed_scaled_field;

// 256 bit value that just contains two 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryPolyval2x128b = [BinaryField128bPolyval; 2]);
