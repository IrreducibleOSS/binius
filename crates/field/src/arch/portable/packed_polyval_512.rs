// Copyright 2024 Ulvetanna Inc.

use crate::BinaryField128bPolyval;

use super::packed_scaled::packed_scaled_field;

// 512 bit value that just contains four 128-bit integers.
// Is used for portable implementation of 512-bit packed fields.
packed_scaled_field!(PackedBinaryPolyval4x128b = [BinaryField128bPolyval; 4]);
