// Copyright 2025 Irreducible Inc.

//! Type aliases for primitive data types.
//!
//! The primitive data types are fields in the canonical tower.

use binius_field::{
	BinaryField1b, BinaryField8b, BinaryField16b, BinaryField32b, BinaryField64b, BinaryField128b,
};

pub type B1 = BinaryField1b;
pub type B8 = BinaryField8b;
pub type B16 = BinaryField16b;
pub type B32 = BinaryField32b;
pub type B64 = BinaryField64b;
pub type B128 = BinaryField128b;
