// Copyright 2024 Ulvetanna Inc.

use super::{
	arithmetic_traits::InvertOrZero,
	binary_field::{binary_field, impl_field_extension, BinaryField, BinaryField1b},
	binary_field_arithmetic::{binary_tower_arithmetic_recursive, TowerFieldArithmetic},
	mul_by_binary_field_1b, BinaryField8b, Error,
};
use crate::{binary_tower, ExtensionField, Field, TowerExtensionField, TowerField};
use bytemuck::{Pod, Zeroable};
use rand::RngCore;
use std::{
	array,
	fmt::{Debug, Display, Formatter},
	iter::{Product, Step, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

// These fields represent a tower based on AES GF(2^8) field (GF(256)/x^8+x^4+x^3+x+1)
// that is isomorphically included into binary tower, i.e.:
//  - AESTowerField16b is GF(2^16) / (x^2 + x * x_2 + 1) where `x_2` is 0x10 from
// BinaryField8b isomorphically projected to AESTowerField8b.
//  - AESTowerField32b is GF(2^32) / (x^2 + x * x_3 + 1), where `x_3` is 0x1000 from AESTowerField16b.
//  ...
binary_field!(pub AESTowerField8b(u8));
binary_field!(pub AESTowerField16b(u16));
binary_field!(pub AESTowerField32b(u32));
binary_field!(pub AESTowerField64b(u64));
binary_field!(pub AESTowerField128b(u128));

unsafe impl Pod for AESTowerField8b {}
unsafe impl Pod for AESTowerField16b {}
unsafe impl Pod for AESTowerField32b {}
unsafe impl Pod for AESTowerField64b {}
unsafe impl Pod for AESTowerField128b {}

binary_tower!(
	AESTowerField8b(u8)
	< AESTowerField16b(u16)
	< AESTowerField32b(u32)
	< AESTowerField64b(u64)
	< AESTowerField128b(u128)
);

impl_field_extension!(BinaryField1b(u8) < @8 => AESTowerField8b(u8));
impl_field_extension!(BinaryField1b(u8) < @16 => AESTowerField16b(u16));
impl_field_extension!(BinaryField1b(u8) < @32 => AESTowerField32b(u32));
impl_field_extension!(BinaryField1b(u8) < @64 => AESTowerField64b(u64));
impl_field_extension!(BinaryField1b(u8) < @128 => AESTowerField128b(u128));

mul_by_binary_field_1b!(AESTowerField8b);
mul_by_binary_field_1b!(AESTowerField16b);
mul_by_binary_field_1b!(AESTowerField32b);
mul_by_binary_field_1b!(AESTowerField64b);
mul_by_binary_field_1b!(AESTowerField128b);

#[rustfmt::skip]
const AES_EXP_TABLE: [u8; 256] = [
    0x01, 0x03, 0x05, 0x0f, 0x11, 0x33, 0x55, 0xff, 0x1a, 0x2e, 0x72, 0x96, 0xa1, 0xf8, 0x13, 0x35,
    0x5f, 0xe1, 0x38, 0x48, 0xd8, 0x73, 0x95, 0xa4, 0xf7, 0x02, 0x06, 0x0a, 0x1e, 0x22, 0x66, 0xaa,
    0xe5, 0x34, 0x5c, 0xe4, 0x37, 0x59, 0xeb, 0x26, 0x6a, 0xbe, 0xd9, 0x70, 0x90, 0xab, 0xe6, 0x31,
    0x53, 0xf5, 0x04, 0x0c, 0x14, 0x3c, 0x44, 0xcc, 0x4f, 0xd1, 0x68, 0xb8, 0xd3, 0x6e, 0xb2, 0xcd,
    0x4c, 0xd4, 0x67, 0xa9, 0xe0, 0x3b, 0x4d, 0xd7, 0x62, 0xa6, 0xf1, 0x08, 0x18, 0x28, 0x78, 0x88,
    0x83, 0x9e, 0xb9, 0xd0, 0x6b, 0xbd, 0xdc, 0x7f, 0x81, 0x98, 0xb3, 0xce, 0x49, 0xdb, 0x76, 0x9a,
    0xb5, 0xc4, 0x57, 0xf9, 0x10, 0x30, 0x50, 0xf0, 0x0b, 0x1d, 0x27, 0x69, 0xbb, 0xd6, 0x61, 0xa3,
    0xfe, 0x19, 0x2b, 0x7d, 0x87, 0x92, 0xad, 0xec, 0x2f, 0x71, 0x93, 0xae, 0xe9, 0x20, 0x60, 0xa0,
    0xfb, 0x16, 0x3a, 0x4e, 0xd2, 0x6d, 0xb7, 0xc2, 0x5d, 0xe7, 0x32, 0x56, 0xfa, 0x15, 0x3f, 0x41,
    0xc3, 0x5e, 0xe2, 0x3d, 0x47, 0xc9, 0x40, 0xc0, 0x5b, 0xed, 0x2c, 0x74, 0x9c, 0xbf, 0xda, 0x75,
    0x9f, 0xba, 0xd5, 0x64, 0xac, 0xef, 0x2a, 0x7e, 0x82, 0x9d, 0xbc, 0xdf, 0x7a, 0x8e, 0x89, 0x80,
    0x9b, 0xb6, 0xc1, 0x58, 0xe8, 0x23, 0x65, 0xaf, 0xea, 0x25, 0x6f, 0xb1, 0xc8, 0x43, 0xc5, 0x54,
    0xfc, 0x1f, 0x21, 0x63, 0xa5, 0xf4, 0x07, 0x09, 0x1b, 0x2d, 0x77, 0x99, 0xb0, 0xcb, 0x46, 0xca,
    0x45, 0xcf, 0x4a, 0xde, 0x79, 0x8b, 0x86, 0x91, 0xa8, 0xe3, 0x3e, 0x42, 0xc6, 0x51, 0xf3, 0x0e,
    0x12, 0x36, 0x5a, 0xee, 0x29, 0x7b, 0x8d, 0x8c, 0x8f, 0x8a, 0x85, 0x94, 0xa7, 0xf2, 0x0d, 0x17,
    0x39, 0x4b, 0xdd, 0x7c, 0x84, 0x97, 0xa2, 0xfd, 0x1c, 0x24, 0x6c, 0xb4, 0xc7, 0x52, 0xf6, 0x0
];

#[rustfmt::skip]
const AES_LOG_TABLE: [u8; 256] = [
	0x00, 0x00, 0x19, 0x01, 0x32, 0x02, 0x1a, 0xc6, 0x4b, 0xc7, 0x1b, 0x68, 0x33, 0xee, 0xdf, 0x03,
	0x64, 0x04, 0xe0, 0x0e, 0x34, 0x8d, 0x81, 0xef, 0x4c, 0x71, 0x08, 0xc8, 0xf8, 0x69, 0x1c, 0xc1,
	0x7d, 0xc2, 0x1d, 0xb5, 0xf9, 0xb9, 0x27, 0x6a, 0x4d, 0xe4, 0xa6, 0x72, 0x9a, 0xc9, 0x09, 0x78,
	0x65, 0x2f, 0x8a, 0x05, 0x21, 0x0f, 0xe1, 0x24, 0x12, 0xf0, 0x82, 0x45, 0x35, 0x93, 0xda, 0x8e,
	0x96, 0x8f, 0xdb, 0xbd, 0x36, 0xd0, 0xce, 0x94, 0x13, 0x5c, 0xd2, 0xf1, 0x40, 0x46, 0x83, 0x38,
	0x66, 0xdd, 0xfd, 0x30, 0xbf, 0x06, 0x8b, 0x62, 0xb3, 0x25, 0xe2, 0x98, 0x22, 0x88, 0x91, 0x10,
	0x7e, 0x6e, 0x48, 0xc3, 0xa3, 0xb6, 0x1e, 0x42, 0x3a, 0x6b, 0x28, 0x54, 0xfa, 0x85, 0x3d, 0xba,
	0x2b, 0x79, 0x0a, 0x15, 0x9b, 0x9f, 0x5e, 0xca, 0x4e, 0xd4, 0xac, 0xe5, 0xf3, 0x73, 0xa7, 0x57,
	0xaf, 0x58, 0xa8, 0x50, 0xf4, 0xea, 0xd6, 0x74, 0x4f, 0xae, 0xe9, 0xd5, 0xe7, 0xe6, 0xad, 0xe8,
	0x2c, 0xd7, 0x75, 0x7a, 0xeb, 0x16, 0x0b, 0xf5, 0x59, 0xcb, 0x5f, 0xb0, 0x9c, 0xa9, 0x51, 0xa0,
	0x7f, 0x0c, 0xf6, 0x6f, 0x17, 0xc4, 0x49, 0xec, 0xd8, 0x43, 0x1f, 0x2d, 0xa4, 0x76, 0x7b, 0xb7,
	0xcc, 0xbb, 0x3e, 0x5a, 0xfb, 0x60, 0xb1, 0x86, 0x3b, 0x52, 0xa1, 0x6c, 0xaa, 0x55, 0x29, 0x9d,
	0x97, 0xb2, 0x87, 0x90, 0x61, 0xbe, 0xdc, 0xfc, 0xbc, 0x95, 0xcf, 0xcd, 0x37, 0x3f, 0x5b, 0xd1,
	0x53, 0x39, 0x84, 0x3c, 0x41, 0xa2, 0x6d, 0x47, 0x14, 0x2a, 0x9e, 0x5d, 0x56, 0xf2, 0xd3, 0xab,
	0x44, 0x11, 0x92, 0xd9, 0x23, 0x20, 0x2e, 0x89, 0xb4, 0x7c, 0xb8, 0x26, 0x77, 0x99, 0xe3, 0xa5,
	0x67, 0x4a, 0xed, 0xde, 0xc5, 0x31, 0xfe, 0x18, 0x0d, 0x63, 0x8c, 0x80, 0xc0, 0xf7, 0x70, 0x07,
];

impl TowerField for AESTowerField8b {}

impl InvertOrZero for AESTowerField8b {
	fn invert_or_zero(self) -> Self {
		// TODO: use lookup table
		if self.0 != 0 {
			Self(AES_EXP_TABLE[(255 - AES_LOG_TABLE[self.0 as usize]) as usize % 255])
		} else {
			Self(0)
		}
	}
}

impl TowerFieldArithmetic for AESTowerField8b {
	fn multiply(self, rhs: Self) -> Self {
		let result = if self.0 != 0 && rhs.0 != 0 {
			let log_table_index =
				AES_LOG_TABLE[self.0 as usize] as usize + AES_LOG_TABLE[rhs.0 as usize] as usize;
			let log_table_index = if log_table_index > 254 {
				log_table_index - 255
			} else {
				log_table_index
			};

			unsafe {
				// Safety: `log_table_index` is smaller than 255 because:
				// - all values in `LOG_TABLE` do not exceed 254
				// - sum of two values do not exceed 254*2
				*AES_EXP_TABLE.get_unchecked(log_table_index)
			}
		} else {
			0
		};

		Self(result)
	}

	fn multiply_alpha(self) -> Self {
		// TODO: use lookup table
		// `0xD3` is the value isomorphic to 0x10 in BinaryField8b
		self * Self(0xD3)
	}

	fn square(self) -> Self {
		// TODO: use lookup table
		let result = if self.0 == 0 {
			0
		} else {
			AES_EXP_TABLE[(2 * AES_LOG_TABLE[self.0 as usize] as usize) % 255]
		};

		Self(result)
	}
}

impl From<AESTowerField8b> for BinaryField8b {
	fn from(value: AESTowerField8b) -> Self {
		const BASIS: [u8; 8] = [0x01, 0x3c, 0x8c, 0x8a, 0x59, 0x7a, 0x53, 0x27];

		(0..8)
			.map(|i| Self(BASIS[i]) * Self(value.0 >> i & 1))
			.sum()
	}
}

impl From<BinaryField8b> for AESTowerField8b {
	fn from(value: BinaryField8b) -> Self {
		const BASIS: [u8; 8] = [0x01, 0xbc, 0xb0, 0xec, 0xd3, 0x8d, 0x2e, 0x58];

		(0..8)
			.map(|i| Self(BASIS[i]) * Self(value.0 >> i & 1))
			.sum()
	}
}

binary_tower_arithmetic_recursive!(AESTowerField16b);
binary_tower_arithmetic_recursive!(AESTowerField32b);
binary_tower_arithmetic_recursive!(AESTowerField64b);
binary_tower_arithmetic_recursive!(AESTowerField128b);

#[cfg(test)]
mod tests {
	use super::*;

	use proptest::{arbitrary::any, proptest};

	fn check_square(f: impl Field) {
		assert_eq!(f.square(), f * f);
	}

	proptest! {
		#[test]
		fn test_square_8(a in any::<u8>()) {
			check_square(AESTowerField8b::from(a))
		}

		#[test]
		fn test_square_16(a in any::<u16>()) {
			check_square(AESTowerField16b::from(a))
		}

		#[test]
		fn test_square_32(a in any::<u32>()) {
			check_square(AESTowerField32b::from(a))
		}

		#[test]
		fn test_square_64(a in any::<u64>()) {
			check_square(AESTowerField64b::from(a))
		}

		#[test]
		fn test_square_128(a in any::<u128>()) {
			check_square(AESTowerField128b::from(a))
		}
	}

	fn check_invert(f: impl Field) {
		let inversed = f.invert();
		if f.is_zero().into() {
			assert!(bool::from(inversed.is_none()));
		} else {
			assert_eq!(inversed.unwrap() * f, Field::ONE);
		}
	}

	proptest! {
		#[test]
		fn test_invert_8(a in any::<u8>()) {
			check_invert(AESTowerField8b::from(a))
		}

		#[test]
		fn test_invert_16(a in any::<u16>()) {
			check_invert(AESTowerField16b::from(a))
		}

		#[test]
		fn test_invert_32(a in any::<u32>()) {
			check_invert(AESTowerField32b::from(a))
		}

		#[test]
		fn test_invert_64(a in any::<u64>()) {
			check_invert(AESTowerField64b::from(a))
		}

		#[test]
		fn test_invert_128(a in any::<u128>()) {
			check_invert(AESTowerField128b::from(a))
		}

		#[test]
		fn test_isomorphism_to_binary_tower8b_roundtrip(a in any::<u8>()) {
			let a_val = AESTowerField8b(a);
			let projected = BinaryField8b::from(a_val);
			let restored = AESTowerField8b::from(projected);
			assert_eq!(a_val, restored);
		}

		#[test]
		fn test_isomorphism_to_binary_tower8b_props(a in any::<u8>(), b in any::<u8>()) {
			let a_val = AESTowerField8b(a);
			let b_val = AESTowerField8b(b);
			assert_eq!(BinaryField8b::from(a_val) * BinaryField8b::from(b_val),
				BinaryField8b::from(a_val * b_val));
			assert_eq!(BinaryField8b::from(a_val) + BinaryField8b::from(b_val),
				BinaryField8b::from(a_val + b_val));
		}

		#[test]
		fn test_isomorphism_from_binary_tower8b_props(a in any::<u8>(), b in any::<u8>()) {
			let a_val = BinaryField8b(a);
			let b_val = BinaryField8b(b);
			assert_eq!(AESTowerField8b::from(a_val) * AESTowerField8b::from(b_val),
				AESTowerField8b::from(a_val * b_val));
			assert_eq!(AESTowerField8b::from(a_val) + AESTowerField8b::from(b_val),
				AESTowerField8b::from(a_val + b_val));
		}
	}

	fn check_mul_by_one<F: Field>(f: F) {
		assert_eq!(F::ONE * f, f);
		assert_eq!(f * F::ONE, f);
	}

	fn check_commutative<F: Field>(f_1: F, f_2: F) {
		assert_eq!(f_1 * f_2, f_2 * f_1);
	}

	fn check_associativity_and_lineraity<F: Field>(f_1: F, f_2: F, f_3: F) {
		assert_eq!(f_1 * (f_2 * f_3), (f_1 * f_2) * f_3);
		assert_eq!(f_1 * (f_2 + f_3), f_1 * f_2 + f_1 * f_3);
	}

	fn check_mul<F: Field>(f_1: F, f_2: F, f_3: F) {
		check_mul_by_one(f_1);
		check_mul_by_one(f_2);
		check_mul_by_one(f_3);

		check_commutative(f_1, f_2);
		check_commutative(f_1, f_3);
		check_commutative(f_2, f_3);

		check_associativity_and_lineraity(f_1, f_2, f_3);
		check_associativity_and_lineraity(f_1, f_3, f_2);
		check_associativity_and_lineraity(f_2, f_1, f_3);
		check_associativity_and_lineraity(f_2, f_3, f_1);
		check_associativity_and_lineraity(f_3, f_1, f_2);
		check_associativity_and_lineraity(f_3, f_2, f_1);
	}

	proptest! {
		#[test]
		fn test_mul_8(a in any::<u8>(), b in any::<u8>(), c in any::<u8>()) {
			check_mul(AESTowerField8b::from(a), AESTowerField8b::from(b), AESTowerField8b::from(c))
		}

		#[test]
		fn test_mul_16(a in any::<u16>(), b in any::<u16>(), c in any::<u16>()) {
			check_mul(AESTowerField16b::from(a), AESTowerField16b::from(b), AESTowerField16b::from(c))
		}

		#[test]
		fn test_mul_32(a in any::<u32>(), b in any::<u32>(), c in any::<u32>()) {
			check_mul(AESTowerField32b::from(a), AESTowerField32b::from(b), AESTowerField32b::from(c))
		}

		#[test]
		fn test_mul_64(a in any::<u64>(), b in any::<u64>(), c in any::<u64>()) {
			check_mul(AESTowerField64b::from(a), AESTowerField64b::from(b), AESTowerField64b::from(c))
		}

		#[test]
		fn test_mul_128(a in any::<u128>(), b in any::<u128>(), c in any::<u128>()) {
			check_mul(AESTowerField128b::from(a), AESTowerField128b::from(b), AESTowerField128b::from(c))
		}
	}
}
