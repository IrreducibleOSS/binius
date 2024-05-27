// Copyright 2024 Ulvetanna Inc.

use super::{
	arithmetic_traits::InvertOrZero,
	binary_field::{binary_field, impl_field_extension, BinaryField, BinaryField1b},
	binary_field_arithmetic::TowerFieldArithmetic,
	mul_by_binary_field_1b, BinaryField8b, Error,
};
use crate::{
	affine_transformation::{FieldAffineTransformation, Transformation},
	as_packed_field::AsPackedField,
	binary_field_arithmetic::{impl_arithmetic_using_packed, impl_mul_primitive},
	binary_tower,
	packed::PackedField,
	underlier::U1,
	BinaryField128b, BinaryField16b, BinaryField32b, BinaryField64b, ExtensionField, Field,
	TowerExtensionField, TowerField,
};
use bytemuck::{Pod, Zeroable};
use rand::RngCore;
use std::{
	array,
	fmt::{Debug, Display, Formatter},
	iter::{Product, Step, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

// These fields represent a tower based on AES GF(2^8) field (GF(256)/x^8+x^4+x^3+x+1)
// that is isomorphically included into binary tower, i.e.:
//  - AESTowerField16b is GF(2^16) / (x^2 + x * x_2 + 1) where `x_2` is 0x10 from
// BinaryField8b isomorphically projected to AESTowerField8b.
//  - AESTowerField32b is GF(2^32) / (x^2 + x * x_3 + 1), where `x_3` is 0x1000 from AESTowerField16b.
//  ...
binary_field!(pub AESTowerField8b(u8), 0xD0);
binary_field!(pub AESTowerField16b(u16), 0x4745);
binary_field!(pub AESTowerField32b(u32), 0xBD478FAB);
binary_field!(pub AESTowerField64b(u64), 0x0DE1555D2BD78EB4);
binary_field!(pub AESTowerField128b(u128), 0x6DB54066349EDB96C33A87244A742678);

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

impl_field_extension!(BinaryField1b(U1) < @8 => AESTowerField8b(u8));
impl_field_extension!(BinaryField1b(U1) < @16 => AESTowerField16b(u16));
impl_field_extension!(BinaryField1b(U1) < @32 => AESTowerField32b(u32));
impl_field_extension!(BinaryField1b(U1) < @64 => AESTowerField64b(u64));
impl_field_extension!(BinaryField1b(U1) < @128 => AESTowerField128b(u128));

mul_by_binary_field_1b!(AESTowerField8b);
mul_by_binary_field_1b!(AESTowerField16b);
mul_by_binary_field_1b!(AESTowerField32b);
mul_by_binary_field_1b!(AESTowerField64b);
mul_by_binary_field_1b!(AESTowerField128b);

impl_arithmetic_using_packed!(AESTowerField8b);
impl_arithmetic_using_packed!(AESTowerField16b);
impl_arithmetic_using_packed!(AESTowerField32b);
impl_arithmetic_using_packed!(AESTowerField64b);
impl_arithmetic_using_packed!(AESTowerField128b);

impl TowerField for AESTowerField8b {
	fn mul_primitive(self, iota: usize) -> Result<Self, Error> {
		match iota {
			0..=1 => Ok(self * ISOMORPHIC_ALPHAS[iota]),
			2 => Ok(self.multiply_alpha()),
			_ => Err(Error::ExtensionDegreeMismatch),
		}
	}
}

impl From<AESTowerField8b> for BinaryField8b {
	fn from(value: AESTowerField8b) -> Self {
		const AFFINE_TRANSFORMATION: FieldAffineTransformation<BinaryField8b> =
			FieldAffineTransformation::new_const(&[
				Self(0x01),
				Self(0x3c),
				Self(0x8c),
				Self(0x8a),
				Self(0x59),
				Self(0x7a),
				Self(0x53),
				Self(0x27),
			]);

		AFFINE_TRANSFORMATION.transform(&value)
	}
}

impl From<BinaryField8b> for AESTowerField8b {
	fn from(value: BinaryField8b) -> Self {
		const AFFINE_TRANSFORMATION: FieldAffineTransformation<AESTowerField8b> =
			FieldAffineTransformation::new_const(&[
				Self(0x01),
				Self(0xbc),
				Self(0xb0),
				Self(0xec),
				Self(0xd3),
				Self(0x8d),
				Self(0x2e),
				Self(0x58),
			]);

		AFFINE_TRANSFORMATION.transform(&value)
	}
}

/// Values isomorphic to 0x02, 0x04 and 0x10 in BinaryField8b
const ISOMORPHIC_ALPHAS: [AESTowerField8b; 3] = [
	AESTowerField8b(0xBC),
	AESTowerField8b(0xB0),
	AESTowerField8b(0xD3),
];

// MulPrimitive implementation for AES tower
impl_mul_primitive!(AESTowerField16b,
	mul_by 0 => ISOMORPHIC_ALPHAS[0],
	mul_by 1 => ISOMORPHIC_ALPHAS[1],
	repack 2 => AESTowerField8b,
	repack 3 => AESTowerField16b,
);
impl_mul_primitive!(AESTowerField32b,
	mul_by 0 => ISOMORPHIC_ALPHAS[0],
	mul_by 1 => ISOMORPHIC_ALPHAS[1],
	repack 2 => AESTowerField8b,
	repack 3 => AESTowerField16b,
	repack 4 => AESTowerField32b,
);
impl_mul_primitive!(AESTowerField64b,
	mul_by 0 => ISOMORPHIC_ALPHAS[0],
	mul_by 1 => ISOMORPHIC_ALPHAS[1],
	repack 2 => AESTowerField8b,
	repack 3 => AESTowerField16b,
	repack 4 => AESTowerField32b,
	repack 5 => AESTowerField64b,
);
impl_mul_primitive!(AESTowerField128b,
	mul_by 0 => ISOMORPHIC_ALPHAS[0],
	mul_by 1 => ISOMORPHIC_ALPHAS[1],
	repack 2 => AESTowerField8b,
	repack 3 => AESTowerField16b,
	repack 4 => AESTowerField32b,
	repack 5 => AESTowerField64b,
	repack 6 => AESTowerField128b,
);

/// We use this function to define isomorphisms between AES and binary tower fields.
/// Repack field as 8b packed field and apply isomorphism for each 8b element
fn convert_as_packed_8b<F1, F2, Scalar1, Scalar2>(val: F1) -> F2
where
	Scalar1: Field,
	Scalar2: Field + From<Scalar1>,
	F1: AsPackedField<Scalar1>,
	F2: AsPackedField<Scalar2>,
{
	assert_eq!(F1::Packed::WIDTH, F2::Packed::WIDTH);

	let val_repacked = val.to_packed();
	let converted_repacked = F2::Packed::from_fn(|i| val_repacked.get(i).into());

	F2::from_packed(converted_repacked)
}

macro_rules! impl_tower_field_conversion {
	($aes_field:ty, $binary_field:ty) => {
		impl From<$aes_field> for $binary_field {
			fn from(value: $aes_field) -> Self {
				convert_as_packed_8b::<_, _, AESTowerField8b, BinaryField8b>(value)
			}
		}

		impl From<$binary_field> for $aes_field {
			fn from(value: $binary_field) -> Self {
				convert_as_packed_8b::<_, _, BinaryField8b, AESTowerField8b>(value)
			}
		}
	};
}

impl_tower_field_conversion!(AESTowerField16b, BinaryField16b);
impl_tower_field_conversion!(AESTowerField32b, BinaryField32b);
impl_tower_field_conversion!(AESTowerField64b, BinaryField64b);
impl_tower_field_conversion!(AESTowerField128b, BinaryField128b);

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{binary_field::tests::is_binary_field_valid_generator, underlier::WithUnderlier};

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
		if f.is_zero() {
			assert!(inversed.is_none());
		} else {
			assert_eq!(inversed.unwrap() * f, Field::ONE);
		}
	}

	fn check_isomorphism_preserves_ops<F1: Field, F2: Field + From<F1>>(a: F1, b: F1) {
		assert_eq!(F2::from(a * b), F2::from(a) * F2::from(b));
		assert_eq!(F2::from(a + b), F2::from(a) + F2::from(b));
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
		fn test_isomorphism_8b(a in any::<u8>(), b in any::<u8>()) {
			check_isomorphism_preserves_ops::<AESTowerField8b, BinaryField8b>(a.into(), b.into());
			check_isomorphism_preserves_ops::<BinaryField8b, AESTowerField8b>(a.into(), b.into());
		}

		#[test]
		fn test_isomorphism_16b(a in any::<u16>(), b in any::<u16>()) {
			check_isomorphism_preserves_ops::<AESTowerField16b, BinaryField16b>(a.into(), b.into());
			check_isomorphism_preserves_ops::<BinaryField16b, AESTowerField16b>(a.into(), b.into());
		}

		#[test]
		fn test_isomorphism_32b(a in any::<u32>(), b in any::<u32>()) {
			check_isomorphism_preserves_ops::<AESTowerField32b, BinaryField32b>(a.into(), b.into());
			check_isomorphism_preserves_ops::<BinaryField32b, AESTowerField32b>(a.into(), b.into());
		}

		#[test]
		fn test_isomorphism_64b(a in any::<u64>(), b in any::<u64>()) {
			check_isomorphism_preserves_ops::<AESTowerField64b, BinaryField64b>(a.into(), b.into());
			check_isomorphism_preserves_ops::<BinaryField64b, AESTowerField64b>(a.into(), b.into());
		}

		#[test]
		fn test_isomorphism_128b(a in any::<u128>(), b in any::<u128>()) {
			check_isomorphism_preserves_ops::<AESTowerField128b, BinaryField128b>(a.into(), b.into());
			check_isomorphism_preserves_ops::<BinaryField128b, AESTowerField128b>(a.into(), b.into());
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

		#[test]
		fn test_conversion_roundtrip(a in any::<u8>()) {
			let a_val = AESTowerField8b(a);
			let converted = BinaryField8b::from(a_val);
			assert_eq!(a_val, AESTowerField8b::from(converted));
		}
	}

	#[test]
	fn test_multiplicative_generators() {
		assert!(is_binary_field_valid_generator::<AESTowerField8b>());
		assert!(is_binary_field_valid_generator::<AESTowerField16b>());
		assert!(is_binary_field_valid_generator::<AESTowerField32b>());
		assert!(is_binary_field_valid_generator::<AESTowerField64b>());
		assert!(is_binary_field_valid_generator::<AESTowerField128b>());
	}

	fn test_mul_primitive<F: TowerField + WithUnderlier<Underlier: From<u8>>>(val: F, iota: usize) {
		let result = val.mul_primitive(iota);
		let expected = match iota {
			0..=2 => Ok(val * F::from(F::Underlier::from(ISOMORPHIC_ALPHAS[iota].to_underlier()))),
			_ => <F as ExtensionField<BinaryField1b>>::basis(1 << iota).map(|b| val * b),
		};
		assert_eq!(result.is_ok(), expected.is_ok());
		if result.is_ok() {
			assert_eq!(result.unwrap(), expected.unwrap());
		} else {
			assert!(matches!(result.unwrap_err(), Error::ExtensionDegreeMismatch));
		}
	}

	proptest! {
		#[test]
		fn test_mul_primitive_8b(val in 0u8.., iota in 3usize..8) {
			test_mul_primitive::<AESTowerField8b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_16b(val in 0u16.., iota in 3usize..8) {
			test_mul_primitive::<AESTowerField16b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_32b(val in 0u32.., iota in 3usize..8) {
			test_mul_primitive::<AESTowerField32b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_64b(val in 0u64.., iota in 3usize..8) {
			test_mul_primitive::<AESTowerField64b>(val.into(), iota)
		}

		#[test]
		fn test_mul_primitive_128b(val in 0u128.., iota in 3usize..8) {
			test_mul_primitive::<AESTowerField128b>(val.into(), iota)
		}
	}
}
