// Copyright 2024-2025 Irreducible Inc.

pub use crate::arch::{
	packed_polyval_128::PackedBinaryPolyval1x128b, packed_polyval_256::PackedBinaryPolyval2x128b,
	packed_polyval_512::PackedBinaryPolyval4x128b,
};

#[cfg(test)]
mod test_utils {
	/// Test if `mult_func` operation is a valid multiply operation on the given values for
	/// all possible packed fields defined on 8-512 bits.
	macro_rules! define_multiply_tests {
		($mult_func:path, $constraint:ty) => {
			$crate::packed_binary_field::test_utils::define_check_packed_mul!(
				$mult_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_mul_packed_128(a_val in any::<u128>(), b_val in any::<u128>()) {
					TestMult::<$crate::arch::packed_polyval_128::PackedBinaryPolyval1x128b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_256(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
					TestMult::<$crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_512(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
					TestMult::<$crate::arch::packed_polyval_512::PackedBinaryPolyval4x128b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}
			}
		};
	}

	/// Test if `square_func` operation is a valid square operation on the given value for
	/// all possible packed fields.
	macro_rules! define_square_tests {
		($square_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_square!(
				$square_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_square_packed_128(a_val in any::<u128>()) {
					TestSquare::<$crate::arch::packed_polyval_128::PackedBinaryPolyval1x128b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_256(a_val in any::<[u128; 2]>()) {
					TestSquare::<$crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_512(a_val in any::<[u128; 4]>()) {
					TestSquare::<$crate::arch::packed_polyval_512::PackedBinaryPolyval4x128b>::test_square(a_val.into());
				}
			}
		};
	}

	/// Test if `invert_func` operation is a valid invert operation on the given value for
	/// all possible packed fields.
	macro_rules! define_invert_tests {
		($invert_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_inverse!(
				$invert_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_invert_packed_128(a_val in any::<u128>()) {
					TestInvert::<$crate::arch::packed_polyval_128::PackedBinaryPolyval1x128b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_256(a_val in any::<[u128; 2]>()) {
					TestInvert::<$crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_512(a_val in any::<[u128; 4]>()) {
					TestInvert::<$crate::arch::packed_polyval_512::PackedBinaryPolyval4x128b>::test_invert(a_val.into());
				}
			}
		};
	}

	macro_rules! define_transformation_tests {
		($constraint:path) => {
			$crate::packed_binary_field::test_utils::define_check_packed_transformation!(
				$constraint
			);

			proptest::proptest! {
				#[test]
				fn test_transformation_packed_128(a_val in proptest::prelude::any::<u128>()) {
					TestTransformation::<$crate::arch::packed_polyval_128::PackedBinaryPolyval1x128b>::test_transformation(a_val.into());
				}

				#[test]
				fn test_transformation_packed_256(a_val in proptest::prelude::any::<[u128; 2]>()) {
					TestTransformation::<$crate::arch::packed_polyval_256::PackedBinaryPolyval2x128b>::test_transformation(a_val.into());
				}

				#[test]
				fn test_transformation_packed_512(a_val in proptest::prelude::any::<[u128; 4]>()) {
					TestTransformation::<$crate::arch::packed_polyval_512::PackedBinaryPolyval4x128b>::test_transformation(a_val.into());
				}
			}
		};
	}

	pub(crate) use define_invert_tests;
	pub(crate) use define_multiply_tests;
	pub(crate) use define_square_tests;
	pub(crate) use define_transformation_tests;
}

#[cfg(test)]
mod tests {
	use std::ops::Mul;

	use proptest::{arbitrary::any, proptest};

	use super::test_utils::{
		define_invert_tests, define_multiply_tests, define_square_tests,
		define_transformation_tests,
	};
	use crate::{
		BinaryField128bPolyval, PackedBinaryField1x128b, PackedBinaryField2x128b,
		PackedBinaryField4x128b, PackedField,
		arch::{
			packed_polyval_128::PackedBinaryPolyval1x128b,
			packed_polyval_256::PackedBinaryPolyval2x128b,
			packed_polyval_512::PackedBinaryPolyval4x128b,
		},
		linear_transformation::PackedTransformationFactory,
		test_utils::implements_transformation_factory,
		underlier::WithUnderlier,
	};

	fn check_get_set<const WIDTH: usize, PT>(a: [u128; WIDTH], b: [u128; WIDTH])
	where
		PT: PackedField<Scalar = BinaryField128bPolyval>
			+ WithUnderlier<Underlier: From<[u128; WIDTH]>>,
	{
		let mut val = PT::from_underlier(a.into());
		for i in 0..WIDTH {
			assert_eq!(val.get(i), BinaryField128bPolyval::from(a[i]));
			val.set(i, BinaryField128bPolyval::from(b[i]));
			assert_eq!(val.get(i), BinaryField128bPolyval::from(b[i]));
		}
	}

	proptest! {
		#[test]
		fn test_get_set_256(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
			check_get_set::<2, PackedBinaryPolyval2x128b>(a, b);
		}

		#[test]
		fn test_get_set_512(a in any::<[u128; 4]>(), b in any::<[u128; 4]>()) {
			check_get_set::<4, PackedBinaryPolyval4x128b>(a, b);
		}
	}

	define_multiply_tests!(Mul::mul, PackedField);

	define_square_tests!(PackedField::square, PackedField);

	define_invert_tests!(PackedField::invert_or_zero, PackedField);

	#[allow(unused)]
	trait SelfTransformationFactory: PackedTransformationFactory<Self> {}

	impl<T: PackedTransformationFactory<T>> SelfTransformationFactory for T {}

	define_transformation_tests!(SelfTransformationFactory);

	/// Compile-time test to ensure packed fields implement `PackedTransformationFactory`.
	#[allow(unused)]
	const fn test_implement_transformation_factory() {
		// 128 bit packed polyval
		implements_transformation_factory::<PackedBinaryPolyval1x128b, PackedBinaryPolyval1x128b>();
		implements_transformation_factory::<PackedBinaryField1x128b, PackedBinaryPolyval1x128b>();

		// 256 bit packed polyval
		implements_transformation_factory::<PackedBinaryPolyval2x128b, PackedBinaryPolyval2x128b>();
		implements_transformation_factory::<PackedBinaryField2x128b, PackedBinaryPolyval2x128b>();

		// 512 bit packed polyval
		implements_transformation_factory::<PackedBinaryPolyval4x128b, PackedBinaryPolyval4x128b>();
		implements_transformation_factory::<PackedBinaryField4x128b, PackedBinaryPolyval4x128b>();
	}
}
