// Copyright 2023 Ulvetanna Inc.

pub use crate::field::arch::{packed_128::*, packed_256::*, packed_512::*};

/// Common code to test different multiply, square and invert implementations
#[cfg(test)]
pub mod test_utils {
	pub struct Unit;

	impl From<u64> for Unit {
		fn from(_: u64) -> Self {
			Self
		}
	}

	impl From<u128> for Unit {
		fn from(_: u128) -> Self {
			Self
		}
	}

	impl From<[u128; 2]> for Unit {
		fn from(_: [u128; 2]) -> Self {
			Self
		}
	}

	impl From<[u128; 4]> for Unit {
		fn from(_: [u128; 4]) -> Self {
			Self
		}
	}

	/// We use such helper macros to run tests only for the
	/// types that implement the `constraint` trait.
	/// The idea is inspired by `impls` trait.
	macro_rules! define_check_packed_mul {
		($mult_func:path, $constraint:path) => {
			#[allow(unused)]
			trait TestMulTrait<T> {
				fn test_mul(_a: T, _b: T) {}
			}

			impl<T> TestMulTrait<$crate::field::packed_binary_field::test_utils::Unit> for T {}

			struct TestMult<T>(std::marker::PhantomData<T>);

			impl<T: $constraint + PackedField> TestMult<T> {
				fn test_mul(a: T, b: T) {
					let c = $mult_func(a, b);
					for i in 0..T::WIDTH {
						assert_eq!(c.get(i), a.get(i) * b.get(i));
					}
				}
			}
		};
	}

	pub(crate) use define_check_packed_mul;

	macro_rules! define_check_packed_square {
		($square_func:path, $constraint:path) => {
			#[allow(unused)]
			trait TestSquareTrait<T> {
				fn test_square(_a: T) {}
			}

			impl<T> TestSquareTrait<$crate::field::packed_binary_field::test_utils::Unit> for T {}

			struct TestSquare<T>(std::marker::PhantomData<T>);

			impl<T: $constraint + PackedField> TestSquare<T> {
				fn test_square(a: T) {
					let c = $square_func(a);
					for i in 0..T::WIDTH {
						assert_eq!(c.get(i), a.get(i) * a.get(i));
					}
				}
			}
		};
	}

	pub(crate) use define_check_packed_square;

	macro_rules! define_check_packed_inverse {
		($invert_func:path, $constraint:path) => {
			#[allow(unused)]
			trait TestInvertTrait<T> {
				fn test_invert(_a: T) {}
			}

			impl<T> TestInvertTrait<$crate::field::packed_binary_field::test_utils::Unit> for T {}

			struct TestInvert<T>(std::marker::PhantomData<T>);

			impl<T: $constraint + PackedField> TestInvert<T> {
				fn test_invert(a: T) {
					use ff::Field;

					let c = $invert_func(a);
					for i in 0..T::WIDTH {
						assert!(
							(c.get(i).is_zero().into()
								&& a.get(i).is_zero().into()
								&& c.get(i).is_zero().into())
								|| T::Scalar::ONE == a.get(i) * c.get(i)
						);
					}
				}
			}
		};
	}

	pub(crate) use define_check_packed_inverse;

	macro_rules! define_check_packed_mul_alpha {
		($mul_alpha_func:path, $constraint:path) => {
			#[allow(unused)]
			trait TestMulAlphaTrait<T> {
				fn test_mul_alpha(_a: T) {}
			}

			impl<T> TestMulAlphaTrait<$crate::field::packed_binary_field::test_utils::Unit> for T {}

			struct TestMulAlpha<T>(std::marker::PhantomData<T>);

			impl<T: $constraint + PackedField> TestMulAlpha<T>
			where
				T::Scalar: $crate::field::arithmetic_traits::MulAlpha,
			{
				fn test_mul_alpha(a: T) {
					use $crate::field::arithmetic_traits::MulAlpha;

					let c = $mul_alpha_func(a);
					for i in 0..T::WIDTH {
						assert_eq!(c.get(i), MulAlpha::mul_alpha(a.get(i)));
					}
				}
			}
		};
	}

	pub(crate) use define_check_packed_mul_alpha;

	/// Test if `mult_func` operation is a valid multiply operation on the given values for
	/// all possible packed fields defined on u128.
	macro_rules! define_multiply_tests {
		($mult_func:path, $constraint:path) => {
			$crate::field::packed_binary_field::test_utils::define_check_packed_mul!(
				$mult_func,
				$constraint
			);

			proptest::proptest! {
				#[test]
				fn test_mul_packed_64(a_val in proptest::prelude::any::<u64>(), b_val in proptest::prelude::any::<u64>()) {
					use $crate::field::arch::packed_64::*;

					TestMult::<PackedBinaryField64x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField32x2b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField16x4b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField8x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField4x16b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField2x32b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField1x64b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_128(a_val in proptest::prelude::any::<u128>(), b_val in proptest::prelude::any::<u128>()) {
					use $crate::field::arch::packed_128::*;

					TestMult::<PackedBinaryField128x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField64x2b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField32x4b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField16x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField8x16b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField4x32b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField2x64b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField1x128b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_256(a_val in proptest::prelude::any::<[u128; 2]>(), b_val in proptest::prelude::any::<[u128; 2]>()) {
					use $crate::field::arch::packed_256::*;

					TestMult::<PackedBinaryField256x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField128x2b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField64x4b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField32x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField16x16b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField8x32b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField4x64b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField2x128b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_512(a_val in proptest::prelude::any::<[u128; 4]>(), b_val in proptest::prelude::any::<[u128; 4]>()) {
					use $crate::field::arch::packed_512::*;

					TestMult::<PackedBinaryField512x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField256x2b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField128x4b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField64x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField32x16b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField16x32b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField8x64b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryField4x128b>::test_mul(a_val.into(), b_val.into());
				}
			}
		};
	}

	/// Test if `square_func` operation is a valid square operation on the given value for
	/// all possible packed fields.
	macro_rules! define_square_tests {
		($square_func:path, $constraint:path) => {
			$crate::field::packed_binary_field::test_utils::define_check_packed_square!(
				$square_func,
				$constraint
			);

			proptest::proptest! {
				#[test]
				fn test_square_packed_64(a_val in proptest::prelude::any::<u64>()) {
					use $crate::field::arch::packed_64::*;

					TestSquare::<PackedBinaryField64x1b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField32x2b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField16x4b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField8x8b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField4x16b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField2x32b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField1x64b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_128(a_val in proptest::prelude::any::<u128>()) {
					use $crate::field::arch::packed_128::*;

					TestSquare::<PackedBinaryField128x1b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField64x2b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField32x4b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField16x8b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField8x16b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField4x32b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField2x64b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField1x128b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_256(a_val in proptest::prelude::any::<[u128; 2]>()) {
					use $crate::field::arch::packed_256::*;

					TestSquare::<PackedBinaryField256x1b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField128x2b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField64x4b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField32x8b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField16x16b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField8x32b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField4x64b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField2x128b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_512(a_val in proptest::prelude::any::<[u128; 4]>()) {
					use $crate::field::arch::packed_512::*;

					TestSquare::<PackedBinaryField512x1b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField256x2b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField128x4b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField64x8b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField32x16b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField16x32b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField8x64b>::test_square(a_val.into());
					TestSquare::<PackedBinaryField4x128b>::test_square(a_val.into());
				}
			}
		};
	}

	/// Test if `invert_func` operation is a valid invert operation on the given value for
	/// all possible packed fields.
	macro_rules! define_invert_tests {
		($invert_func:path, $constraint:path) => {
			$crate::field::packed_binary_field::test_utils::define_check_packed_inverse!(
				$invert_func,
				$constraint
			);

			proptest::proptest! {
				#[test]
				fn test_invert_packed_64(a_val in proptest::prelude::any::<u64>()) {
					use $crate::field::arch::packed_64::*;

					TestInvert::<PackedBinaryField64x1b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField32x2b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField16x4b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField8x8b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField4x16b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField2x32b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField1x64b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_128(a_val in proptest::prelude::any::<u128>()) {
					use $crate::field::arch::packed_128::*;

					TestInvert::<PackedBinaryField128x1b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField64x2b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField32x4b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField16x8b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField8x16b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField4x32b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField2x64b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField1x128b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_256(a_val in proptest::prelude::any::<[u128; 2]>()) {
					use $crate::field::arch::packed_256::*;

					TestInvert::<PackedBinaryField256x1b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField128x2b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField64x4b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField32x8b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField16x16b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField8x32b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField4x64b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField2x128b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_512(a_val in proptest::prelude::any::<[u128; 4]>()) {
					use $crate::field::arch::packed_512::*;

					TestInvert::<PackedBinaryField512x1b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField256x2b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField128x4b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField64x8b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField32x16b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField16x32b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField8x64b>::test_invert(a_val.into());
					TestInvert::<PackedBinaryField4x128b>::test_invert(a_val.into());
				}
			}
		};
	}

	/// Test if `mul_alpha_func` operation is a valid multiply by alpha operation on the given value for
	/// all possible packed fields.
	macro_rules! define_mul_alpha_tests {
		($mul_alpha_func:path, $constraint:path) => {
			$crate::field::packed_binary_field::test_utils::define_check_packed_mul_alpha!(
				$mul_alpha_func,
				$constraint
			);

			proptest::proptest! {
				#[test]
				fn test_mul_alpha_packed_64(a_val in proptest::prelude::any::<u64>()) {
					use $crate::field::arch::packed_64::*;

					TestMulAlpha::<PackedBinaryField64x1b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField32x2b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField16x4b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField8x8b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField4x16b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField2x32b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField1x64b>::test_mul_alpha(a_val.into());
				}

				#[test]
				fn test_mul_alpha_packed_128(a_val in proptest::prelude::any::<u128>()) {
					use $crate::field::arch::packed_128::*;

					TestMulAlpha::<PackedBinaryField128x1b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField64x2b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField32x4b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField16x8b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField8x16b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField4x32b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField2x64b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField1x128b>::test_mul_alpha(a_val.into());
				}

				#[test]
				fn test_mul_alpha_packed_256(a_val in proptest::prelude::any::<[u128; 2]>()) {
					use $crate::field::arch::packed_256::*;

					TestMulAlpha::<PackedBinaryField256x1b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField128x2b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField64x4b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField32x8b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField16x16b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField8x32b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField4x64b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField2x128b>::test_mul_alpha(a_val.into());
				}

				#[test]
				fn test_mul_alpha_packed_512(a_val in proptest::prelude::any::<[u128; 4]>()) {
					use $crate::field::arch::packed_512::*;

					TestMulAlpha::<PackedBinaryField512x1b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField256x2b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField128x4b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField64x8b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField32x16b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField16x32b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField8x64b>::test_mul_alpha(a_val.into());
					TestMulAlpha::<PackedBinaryField4x128b>::test_mul_alpha(a_val.into());
				}
			}
		};
	}

	pub(crate) use define_invert_tests;
	pub(crate) use define_mul_alpha_tests;
	pub(crate) use define_multiply_tests;
	pub(crate) use define_square_tests;
}

#[cfg(test)]
mod tests {
	use super::{
		test_utils::{
			define_invert_tests, define_mul_alpha_tests, define_multiply_tests, define_square_tests,
		},
		*,
	};
	use crate::field::{arithmetic_traits::MulAlpha, BinaryField8b, Field, PackedField};
	use proptest::prelude::*;
	use rand::{rngs::StdRng, thread_rng, SeedableRng};
	use std::{iter::repeat_with, ops::Mul};

	fn test_add_packed<P: PackedField + From<u128>>(a_val: u128, b_val: u128) {
		let a = P::from(a_val);
		let b = P::from(b_val);
		let c = a + b;
		for i in 0..P::WIDTH {
			assert_eq!(c.get(i), a.get(i) + b.get(i));
		}
	}

	fn test_mul_packed<P: PackedField>(a: P, b: P) {
		let c = a * b;
		for i in 0..P::WIDTH {
			assert_eq!(c.get(i), a.get(i) * b.get(i));
		}
	}

	fn test_mul_packed_random<P: PackedField>(mut rng: impl Rng) {
		test_mul_packed(P::random(&mut rng), P::random(&mut rng))
	}

	fn test_set_then_get<P: PackedField>() {
		let mut rng = StdRng::seed_from_u64(0);
		let mut elem = P::random(&mut rng);

		let scalars = repeat_with(|| Field::random(&mut rng))
			.take(P::WIDTH)
			.collect::<Vec<P::Scalar>>();

		for (i, val) in scalars.iter().enumerate() {
			elem.set(i, *val);
		}
		for (i, val) in scalars.iter().enumerate() {
			assert_eq!(elem.get(i), *val);
		}
	}

	#[test]
	fn test_set_then_get_4b() {
		test_set_then_get::<PackedBinaryField4x32b>();
	}

	#[test]
	fn test_set_then_get_32b() {
		test_set_then_get::<PackedBinaryField4x32b>();
	}

	#[test]
	fn test_set_then_get_128b() {
		test_set_then_get::<PackedBinaryField1x128b>();
	}

	// TODO: Generate lots more proptests using macros
	proptest! {
		#[test]
		fn test_add_packed_128x1b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField128x1b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_16x8b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField16x8b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_8x16b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField8x16b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_4x32b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField4x32b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_2x64b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField2x64b>(a_val, b_val)
		}

		#[test]
		fn test_add_packed_1x128b(a_val in any::<u128>(), b_val in any::<u128>()) {
			test_add_packed::<PackedBinaryField1x128b>(a_val, b_val)
		}
	}

	#[test]
	fn test_mul_packed_256x1b() {
		test_mul_packed_random::<PackedBinaryField256x1b>(thread_rng())
	}

	#[test]
	fn test_mul_packed_32x8b() {
		test_mul_packed_random::<PackedBinaryField32x8b>(thread_rng())
	}

	#[test]
	fn test_mul_packed_16x16b() {
		test_mul_packed_random::<PackedBinaryField16x16b>(thread_rng())
	}

	#[test]
	fn test_mul_packed_8x32b() {
		test_mul_packed_random::<PackedBinaryField8x32b>(thread_rng())
	}

	#[test]
	fn test_mul_packed_4x64b() {
		test_mul_packed_random::<PackedBinaryField4x64b>(thread_rng())
	}

	#[test]
	fn test_mul_packed_2x128b() {
		test_mul_packed_random::<PackedBinaryField2x128b>(thread_rng())
	}

	#[test]
	#[rustfmt::skip]
	fn test_interleave_8b() {
		let a = PackedBinaryField16x8b::from(
			[
				0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
				0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
			]
				.map(BinaryField8b::new),
		);
		let b = PackedBinaryField16x8b::from(
			[
				0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
				0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
			]
				.map(BinaryField8b::new),
		);

		let (c, d) = a.interleave(b, 0);
		let expected_c = PackedBinaryField16x8b::from(
			[
				0x00, 0x10, 0x02, 0x12, 0x04, 0x14, 0x06, 0x16,
				0x08, 0x18, 0x0a, 0x1a, 0x0c, 0x1c, 0x0e, 0x1e,
			]
				.map(BinaryField8b::new),
		);
		let expected_d = PackedBinaryField16x8b::from(
			[
				0x01, 0x11, 0x03, 0x13, 0x05, 0x15, 0x07, 0x17,
				0x09, 0x19, 0x0b, 0x1b, 0x0d, 0x1d, 0x0f, 0x1f,
			]
				.map(BinaryField8b::new),
		);
		assert_eq!(c, expected_c);
		assert_eq!(d, expected_d);

		let (c, d) = a.interleave(b, 1);
		let expected_c = PackedBinaryField16x8b::from(
			[
				0x00, 0x01, 0x10, 0x11, 0x04, 0x05, 0x14, 0x15,
				0x08, 0x09, 0x18, 0x19, 0x0c, 0x0d, 0x1c, 0x1d,
			]
				.map(BinaryField8b::new),
		);
		let expected_d = PackedBinaryField16x8b::from(
			[
				0x02, 0x03, 0x12, 0x13, 0x06, 0x07, 0x16, 0x17,
				0x0a, 0x0b, 0x1a, 0x1b, 0x0e, 0x0f, 0x1e, 0x1f,
			]
				.map(BinaryField8b::new),
		);
		assert_eq!(c, expected_c);
		assert_eq!(d, expected_d);

		let (c, d) = a.interleave(b, 2);
		let expected_c = PackedBinaryField16x8b::from(
			[
				0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
				0x08, 0x09, 0x0a, 0x0b, 0x18, 0x19, 0x1a, 0x1b,
			]
				.map(BinaryField8b::new),
		);
		let expected_d = PackedBinaryField16x8b::from(
			[
				0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17,
				0x0c, 0x0d, 0x0e, 0x0f, 0x1c, 0x1d, 0x1e, 0x1f,
			]
				.map(BinaryField8b::new),
		);
		assert_eq!(c, expected_c);
		assert_eq!(d, expected_d);

		let (c, d) = a.interleave(b, 3);
		let expected_c = PackedBinaryField16x8b::from(
			[
				0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
				0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
			]
				.map(BinaryField8b::new),
		);
		let expected_d = PackedBinaryField16x8b::from(
			[
				0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
				0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
			]
				.map(BinaryField8b::new),
		);
		assert_eq!(c, expected_c);
		assert_eq!(d, expected_d);
	}

	#[test]
	fn test_interleave_1b() {
		let a = PackedBinaryField128x1b::from(0x0000000000000000ffffffffffffffffu128);
		let b = PackedBinaryField128x1b::from(0xffffffffffffffff0000000000000000u128);

		let c = PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaa5555555555555555u128);
		let d = PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaa5555555555555555u128);
		assert_eq!(a.interleave(b, 0), (c, d));
		assert_eq!(c.interleave(d, 0), (a, b));

		let c = PackedBinaryField128x1b::from(0xcccccccccccccccc3333333333333333u128);
		let d = PackedBinaryField128x1b::from(0xcccccccccccccccc3333333333333333u128);
		assert_eq!(a.interleave(b, 1), (c, d));
		assert_eq!(c.interleave(d, 1), (a, b));

		let c = PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f00f0f0f0f0f0f0f0fu128);
		let d = PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f00f0f0f0f0f0f0f0fu128);
		assert_eq!(a.interleave(b, 2), (c, d));
		assert_eq!(c.interleave(d, 2), (a, b));
	}

	#[test]
	fn test_iter_size_hint() {
		assert_valid_iterator_with_exact_size_hint::<crate::field::BinaryField128b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::BinaryField32b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::BinaryField1b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::PackedBinaryField128x1b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::PackedBinaryField64x2b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::PackedBinaryField32x4b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::PackedBinaryField16x16b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::PackedBinaryField8x32b>();
		assert_valid_iterator_with_exact_size_hint::<crate::field::PackedBinaryField4x64b>();
	}

	fn assert_valid_iterator_with_exact_size_hint<P: PackedField>() {
		assert_eq!(P::default().iter().size_hint(), (P::WIDTH, Some(P::WIDTH)));
		assert_eq!(P::default().into_iter().size_hint(), (P::WIDTH, Some(P::WIDTH)));
		assert_eq!(P::default().iter().count(), P::WIDTH);
		assert_eq!(P::default().into_iter().count(), P::WIDTH);
	}

	define_multiply_tests!(Mul::mul, PackedField);

	define_square_tests!(PackedField::square, PackedField);

	define_invert_tests!(PackedField::invert_or_zero, PackedField);

	define_mul_alpha_tests!(MulAlpha::mul_alpha, MulAlpha);
}
