// Copyright 2023 Ulvetanna Inc.

pub use crate::field::arch::{packed_128::*, packed_256::*};

/// Common code to test different multiply, square and invert implementations
#[cfg(test)]
pub mod test_utils {
	/// Test if `mult_func` operation is a valid multiply operation on the given values for
	/// all possible packed fields defined on u128.
	macro_rules! define_multiply_tests {
		($mult_func:ident, $constraint:ident) => {
			fn check_mul_packed<P: $constraint>(a: P, b: P) {
				let c = $mult_func(a, b);
				for i in 0..P::WIDTH {
					assert_eq!(c.get(i), a.get(i) * b.get(i));
				}
			}

			fn check_mul_packed_u128(a_val: u128, b_val: u128) {
				check_mul_packed::<PackedBinaryField128x1b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField64x2b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField32x4b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField16x8b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField8x16b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField4x32b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField2x64b>(a_val.into(), b_val.into());
				check_mul_packed::<PackedBinaryField1x128b>(a_val.into(), b_val.into());
			}

			proptest! {
				#[test]
				fn test_mul_packed_128(a_val in any::<u128>(), b_val in any::<u128>()) {
					check_mul_packed_u128(a_val, b_val);
				}
			}
		};
	}

	/// Test if `square_func` operation is a valid square operation on the given value for
	/// all possible packed fields.
	macro_rules! define_square_tests {
		($square_func:ident, $constraint:ident) => {
			fn check_square_packed<P: $constraint>(a: P) {
				let c = $square_func(a);
				for i in 0..P::WIDTH {
					assert_eq!(c.get(i), a.get(i) * a.get(i));
				}
			}

			fn check_square_packed_128(a_val: u128) {
				check_square_packed::<PackedBinaryField128x1b>(a_val.into());
				check_square_packed::<PackedBinaryField64x2b>(a_val.into());
				check_square_packed::<PackedBinaryField32x4b>(a_val.into());
				check_square_packed::<PackedBinaryField16x8b>(a_val.into());
				check_square_packed::<PackedBinaryField8x16b>(a_val.into());
				check_square_packed::<PackedBinaryField4x32b>(a_val.into());
				check_square_packed::<PackedBinaryField2x64b>(a_val.into());
				check_square_packed::<PackedBinaryField1x128b>(a_val.into());
			}

			proptest! {
				#[test]
				fn test_square_packed_128(a_val in any::<u128>()) {
					check_square_packed_128(a_val);
				}
			}
		};
	}

	/// Test if `invert_func` operation is a valid invert operation on the given value for
	/// all possible packed fields.
	macro_rules! define_invert_tests {
		($invert_func:ident, $constraint:ident) => {
			fn test_inverse_packed<P: $constraint>(a: P) {
				use $crate::field::Field;

				let c = $invert_func(a);
				for i in 0..P::WIDTH {
					assert!(
						(c.get(i).is_zero().into()
							&& a.get(i).is_zero().into()
							&& c.get(i).is_zero().into())
							|| P::Scalar::ONE == a.get(i) * c.get(i)
					);
				}
			}

			pub fn check_inverse_packed_128(a_val: u128) {
				test_inverse_packed::<PackedBinaryField128x1b>(a_val.into());
				test_inverse_packed::<PackedBinaryField64x2b>(a_val.into());
				test_inverse_packed::<PackedBinaryField32x4b>(a_val.into());
				test_inverse_packed::<PackedBinaryField16x8b>(a_val.into());
				test_inverse_packed::<PackedBinaryField8x16b>(a_val.into());
				test_inverse_packed::<PackedBinaryField4x32b>(a_val.into());
				test_inverse_packed::<PackedBinaryField2x64b>(a_val.into());
				test_inverse_packed::<PackedBinaryField1x128b>(a_val.into());
			}

			proptest! {
				#[test]
				fn test_invert_packed_128(a_val in any::<u128>()) {
					check_inverse_packed_128(a_val);
				}
			}
		};
	}

	pub(crate) use define_invert_tests;
	pub(crate) use define_multiply_tests;
	pub(crate) use define_square_tests;
}

#[cfg(test)]
mod tests {
	use super::*;

	use crate::field::{BinaryField8b, Field, PackedField};

	use proptest::prelude::*;
	use rand::{rngs::StdRng, thread_rng, SeedableRng};
	use std::iter::repeat_with;

	use super::test_utils::{define_invert_tests, define_multiply_tests, define_square_tests};

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

	fn mult<P: PackedField>(a: P, b: P) -> P {
		a * b
	}

	define_multiply_tests!(mult, PackedField);

	fn square<P: PackedField>(a: P) -> P {
		a.square()
	}

	define_square_tests!(square, PackedField);

	fn invert<P: PackedField>(a: P) -> P {
		a.invert()
	}

	define_invert_tests!(invert, PackedField);
}
