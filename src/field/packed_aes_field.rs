// Copyright 2024 Ulvetanna Inc.

pub use crate::field::arch::packed_aes_128::*;

#[cfg(test)]
mod test_utils {
	/// Test if `mult_func` operation is a valid multiply operation on the given values for
	/// all possible packed fields defined on u128.
	macro_rules! define_multiply_tests {
		($mult_func:path, $constraint:ident) => {
			$crate::field::packed_binary_field::test_utils::define_check_packed_mul!(
				$mult_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_mul_packed_128(a_val in any::<u128>(), b_val in any::<u128>()) {
					check_packed_mul::<$crate::field::PackedAESBinaryField16x8b>(
						a_val.into(),
						b_val.into(),
					);
					check_packed_mul::<$crate::field::PackedAESBinaryField8x16b>(
						a_val.into(),
						b_val.into(),
					);
					check_packed_mul::<$crate::field::PackedAESBinaryField4x32b>(
						a_val.into(),
						b_val.into(),
					);
					check_packed_mul::<$crate::field::PackedAESBinaryField2x64b>(
						a_val.into(),
						b_val.into(),
					);
					check_packed_mul::<$crate::field::PackedAESBinaryField1x128b>(
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
			$crate::field::packed_binary_field::test_utils::define_check_packed_square!(
				$square_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_square_packed_128(a_val in any::<u128>()) {
					check_packed_square::<PackedAESBinaryField16x8b>(a_val.into());
					check_packed_square::<PackedAESBinaryField8x16b>(a_val.into());
					check_packed_square::<PackedAESBinaryField4x32b>(a_val.into());
					check_packed_square::<PackedAESBinaryField2x64b>(a_val.into());
					check_packed_square::<PackedAESBinaryField1x128b>(a_val.into());
				}
			}
		};
	}

	/// Test if `invert_func` operation is a valid invert operation on the given value for
	/// all possible packed fields.
	macro_rules! define_invert_tests {
		($invert_func:path, $constraint:ident) => {
			$crate::field::packed_binary_field::test_utils::define_check_packed_inverse!(
				$invert_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_invert_packed_128(a_val in any::<u128>()) {
					check_packed_inverse::<PackedAESBinaryField16x8b>(a_val.into());
					check_packed_inverse::<PackedAESBinaryField8x16b>(a_val.into());
					check_packed_inverse::<PackedAESBinaryField4x32b>(a_val.into());
					check_packed_inverse::<PackedAESBinaryField2x64b>(a_val.into());
					check_packed_inverse::<PackedAESBinaryField1x128b>(a_val.into());
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
	use super::{
		test_utils::{define_invert_tests, define_multiply_tests, define_square_tests},
		*,
	};
	use crate::field::PackedField;
	use proptest::prelude::*;
	use std::ops::Mul;

	define_multiply_tests!(Mul::mul, PackedField);

	define_square_tests!(PackedField::square, PackedField);

	define_invert_tests!(PackedField::invert_or_zero, PackedField);
}
