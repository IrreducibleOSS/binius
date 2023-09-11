#![allow(clippy::non_canonical_partial_ord_impl)]
#![allow(clippy::derive_ord_xor_partial_ord)]
// Copyright 2023 Ulvetanna Inc.

//! Lexographic ordered Step implementations for binary fields.
//!
//! Steppers define an ordering and successor/predecessor relations on the field elements, as
//! formalized by the Step trait.

use crate::field::*;
use std::iter::Step;

macro_rules! binary_field_lexographic_stepper {
	($vis:vis $name:ident($field:ident($typ:ty))) => {
		#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
		pub struct $name($field);

		impl $name {
			pub fn into_inner(self) -> $field {
				self.0
			}
		}

		impl From<$field> for $name {
			fn from(value: $field) -> Self {
				Self(value)
			}
		}

		impl Step for $name {
			fn steps_between(start: &Self, end: &Self) -> Option<usize> {
				let diff = end.0.val().checked_sub(start.0.val())?;
				usize::try_from(diff).ok()
			}

			fn forward_checked(start: Self, count: usize) -> Option<Self> {
				let val = start.0.val().checked_add(count as $typ)?;
				let inner = <$field>::new_checked(val).ok()?;
				Some(Self(inner))
			}

			fn backward_checked(start: Self, count: usize) -> Option<Self> {
				let val = start.0.val().checked_sub(count as $typ)?;
				let inner = <$field>::new_checked(val).ok()?;
				Some(Self(inner))
			}
		}
	};
}

binary_field_lexographic_stepper!(pub BinaryField1bLexographic(BinaryField1b(u8)));
binary_field_lexographic_stepper!(pub BinaryField2bLexographic(BinaryField2b(u8)));
binary_field_lexographic_stepper!(pub BinaryField4bLexographic(BinaryField4b(u8)));
binary_field_lexographic_stepper!(pub BinaryField8bLexographic(BinaryField8b(u8)));
binary_field_lexographic_stepper!(pub BinaryField16bLexographic(BinaryField16b(u16)));
binary_field_lexographic_stepper!(pub BinaryField32bLexographic(BinaryField32b(u32)));
binary_field_lexographic_stepper!(pub BinaryField64bLexographic(BinaryField64b(u64)));
binary_field_lexographic_stepper!(pub BinaryField128bLexographic(BinaryField128b(u128)));

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_step_32b() {
		let step0 = BinaryField32bLexographic(BinaryField32b::ZERO);
		let step1 = BinaryField32bLexographic::forward_checked(step0, 0x10000000);
		assert_eq!(step1, Some(BinaryField32bLexographic(BinaryField32b::new(0x10000000))));
		let step2 = BinaryField32bLexographic::forward_checked(step1.unwrap(), 0x01000000);
		assert_eq!(step2, Some(BinaryField32bLexographic(BinaryField32b::new(0x11000000))));
		let step3 = BinaryField32bLexographic::forward_checked(step2.unwrap(), 0xF0000000);
		assert_eq!(step3, None);
	}
}
