// Copyright 2024-2025 Irreducible Inc.

use binius_field::BinaryField;
use binius_utils::iter::IterExtensions;

use super::error::Error;

/// An $F_2$-linear subspace of a binary field.
///
/// The subspace is defined by a basis of elements from a binary field. The basis elements are
/// ordered, which implies an ordering on the subspace elements.
#[derive(Debug, Clone)]
pub struct BinarySubspace<F: BinaryField> {
	basis: Vec<F>,
}

impl<F: BinaryField> BinarySubspace<F> {
	pub fn basis(&self) -> &[F] {
		&self.basis
	}

	pub fn get(&self, index: usize) -> F {
		self.basis
			.iter()
			.take(usize::BITS as usize)
			.enumerate()
			.fold(F::ZERO, |acc, (i, basis_i)| {
				if (index >> i) & 1 != 0 {
					acc + basis_i
				} else {
					acc
				}
			})
	}

	pub fn get_checked(&self, index: usize) -> Result<F, Error> {
		if index >= 1 << self.basis.len() {
			return Err(Error::ArgumentRangeError {
				arg: "index".to_string(),
				range: 0..1 << self.basis.len(),
			});
		}
		Ok(self.get(index))
	}

	/// Returns an iterator over all elements of the subspace in order.
	///
	/// This has a limitation that the iterator only yields the first `2^usize::BITS` elements.
	pub fn iter(&self) -> impl Iterator<Item = F> + '_ {
		let last = if self.basis.len() < usize::BITS as usize {
			(1 << self.basis.len()) - 1
		} else {
			usize::MAX
		};
		(0..=last).map_skippable(|i| self.get(i))
	}
}

impl<F: BinaryField> Default for BinarySubspace<F> {
	fn default() -> Self {
		let basis = (0..F::DEGREE)
			.map(|i| F::basis(i).expect("index is in range"))
			.collect::<Vec<_>>();
		Self { basis }
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{BinaryField128b, BinaryField8b};

	use super::*;

	#[test]
	fn test_default_binary_subspace_iterates_elements() {
		let subspace = BinarySubspace::<BinaryField8b>::default();
		for i in 0..=255 {
			assert_eq!(subspace.get(i), BinaryField8b::new(i as u8));
		}
	}

	#[test]
	fn test_binary_subspace_range_error() {
		let subspace = BinarySubspace::<BinaryField8b>::default();
		assert_matches!(subspace.get_checked(256), Err(Error::ArgumentRangeError { .. }));
	}

	#[test]
	fn test_default_large_binary_subspace_iterates_elements() {
		let subspace = BinarySubspace::<BinaryField128b>::default();
		for i in 0..=255 {
			assert_eq!(subspace.get(i), BinaryField128b::new(i as u128));
		}
	}
}
