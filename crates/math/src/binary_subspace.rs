// Copyright 2024-2025 Irreducible Inc.

use binius_field::BinaryField;
use binius_utils::{bail, iter::IterExtensions};

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
	/// Creates a new subspace from a vector of ordered basis elements.
	///
	/// This constructor does not check that the basis elements are linearly independent.
	pub const fn new_unchecked(basis: Vec<F>) -> Self {
		Self { basis }
	}

	/// Creates a new subspace of this binary subspace with the given dimension.
	///
	/// This creates a new sub-subspace using a prefix of the default $\mathbb{F}_2$ basis elements
	/// of the field.
	///
	/// ## Throws
	///
	/// * `Error::DomainSizeTooLarge` if `dim` is greater than this subspace's dimension.
	pub fn with_dim(dim: usize) -> Result<Self, Error> {
		let basis = (0..dim)
			.map(|i| F::basis(i).map_err(|_| Error::DomainSizeTooLarge))
			.collect::<Result<_, _>>()?;
		Ok(Self { basis })
	}

	/// Creates a new subspace of this binary subspace with reduced dimension.
	///
	/// This creates a new sub-subspace using a prefix of the ordered basis elements.
	///
	/// ## Throws
	///
	/// * `Error::DomainSizeTooLarge` if `dim` is greater than this subspace's dimension.
	pub fn reduce_dim(&self, dim: usize) -> Result<Self, Error> {
		if dim > self.dim() {
			bail!(Error::DomainSizeTooLarge);
		}
		Ok(Self {
			basis: self.basis[..dim].to_vec(),
		})
	}

	/// Creates a new subspace isomorphic to the given one.
	pub fn isomorphic<FIso>(&self) -> BinarySubspace<FIso>
	where
		FIso: BinaryField + From<F>,
	{
		BinarySubspace {
			basis: self.basis.iter().copied().map(Into::into).collect(),
		}
	}

	/// Returns the dimension of the subspace.
	pub fn dim(&self) -> usize {
		self.basis.len()
	}

	/// Returns the slice of ordered basis elements.
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
			.collect();
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
