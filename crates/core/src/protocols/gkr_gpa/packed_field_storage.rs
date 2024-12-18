// Copyright 2024 Irreducible Inc.

use std::ops::Deref;

use binius_field::PackedField;
use binius_math::{Error, MultilinearExtension};
use binius_utils::checked_arithmetics::checked_log_2;

/// Packed field storage that can either reference a full slice of packed field elements or
/// store a small number of packed field elements (not greater than `P::WIDTH``) in-place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedFieldStorage<'a, P: PackedField> {
	SliceRef(&'a [P]),
	Inline { data: P, size: usize },
}

impl<'a, P: PackedField> PackedFieldStorage<'a, P> {
	/// Creates a new packed field storage from an iterator of scalar elements.
	/// The number of elements in the iterator must not exceed `P::WIDTH`.
	pub fn new_inline(values: impl Iterator<Item = P::Scalar>) -> Result<Self, Error> {
		let mut data = P::default();
		let mut size = 0;
		for (i, val) in values.enumerate() {
			data.set_checked(i, val)?;
			size += 1;
		}
		Ok(Self::Inline { data, size })
	}

	/// Creates a new packed field storage from a slice of packed field elements.
	pub fn new_slice(data: &'a [P]) -> Self {
		Self::SliceRef(data)
	}

	/// Returns the number of scalar elements in the packed field storage.
	pub fn n_scalars(&self) -> usize {
		match self {
			PackedFieldStorage::SliceRef(data) => data.len() * P::WIDTH,
			PackedFieldStorage::Inline { size, .. } => *size,
		}
	}

	/// Returns the logarithm of the number of scalar elements in the packed field storage.
	/// Panics if the number of scalar elements is not a power of 2.
	pub fn log_n_scalars(&self) -> Option<usize> {
		self.n_scalars()
			.is_power_of_two()
			.then(|| checked_log_2(self.n_scalars()))
	}
}

impl<'a, P: PackedField> From<&'a [P]> for PackedFieldStorage<'a, P> {
	fn from(data: &'a [P]) -> Self {
		PackedFieldStorage::new_slice(data)
	}
}

impl<P: PackedField> Deref for PackedFieldStorage<'_, P> {
	type Target = [P];

	fn deref(&self) -> &Self::Target {
		match self {
			PackedFieldStorage::SliceRef(data) => data,
			PackedFieldStorage::Inline { data, .. } => std::slice::from_ref(data),
		}
	}
}

impl<'a, P: PackedField> TryFrom<PackedFieldStorage<'a, P>>
	for MultilinearExtension<P, PackedFieldStorage<'a, P>>
{
	type Error = Error;

	fn try_from(storage: PackedFieldStorage<'a, P>) -> Result<Self, Error> {
		Self::new(
			storage
				.log_n_scalars()
				.ok_or(binius_math::Error::PowerOfTwoLengthRequired)?,
			storage,
		)
	}
}
