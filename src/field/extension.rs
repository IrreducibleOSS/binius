// Copyright 2023 Ulvetanna Inc.

use super::{error::Error, packed::PackedField, Field};
use std::{
	iter,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
	slice,
};

pub trait ExtensionField<F: Field>:
	Field
	+ From<F>
	+ TryInto<F>
	+ Add<F, Output = Self>
	+ Sub<F, Output = Self>
	+ Mul<F, Output = Self>
	+ AddAssign<F>
	+ SubAssign<F>
	+ MulAssign<F>
{
	type Iterator: Iterator<Item = F>;

	const DEGREE: usize;

	fn basis(i: usize) -> Result<Self, Error>;

	fn from_bases(base_elems: &[F]) -> Result<Self, Error>;

	fn iter_bases(&self) -> Self::Iterator;
}

impl<F: Field> ExtensionField<F> for F {
	type Iterator = iter::Once<F>;

	const DEGREE: usize = 1;

	fn basis(i: usize) -> Result<Self, Error> {
		if i != 0 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		Ok(Self::ONE)
	}

	fn from_bases(base_elems: &[F]) -> Result<Self, Error> {
		match base_elems.len() {
			0 => Ok(F::ZERO),
			1 => Ok(base_elems[0]),
			_ => Err(Error::ExtensionDegreeMismatch),
		}
	}

	fn iter_bases(&self) -> Self::Iterator {
		iter::once(*self)
	}
}

/// Trait represents a relationship between a packed struct of field elements and a packed struct
/// of elements from an extension field.
///
/// This trait relation guarantees that the following iterators yield the same sequence of scalar
/// elements:
///
/// ```
/// use binius::field::{ExtensionField, iter_packed_slice, PackedExtensionField, PackedField};
///
/// fn iter_ext_then_bases<'a, P, PE>(packed: &'a [PE]) -> impl Iterator<Item=P::Scalar> + 'a
///     where
///         P: PackedField + 'a,
///         PE: PackedExtensionField<P>,
///         PE::Scalar: ExtensionField<P::Scalar>,
/// {
///     iter_packed_slice(packed).flat_map(|ext| ext.iter_bases())
/// }
///
/// fn iter_cast_then_iter<'a, P, PE>(packed: &'a [PE]) -> impl Iterator<Item=P::Scalar> + 'a
///     where
///         P: PackedField + 'a,
///         PE: PackedExtensionField<P>,
///         PE::Scalar: ExtensionField<P::Scalar>,
/// {
///     iter_packed_slice(PE::cast_to_bases(packed)).flat_map(|p| p.into_iter())
/// }
/// ```
///
/// # Safety
///
/// In order for the above relation to be guaranteed, the memory representation of a slice of
/// `PackedExtensionField` elements must be the same as a slice of the underlying `PackedField`
/// elements, differing only in the slice lengths.
pub unsafe trait PackedExtensionField<P: PackedField>: PackedField
where
	Self::Scalar: ExtensionField<P::Scalar>,
{
	fn cast_to_bases(packed: &[Self]) -> &[P];
	fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P];

	fn as_bases(&self) -> &[P] {
		Self::cast_to_bases(slice::from_ref(self))
	}

	fn as_bases_mut(&mut self) -> &mut [P] {
		Self::cast_to_bases_mut(slice::from_mut(self))
	}

	/// Try to cast a slice of base field elements to extension field elements.
	///
	/// Returns None if the extension degree does not divide the number of base field elements.
	fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]>;

	/// Try to cast a mutable slice of base field elements to extension field elements.
	///
	/// Returns None if the extension degree does not divide the number of base field elements.
	fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]>;
}

unsafe impl<P: PackedField> PackedExtensionField<P> for P {
	fn cast_to_bases(packed: &[Self]) -> &[P] {
		packed
	}

	fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P] {
		packed
	}

	fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]> {
		Some(packed)
	}

	fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]> {
		Some(packed)
	}
}

pub fn unpack_scalars<P, S>(packed: &[P]) -> &[S]
where
	S: Field,
	P: PackedExtensionField<S, Scalar = S>,
{
	P::cast_to_bases(packed)
}

pub fn unpack_scalars_mut<P, S>(packed: &mut [P]) -> &mut [S]
where
	S: Field,
	P: PackedExtensionField<S, Scalar = S>,
{
	P::cast_to_bases_mut(packed)
}
