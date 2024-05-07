// Copyright 2023-2024 Ulvetanna Inc.

use crate::{ExtensionField, PackedField};
use ff::Field;
use std::slice;

/// Trait represents a relationship between a packed struct of field elements and a packed struct
/// of elements from an extension field.
///
/// This trait relation guarantees that the following iterators yield the same sequence of scalar
/// elements:
///
/// ```
/// use binius_field::{ExtensionField, packed::iter_packed_slice, PackedExtensionField, PackedField};
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

/// A [`PackedField`] that can be safely cast to indexable slices of scalars.
///
/// Not all packed fields can index individual scalar elements. Notably, packed fields of
/// $\mathbb{F}_2$ elements can pack multiple scalars into a single byte.
///
///
/// # Safety
///
/// In order for the above relation to be guaranteed, the memory representation of a slice of
/// `PackedExtensionIndexable` elements must be the same as a slice of the underlying scalar
/// elements, differing only in the slice lengths.
pub unsafe trait PackedFieldIndexable: PackedField {
	fn unpack_scalars(packed: &[Self]) -> &[Self::Scalar];
	fn unpack_scalars_mut(packed: &mut [Self]) -> &mut [Self::Scalar];
}

unsafe impl<S, P> PackedFieldIndexable for P
where
	S: Field,
	P: PackedExtensionField<S, Scalar = S>,
{
	fn unpack_scalars(packed: &[Self]) -> &[Self::Scalar] {
		Self::cast_to_bases(packed)
	}

	fn unpack_scalars_mut(packed: &mut [Self]) -> &mut [Self::Scalar] {
		Self::cast_to_bases_mut(packed)
	}
}
