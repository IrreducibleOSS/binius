// Copyright 2023-2024 Ulvetanna Inc.

use crate::{
	as_packed_field::PackScalar,
	underlier::{Divisible, WithUnderlier},
	ExtensionField, Field, PackedField,
};
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

/// The trait represents the relation between the packed fields of the same bit size
/// where one scalar is the extension of the other one.
pub trait PackedExtension<FS: Field>: PackedField
where
	Self::Scalar: ExtensionField<FS>,
{
	type PackedSubfield: PackedField<Scalar = FS>;

	fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield];
	fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield];

	fn cast_exts(packed: &[Self::PackedSubfield]) -> &[Self];
	fn cast_exts_mut(packed: &mut [Self::PackedSubfield]) -> &mut [Self];

	fn cast_base(&self) -> &Self::PackedSubfield;
	fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield;

	fn cast_ext(base: &Self::PackedSubfield) -> &Self;
	fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self;
}

impl<PT, FS> PackedExtension<FS> for PT
where
	FS: Field,
	PT: PackedField<Scalar: ExtensionField<FS>> + WithUnderlier<Underlier: PackScalar<FS>>,
{
	type PackedSubfield = <PT::Underlier as PackScalar<FS>>::Packed;

	fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield] {
		Self::PackedSubfield::from_underliers_ref(Self::to_underliers_ref(packed))
	}

	fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield] {
		Self::PackedSubfield::from_underliers_ref_mut(Self::to_underliers_ref_mut(packed))
	}

	fn cast_exts(base: &[Self::PackedSubfield]) -> &[Self] {
		Self::from_underliers_ref(Self::PackedSubfield::to_underliers_ref(base))
	}

	fn cast_exts_mut(base: &mut [Self::PackedSubfield]) -> &mut [Self] {
		Self::from_underliers_ref_mut(Self::PackedSubfield::to_underliers_ref_mut(base))
	}

	fn cast_base(&self) -> &Self::PackedSubfield {
		Self::PackedSubfield::from_underlier_ref(self.to_underlier_ref())
	}

	fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
		Self::PackedSubfield::from_underlier_ref_mut(self.to_underlier_ref_mut())
	}

	fn cast_ext(base: &Self::PackedSubfield) -> &Self {
		Self::from_underlier_ref(base.to_underlier_ref())
	}

	fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
		Self::from_underlier_ref_mut(base.to_underlier_ref_mut())
	}
}

/// Trait represents a relationship between a packed struct of field elements and a smaller packed
/// struct the same field elements.
///
/// This trait can be used to safely cast memory slices from larger packed fields to smaller ones.
///
/// # Safety
///
/// In order for the above relation to be guaranteed, the memory representation of a slice of
/// `PackedDivisible` elements must be the same as a slice of the underlying `PackedField`
/// elements, differing only in the slice lengths.
pub unsafe trait PackedDivisible<P>: PackedField
where
	P: PackedField<Scalar = Self::Scalar>,
{
	fn divide(packed: &[Self]) -> &[P];
	fn divide_mut(packed: &mut [Self]) -> &mut [P];
}

unsafe impl<PT1, PT2> PackedDivisible<PT2> for PT1
where
	PT2: PackedField + WithUnderlier,
	PT1: PackedField<Scalar = PT2::Scalar> + WithUnderlier<Underlier: Divisible<PT2::Underlier>>,
{
	fn divide(packed: &[Self]) -> &[PT2] {
		let underliers = PT1::to_underliers_ref(packed);
		let underliers: &[PT2::Underlier] = PT1::Underlier::split_slice(underliers);
		PT2::from_underliers_ref(underliers)
	}

	fn divide_mut(packed: &mut [Self]) -> &mut [PT2] {
		let underliers = PT1::to_underliers_ref_mut(packed);
		let underliers: &mut [PT2::Underlier] = PT1::Underlier::split_slice_mut(underliers);
		PT2::from_underliers_ref_mut(underliers)
	}
}
