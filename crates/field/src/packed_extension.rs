// Copyright 2023-2024 Irreducible Inc.

use crate::{
	as_packed_field::PackScalar,
	underlier::{Divisible, WithUnderlier},
	Error, ExtensionField, Field, PackedField,
};

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
	P: PackedDivisible<S, Scalar = S>,
{
	fn unpack_scalars(packed: &[Self]) -> &[Self::Scalar] {
		P::divide(packed)
	}

	fn unpack_scalars_mut(packed: &mut [Self]) -> &mut [Self::Scalar] {
		P::divide_mut(packed)
	}
}

/// Trait represents a relationship between a packed struct of field elements and a packed struct
/// of elements from an extension field.
///
/// This trait guarantees that one packed type has the same
/// memory representation as the other, differing only in the scalar type and preserving the order
/// of smaller elements.
///
/// This trait relation guarantees that the following iterators yield the same sequence of scalar
/// elements:
///
/// ```
/// use binius_field::{ExtensionField, packed::iter_packed_slice, PackedExtension, PackedField, Field};
///
/// fn ext_then_bases<'a, F, PE>(packed: &'a PE) -> impl Iterator<Item=F> + 'a
///     where
///         PE: PackedField<Scalar: ExtensionField<F>>,
///         F: Field,
/// {
///     packed.iter().flat_map(|ext| ext.iter_bases())
/// }
///
/// fn cast_then_iter<'a, F, PE>(packed: &'a PE) -> impl Iterator<Item=F> + 'a
///     where
///         PE: PackedExtension<F, Scalar: ExtensionField<F>>,
///         F: Field,
/// {
///     PE::cast_base_ref(packed).into_iter()
/// }
/// ```
///
/// # Safety
///
/// In order for the above relation to be guaranteed, the memory representation of
/// `PackedExtensionField` element must be the same as a slice of the underlying `PackedField`
/// element.
pub trait PackedExtension<FS: Field>: PackedField
where
	Self::Scalar: ExtensionField<FS>,
{
	type PackedSubfield: PackedField<Scalar = FS>;

	fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield];
	fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield];

	fn cast_exts(packed: &[Self::PackedSubfield]) -> &[Self];
	fn cast_exts_mut(packed: &mut [Self::PackedSubfield]) -> &mut [Self];

	fn cast_base(self) -> Self::PackedSubfield;
	fn cast_base_ref(&self) -> &Self::PackedSubfield;
	fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield;

	fn cast_ext(base: Self::PackedSubfield) -> Self;
	fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self;
	fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self;
}

pub fn ext_base_mul<PE, F>(
	lhs: &[PE],
	rhs: &[PE::PackedSubfield],
	out: &mut [PE],
) -> Result<(), Error>
where
	PE: PackedExtension<F>,
	PE::Scalar: ExtensionField<F>,
	F: Field,
{
	if lhs.len() != rhs.len() * PE::Scalar::DEGREE {
		return Err(Error::MismatchedLengths);
	}

	for i in 0..lhs.len() {
		let bottom_most_scalar_idx = i * PE::WIDTH;
		let bottom_most_scalar_idx_in_subfield_arr =
			bottom_most_scalar_idx / PE::PackedSubfield::WIDTH;
		let bottom_most_scalar_idx_within_packed_subfield =
			bottom_most_scalar_idx % PE::PackedSubfield::WIDTH;
		let block_idx = bottom_most_scalar_idx_within_packed_subfield / PE::WIDTH;

		// SAFETY: Width of PackedSubfield is always >= the width of the field implementing PackedExtension
		let broadcasted_rhs = unsafe {
			rhs[bottom_most_scalar_idx_in_subfield_arr].spread_unchecked(PE::LOG_WIDTH, block_idx)
		};

		let result_base = lhs[i].cast_base() * broadcasted_rhs;
		out[i] = PE::cast_ext(result_base);
	}
	Ok(())
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

	fn cast_base(self) -> Self::PackedSubfield {
		Self::PackedSubfield::from_underlier(self.to_underlier())
	}

	fn cast_base_ref(&self) -> &Self::PackedSubfield {
		Self::PackedSubfield::from_underlier_ref(self.to_underlier_ref())
	}

	fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
		Self::PackedSubfield::from_underlier_ref_mut(self.to_underlier_ref_mut())
	}

	fn cast_ext(base: Self::PackedSubfield) -> Self {
		Self::from_underlier(base.to_underlier())
	}

	fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self {
		Self::from_underlier_ref(base.to_underlier_ref())
	}

	fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
		Self::from_underlier_ref_mut(base.to_underlier_ref_mut())
	}
}

/// This trait is a shorthand for the case `PackedExtension<P::Scalar, PackedSubfield = P>` which is a
/// quite common case in our codebase.
pub trait RepackedExtension<P: PackedField>:
	PackedExtension<P::Scalar, PackedSubfield = P>
where
	Self::Scalar: ExtensionField<P::Scalar>,
{
}

impl<PT1, PT2> RepackedExtension<PT1> for PT2
where
	PT1: PackedField,
	PT2: PackedExtension<PT1::Scalar, PackedSubfield = PT1, Scalar: ExtensionField<PT1::Scalar>>,
{
}

/// This trait adds shortcut methods for the case `PackedExtension<F, PackedSubfield: PackedFieldIndexable>` which is a
/// quite common case in our codebase.
pub trait PackedExtensionIndexable<F: Field>: PackedExtension<F>
where
	Self::Scalar: ExtensionField<F>,
	Self::PackedSubfield: PackedFieldIndexable,
{
	fn unpack_base_scalars(packed: &[Self]) -> &[F] {
		Self::PackedSubfield::unpack_scalars(Self::cast_bases(packed))
	}

	fn unpack_base_scalars_mut(packed: &mut [Self]) -> &mut [F] {
		Self::PackedSubfield::unpack_scalars_mut(Self::cast_bases_mut(packed))
	}
}

impl<F, PT> PackedExtensionIndexable<F> for PT
where
	F: Field,
	PT: PackedExtension<F, Scalar: ExtensionField<F>, PackedSubfield: PackedFieldIndexable>,
{
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

#[cfg(test)]
mod tests {
	use crate::{
		ext_base_mul,
		packed::{get_packed_slice, set_packed_slice},
		underlier::WithUnderlier,
		BinaryField128b, BinaryField16b, BinaryField8b, PackedBinaryField16x16b,
		PackedBinaryField2x128b, PackedBinaryField32x8b, PackedField,
	};
	use proptest::prelude::*;

	fn strategy_8b_scalars() -> impl Strategy<Value = [BinaryField8b; 32]> {
		any::<[<BinaryField8b as WithUnderlier>::Underlier; 32]>()
			.prop_map(|arr| arr.map(<BinaryField8b>::from_underlier))
	}

	fn strategy_16b_scalars() -> impl Strategy<Value = [BinaryField16b; 32]> {
		any::<[<BinaryField16b as WithUnderlier>::Underlier; 32]>()
			.prop_map(|arr| arr.map(<BinaryField16b>::from_underlier))
	}

	fn strategy_128b_scalars() -> impl Strategy<Value = [BinaryField128b; 32]> {
		any::<[<BinaryField128b as WithUnderlier>::Underlier; 32]>()
			.prop_map(|arr| arr.map(<BinaryField128b>::from_underlier))
	}

	fn pack_slice<P: PackedField>(scalar_slice: &[P::Scalar]) -> Vec<P> {
		let mut packed_slice = vec![P::default(); scalar_slice.len() / P::WIDTH];

		for (i, scalar) in scalar_slice.iter().enumerate() {
			set_packed_slice(&mut packed_slice, i, *scalar);
		}

		packed_slice
	}

	proptest! {
		#[test]
		fn test_base_ext_mul_8(base_scalars in strategy_8b_scalars(), ext_scalars in strategy_128b_scalars()){
			let base_packed = pack_slice::<PackedBinaryField32x8b>(&base_scalars);
			let ext_packed = pack_slice::<PackedBinaryField2x128b>(&ext_scalars);

			let mut result = vec![PackedBinaryField2x128b::default(); ext_packed.len()];

			ext_base_mul(&ext_packed, &base_packed, &mut result).unwrap();

			for (i, (base, ext)) in base_scalars.iter().zip(ext_scalars).enumerate(){
				assert_eq!(ext * *base, get_packed_slice(&result, i));
			}
		}

		#[test]
		fn test_base_ext_mul_16(base_scalars in strategy_16b_scalars(), ext_scalars in strategy_128b_scalars()){
			let base_packed = pack_slice::<PackedBinaryField16x16b>(&base_scalars);
			let ext_packed = pack_slice::<PackedBinaryField2x128b>(&ext_scalars);

			let mut result = vec![PackedBinaryField2x128b::default(); ext_packed.len()];

			ext_base_mul(&ext_packed, &base_packed, &mut result).unwrap();

			for (i, (base, ext)) in base_scalars.iter().zip(ext_scalars).enumerate(){
				assert_eq!(ext * *base, get_packed_slice(&result, i));
			}
		}
	}
}
