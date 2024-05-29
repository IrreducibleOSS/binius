// Copyright 2024 Ulvetanna Inc.

use super::packed_arithmetic::UnderlierWithBitConstants;
use crate::{
	arithmetic_traits::{Broadcast, InvertOrZero, MulAlpha, Square},
	as_packed_field::PackScalar,
	underlier::{NumCast, UnderlierType, UnderlierWithBitOps, WithUnderlier},
	BinaryField, Error, PackedField,
};
use bytemuck::{Pod, Zeroable};
use rand::RngCore;
use std::{
	fmt::Debug,
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use subtle::{Choice, ConstantTimeEq};

#[derive(PartialEq, Eq, Clone, Copy, Default)]
#[repr(transparent)]
pub struct PackedPrimitiveType<U: UnderlierType, Scalar: BinaryField>(
	pub U,
	pub PhantomData<Scalar>,
);

impl<U: UnderlierType, Scalar: BinaryField> PackedPrimitiveType<U, Scalar> {
	pub const WIDTH: usize = {
		assert!(U::BITS % Scalar::N_BITS == 0);

		U::BITS / Scalar::N_BITS
	};

	pub const LOG_WIDTH: usize = {
		let result = Self::WIDTH.ilog2();

		assert!(2usize.pow(result) == Self::WIDTH);

		result as usize
	};

	#[inline]
	pub fn from_underlier(val: U) -> Self {
		Self(val, PhantomData)
	}

	#[inline]
	pub const fn to_underlier(self) -> U {
		self.0
	}
}

unsafe impl<U: UnderlierType, Scalar: BinaryField> WithUnderlier for PackedPrimitiveType<U, Scalar>
where
	U: From<Self>,
	Self: From<U>,
{
	type Underlier = U;
}

impl<U: UnderlierType, Scalar: BinaryField> Debug for PackedPrimitiveType<U, Scalar>
where
	Self: PackedField,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let width = U::BITS / Scalar::N_BITS;
		let values: Vec<_> = (0..width).map(|i| self.get(i)).collect();
		write!(f, "Packed{}x{}({:?})", width, Scalar::N_BITS, values)
	}
}

impl<U: UnderlierType, Scalar: BinaryField> From<U> for PackedPrimitiveType<U, Scalar> {
	#[inline]
	fn from(val: U) -> Self {
		Self(val, PhantomData)
	}
}

impl<U: UnderlierType, Scalar: BinaryField> ConstantTimeEq for PackedPrimitiveType<U, Scalar> {
	fn ct_eq(&self, other: &Self) -> Choice {
		self.0.ct_eq(&other.0)
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Add for PackedPrimitiveType<U, Scalar> {
	type Output = Self;

	#[inline]
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn add(self, rhs: Self) -> Self::Output {
		(self.0 ^ rhs.0).into()
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Sub for PackedPrimitiveType<U, Scalar> {
	type Output = Self;

	#[inline]
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn sub(self, rhs: Self) -> Self::Output {
		(self.0 ^ rhs.0).into()
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> AddAssign for PackedPrimitiveType<U, Scalar> {
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> SubAssign for PackedPrimitiveType<U, Scalar> {
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> MulAssign for PackedPrimitiveType<U, Scalar>
where
	Self: Mul<Output = Self>,
{
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Add<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar>,
{
	type Output = Self;

	fn add(self, rhs: Scalar) -> Self::Output {
		self + Self::broadcast(rhs)
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Sub<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar>,
{
	type Output = Self;

	fn sub(self, rhs: Scalar) -> Self::Output {
		self - Self::broadcast(rhs)
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Mul<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Mul<Output = Self>,
{
	type Output = Self;

	fn mul(self, rhs: Scalar) -> Self::Output {
		self * Self::broadcast(rhs)
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> AddAssign<Scalar>
	for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar>,
{
	fn add_assign(&mut self, rhs: Scalar) {
		*self += Self::broadcast(rhs);
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> SubAssign<Scalar>
	for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar>,
{
	fn sub_assign(&mut self, rhs: Scalar) {
		*self -= Self::broadcast(rhs);
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> MulAssign<Scalar>
	for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + MulAssign<Self>,
{
	fn mul_assign(&mut self, rhs: Scalar) {
		*self *= Self::broadcast(rhs);
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Sum for PackedPrimitiveType<U, Scalar> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::from(U::default()), |result, next| result + next)
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Product for PackedPrimitiveType<U, Scalar>
where
	PackedPrimitiveType<U, Scalar>: Broadcast<Scalar> + Mul<Output = Self>,
{
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::broadcast(Scalar::ONE), |result, next| result * next)
	}
}

unsafe impl<U: UnderlierWithBitOps + Zeroable, Scalar: BinaryField> Zeroable
	for PackedPrimitiveType<U, Scalar>
{
}

unsafe impl<U: UnderlierWithBitOps + Pod, Scalar: BinaryField> Pod
	for PackedPrimitiveType<U, Scalar>
{
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> PackedField for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Square + InvertOrZero + Mul<Output = Self>,
	U: UnderlierWithBitConstants + Send + Sync + 'static,
	Scalar: WithUnderlier<Underlier: UnderlierWithBitOps>,
	U: From<Scalar::Underlier>,
	Scalar::Underlier: NumCast<U>,
{
	type Scalar = Scalar;

	const LOG_WIDTH: usize = (U::BITS / Scalar::N_BITS).ilog2() as usize;

	#[inline]
	fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
		(i < Self::WIDTH)
			.then(|| self.0.get_subvalue(i))
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
	}

	#[inline]
	fn set_checked(&mut self, i: usize, scalar: Scalar) -> Result<(), Error> {
		(i < Self::WIDTH)
			.then(|| self.0.set_subvalue(i, scalar))
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
	}

	fn random(rng: impl RngCore) -> Self {
		U::random(rng).into()
	}

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_WIDTH);
		let log_bit_len = Self::Scalar::N_BITS.ilog2() as usize;
		let (c, d) = self.0.interleave(other.0, log_block_len + log_bit_len);
		(c.into(), d.into())
	}

	#[inline]
	fn broadcast(scalar: Self::Scalar) -> Self {
		<Self as Broadcast<Self::Scalar>>::broadcast(scalar)
	}

	#[inline]
	fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self {
		U::from_fn(f).into()
	}

	#[inline]
	fn square(self) -> Self {
		<Self as Square>::square(self)
	}

	#[inline]
	fn invert_or_zero(self) -> Self {
		<Self as InvertOrZero>::invert_or_zero(self)
	}
}

/// Implement the PackedExtensionField trait for binary fields that are subfields of the
/// scalar type.
///
/// For example, `PackedField2x64b` is `PackedExtensionField<BinaryField64b>` and also
/// `PackedExtensionField<BinaryField32b>`, and so on.
/// We are using macro because otherwise we will get the conflicting implementation with
/// `PackedExtensionField<P> for P`
macro_rules! impl_packed_extension_field {
	($name:ty) => {
		#[cfg(target_endian = "little")]
		unsafe impl<P> $crate::PackedExtensionField<P> for $name
		where
			P: $crate::PackedField,
			Self::Scalar: $crate::PackedExtensionField<P>,
			Self::Scalar: $crate::ExtensionField<P::Scalar>,
		{
			fn cast_to_bases(packed: &[Self]) -> &[P] {
				Self::Scalar::cast_to_bases(bytemuck::must_cast_slice(packed))
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P] {
				Self::Scalar::cast_to_bases_mut(bytemuck::must_cast_slice_mut(packed))
			}

			fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]> {
				Self::Scalar::try_cast_to_ext(packed)
					.and_then(|scalars| bytemuck::try_cast_slice(scalars).ok())
			}

			fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]> {
				Self::Scalar::try_cast_to_ext_mut(packed)
					.and_then(|scalars| bytemuck::try_cast_slice_mut(scalars).ok())
			}
		}
	};
}

pub(crate) use impl_packed_extension_field;

macro_rules! impl_broadcast {
	($name:ty, BinaryField1b) => {
		impl $crate::arithmetic_traits::Broadcast<BinaryField1b>
			for PackedPrimitiveType<$name, BinaryField1b>
		{
			#[inline]
			fn broadcast(scalar: BinaryField1b) -> Self {
				use $crate::underlier::{UnderlierWithBitOps, WithUnderlier};

				<Self as WithUnderlier>::Underlier::fill_with_bit(scalar.0.into()).into()
			}
		}
	};
	($name:ty, $scalar_type:ty) => {
		impl $crate::arithmetic_traits::Broadcast<$scalar_type>
			for PackedPrimitiveType<$name, $scalar_type>
		{
			#[inline]
			fn broadcast(scalar: $scalar_type) -> Self {
				let mut value = <$name>::from(scalar.0);
				// For PackedBinaryField1x128b, the log bits is 7, so this is
				// an empty range. This is safe behavior.
				#[allow(clippy::reversed_empty_ranges)]
				for i in <$scalar_type as $crate::binary_field::BinaryField>::N_BITS.ilog2()
					..<$name>::BITS.ilog2()
				{
					value = value << (1 << i) | value;
				}

				value.into()
			}
		}
	};
}

pub(crate) use impl_broadcast;

/// We can't define conversions to underlier and from array of scalars in a generic way due to Rust restrictions
macro_rules! impl_conversion {
	($underlier:ty, $name:ty) => {
		impl From<$name> for $underlier {
			#[inline]
			fn from(value: $name) -> Self {
				return value.0;
			}
		}

		impl From<[<$name as $crate::PackedField>::Scalar; <$name>::WIDTH]> for $name {
			#[inline]
			fn from(val: [<$name as $crate::PackedField>::Scalar; <$name>::WIDTH]) -> Self {
				use $crate::PackedField;

				Self::from_fn(|i| val[i])
			}
		}
	};
}

pub(crate) use impl_conversion;

macro_rules! packed_binary_field_tower_extension {
	($subfield_name:ident < $name:ident) => {
		#[cfg(target_endian = "little")]
		unsafe impl $crate::PackedExtensionField<$subfield_name> for $name {
			#[inline]
			fn cast_to_bases(packed: &[Self]) -> &[$subfield_name] {
				bytemuck::must_cast_slice(packed)
			}

			#[inline]
			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [$subfield_name] {
				bytemuck::must_cast_slice_mut(packed)
			}

			#[inline]
			fn try_cast_to_ext(packed: &[$subfield_name]) -> Option<&[Self]> {
				Some(bytemuck::must_cast_slice(packed))
			}

			#[inline]
			fn try_cast_to_ext_mut(packed: &mut [$subfield_name]) -> Option<&mut [Self]> {
				Some(bytemuck::must_cast_slice_mut(packed))
			}
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		$crate::arch::portable::packed::packed_binary_field_tower_extension!($subfield_name < $name);
		$(
			$crate::arch::portable::packed::packed_binary_field_tower_extension!($subfield_name < $extfield_name);
		)+
		$crate::arch::portable::packed::packed_binary_field_tower_extension!($name $(< $extfield_name)+);
	};
}

pub(crate) use packed_binary_field_tower_extension;

macro_rules! packed_binary_field_tower_impl {
	($subfield_name:ident < $name:ident) => {
		impl $crate::arch::portable::packed_arithmetic::PackedTowerField for $name {
			type Underlier = <Self as $crate::underlier::WithUnderlier>::Underlier;
			type DirectSubfield = <$subfield_name as $crate::PackedField>::Scalar;

			type PackedDirectSubfield = $subfield_name;
		}
	};
	($subfield_name:ident < $name:ident $(< $extfield_name:ident)+) => {
		impl $crate::arch::portable::packed_arithmetic::PackedTowerField for $name {
			type Underlier = <Self as $crate::underlier::WithUnderlier>::Underlier;
			type DirectSubfield = <$subfield_name as $crate::PackedField>::Scalar;

			type PackedDirectSubfield = $subfield_name;
		}

		$crate::arch::portable::packed::packed_binary_field_tower_impl!($name $(< $extfield_name)+);
	}
}

pub(crate) use packed_binary_field_tower_impl;

macro_rules! packed_binary_field_tower {
	($name:ident $(< $extfield_name:ident)+) => {
		$crate::arch::portable::packed::packed_binary_field_tower_extension!($name $(< $extfield_name)+ );
		$crate::arch::portable::packed::packed_binary_field_tower_impl!($name $(< $extfield_name)+ );
	}
}

pub(crate) use packed_binary_field_tower;

macro_rules! impl_ops_for_zero_height {
	($name:ty) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			#[inline]
			fn mul(self, b: Self) -> Self {
				(self.to_underlier() & b.to_underlier()).into()
			}
		}

		impl $crate::arithmetic_traits::MulAlpha for $name {
			#[inline]
			fn mul_alpha(self) -> Self {
				self
			}
		}

		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				self
			}
		}

		impl $crate::arithmetic_traits::InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				self
			}
		}
	};
}

pub(crate) use impl_ops_for_zero_height;

/// Multiply `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD multiplication is faster.
pub fn mul_as_bigger_type<PT1, PT2>(lhs: PT1, rhs: PT1) -> PT1
where
	PT1: PackedField + WithUnderlier,
	PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier,
	PT2::Underlier: From<PT1::Underlier>,
	PT1::Underlier: NumCast<PT2::Underlier>,
{
	let bigger_lhs = PT2::from(lhs.to_underlier().into());
	let bigger_rhs = PT2::from(rhs.to_underlier().into());

	let bigger_result = bigger_lhs * bigger_rhs;

	PT1::Underlier::num_cast_from(bigger_result.to_underlier()).into()
}

/// Square `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD square is faster.
pub fn square_as_bigger_type<PT1, PT2>(val: PT1) -> PT1
where
	PT1: PackedField + WithUnderlier,
	PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier,
	PT2::Underlier: From<PT1::Underlier>,
	PT1::Underlier: NumCast<PT2::Underlier>,
{
	let bigger_val = PT2::from(val.to_underlier().into());

	let bigger_result = bigger_val.square();

	PT1::Underlier::num_cast_from(bigger_result.to_underlier()).into()
}

/// Invert `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD invert is faster.
pub fn invert_as_bigger_type<PT1, PT2>(val: PT1) -> PT1
where
	PT1: PackedField + WithUnderlier,
	PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier,
	PT2::Underlier: From<PT1::Underlier>,
	PT1::Underlier: NumCast<PT2::Underlier>,
{
	let bigger_val = PT2::from(val.to_underlier().into());

	let bigger_result = bigger_val.invert_or_zero();

	PT1::Underlier::num_cast_from(bigger_result.to_underlier()).into()
}

/// Multiply by alpha `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD multiply by alpha is faster.
pub fn mul_alpha_as_bigger_type<PT1, PT2>(val: PT1) -> PT1
where
	PT1: PackedField + WithUnderlier,
	PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier + MulAlpha,
	PT2::Underlier: From<PT1::Underlier>,
	PT1::Underlier: NumCast<PT2::Underlier>,
{
	let bigger_val = PT2::from(val.to_underlier().into());

	let bigger_result = bigger_val.mul_alpha();

	PT1::Underlier::num_cast_from(bigger_result.to_underlier()).into()
}

impl<U, F> PackScalar<F> for U
where
	U: UnderlierType,
	F: BinaryField,
	PackedPrimitiveType<U, F>: PackedField<Scalar = F> + WithUnderlier<Underlier = U>,
{
	type Packed = PackedPrimitiveType<U, F>;
}
