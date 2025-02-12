// Copyright 2024-2025 Irreducible Inc.

// This is because derive(bytemuck::TransparentWrapper) adds some type constraints to
// PackedPrimitiveType in addition to the type constraints we define. Even more, annoying, the
// allow attribute has to be added to the module, it doesn't work to add it to the struct
// definition.
#![allow(clippy::multiple_bound_locations)]

use std::{
	fmt::Debug,
	iter::{Product, Sum},
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::{checked_arithmetics::checked_int_div, iter::IterExtensions};
use bytemuck::{Pod, TransparentWrapper, Zeroable};
use rand::RngCore;
use subtle::{Choice, ConstantTimeEq};

use super::packed_arithmetic::UnderlierWithBitConstants;
use crate::{
	arithmetic_traits::{Broadcast, InvertOrZero, MulAlpha, Square},
	underlier::{
		IterationMethods, IterationStrategy, NumCast, UnderlierType, UnderlierWithBitOps,
		WithUnderlier, U1, U2, U4,
	},
	BinaryField, PackedField,
};

#[derive(PartialEq, Eq, Clone, Copy, Default, bytemuck::TransparentWrapper)]
#[repr(transparent)]
#[transparent(U)]
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
	pub const fn from_underlier(val: U) -> Self {
		Self(val, PhantomData)
	}

	#[inline]
	pub const fn to_underlier(self) -> U {
		self.0
	}
}

unsafe impl<U: UnderlierType, Scalar: BinaryField> WithUnderlier
	for PackedPrimitiveType<U, Scalar>
{
	type Underlier = U;

	#[inline(always)]
	fn to_underlier(self) -> Self::Underlier {
		TransparentWrapper::peel(self)
	}

	#[inline(always)]
	fn to_underlier_ref(&self) -> &Self::Underlier {
		TransparentWrapper::peel_ref(self)
	}

	#[inline(always)]
	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		TransparentWrapper::peel_mut(self)
	}

	#[inline(always)]
	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		TransparentWrapper::peel_slice(val)
	}

	#[inline(always)]
	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		TransparentWrapper::peel_slice_mut(val)
	}

	#[inline(always)]
	fn from_underlier(val: Self::Underlier) -> Self {
		TransparentWrapper::wrap(val)
	}

	#[inline(always)]
	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		TransparentWrapper::wrap_ref(val)
	}

	#[inline(always)]
	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		TransparentWrapper::wrap_mut(val)
	}

	#[inline(always)]
	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		TransparentWrapper::wrap_slice(val)
	}

	#[inline(always)]
	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		TransparentWrapper::wrap_slice_mut(val)
	}
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Debug for PackedPrimitiveType<U, Scalar>
where
	Self: PackedField,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let width = checked_int_div(U::BITS, Scalar::N_BITS);
		let values_str = self
			.iter()
			.map(|value| format!("{}", value))
			.collect::<Vec<_>>()
			.join(",");

		write!(f, "Packed{}x{}([{}])", width, Scalar::N_BITS, values_str)
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

impl<U: UnderlierType, Scalar: BinaryField> AddAssign for PackedPrimitiveType<U, Scalar>
where
	Self: Add<Output = Self>,
{
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<U: UnderlierType, Scalar: BinaryField> SubAssign for PackedPrimitiveType<U, Scalar>
where
	Self: Sub<Output = Self>,
{
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<U: UnderlierType, Scalar: BinaryField> MulAssign for PackedPrimitiveType<U, Scalar>
where
	Self: Mul<Output = Self>,
{
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<U: UnderlierType, Scalar: BinaryField> Add<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Add<Output = Self>,
{
	type Output = Self;

	fn add(self, rhs: Scalar) -> Self::Output {
		self + Self::broadcast(rhs)
	}
}

impl<U: UnderlierType, Scalar: BinaryField> Sub<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Sub<Output = Self>,
{
	type Output = Self;

	fn sub(self, rhs: Scalar) -> Self::Output {
		self - Self::broadcast(rhs)
	}
}

impl<U: UnderlierType, Scalar: BinaryField> Mul<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Mul<Output = Self>,
{
	type Output = Self;

	fn mul(self, rhs: Scalar) -> Self::Output {
		self * Self::broadcast(rhs)
	}
}

impl<U: UnderlierType, Scalar: BinaryField> AddAssign<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + AddAssign<Self>,
{
	fn add_assign(&mut self, rhs: Scalar) {
		*self += Self::broadcast(rhs);
	}
}

impl<U: UnderlierType, Scalar: BinaryField> SubAssign<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + SubAssign<Self>,
{
	fn sub_assign(&mut self, rhs: Scalar) {
		*self -= Self::broadcast(rhs);
	}
}

impl<U: UnderlierType, Scalar: BinaryField> MulAssign<Scalar> for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + MulAssign<Self>,
{
	fn mul_assign(&mut self, rhs: Scalar) {
		*self *= Self::broadcast(rhs);
	}
}

impl<U: UnderlierType, Scalar: BinaryField> Sum for PackedPrimitiveType<U, Scalar>
where
	Self: Add<Output = Self>,
{
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::from(U::default()), |result, next| result + next)
	}
}

impl<U: UnderlierType, Scalar: BinaryField> Product for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Mul<Output = Self>,
{
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::broadcast(Scalar::ONE), |result, next| result * next)
	}
}

unsafe impl<U: UnderlierType + Zeroable, Scalar: BinaryField> Zeroable
	for PackedPrimitiveType<U, Scalar>
{
}

unsafe impl<U: UnderlierType + Pod, Scalar: BinaryField> Pod for PackedPrimitiveType<U, Scalar> {}

impl<U: UnderlierWithBitOps, Scalar> PackedField for PackedPrimitiveType<U, Scalar>
where
	Self: Broadcast<Scalar> + Square + InvertOrZero + Mul<Output = Self>,
	U: UnderlierWithBitConstants + From<Scalar::Underlier> + Send + Sync + 'static,
	Scalar: BinaryField + WithUnderlier<Underlier: UnderlierWithBitOps>,
	Scalar::Underlier: NumCast<U>,
	IterationMethods<Scalar::Underlier, U>: IterationStrategy<Scalar::Underlier, U>,
{
	type Scalar = Scalar;

	const LOG_WIDTH: usize = (U::BITS / Scalar::N_BITS).ilog2() as usize;

	#[inline]
	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
		Scalar::from_underlier(self.0.get_subvalue(i))
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, i: usize, scalar: Scalar) {
		self.0.set_subvalue(i, scalar.to_underlier());
	}

	#[inline]
	fn zero() -> Self {
		Self::from_underlier(U::ZERO)
	}

	fn random(rng: impl RngCore) -> Self {
		U::random(rng).into()
	}

	#[inline]
	fn iter(&self) -> impl Iterator<Item = Self::Scalar> + Send + '_ {
		IterationMethods::<Scalar::Underlier, U>::ref_iter(&self.0)
			.map(|underlier| Scalar::from_underlier(underlier))
	}

	#[inline]
	fn into_iter(self) -> impl Iterator<Item = Self::Scalar> + Send {
		IterationMethods::<Scalar::Underlier, U>::value_iter(self.0)
			.map(|underlier| Scalar::from_underlier(underlier))
	}

	#[inline]
	fn iter_slice(slice: &[Self]) -> impl Iterator<Item = Self::Scalar> + Send + '_ {
		IterationMethods::<Scalar::Underlier, U>::slice_iter(Self::to_underliers_ref(slice))
			.map_skippable(|underlier| Scalar::from_underlier(underlier))
	}

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_WIDTH);
		let log_bit_len = Self::Scalar::N_BITS.ilog2() as usize;
		let (c, d) = self.0.interleave(other.0, log_block_len + log_bit_len);
		(c.into(), d.into())
	}

	#[inline]
	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_WIDTH);
		let log_bit_len = Self::Scalar::N_BITS.ilog2() as usize;
		let (c, d) = self.0.transpose(other.0, log_block_len + log_bit_len);
		(c.into(), d.into())
	}

	#[inline]
	unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
		debug_assert!(log_block_len <= Self::LOG_WIDTH, "{} <= {}", log_block_len, Self::LOG_WIDTH);
		debug_assert!(
			block_idx < 1 << (Self::LOG_WIDTH - log_block_len),
			"{} < {}",
			block_idx,
			1 << (Self::LOG_WIDTH - log_block_len)
		);

		self.0
			.spread::<<Self::Scalar as WithUnderlier>::Underlier>(log_block_len, block_idx)
			.into()
	}

	#[inline]
	fn broadcast(scalar: Self::Scalar) -> Self {
		<Self as Broadcast<Self::Scalar>>::broadcast(scalar)
	}

	#[inline]
	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		U::from_fn(move |i| f(i).to_underlier()).into()
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

macro_rules! impl_ops_for_zero_height {
	($name:ty) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			#[inline]
			fn mul(self, b: Self) -> Self {
				crate::tracing::trace_multiplication!($name);

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
	let bigger_lhs = PT2::from_underlier(lhs.to_underlier().into());
	let bigger_rhs = PT2::from_underlier(rhs.to_underlier().into());

	let bigger_result = bigger_lhs * bigger_rhs;

	PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
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
	let bigger_val = PT2::from_underlier(val.to_underlier().into());

	let bigger_result = bigger_val.square();

	PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
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
	let bigger_val = PT2::from_underlier(val.to_underlier().into());

	let bigger_result = bigger_val.invert_or_zero();

	PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
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
	let bigger_val = PT2::from_underlier(val.to_underlier().into());

	let bigger_result = bigger_val.mul_alpha();

	PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
}

macro_rules! impl_pack_scalar {
	($underlier:ty) => {
		impl<F> $crate::as_packed_field::PackScalar<F> for $underlier
		where
			F: BinaryField,
			PackedPrimitiveType<$underlier, F>:
				$crate::packed::PackedField<Scalar = F> + WithUnderlier<Underlier = $underlier>,
		{
			type Packed = PackedPrimitiveType<$underlier, F>;
		}
	};
}

pub(crate) use impl_pack_scalar;

impl_pack_scalar!(U1);
impl_pack_scalar!(U2);
impl_pack_scalar!(U4);
impl_pack_scalar!(u8);
impl_pack_scalar!(u16);
impl_pack_scalar!(u32);
impl_pack_scalar!(u64);
impl_pack_scalar!(u128);
