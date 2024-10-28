// Copyright 2024 Irreducible Inc.

use crate::{
	arithmetic_traits::MulAlpha,
	as_packed_field::PackScalar,
	linear_transformation::{
		FieldLinearTransformation, PackedTransformationFactory, Transformation,
	},
	packed::PackedBinaryField,
	underlier::{ScaledUnderlier, UnderlierType, WithUnderlier},
	Field, PackedField,
};
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{Pod, TransparentWrapper, Zeroable};
use rand::RngCore;
use std::{
	array,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Deref, Mul, MulAssign, Sub, SubAssign},
};
use subtle::ConstantTimeEq;

/// Packed field that just stores smaller packed field N times and performs all operations
/// one by one.
/// This makes sense for creating portable implementations for 256 and 512 packed sizes.
#[derive(PartialEq, Eq, Clone, Copy, Debug, bytemuck::TransparentWrapper)]
#[repr(transparent)]
pub struct ScaledPackedField<PT, const N: usize>(pub(super) [PT; N]);

impl<PT, const N: usize> ScaledPackedField<PT, N> {
	pub const WIDTH_IN_PT: usize = N;

	/// In general case PT != Self::Scalar, so this function has a different name from `PackedField::from_fn`
	pub fn from_direct_packed_fn(f: impl FnMut(usize) -> PT) -> Self {
		Self(std::array::from_fn(f))
	}
}

impl<PT, const N: usize> Default for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	fn default() -> Self {
		Self(Default::default())
	}
}

impl<U, PT, const N: usize> From<[U; N]> for ScaledPackedField<PT, N>
where
	PT: From<U>,
{
	fn from(value: [U; N]) -> Self {
		Self(value.map(Into::into))
	}
}

impl<U, PT, const N: usize> From<ScaledPackedField<PT, N>> for [U; N]
where
	U: From<PT>,
{
	fn from(value: ScaledPackedField<PT, N>) -> Self {
		value.0.map(Into::into)
	}
}

unsafe impl<PT: Zeroable, const N: usize> Zeroable for ScaledPackedField<PT, N> {}

unsafe impl<PT: Pod, const N: usize> Pod for ScaledPackedField<PT, N> {}

impl<PT: ConstantTimeEq, const N: usize> ConstantTimeEq for ScaledPackedField<PT, N> {
	fn ct_eq(&self, other: &Self) -> subtle::Choice {
		self.0.ct_eq(&other.0)
	}
}

impl<PT: Copy + Add<Output = PT>, const N: usize> Add for ScaledPackedField<PT, N>
where
	Self: Default,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self::from_direct_packed_fn(|i| self.0[i] + rhs.0[i])
	}
}

impl<PT: Copy + AddAssign, const N: usize> AddAssign for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn add_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] += rhs.0[i];
		}
	}
}

impl<PT: Copy + Sub<Output = PT>, const N: usize> Sub for ScaledPackedField<PT, N>
where
	Self: Default,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		Self::from_direct_packed_fn(|i| self.0[i] - rhs.0[i])
	}
}

impl<PT: Copy + SubAssign, const N: usize> SubAssign for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn sub_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] -= rhs.0[i];
		}
	}
}

impl<PT: Copy + Mul<Output = PT>, const N: usize> Mul for ScaledPackedField<PT, N>
where
	Self: Default,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self::from_direct_packed_fn(|i| self.0[i] * rhs.0[i])
	}
}

impl<PT: Copy + MulAssign, const N: usize> MulAssign for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn mul_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] *= rhs.0[i];
		}
	}
}

/// Currently we use this trait only in this file for compactness.
/// If it is useful in some other place it worth moving it to `arithmetic_traits.rs`
trait ArithmeticOps<Rhs>:
	Add<Rhs, Output = Self>
	+ AddAssign<Rhs>
	+ Sub<Rhs, Output = Self>
	+ SubAssign<Rhs>
	+ Mul<Rhs, Output = Self>
	+ MulAssign<Rhs>
{
}

impl<T, Rhs> ArithmeticOps<Rhs> for T where
	T: Add<Rhs, Output = Self>
		+ AddAssign<Rhs>
		+ Sub<Rhs, Output = Self>
		+ SubAssign<Rhs>
		+ Mul<Rhs, Output = Self>
		+ MulAssign<Rhs>
{
}

impl<PT: Add<Output = PT> + Copy, const N: usize> Sum for ScaledPackedField<PT, N>
where
	Self: Default,
{
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |l, r| l + r)
	}
}

impl<PT: PackedField, const N: usize> Product for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		let one = Self([PT::one(); N]);
		iter.fold(one, |l, r| l * r)
	}
}

impl<PT: PackedField, const N: usize> PackedField for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
	Self: ArithmeticOps<PT::Scalar>,
{
	type Scalar = PT::Scalar;

	const LOG_WIDTH: usize = PT::LOG_WIDTH + checked_log_2(N);

	#[inline]
	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
		let outer_i = i / PT::WIDTH;
		let inner_i = i % PT::WIDTH;
		self.0.get_unchecked(outer_i).get_unchecked(inner_i)
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
		let outer_i = i / PT::WIDTH;
		let inner_i = i % PT::WIDTH;
		self.0
			.get_unchecked_mut(outer_i)
			.set_unchecked(inner_i, scalar);
	}

	#[inline]
	fn zero() -> Self {
		Self(array::from_fn(|_| PT::zero()))
	}

	fn random(mut rng: impl RngCore) -> Self {
		Self(array::from_fn(|_| PT::random(&mut rng)))
	}

	fn broadcast(scalar: Self::Scalar) -> Self {
		Self(array::from_fn(|_| PT::broadcast(scalar)))
	}

	fn square(self) -> Self {
		Self(self.0.map(|v| v.square()))
	}

	fn invert_or_zero(self) -> Self {
		Self(self.0.map(|v| v.invert_or_zero()))
	}

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let mut first = [Default::default(); N];
		let mut second = [Default::default(); N];

		if log_block_len >= PT::LOG_WIDTH {
			let block_in_pts = 1 << (log_block_len - PT::LOG_WIDTH);
			for i in (0..N).step_by(block_in_pts * 2) {
				first[i..i + block_in_pts].copy_from_slice(&self.0[i..i + block_in_pts]);
				first[i + block_in_pts..i + 2 * block_in_pts]
					.copy_from_slice(&other.0[i..i + block_in_pts]);

				second[i..i + block_in_pts]
					.copy_from_slice(&self.0[i + block_in_pts..i + 2 * block_in_pts]);
				second[i + block_in_pts..i + 2 * block_in_pts]
					.copy_from_slice(&other.0[i + block_in_pts..i + 2 * block_in_pts]);
			}
		} else {
			for i in 0..N {
				(first[i], second[i]) = self.0[i].interleave(other.0[i], log_block_len);
			}
		}

		(Self(first), Self(second))
	}

	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		Self(array::from_fn(|i| PT::from_fn(|j| f(i * PT::WIDTH + j))))
	}
}

impl<PT: PackedField + MulAlpha, const N: usize> MulAlpha for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		Self(self.0.map(|v| v.mul_alpha()))
	}
}

/// Per-element transformation as a scaled packed field.
pub struct ScaledTransformation<I> {
	inner: I,
}

impl<I> ScaledTransformation<I> {
	fn new(inner: I) -> Self {
		Self { inner }
	}
}

impl<OP, IP, const N: usize, I> Transformation<ScaledPackedField<IP, N>, ScaledPackedField<OP, N>>
	for ScaledTransformation<I>
where
	I: Transformation<IP, OP>,
{
	fn transform(&self, data: &ScaledPackedField<IP, N>) -> ScaledPackedField<OP, N> {
		ScaledPackedField::from_direct_packed_fn(|i| self.inner.transform(&data.0[i]))
	}
}

impl<OP, IP, const N: usize> PackedTransformationFactory<ScaledPackedField<OP, N>>
	for ScaledPackedField<IP, N>
where
	ScaledPackedField<IP, N>: PackedBinaryField,
	ScaledPackedField<OP, N>: PackedBinaryField<Scalar = OP::Scalar>,
	OP: PackedBinaryField,
	IP: PackedTransformationFactory<OP>,
{
	type PackedTransformation<Data: Deref<Target = [OP::Scalar]>> =
		ScaledTransformation<IP::PackedTransformation<Data>>;

	fn make_packed_transformation<Data: Deref<Target = [OP::Scalar]>>(
		transformation: FieldLinearTransformation<
			<ScaledPackedField<OP, N> as PackedField>::Scalar,
			Data,
		>,
	) -> Self::PackedTransformation<Data> {
		ScaledTransformation::new(IP::make_packed_transformation(transformation))
	}
}

/// The only thing that prevents us from having pure generic implementation of `ScaledPackedField`
/// is that we can't have generic operations both with `Self` and `PT::Scalar`
/// (it leads to `conflicting implementations of trait` error).
/// That's why we implement one of those in a macro.
macro_rules! packed_scaled_field {
	($name:ident = [$inner:ty;$size:literal]) => {
		pub type $name = $crate::arch::portable::packed_scaled::ScaledPackedField<$inner, $size>;

		impl std::ops::Add<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			type Output = Self;

			fn add(self, rhs: <$inner as $crate::packed::PackedField>::Scalar) -> Self {
				let mut result = Self::default();
				for i in 0..Self::WIDTH_IN_PT {
					result.0[i] = self.0[i] + rhs;
				}

				result
			}
		}

		impl std::ops::AddAssign<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			fn add_assign(&mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) {
				for i in 0..Self::WIDTH_IN_PT {
					self.0[i] += rhs;
				}
			}
		}

		impl std::ops::Sub<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			type Output = Self;

			fn sub(self, rhs: <$inner as $crate::packed::PackedField>::Scalar) -> Self {
				let mut result = Self::default();
				for i in 0..Self::WIDTH_IN_PT {
					result.0[i] = self.0[i] - rhs;
				}

				result
			}
		}

		impl std::ops::SubAssign<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			fn sub_assign(&mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) {
				for i in 0..Self::WIDTH_IN_PT {
					self.0[i] -= rhs;
				}
			}
		}

		impl std::ops::Mul<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			type Output = Self;

			fn mul(self, rhs: <$inner as $crate::packed::PackedField>::Scalar) -> Self {
				let mut result = Self::default();
				for i in 0..Self::WIDTH_IN_PT {
					result.0[i] = self.0[i] * rhs;
				}

				result
			}
		}

		impl std::ops::MulAssign<<$inner as $crate::packed::PackedField>::Scalar> for $name {
			fn mul_assign(&mut self, rhs: <$inner as $crate::packed::PackedField>::Scalar) {
				for i in 0..Self::WIDTH_IN_PT {
					self.0[i] *= rhs;
				}
			}
		}
	};
}

pub(crate) use packed_scaled_field;

unsafe impl<PT, const N: usize> WithUnderlier for ScaledPackedField<PT, N>
where
	PT: WithUnderlier<Underlier: Pod>,
{
	type Underlier = ScaledUnderlier<PT::Underlier, N>;

	fn to_underlier(self) -> Self::Underlier {
		TransparentWrapper::peel(self)
	}

	fn to_underlier_ref(&self) -> &Self::Underlier {
		TransparentWrapper::peel_ref(self)
	}

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		TransparentWrapper::peel_mut(self)
	}

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		TransparentWrapper::peel_slice(val)
	}

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		TransparentWrapper::peel_slice_mut(val)
	}

	fn from_underlier(val: Self::Underlier) -> Self {
		TransparentWrapper::wrap(val)
	}

	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		TransparentWrapper::wrap_ref(val)
	}

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		TransparentWrapper::wrap_mut(val)
	}

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		TransparentWrapper::wrap_slice(val)
	}

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		TransparentWrapper::wrap_slice_mut(val)
	}
}

impl<U, F, const N: usize> PackScalar<F> for ScaledUnderlier<U, N>
where
	U: PackScalar<F> + UnderlierType + Pod,
	F: Field,
	ScaledPackedField<U::Packed, N>:
		PackedField<Scalar = F> + WithUnderlier<Underlier = ScaledUnderlier<U, N>>,
{
	type Packed = ScaledPackedField<U::Packed, N>;
}

unsafe impl<PT, U, const N: usize> TransparentWrapper<ScaledUnderlier<U, N>>
	for ScaledPackedField<PT, N>
where
	PT: WithUnderlier<Underlier = U>,
{
}
