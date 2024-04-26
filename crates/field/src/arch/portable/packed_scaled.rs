// Copyright 2024 Ulvetanna Inc.

use crate::{
	affine_transformation::{
		FieldAffineTransformation, PackedTransformationFactory, Transformation,
	},
	arithmetic_traits::MulAlpha,
	packed::PackedBinaryField,
	Error, PackedField,
};
use bytemuck::{Pod, Zeroable};
use std::{
	array,
	iter::{Product, Sum},
	ops::{Add, AddAssign, Deref, Mul, MulAssign, Sub, SubAssign},
};
use subtle::ConstantTimeEq;

/// Packed field that just stores smaller packed field N times and performs all operations
/// one by one.
/// This makes sense for creating portable implementations for 256 and 512 packed sizes.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ScaledPackedField<PT, const N: usize>(pub(super) [PT; N]);

impl<PT, const N: usize> ScaledPackedField<PT, N> {
	pub const WIDTH_IN_PT: usize = N;

	pub fn from_fn(f: impl FnMut(usize) -> PT) -> Self {
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

impl<PT, const N: usize> From<ScaledPackedField<PT, N>> for [PT; N] {
	fn from(value: ScaledPackedField<PT, N>) -> Self {
		value.0
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
		Self::from_fn(|i| self.0[i] + rhs.0[i])
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
		Self::from_fn(|i| self.0[i] - rhs.0[i])
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
		Self::from_fn(|i| self.0[i] * rhs.0[i])
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

	const LOG_WIDTH: usize = { PT::LOG_WIDTH + N.ilog2() as usize };

	fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
		let outer_i = i / PT::WIDTH;
		let inner_i = i % PT::WIDTH;
		self.0
			.get(outer_i)
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
			.and_then(|inner| inner.get_checked(inner_i))
	}

	fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error> {
		let outer_i = i / PT::WIDTH;
		let inner_i = i % PT::WIDTH;
		self.0
			.get_mut(outer_i)
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
			.and_then(|inner| inner.set_checked(inner_i, scalar))
	}

	fn random(mut rng: impl rand::prelude::RngCore) -> Self {
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

				first[i..i + block_in_pts]
					.copy_from_slice(&self.0[i + block_in_pts..i + 2 * block_in_pts]);
				first[i + block_in_pts..i + 2 * block_in_pts]
					.copy_from_slice(&other.0[i + block_in_pts..i + 2 * block_in_pts]);
			}
		} else {
			for i in (0..N).step_by(2) {
				(first[i], first[i + 1]) = self.0[i].interleave(other.0[i], log_block_len);
				(second[i], second[i + 1]) =
					self.0[i + 1].interleave(other.0[i + 1], log_block_len);
			}
		}

		(Self(first), Self(second))
	}

	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		Self(std::array::from_fn(|i| PT::from_fn(|j| f(i * PT::WIDTH + j))))
	}
}

impl<PT: PackedField + MulAlpha, const N: usize> MulAlpha for ScaledPackedField<PT, N>
where
	[PT; N]: Default,
{
	fn mul_alpha(self) -> Self {
		Self(self.0.map(|v| v.mul_alpha()))
	}
}

/// Per-element transformation as a scaled packed field.
struct ScaledTransformation<I> {
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
		ScaledPackedField::from_fn(|i| self.inner.transform(&data.0[i]))
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
	fn make_packed_transformation<Data: Deref<Target = [OP::Scalar]>>(
		transformation: FieldAffineTransformation<
			<ScaledPackedField<OP, N> as PackedField>::Scalar,
			Data,
		>,
	) -> impl Transformation<Self, ScaledPackedField<OP, N>> {
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

		unsafe impl<P> $crate::extension::PackedExtensionField<P> for $name
		where
			P: $crate::packed::PackedField,
			$inner: $crate::extension::PackedExtensionField<P>,
			<$inner as $crate::packed::PackedField>::Scalar:
				$crate::extension::ExtensionField<P::Scalar>,
		{
			fn cast_to_bases(packed: &[Self]) -> &[P] {
				<$inner>::cast_to_bases(bytemuck::must_cast_slice(packed))
			}

			fn cast_to_bases_mut(packed: &mut [Self]) -> &mut [P] {
				<$inner>::cast_to_bases_mut(bytemuck::must_cast_slice_mut(packed))
			}

			fn try_cast_to_ext(packed: &[P]) -> Option<&[Self]> {
				<$inner>::try_cast_to_ext(packed)
					.and_then(|bases| bytemuck::try_cast_slice(bases).ok())
			}

			fn try_cast_to_ext_mut(packed: &mut [P]) -> Option<&mut [Self]> {
				<$inner>::try_cast_to_ext_mut(packed)
					.and_then(|bases| bytemuck::try_cast_slice_mut(bases).ok())
			}
		}
	};
}

pub(crate) use packed_scaled_field;

macro_rules! impl_scaled_512_bit_conversion_from_u128_array {
	($name:ty, $inner:ty) => {
		impl From<[u128; 4]> for $name {
			fn from(value: [u128; 4]) -> Self {
				Self([
					<$inner>::from([value[0], value[1]]),
					<$inner>::from([value[2], value[3]]),
				])
			}
		}
	};
}

pub(crate) use impl_scaled_512_bit_conversion_from_u128_array;
