// Copyright 2025 Irreducible Inc.

use std::{
	fmt::Debug,
	marker::PhantomData,
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};

use crate::{
	arch::byte_sliced::{invert::invert_or_zero, multiply::mul, square::square},
	tower_levels::{TowerLevel16, TowerLevelWithArithOps},
	underlier::WithUnderlier,
	AESTowerField128b, AESTowerField8b, ExtensionField, PackedAESBinaryField16x8b,
	PackedAESBinaryField1x128b, PackedField,
};

trait PodProps: Zeroable + Copy + Pod + Eq + Send + Sync {}

impl<T: Zeroable + Copy + Pod + Eq + Send + Sync> PodProps for T {}

#[repr(transparent)]
#[allow(private_bounds)]
pub struct ByteSliced2D<
	F: ExtensionField<AESTowerField8b>,
	PackedStorage: PackedField<Scalar = AESTowerField8b>,
	TL: TowerLevelWithArithOps,
> where
	TL::Data<PackedStorage>: PodProps,
{
	data: TL::Data<PackedStorage>,
	_pd: PhantomData<F>,
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Clone for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	fn clone(&self) -> Self {
		Self {
			data: self.data.clone(),
			_pd: PhantomData,
		}
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Copy for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
}

unsafe impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Zeroable for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
}

unsafe impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Pod for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
}

unsafe impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Send for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
}

unsafe impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Sync for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b> + std::fmt::Debug,
		TL: TowerLevelWithArithOps,
	> PartialEq for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	fn eq(&self, other: &Self) -> bool {
		self.data.as_ref() == other.data.as_ref()
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b> + std::fmt::Debug,
		TL: TowerLevelWithArithOps,
	> Eq for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b> + std::fmt::Debug,
		TL: TowerLevelWithArithOps,
	> Debug for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let values_str = (0..TL::WIDTH)
			.map(|i| format!("{:?}", self.data[i]))
			.collect::<Vec<_>>()
			.join(", ");
		write!(f, "ByteSliced2D([{}])", values_str)
	}
}

#[allow(private_bounds)]
impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	pub const BYTES: usize = PackedStorage::WIDTH * TL::WIDTH;

	/// Get the byte at the given index.
	///
	/// # Safety
	/// The caller must ensure that `byte_index` is less than `BYTES`.
	#[allow(clippy::modulo_one)]
	#[inline(always)]
	pub unsafe fn get_byte_unchecked(&self, byte_index: usize) -> u8 {
		self.data
			.as_ref()
			.get_unchecked(byte_index % TL::WIDTH)
			.get_unchecked(byte_index / TL::WIDTH)
			.to_underlier()
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> PackedField for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Scalar = F;

	const LOG_WIDTH: usize = PackedStorage::LOG_WIDTH;

	#[inline(always)]
	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
		F::from_bases((0..F::DEGREE).map(|byte_index| {
			self.data
				.as_ref()
				.get_unchecked(byte_index)
				.get_unchecked(i)
				.into()
		}))
		.expect("number of bases is correct")
	}

	#[inline(always)]
	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
		for byte_index in 0..TL::WIDTH {
			self.data[byte_index].set_unchecked(i, scalar.get_base(byte_index));
		}
	}

	fn random(mut rng: impl rand::RngCore) -> Self {
		Self {
			data: TL::from_fn(|_| PackedStorage::random(&mut rng)),
			_pd: PhantomData,
		}
	}

	#[inline]
	fn broadcast(scalar: Self::Scalar) -> Self {
		Self {
			data: TL::from_fn(|byte_index| {
				PackedStorage::broadcast(unsafe { scalar.get_base_unchecked(byte_index) })
			}),
			_pd: PhantomData,
		}
	}

	#[inline]
	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		let data = TL::from_fn(|byte_index| PackedStorage::from_fn(|i| f(i).get_base(byte_index)));

		Self {
			data,
			_pd: PhantomData,
		}
	}

	#[inline]
	fn square(self) -> Self {
		let mut result = Self::default();

		square::<PackedStorage, TL>(&self.data, &mut result.data);

		result
	}

	#[inline]
	fn invert_or_zero(self) -> Self {
		let mut result = Self::default();
		invert_or_zero::<PackedStorage, TL>(&self.data, &mut result.data);
		result
	}

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let mut result1 = Self::default();
		let mut result2 = Self::default();

		for byte_num in 0..TL::WIDTH {
			(result1.data[byte_num], result2.data[byte_num]) =
				self.data[byte_num].interleave(other.data[byte_num], log_block_len);
		}

		(result1, result2)
	}

	#[inline]
	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let mut result1 = Self::default();
		let mut result2 = Self::default();

		for byte_num in 0..TL::WIDTH {
			(result1.data[byte_num], result2.data[byte_num]) =
				self.data[byte_num].unzip(other.data[byte_num], log_block_len);
		}

		(result1, result2)
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Mul for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		let mut result = Self::default();

		mul::<PackedStorage, TL>(&self.data, &rhs.data, &mut result.data);

		result
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Add<F> for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Output = Self;

	#[inline]
	fn add(self, rhs: F) -> Self {
		self + Self::broadcast(rhs)
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> AddAssign<F> for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	#[inline]
	fn add_assign(&mut self, rhs: F) {
		*self += Self::broadcast(rhs)
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Sub<F> for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Output = Self;

	#[inline]
	fn sub(self, rhs: F) -> Self {
		self.add(rhs)
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> SubAssign<F> for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	#[inline]
	fn sub_assign(&mut self, rhs: F) {
		self.add_assign(rhs)
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Mul<F> for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Output = Self;

	#[inline]
	fn mul(self, rhs: F) -> Self {
		self * Self::broadcast(rhs)
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> MulAssign<F> for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	#[inline]
	fn mul_assign(&mut self, rhs: F) {
		*self *= Self::broadcast(rhs);
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Default for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	fn default() -> Self {
		Self {
			data: TL::from_fn(|_| PackedStorage::zero()),
			_pd: PhantomData,
		}
	}
}
impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Add for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self {
			data: TL::from_fn(|byte_number| self.data[byte_number] + rhs.data[byte_number]),
			_pd: PhantomData,
		}
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> AddAssign for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		for (data, rhs) in self.data.as_mut().iter_mut().zip(rhs.data.as_ref()) {
			*data += *rhs;
		}
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> Sub for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self {
			data: TL::from_fn(|byte_number| self.data[byte_number] - rhs.data[byte_number]),
			_pd: PhantomData,
		}
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> SubAssign for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		for (data, rhs) in self.data.as_mut().iter_mut().zip(rhs.data.as_ref()) {
			*data -= *rhs;
		}
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> MulAssign for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	#[inline]
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> std::iter::Product for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		let mut result = Self::default();

		let mut is_first_item = true;
		for item in iter {
			if is_first_item {
				result = item;
			} else {
				result *= item;
			}

			is_first_item = false;
		}

		result
	}
}

impl<
		F: ExtensionField<AESTowerField8b>,
		PackedStorage: PackedField<Scalar = AESTowerField8b>,
		TL: TowerLevelWithArithOps,
	> std::iter::Sum for ByteSliced2D<F, PackedStorage, TL>
where
	TL::Data<PackedStorage>: PodProps,
{
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		let mut result = Self::default();

		for item in iter {
			result += item;
		}

		result
	}
}

pub type ByteSlicedAES16x128b =
	ByteSliced2D<AESTowerField128b, PackedAESBinaryField16x8b, TowerLevel16>;
