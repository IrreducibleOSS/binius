// Copyright 2023-2024 Irreducible Inc.

//! Traits for packed field elements which support SIMD implementations.
//!
//! Interfaces are derived from [`plonky2`](https://github.com/mir-protocol/plonky2).

use super::{
	arithmetic_traits::{Broadcast, MulAlpha, Square},
	binary_field_arithmetic::TowerFieldArithmetic,
	Error,
};
use crate::{
	arithmetic_traits::InvertOrZero, underlier::WithUnderlier, BinaryField, ExtensionField, Field,
	PackedExtension,
};
use binius_utils::iter::IterExtensions;
use bytemuck::Zeroable;
use rand::RngCore;
use std::{
	fmt::Debug,
	iter::{self, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A packed field represents a vector of underlying field elements.
///
/// Arithmetic operations on packed field elements can be accelerated with SIMD CPU instructions.
/// The vector width is a constant, `WIDTH`. This trait requires that the width must be a power of
/// two.
pub trait PackedField:
	Default
	+ Debug
	+ Clone
	+ Copy
	+ Eq
	+ Sized
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Mul<Output = Self>
	+ AddAssign
	+ SubAssign
	+ MulAssign
	+ Add<Self::Scalar, Output = Self>
	+ Sub<Self::Scalar, Output = Self>
	+ Mul<Self::Scalar, Output = Self>
	+ AddAssign<Self::Scalar>
	+ SubAssign<Self::Scalar>
	+ MulAssign<Self::Scalar>
	// TODO: Get rid of Sum and Product. It's confusing with nested impls of Packed.
	+ Sum
	+ Product
	+ Send
	+ Sync
	+ Zeroable
	+ 'static
{
	type Scalar: Field;

	/// Base-2 logarithm of the number of field elements packed into one packed element.
	const LOG_WIDTH: usize;

	/// The number of field elements packed into one packed element.
	///
	/// WIDTH is guaranteed to equal 2^LOG_WIDTH.
	const WIDTH: usize = 1 << Self::LOG_WIDTH;

	/// Get the scalar at a given index without bounds checking.
	/// # Safety
	/// The caller must ensure that `i` is less than `WIDTH`.
	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar;

	/// Set the scalar at a given index without bounds checking.
	/// # Safety
	/// The caller must ensure that `i` is less than `WIDTH`.
	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar);

	/// Get the scalar at a given index.
	#[inline]
	fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
		(i < Self::WIDTH)
			.then_some(unsafe { self.get_unchecked(i) })
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
	}

	/// Set the scalar at a given index.
	#[inline]
	fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error> {
		(i < Self::WIDTH)
			.then(|| unsafe { self.set_unchecked(i, scalar) })
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
	}

	/// Get the scalar at a given index.
	#[inline]
	fn get(&self, i: usize) -> Self::Scalar {
		self.get_checked(i).expect("index must be less than width")
	}

	/// Set the scalar at a given index.
	#[inline]
	fn set(&mut self, i: usize, scalar: Self::Scalar) {
		self.set_checked(i, scalar).expect("index must be less than width")
	}

	#[inline]
	fn into_iter(self) -> impl Iterator<Item=Self::Scalar> {
		(0..Self::WIDTH).map_skippable(move |i|
			// Safety: `i` is always less than `WIDTH`
			unsafe { self.get_unchecked(i) })
	}

	#[inline]
	fn iter(&self) -> impl Iterator<Item=Self::Scalar> + Send {
		(0..Self::WIDTH).map_skippable(move |i|
			// Safety: `i` is always less than `WIDTH`
			unsafe { self.get_unchecked(i) })
	}

	#[inline]
	fn zero() -> Self {
		Self::broadcast(Self::Scalar::ZERO)
	}

	#[inline]
	fn one() -> Self {
		Self::broadcast(Self::Scalar::ONE)
	}

	/// Initialize zero position with `scalar`, set other elements to zero.
	#[inline(always)]
	fn set_single(scalar: Self::Scalar) -> Self {
		let mut result = Self::default();
		result.set(0, scalar);

		result
	}

	fn random(rng: impl RngCore) -> Self;
	fn broadcast(scalar: Self::Scalar) -> Self;

	/// Construct a packed field element from a function that returns scalar values by index.
	fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self;

	/// Construct a packed field element from a sequence of scalars.
	///
	/// If the number of values in the sequence is less than the packing width, the remaining
	/// elements are set to zero. If greater than the packing width, the excess elements are
	/// ignored.
	fn from_scalars(values: impl IntoIterator<Item=Self::Scalar>) -> Self {
		let mut result = Self::default();
		for (i, val) in values.into_iter().take(Self::WIDTH).enumerate() {
			result.set(i, val);
		}
		result
	}

	/// Returns the value multiplied by itself
	fn square(self) -> Self;

	/// Returns the packed inverse values or zeroes at indices where `self` is zero.
	fn invert_or_zero(self) -> Self;

	/// Interleaves blocks of this packed vector with another packed vector.
	///
	/// The operation can be seen as stacking the two vectors, dividing them into 2x2 matrices of
	/// blocks, where each block is 2^`log_block_width` elements, and transposing the matrices.
	///
	/// Consider this example, where `LOG_WIDTH` is 3 and `log_block_len` is 1:
	///     A = [a0, a1, a2, a3, a4, a5, a6, a7]
	///     B = [b0, b1, b2, b3, b4, b5, b6, b7]
	///
	/// The interleaved result is
	///     A' = [a0, a1, b0, b1, a4, a5, b4, b5]
	///     B' = [a2, a3, b2, b3, a6, a7, b6, b7]
	///
	/// ## Preconditions
	/// * `log_block_len` must be strictly less than `LOG_WIDTH`.
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self);
}

pub fn iter_packed_slice<P: PackedField>(
	packed: &[P],
) -> impl Iterator<Item = P::Scalar> + '_ + Send {
	packed.iter().flat_map(|packed_i| packed_i.iter())
}

#[inline]
pub fn get_packed_slice<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	// Safety: `i % P::WIDTH` is always less than `P::WIDTH
	unsafe { packed[i / P::WIDTH].get_unchecked(i % P::WIDTH) }
}

/// Returns the scalar at the given index without bounds checking.
/// # Safety
/// The caller must ensure that `i` is less than `P::WIDTH * packed.len()`.
#[inline]
pub unsafe fn get_packed_slice_unchecked<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	packed
		.get_unchecked(i / P::WIDTH)
		.get_unchecked(i % P::WIDTH)
}

pub fn get_packed_slice_checked<P: PackedField>(
	packed: &[P],
	i: usize,
) -> Result<P::Scalar, Error> {
	packed
		.get(i / P::WIDTH)
		.map(|el| el.get(i % P::WIDTH))
		.ok_or(Error::IndexOutOfRange {
			index: i,
			max: packed.len() * P::WIDTH,
		})
}

/// Sets the scalar at the given index without bounds checking.
/// # Safety
/// The caller must ensure that `i` is less than `P::WIDTH * packed.len()`.
pub unsafe fn set_packed_slice_unchecked<P: PackedField>(
	packed: &mut [P],
	i: usize,
	scalar: P::Scalar,
) {
	unsafe {
		packed
			.get_unchecked_mut(i / P::WIDTH)
			.set_unchecked(i % P::WIDTH, scalar)
	}
}

pub fn set_packed_slice<P: PackedField>(packed: &mut [P], i: usize, scalar: P::Scalar) {
	// Safety: `i % P::WIDTH` is always less than `P::WIDTH
	unsafe { packed[i / P::WIDTH].set_unchecked(i % P::WIDTH, scalar) }
}

pub fn set_packed_slice_checked<P: PackedField>(
	packed: &mut [P],
	i: usize,
	scalar: P::Scalar,
) -> Result<(), Error> {
	packed
		.get_mut(i / P::WIDTH)
		.map(|el| el.set(i % P::WIDTH, scalar))
		.ok_or(Error::IndexOutOfRange {
			index: i,
			max: packed.len() * P::WIDTH,
		})
}

pub fn len_packed_slice<P: PackedField>(packed: &[P]) -> usize {
	packed.len() * P::WIDTH
}

/// Multiply packed field element by a subfield scalar.
pub fn mul_by_subfield_scalar<P, FS>(val: P, multiplier: FS) -> P
where
	P: PackedExtension<FS, Scalar: ExtensionField<FS>>,
	FS: Field,
{
	use crate::underlier::UnderlierType;

	// This is a workaround not to make the multiplication slower in certain cases.
	// TODO: implement efficient strategy to multiply packed field by a subfield scalar.
	let subfield_bits = FS::Underlier::BITS;
	let extension_bits = <<P as PackedField>::Scalar as WithUnderlier>::Underlier::BITS;

	if (subfield_bits == 1 && extension_bits > 8) || extension_bits >= 32 {
		P::from_fn(|i| unsafe { val.get_unchecked(i) } * multiplier)
	} else {
		*P::cast_ext(&(*P::cast_base(&val) * P::PackedSubfield::broadcast(multiplier)))
	}
}

impl<F: Field> Broadcast<F> for F {
	fn broadcast(scalar: F) -> Self {
		scalar
	}
}

impl<T: TowerFieldArithmetic> MulAlpha for T {
	#[inline]
	fn mul_alpha(self) -> Self {
		<Self as TowerFieldArithmetic>::multiply_alpha(self)
	}
}

impl<F: Field> PackedField for F {
	type Scalar = F;

	const LOG_WIDTH: usize = 0;

	#[inline]
	unsafe fn get_unchecked(&self, _i: usize) -> Self::Scalar {
		*self
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, _i: usize, scalar: Self::Scalar) {
		*self = scalar;
	}

	fn iter(&self) -> impl Iterator<Item = Self::Scalar> {
		iter::once(*self)
	}

	fn random(rng: impl RngCore) -> Self {
		<Self as Field>::random(rng)
	}

	fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
		panic!("cannot interleave when WIDTH = 1");
	}

	fn broadcast(scalar: Self::Scalar) -> Self {
		scalar
	}

	fn square(self) -> Self {
		<Self as Square>::square(self)
	}

	fn invert_or_zero(self) -> Self {
		<Self as InvertOrZero>::invert_or_zero(self)
	}

	#[inline]
	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		f(0)
	}
}

/// A helper trait to make the generic bunds shorter
pub trait PackedBinaryField: PackedField<Scalar: BinaryField> {}

impl<PT> PackedBinaryField for PT where PT: PackedField<Scalar: BinaryField> {}
