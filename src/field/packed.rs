// Copyright 2023 Ulvetanna Inc.

//! Traits for packed field elements which support SIMD implementations.
//!
//! Interfaces are derived from [`plonky2`](https://github.com/mir-protocol/plonky2).

use ff::Field;
use rand::RngCore;
use std::{
	iter::{self, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use subtle::ConstantTimeEq;

use super::Error;

/// A packed field represents a vector of underlying field elements.
///
/// Arithmetic operations on packed field elements can be accelerated with SIMD CPU instructions.
/// The vector width is a constant, `WIDTH`. This trait requires that the width must be a power of
/// two.
pub trait PackedField:
	Default
	+ Clone
	+ Copy
    + Eq
	+ Sized
	+ ConstantTimeEq
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
	+ 'static
{
	type Scalar: Field;

	/// Base-2 logarithm of the number of field elements packed into one packed element.
	const LOG_WIDTH: usize;

	/// The number of field elements packed into one packed element.
	///
	/// WIDTH is guaranteed to equal 2^LOG_WIDTH.
	const WIDTH: usize = 1 << Self::LOG_WIDTH;

	/// Get the scalar at a given index.
	fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error>;

	/// Set the scalar at a given index.
	fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error>;

	/// Get the scalar at a given index.
	fn get(&self, i: usize) -> Self::Scalar {
		self.get_checked(i).expect("index must be less than width")
	}

	/// Set the scalar at a given index.
	fn set(&mut self, i: usize, scalar: Self::Scalar) {
		self.set_checked(i, scalar).expect("index must be less than width")
	}

	fn into_iter(self) -> impl Iterator<Item=Self::Scalar> {
		(0..Self::WIDTH).map(move |i| self.get(i))
	}

	fn iter(&self) -> impl Iterator<Item=Self::Scalar> {
		(0..Self::WIDTH).map(|i| self.get(i))
	}

	fn zero() -> Self {
		Self::broadcast(Self::Scalar::ZERO)
	}

	fn one() -> Self {
		Self::broadcast(Self::Scalar::ONE)
	}

	fn random(rng: impl RngCore) -> Self;
	fn broadcast(scalar: Self::Scalar) -> Self;

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

pub fn iter_packed_slice<P: PackedField>(packed: &[P]) -> impl Iterator<Item = P::Scalar> + '_ {
	packed.iter().flat_map(|packed_i| packed_i.iter())
}

pub fn get_packed_slice<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	packed[i / P::WIDTH].get(i % P::WIDTH)
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

pub fn set_packed_slice<P: PackedField>(packed: &mut [P], i: usize, scalar: P::Scalar) {
	packed[i / P::WIDTH].set(i % P::WIDTH, scalar)
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

impl<F: Field> PackedField for F {
	type Scalar = F;

	const LOG_WIDTH: usize = 0;

	fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
		(i == 0)
			.then_some(*self)
			.ok_or(Error::IndexOutOfRange { index: i, max: 1 })
	}

	fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error> {
		(i == 0)
			.then(|| *self = scalar)
			.ok_or(Error::IndexOutOfRange { index: i, max: 1 })
	}

	fn iter(&self) -> impl Iterator<Item = Self::Scalar> {
		iter::once(*self)
	}

	fn random(rng: impl RngCore) -> Self {
		<Self as Field>::random(rng)
	}

	fn broadcast(scalar: Self::Scalar) -> Self {
		scalar
	}

	fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
		panic!("cannot interleave when WIDTH = 1");
	}
}
