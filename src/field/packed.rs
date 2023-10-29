// Copyright 2023 Ulvetanna Inc.

//! Traits for packed field elements which support SIMD implementations.
//!
//! Interfaces are derived from [`plonky2`](https://github.com/mir-protocol/plonky2).

use ff::Field;
use p3_util::log2_strict_usize;
use rand::RngCore;
use std::{
	iter::{self, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use subtle::ConstantTimeEq;

use super::Error;

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
	+ 'static
{
	type Scalar: Field;
	type Iterator: Iterator<Item=Self::Scalar>;

	// TODO: WIDTH should probably be required to be power of two
	const WIDTH: usize;

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

	fn iter(&self) -> Self::Iterator;

	fn random(rng: impl RngCore) -> Self;
	fn broadcast(scalar: Self::Scalar) -> Self;

	// TODO: This should return a Result
	// TODO: Change block_len to log_block_len
	fn interleave(self, other: Self, block_len: usize) -> (Self, Self);

	fn square_transpose(elems: &mut [Self]) -> Result<(), Error> {
		let size = elems.len();
		if !size.is_power_of_two() {
			return Err(Error::SquareTransposePowerOfTwoSizeRequired);
		}
		if Self::WIDTH % size != 0 {
			return Err(Error::SquareTransposeSizeMustDivideWidth);
		}

		// See Hacker's Delight, Section 7-3.
		// https://dl.acm.org/doi/10.5555/2462741
		let log_size = log2_strict_usize(size);
		for i in 0..log_size {
			for j in 0..1 << (log_size - i - 1) {
				for k in 0..1 << i {
					let idx0 = j << (i + 1) | k;
					let idx1 = idx0 | (1 << i);

					let v0 = elems[idx0];
					let v1 = elems[idx1];
					let (v0, v1) = v0.interleave(v1, 1 << i);
					elems[idx0] = v0;
					elems[idx1] = v1;
				}
			}
		}

		Ok(())
	}
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
	type Iterator = iter::Once<F>;

	const WIDTH: usize = 1;

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

	fn iter(&self) -> Self::Iterator {
		iter::once(*self)
	}

	fn random(rng: impl RngCore) -> Self {
		<Self as Field>::random(rng)
	}

	fn broadcast(scalar: Self::Scalar) -> Self {
		scalar
	}

	fn interleave(self, _other: Self, _block_len: usize) -> (Self, Self) {
		panic!("cannot interleave when WIDTH = 1");
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::field::{PackedBinaryField128x1b, PackedBinaryField64x2b};

	#[test]
	fn test_square_transpose_128x1b() {
		let mut elems = [
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		PackedField::square_transpose(&mut elems).unwrap();

		let expected = [
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
		];
		assert_eq!(elems, expected);
	}

	#[test]
	fn test_square_transpose_64x2b() {
		let mut elems = [
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		PackedField::square_transpose(&mut elems).unwrap();

		let expected = [
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
			PackedBinaryField64x2b::from(0xff00ff00ff00ff00ff00ff00ff00ff00u128),
		];
		assert_eq!(elems, expected);
	}
}
