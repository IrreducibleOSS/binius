// Copyright 2024 Irreducible Inc.

use super::{Divisible, Random, UnderlierType};
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{must_cast_mut, must_cast_ref, NoUninit, Pod, Zeroable};
use rand::RngCore;
use std::array;
use subtle::{Choice, ConstantTimeEq};

/// A type that represents a pair of elements of the same underlier type.
/// We use it as an underlier for the `ScaledPAckedField` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ScaledUnderlier<U, const N: usize>(pub [U; N]);

impl<U: Default, const N: usize> Default for ScaledUnderlier<U, N> {
	fn default() -> Self {
		ScaledUnderlier(array::from_fn(|_| U::default()))
	}
}

impl<U: Random, const N: usize> Random for ScaledUnderlier<U, N> {
	fn random(mut rng: impl RngCore) -> Self {
		ScaledUnderlier(array::from_fn(|_| U::random(&mut rng)))
	}
}

impl<U, const N: usize> From<ScaledUnderlier<U, N>> for [U; N] {
	fn from(val: ScaledUnderlier<U, N>) -> Self {
		val.0
	}
}

impl<T, U: From<T>, const N: usize> From<[T; N]> for ScaledUnderlier<U, N> {
	fn from(value: [T; N]) -> Self {
		Self(value.map(U::from))
	}
}

impl<T: Copy, U: From<[T; 2]>> From<[T; 4]> for ScaledUnderlier<U, 2> {
	fn from(value: [T; 4]) -> Self {
		Self([[value[0], value[1]], [value[2], value[3]]].map(Into::into))
	}
}

impl<U: ConstantTimeEq, const N: usize> ConstantTimeEq for ScaledUnderlier<U, N> {
	fn ct_eq(&self, other: &Self) -> Choice {
		self.0.ct_eq(&other.0)
	}
}

unsafe impl<U: Zeroable, const N: usize> Zeroable for ScaledUnderlier<U, N> {}

unsafe impl<U: Pod, const N: usize> Pod for ScaledUnderlier<U, N> {}

impl<U: UnderlierType + Pod, const N: usize> UnderlierType for ScaledUnderlier<U, N> {
	const LOG_BITS: usize = U::LOG_BITS + checked_log_2(N);
}

unsafe impl<U, const N: usize> Divisible<U> for ScaledUnderlier<U, N>
where
	ScaledUnderlier<U, N>: UnderlierType,
	U: UnderlierType,
{
	fn split_ref(&self) -> &[U] {
		&self.0
	}

	fn split_mut(&mut self) -> &mut [U] {
		&mut self.0
	}
}

unsafe impl<U> Divisible<U> for ScaledUnderlier<ScaledUnderlier<U, 2>, 2>
where
	ScaledUnderlier<ScaledUnderlier<U, 2>, 2>: UnderlierType + NoUninit,
	U: UnderlierType + Pod,
{
	fn split_ref(&self) -> &[U] {
		must_cast_ref::<Self, [U; 4]>(self)
	}

	fn split_mut(&mut self) -> &mut [U] {
		must_cast_mut::<Self, [U; 4]>(self)
	}
}
