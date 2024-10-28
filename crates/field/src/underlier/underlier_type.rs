// Copyright 2024 Irreducible Inc.

use bytemuck::{NoUninit, Zeroable};
use rand::{
	distributions::{Distribution, Standard},
	Rng, RngCore,
};
use std::fmt::Debug;
use subtle::ConstantTimeEq;

/// Primitive integer underlying a binary field or packed binary field implementation.
/// Note that this type is not guaranteed to be POD, U1, U2 and U4 have some unused bits.
pub trait UnderlierType:
	Debug
	+ Default
	+ PartialEq
	+ Eq
	+ PartialOrd
	+ Ord
	+ ConstantTimeEq
	+ Copy
	+ Random
	+ NoUninit
	+ Zeroable
	+ Sized
	+ Send
	+ Sync
	+ 'static
{
	/// Number of bits in value
	const LOG_BITS: usize;
	/// Number of bits used to represent a value.
	/// This may not be equal to the number of bits in a type instance.
	const BITS: usize = 1 << Self::LOG_BITS;
}

/// A type that is transparently backed by an underlier.
///
/// This trait is needed to make it possible getting the underlier type from already defined type.
/// Bidirectional `From` trait implementations are not enough, because they do not allow getting underlier type
/// in a generic code.
///
/// # Safety
/// `WithUnderlier` can be implemented for a type only if it's representation is a transparent `Underlier`'s representation.
/// That's allows us casting references of type and it's underlier in both directions.
pub unsafe trait WithUnderlier: Sized + Zeroable + Copy + Send + Sync + 'static {
	/// Underlier primitive type
	type Underlier: UnderlierType;

	/// Convert value to underlier.
	fn to_underlier(self) -> Self::Underlier;

	fn to_underlier_ref(&self) -> &Self::Underlier;

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier;

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier];

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier];

	fn from_underlier(val: Self::Underlier) -> Self;

	fn from_underlier_ref(val: &Self::Underlier) -> &Self;

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self;

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self];

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self];

	#[inline]
	fn mutate_underlier(self, f: impl FnOnce(Self::Underlier) -> Self::Underlier) -> Self {
		Self::from_underlier(f(self.to_underlier()))
	}
}

unsafe impl<U: UnderlierType> WithUnderlier for U {
	type Underlier = U;

	#[inline]
	fn to_underlier(self) -> Self::Underlier {
		self
	}

	#[inline]
	fn to_underlier_ref(&self) -> &Self::Underlier {
		self
	}

	#[inline]
	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		self
	}

	#[inline]
	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		val
	}

	#[inline]
	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		val
	}

	#[inline]
	fn from_underlier(val: Self::Underlier) -> Self {
		val
	}

	#[inline]
	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		val
	}

	#[inline]
	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		val
	}

	#[inline]
	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		val
	}

	#[inline]
	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		val
	}
}

/// A value that can be randomly generated
pub trait Random {
	/// Generate random value
	fn random(rng: impl RngCore) -> Self;
}

impl<T> Random for T
where
	Standard: Distribution<T>,
{
	fn random(mut rng: impl RngCore) -> Self {
		rng.gen()
	}
}

/// A trait that represents potentially lossy numeric cast.
/// Is a drop-in replacement of `as _` in a generic code.
pub trait NumCast<From> {
	fn num_cast_from(val: From) -> Self;
}

impl<U: UnderlierType> NumCast<U> for U {
	#[inline(always)]
	fn num_cast_from(val: U) -> Self {
		val
	}
}
