// Copyright 2024 Ulvetanna Inc.

use bytemuck::NoUninit;
use rand::{
	distributions::{Distribution, Standard},
	Rng, RngCore,
};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use subtle::ConstantTimeEq;

/// Primitive integer underlying a binary field or packed binary field implementation.
/// Note that this type is not guaranteed to be POD, U1, U2 and U4 have some unused bits.
pub trait UnderlierType:
	Default + PartialEq + Eq + ConstantTimeEq + Copy + Random + NoUninit
{
	/// Number of bits in value
	const LOG_BITS: usize;
	/// Number of bits used to represent a value.
	/// This may not be equal to the number of bits in a type instance.
	const BITS: usize = 1 << Self::LOG_BITS;
}

/// This trait is needed to make it possible getting the underlier type from already defined type.
/// Bidirectional `From` trait implementations are not enough, because they do not allow getting underlier type
/// in a generic code.
///
/// # Safety
/// `WithUnderlier` can be implemented for a type only if it's representation is a transparent `Underlier`'s representation.
/// That's allows us casting references of type and it's underlier in both directions.
pub unsafe trait WithUnderlier: From<Self::Underlier>
where
	Self::Underlier: From<Self>,
{
	/// Underlier primitive type
	type Underlier: UnderlierType;

	/// Cast value to underlier.
	fn to_underlier(self) -> Self::Underlier {
		self.into()
	}

	fn to_underlier_ref(&self) -> &Self::Underlier {
		unsafe { &*(self as *const Self as *const Self::Underlier) }
	}

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		unsafe { &mut *(self as *mut Self as *mut Self::Underlier) }
	}

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		unsafe { from_raw_parts(val.as_ptr() as *const Self::Underlier, val.len()) }
	}

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		unsafe { from_raw_parts_mut(val.as_mut_ptr() as *mut Self::Underlier, val.len()) }
	}

	fn from_underlier(value: Self::Underlier) -> Self {
		Self::from(value)
	}

	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		unsafe { &*(val as *const Self::Underlier as *const Self) }
	}

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		unsafe { &mut *(val as *mut Self::Underlier as *mut Self) }
	}

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		unsafe { from_raw_parts(val.as_ptr() as *const Self, val.len()) }
	}

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		unsafe { from_raw_parts_mut(val.as_mut_ptr() as *mut Self, val.len()) }
	}
}

unsafe impl<U: UnderlierType> WithUnderlier for U {
	type Underlier = U;
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
	fn num_cast_from(val: U) -> Self {
		val
	}
}
