// Copyright 2024 Ulvetanna Inc.

#[cfg(not(feature = "linerate-backend"))]
use binius_field::PackedField;
#[cfg(not(feature = "linerate-backend"))]
use std::ops::{Deref, DerefMut};

/// Wrapper for compatibility between Vec and ImmutableSlice.
#[cfg(not(feature = "linerate-backend"))]
#[derive(Debug)]
pub enum VecOrImmutableSlice<P> {
	V(Vec<P>),
}

#[cfg(not(feature = "linerate-backend"))]
impl<P> Deref for VecOrImmutableSlice<P> {
	type Target = [P];

	fn deref(&self) -> &Self::Target {
		match self {
			VecOrImmutableSlice::V(v) => v,
		}
	}
}

#[cfg(not(feature = "linerate-backend"))]
impl<P: PackedField> VecOrImmutableSlice<P> {
	pub fn into_expansion(self, len: usize) -> VecOrImmutableSlice<P> {
		match self {
			VecOrImmutableSlice::V(mut v) => {
				v.resize(len, P::zero());
				VecOrImmutableSlice::V(v)
			}
		}
	}
}

#[cfg(not(feature = "linerate-backend"))]
impl<P> DerefMut for VecOrImmutableSlice<P> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		match self {
			VecOrImmutableSlice::V(v) => v,
		}
	}
}
