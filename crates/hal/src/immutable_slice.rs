// Copyright 2024 Ulvetanna Inc.

use binius_field::PackedField;
#[cfg(feature = "linerate-backend")]
use binius_linerate_slice::ImmutableSlice;
use std::ops::{Deref, DerefMut};

/// Wrapper for compatibility between Vec and ImmutableSlice.
#[allow(dead_code)]
#[derive(Debug)]
pub enum VecOrImmutableSlice<P> {
	V(Vec<P>),
	#[cfg(feature = "linerate-backend")]
	// Supports only read-only operations.
	IS(ImmutableSlice<P>),
}

impl<P> Deref for VecOrImmutableSlice<P> {
	type Target = [P];

	fn deref(&self) -> &Self::Target {
		match self {
			VecOrImmutableSlice::V(v) => v,
			#[cfg(feature = "linerate-backend")]
			VecOrImmutableSlice::IS(s) => s,
		}
	}
}

impl<P: PackedField> VecOrImmutableSlice<P> {
	#[allow(dead_code)]
	pub fn into_expansion(self, len: usize) -> VecOrImmutableSlice<P> {
		match self {
			VecOrImmutableSlice::V(mut v) => {
				v.resize(len, P::zero());
				VecOrImmutableSlice::V(v)
			}
			#[cfg(feature = "linerate-backend")]
			VecOrImmutableSlice::IS(ref is) => {
				assert_eq!(is.len(), len);
				self
			}
		}
	}
}

impl<P> DerefMut for VecOrImmutableSlice<P> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		match self {
			VecOrImmutableSlice::V(v) => v,
			#[cfg(feature = "linerate-backend")]
			VecOrImmutableSlice::IS(_) => panic!("Unsupported"),
		}
	}
}
