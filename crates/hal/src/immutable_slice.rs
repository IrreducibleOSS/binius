// Copyright 2024 Ulvetanna Inc.

use binius_field::PackedField;
use linerate_binius_tensor_product::ImmutableSlice;
use std::ops::{Deref, DerefMut};

/// Wrapper for compatibility between Vec and ImmutableSlice.
#[derive(Debug)]
pub enum VecOrImmutableSlice<P> {
	V(Vec<P>),
	// Supports only read-only operations.
	IS(ImmutableSlice<P>),
}

impl<P> Deref for VecOrImmutableSlice<P> {
	type Target = [P];

	fn deref(&self) -> &Self::Target {
		match self {
			VecOrImmutableSlice::V(v) => v,
			VecOrImmutableSlice::IS(s) => s,
		}
	}
}

impl<P: PackedField> VecOrImmutableSlice<P> {
	pub fn into_expansion(self, len: usize) -> VecOrImmutableSlice<P> {
		match self {
			VecOrImmutableSlice::V(mut v) => {
				v.resize(len, P::zero());
				VecOrImmutableSlice::V(v)
			}
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
			VecOrImmutableSlice::IS(_) => panic!("Unsupported"),
		}
	}
}
