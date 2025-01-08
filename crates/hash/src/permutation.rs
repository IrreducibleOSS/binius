// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 Authors

//! These interfaces are taken from [p3_symmetric](https://github.com/Plonky3/Plonky3/blob/main/symmetric/src/permutation.rs) in [Plonky3].
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

/// A permutation in the mathematical sense.
pub trait Permutation<T: Clone>: Clone + Sync {
	fn permute(&self, mut input: T) -> T {
		self.permute_mut(&mut input);
		input
	}

	fn permute_mut(&self, input: &mut T);
}

/// A permutation thought to be cryptographically secure, in the sense that it is thought to be
/// difficult to distinguish (in a nontrivial way) from a random permutation.
pub trait CryptographicPermutation<T: Clone>: Permutation<T> {}
