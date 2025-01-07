// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 authors

//! Traits used to sample random values in a public-coin interactive protocol.
//!
//! These interfaces are taken from [p3_challenger](https://github.com/Plonky3/Plonky3/blob/main/challenger/src/lib.rs) in [Plonky3].
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

use std::array;

#[auto_impl::auto_impl(&mut)]
pub trait CanSample<T> {
	fn sample(&mut self) -> T;

	fn sample_array<const N: usize>(&mut self) -> [T; N] {
		array::from_fn(|_| self.sample())
	}

	fn sample_vec(&mut self, n: usize) -> Vec<T> {
		(0..n).map(|_| self.sample()).collect()
	}
}

#[auto_impl::auto_impl(&mut)]
pub trait CanSampleBits<T> {
	fn sample_bits(&mut self, bits: usize) -> T;
}
