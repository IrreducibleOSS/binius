// Copyright 2024 Irreducible Inc.

use binius_field::PackedField;
use itertools::Either;
use std::iter;

/// Given a slice of packed fields representing `2^log_scalar_count` scalars, returns an iterator
/// that yields pairs of packed fields that can be unzipped into a deinterleaved representation
/// of the slice.
///
/// For example, for a  slice of two packed fields of four scalars:
/// [[a0, b0, a1, b1] [a2, b2, a3, b3]] -> ([a0, a1, a2, a3], [b0, b1, b2, b3])
pub fn deinterleave<P: PackedField>(
	log_scalar_count: usize,
	interleaved: &[P],
) -> impl Iterator<Item = (usize, P, P)> + '_ {
	assert_eq!(interleaved.len(), 1 << (log_scalar_count + 1).saturating_sub(P::LOG_WIDTH));

	if log_scalar_count < P::LOG_WIDTH {
		let mut even = P::zero();
		let mut odd = P::zero();

		for i in 0..1 << log_scalar_count {
			even.set(i, interleaved[0].get(2 * i));
			odd.set(i, interleaved[0].get(2 * i + 1));
		}

		return Either::Left(iter::once((0, even, odd)));
	}

	let deinterleaved = (0..1 << (log_scalar_count - P::LOG_WIDTH)).map(|i| {
		let mut even = interleaved[2 * i];
		let mut odd = interleaved[2 * i + 1];

		for log_block_len in (0..P::LOG_WIDTH).rev() {
			let (even_interleaved, odd_interleaved) = even.interleave(odd, log_block_len);
			even = even_interleaved;
			odd = odd_interleaved;
		}

		(i, even, odd)
	});

	Either::Right(deinterleaved)
}
