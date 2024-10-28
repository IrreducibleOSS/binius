// Copyright 2024 Irreducible Inc.

use crate::polynomial::MultivariatePoly;
use binius_field::{packed::set_packed_slice, BinaryField1b, Field, PackedField};

pub fn hypercube_evals_from_oracle<F: Field>(oracle: &dyn MultivariatePoly<F>) -> Vec<F> {
	(0..(1 << oracle.n_vars()))
		.map(|i| {
			oracle
				.evaluate(&decompose_index_to_hypercube_point(oracle.n_vars(), i))
				.unwrap()
		})
		.collect()
}

pub fn decompose_index_to_hypercube_point<F: Field>(n_vars: usize, index: usize) -> Vec<F> {
	(0..n_vars)
		.map(|k| {
			if (index >> k) % 2 == 1 {
				F::ONE
			} else {
				F::ZERO
			}
		})
		.collect::<Vec<_>>()
}

pub fn packed_slice<P>(assignments: &[(std::ops::Range<usize>, u8)]) -> Vec<P>
where
	P: PackedField<Scalar = BinaryField1b>,
{
	assert_eq!(assignments[0].0.start, 0, "First assignment must start at index 0");
	assert_eq!(
		assignments[assignments.len() - 1].0.end % P::WIDTH,
		0,
		"Last assignment must end at an index divisible by packing width"
	);
	for i in 1..assignments.len() {
		assert_eq!(
			assignments[i].0.start,
			assignments[i - 1].0.end,
			"2 assignments following each other can't be overlapping or have holes in between"
		);
	}
	assignments
		.iter()
		.for_each(|(r, _)| assert!(r.end > r.start, "Range must have positive size"));
	let packed_len = assignments
		.iter()
		.map(|(range, _)| range.end)
		.max()
		.unwrap_or(0)
		.div_ceil(P::WIDTH);
	let mut result: Vec<P> = vec![P::default(); packed_len];
	for (range, value) in assignments.iter() {
		for i in range.clone() {
			set_packed_slice(&mut result, i, P::Scalar::from(*value));
		}
	}
	result
}
