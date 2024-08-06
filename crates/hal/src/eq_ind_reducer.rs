use binius_field::Field;
use linerate_binius_sumcheck::reducer::Reducer;
use rayon::prelude::*;
use std::{mem::size_of, slice::from_raw_parts};
use tracing::instrument;

/// Aggregates the response by multiplying it with the corresponding value of eq_ind, which is the sum of a chunk of INTERLEAVE values.
pub(crate) struct EqIndReducer<'a, F> {
	eq_ind: &'a [F],
}

impl<'a, F: Field> EqIndReducer<'a, F> {
	#[inline]
	pub(crate) fn new(eq_ind: &'a [F]) -> Self {
		assert_eq!(size_of::<F>(), size_of::<u128>());
		Self { eq_ind }
	}

	#[inline]
	fn eq_ind<const INTERLEAVE: usize>(&self, index: usize) -> F {
		self.eq_ind[index * INTERLEAVE..(index + 1) * INTERLEAVE]
			.iter()
			.sum()
	}
}

impl<'a, F: Field> Reducer<F> for EqIndReducer<'a, F> {
	#[instrument(skip_all)]
	fn reduce<const BLOW_UP: usize, const INTERLEAVE: usize>(
		&self,
		data: impl IndexedParallelIterator<Item = Vec<u128>>,
		degree: usize,
	) -> Vec<F> {
		data.enumerate()
			.fold(
				|| vec![F::ZERO; degree],
				|mut agg, (row, v)| {
					let v = &v;
					let v = unsafe { from_raw_parts(v.as_ptr() as *const F, v.len()) };
					let eq_ind_factor = self.eq_ind::<INTERLEAVE>(row);
					for j in 0..degree {
						agg[j] += eq_ind_factor * v[j];
					}
					agg
				},
			)
			.reduce(
				|| vec![F::ZERO; degree],
				|mut a, b| {
					for j in 0..degree {
						a[j] += b[j];
					}
					a
				},
			)
	}
}
