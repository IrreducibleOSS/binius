// Copyright 2025 Irreducible Inc.

use std::{
	marker::PhantomData,
	ops::{Bound, RangeBounds},
};

use binius_field::{BinaryField, PackedField, TowerField};

use crate::v2_cpu::ComputeLayer;

/// Slice of SIMD-optimized packed field elements in host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackedFieldSlice<'a, P: PackedField>(pub &'a [P]);

#[derive(Debug)]
pub struct FastCpuBackend<P>(PhantomData<P>);

impl<F, P> ComputeLayer<F> for FastCpuBackend<P>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
{
	const MIN_SLICE_LEN: usize = P::WIDTH;

	type FSlice<'a> = &'a [P];
	type FSliceMut<'a> = &'a mut [P];

	fn as_const<'a, 'b>(data: &'a &'b mut [P]) -> &'a [P] {
		data
	}

	fn try_slice<'a>(
		data: Self::FSlice<'a>,
		range: impl RangeBounds<usize>,
	) -> Option<Self::FSlice<'a>> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + P::WIDTH,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + P::WIDTH,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => data.len() * P::WIDTH,
		};

		if start % P::WIDTH != 0 {
			return None;
		}
		if end % P::WIDTH != 0 {
			return None;
		}
		Some(&data[start / P::WIDTH..end / P::WIDTH])
	}

	fn try_slice_mut<'a, 'b>(
		data: &'a mut &'b mut [P],
		range: impl RangeBounds<usize>,
	) -> Option<&'a mut [P]> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + P::WIDTH,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + P::WIDTH,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => data.len() * P::WIDTH,
		};

		if start % P::WIDTH != 0 {
			return None;
		}
		if end % P::WIDTH != 0 {
			return None;
		}
		Some(&mut data[start / P::WIDTH..end / P::WIDTH])
	}

	fn try_split_at_mut<'a, 'b>(
		data: &'a mut &'b mut [P],
		mid: usize,
	) -> Option<(&'a mut [P], &'a mut [P])> {
		if mid % P::WIDTH != 0 {
			return None;
		}
		Some(data.split_at_mut(mid / P::WIDTH))
	}
}
