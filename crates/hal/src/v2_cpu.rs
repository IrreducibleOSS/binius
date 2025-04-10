// Copyright 2025 Irreducible Inc.

use std::ops::{Bound, RangeBounds};

use binius_field::{Field, PackedField};

/// Immutable slice of elements in a compute-abstracted device.
pub trait DevSlice<'a, T>: Copy {
	const MIN_LEN: usize;

	// This doesn't work for ranges too small or unaligned
	fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self>;
}

/// Mutable slice of elements in a compute-abstracted device.
pub trait DevSliceMut<'a, T>: 'a + Into<Self::ConstSlice<'a>> {
	const MIN_LEN: usize = Self::ConstSlice::MIN_LEN;
	type ConstSlice<'b>: DevSlice<'b, T>
	where
		Self: 'b;

	fn try_slice<'b>(&'b self, range: impl RangeBounds<usize>) -> Option<Self::ConstSlice<'b>>;
	fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<Self>;

	fn try_split_at_mut(&mut self, mid: usize) -> Option<(Self, Self)>;
}

/*
pub trait HAL<F: BinaryField> {
	type FSlice<'a>: DevSlice<'a, F>;
	type FSliceMut<'a>: DevSliceMut<'a, F, ConstSlice=Self::FSlice>; // Needs indexing
}

struct BasicCpuBackend;

impl<F: BinaryField> HAL<F> for BasicCpuBackend {
	//type FSlice = &'a [F];
	type FSliceMut<'a> = &'a mut [F];
}
 */

/*
/// General slice of data elements in host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemorySlice<'a, T>(&'a [T]);
 */

impl<'a, T> DevSlice<'a, T> for &'a [T] {
	const MIN_LEN: usize = 1;

	fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => self.len(),
		};
		Some(&self[start..end])
	}
}

/// Slice of SIMD-optimized packed field elements in host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackedFieldSlice<'a, P: PackedField>(pub &'a [P]);

impl<'a, F, P> DevSlice<'a, F> for PackedFieldSlice<'a, P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	const MIN_LEN: usize = P::WIDTH;

	fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + P::WIDTH,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + P::WIDTH,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => self.0.len() * P::WIDTH,
		};

		if start % P::WIDTH != 0 {
			return None;
		}
		if end % P::WIDTH != 0 {
			return None;
		}
		Some(PackedFieldSlice(&self.0[start / P::WIDTH..end / P::WIDTH]))
	}
}

/*
impl<'a, T> DevSliceMut<'a, T> for &'a mut [T] {
	const MIN_LEN: usize = 0;
	type ConstSlice<'b> = ();

	fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self::ConstSlice> {
		todo!()
	}

	fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<Self> {
		todo!()
	}

	fn try_split_at_mut(&mut self, mid: usize) -> Option<(Self, Self)> {
		todo!()
	}
}
 */

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::PackedBinaryField4x32b;
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;

	#[test]
	fn test_try_slice_on_mem_slice() {
		let data = &vec![4u32, 5, 6];
		assert_eq!(DevSlice::try_slice(&data.as_slice(), 0..2), Some(&data[0..2]));
		assert_eq!(DevSlice::try_slice(&data.as_slice(), ..2), Some(&data[..2]));
		assert_eq!(DevSlice::try_slice(&data.as_slice(), 1..), Some(&data[1..]));
		assert_eq!(DevSlice::try_slice(&data.as_slice(), ..), Some(&data[..]));
	}

	#[test]
	fn test_try_slice_on_packed_slice() {
		let mut rng = StdRng::seed_from_u64(0);
		let data = repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
			.take(5)
			.collect::<Vec<_>>();
		let fslice = PackedFieldSlice(&data);
		assert_eq!(fslice.try_slice(4..12), Some(PackedFieldSlice(&data[1..3])));
		assert_eq!(fslice.try_slice(..12), Some(PackedFieldSlice(&data[..3])));
		assert_eq!(fslice.try_slice(5..12), None);
		assert_eq!(fslice.try_slice(4..11), None);
		assert_eq!(fslice.try_slice(..11), None);
	}
}
