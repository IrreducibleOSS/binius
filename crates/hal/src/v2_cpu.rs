// Copyright 2025 Irreducible Inc.

use std::ops::{Bound, RangeBounds};

use binius_field::BinaryField;

pub trait ComputeLayer<F: BinaryField> {
	const MIN_SLICE_LEN: usize;

	type FSlice<'a>: Copy;
	type FSliceMut<'a>;

	fn as_const<'a, 'b>(data: &'a Self::FSliceMut<'b>) -> Self::FSlice<'a>;

	fn try_slice<'a>(
		data: Self::FSlice<'a>,
		range: impl RangeBounds<usize>,
	) -> Option<Self::FSlice<'a>>;

	fn try_slice_mut<'a, 'b>(
		data: &'a mut Self::FSliceMut<'b>,
		range: impl RangeBounds<usize>,
	) -> Option<Self::FSliceMut<'a>>;

	fn try_split_at_mut<'a, 'b>(
		data: &'a mut Self::FSliceMut<'b>,
		mid: usize,
	) -> Option<(Self::FSliceMut<'a>, Self::FSliceMut<'a>)>;
}

#[derive(Debug)]
pub struct BasicCpuBackend;

impl<F: BinaryField> ComputeLayer<F> for BasicCpuBackend {
	const MIN_SLICE_LEN: usize = 1;

	type FSlice<'a> = &'a [F];
	type FSliceMut<'a> = &'a mut [F];

	fn as_const<'a, 'b>(data: &'a &'b mut [F]) -> &'a [F] {
		data
	}

	fn try_slice<'a>(
		data: Self::FSlice<'a>,
		range: impl RangeBounds<usize>,
	) -> Option<Self::FSlice<'a>> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => data.len(),
		};
		Some(&data[start..end])
	}

	fn try_slice_mut<'a, 'b>(
		data: &'a mut &'b mut [F],
		range: impl RangeBounds<usize>,
	) -> Option<&'a mut [F]> {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => data.len(),
		};
		Some(&mut data[start..end])
	}

	fn try_split_at_mut<'a, 'b>(
		fslice: &'a mut Self::FSliceMut<'b>,
		mid: usize,
	) -> Option<(Self::FSliceMut<'a>, Self::FSliceMut<'a>)> {
		Some(fslice.split_at_mut(mid))
	}
}

/*
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
 */

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
	use binius_field::BinaryField32b;

	use super::*;

	#[test]
	fn test_try_slice_on_mem_slice() {
		let data = [4, 5, 6].map(BinaryField32b::new);
		assert_eq!(BasicCpuBackend::try_slice(&data, 0..2), Some(&data[0..2]));
		assert_eq!(BasicCpuBackend::try_slice(&data, ..2), Some(&data[..2]));
		assert_eq!(BasicCpuBackend::try_slice(&data, 1..), Some(&data[1..]));
		assert_eq!(BasicCpuBackend::try_slice(&data, ..), Some(&data[..]));
	}

	#[test]
	fn test_convert_mut_mem_slice_to_const() {
		let mut data = [4, 5, 6].map(BinaryField32b::new);
		let data_clone = data.clone();
		let data = &mut data[..];
		let data = BasicCpuBackend::as_const(&data);
		assert_eq!(data, &data_clone);
	}

	#[test]
	fn test_try_slice_on_mut_mem_slice() {
		let mut data = [4, 5, 6].map(BinaryField32b::new);
		let mut data_clone = data.clone();
		let mut data = &mut data[..];
		assert_eq!(BasicCpuBackend::try_slice_mut(&mut data, 0..2), Some(&mut data_clone[0..2]));
		assert_eq!(BasicCpuBackend::try_slice_mut(&mut data, ..2), Some(&mut data_clone[..2]));
		assert_eq!(BasicCpuBackend::try_slice_mut(&mut data, 1..), Some(&mut data_clone[1..]));
		assert_eq!(BasicCpuBackend::try_slice_mut(&mut data, ..), Some(&mut data_clone[..]));
	}

	/*
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

	fn extrapolate<'a, F: BinaryField, H: ComputeLayer<F>>(hal: &'a H, mut data: H::FSliceMut<'a>) {
		let (mut lhs, rhs) = H::try_split_at_mut(&mut data, 16).unwrap();
		hal.extrapolate_line(&mut lhs, H::as_const(&rhs), F::ONE);
	}
	 */
}
