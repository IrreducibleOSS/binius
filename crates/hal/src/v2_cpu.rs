use std::{
	ops::{Range, RangeBounds},
	slice::SliceIndex,
};

use binius_field::{BinaryField, Field};

pub trait DevSlice<'a, T>: Copy {
	const MIN_LEN: usize;

	// This doesn't work for ranges too small or unaligned
	fn try_slice(&self, range: Range<usize>) -> Option<Self>;
}

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

impl<'a, T> DevSlice<'a, T> for &'a [T] {
	const MIN_LEN: usize = 1;

	fn try_slice(&self, range: Range<usize>) -> Option<Self> {
		Some(&self[range])
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
	use super::*;

	#[test]
	fn test_try_slice_on_mem_slice() {
		let data = vec![4u32, 5, 6];
		assert_eq!(DevSlice::try_slice(&data.as_slice(), 0..2), Some(&data[0..2]));
	}
}
