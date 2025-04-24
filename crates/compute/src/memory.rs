// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

pub trait OpaqueSlice<T> {
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	fn len(&self) -> usize;
}

impl<T> OpaqueSlice<T> for &[T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

impl<T> OpaqueSlice<T> for &mut [T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

/// Interface for manipulating handles to memory in a compute device.
pub trait ComputeMemory<F> {
	// The minimum length of a slice that can be allocated and used in a compute kernel.
	const MIN_SLICE_LEN: usize;
	// The required alignment of a slice that can be allocated and used in a compute kernel.
	const REQUIRED_SLICE_ALIGNMENT: usize;

	/// An opaque handle to an immutable slice of elements stored in a compute memory.
	type FSlice<'a>: Copy + OpaqueSlice<F>;

	/// An opaque handle to a mutable slice of elements stored in a compute memory.
	type FSliceMut<'a>: OpaqueSlice<F>;

	/// Borrows a mutable memory slice as immutable.
	///
	/// This allows the immutable reference to be copied.
	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a>;

	/// Borrows a subslice of an immutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_SLICE_LEN`]
	/// - the range bounds must be aligned to [`Self::REQUIRED_SLICE_ALIGNMENT`]
	fn slice(data: Self::FSlice<'_>, range: impl RangeBounds<usize>) -> Self::FSlice<'_>;

	/// Borrows a subslice of a mutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_SLICE_LEN`]
	/// - the range bounds must be aligned to [`Self::REQUIRED_SLICE_ALIGNMENT`]
	fn slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl RangeBounds<usize>,
	) -> Self::FSliceMut<'a>;

	/// Splits a mutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of [`Self::MIN_SLICE_LEN`]
	/// - `mid` must be aligned to [`Self::REQUIRED_SLICE_ALIGNMENT`]
	fn split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>);

	fn split_at_mut_borrowed<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'a>, Self::FSliceMut<'a>) {
		let borrowed = Self::slice_mut(data, ..);
		Self::split_at_mut(borrowed, mid)
	}
}
