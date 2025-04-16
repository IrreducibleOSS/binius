// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

pub trait DevSlice<T> {
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	fn len(&self) -> usize;
}

impl<T> DevSlice<T> for &[T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

impl<T> DevSlice<T> for &mut [T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

/// Interface for manipulating handles to memory in a compute device.
pub trait ComputeMemory<F> {
	const MIN_DEVICE_SLICE_LEN: usize;
	const MIN_HOST_SLICE_LEN: usize;

	/// An opaque handle to an immutable slice of elements stored in a compute memory.
	type FSlice<'a>: Copy + DevSlice<F>;

	/// An opaque handle to a mutable slice of elements stored in a compute memory.
	type FSliceMut<'a>: DevSlice<F>;

	/// A transparent handle to immutable host memory
	type HostSlice<'a>: AsRef<[F]>;

	/// A transparent handle to mutable host memory
	type HostSliceMut<'a>: AsMut<[F]>;

	/// Borrows a mutable memory slice as immutable.
	///
	/// This allows the immutable reference to be copied.
	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a>;

	/// Borrows a subslice of an immutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_DEVICE_SLICE_LEN`]
	fn device_slice(data: Self::FSlice<'_>, range: impl RangeBounds<usize>) -> Self::FSlice<'_>;

	/// Borrows a subslice of an immutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_HOST_SLICE_LEN`]
	fn host_slice(data: Self::HostSlice<'_>, range: impl RangeBounds<usize>)
		-> Self::HostSlice<'_>;

	/// Borrows a subslice of a mutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_DEVICE_SLICE_LEN`]
	fn device_slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl RangeBounds<usize>,
	) -> Self::FSliceMut<'a>;

	/// Borrows a subslice of a mutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_HOST_SLICE_LEN`]
	fn host_slice_mut<'a>(
		data: &'a mut Self::HostSliceMut<'_>,
		range: impl RangeBounds<usize>,
	) -> Self::HostSliceMut<'a>;

	/// Splits a mutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of [`Self::MIN_DEVICE_SLICE_LEN`]
	fn device_split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>);

	fn device_split_at_mut_borrowed<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'a>, Self::FSliceMut<'a>) {
		let borrowed = Self::device_slice_mut(data, ..);
		Self::device_split_at_mut(borrowed, mid)
	}

	/// Splits a mutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of [`Self::MIN_HOST_SLICE_LEN`]
	fn host_split_at_mut(
		data: Self::HostSliceMut<'_>,
		mid: usize,
	) -> (Self::HostSliceMut<'_>, Self::HostSliceMut<'_>);

	fn host_split_at_mut_borrowed<'a>(
		data: &'a mut Self::HostSliceMut<'_>,
		mid: usize,
	) -> (Self::HostSliceMut<'a>, Self::HostSliceMut<'a>) {
		let borrowed = Self::host_slice_mut(data, ..);
		Self::host_split_at_mut(borrowed, mid)
	}
}
