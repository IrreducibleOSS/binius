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
	const MIN_SLICE_LEN: usize;

	/// An opaque handle to an immutable slice of elements stored in a compute memory.
	type FSlice<'a>: Copy + DevSlice<F>;

	/// An opaque handle to a mutable slice of elements stored in a compute memory.
	type FSliceMut<'a>: DevSlice<F>;

	/// Borrows a mutable memory slice as immutable.
	///
	/// This allows the immutable reference to be copied.
	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a>;

	/// Borrows a subslice of an immutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_SLICE_LEN`]
	fn slice(data: Self::FSlice<'_>, range: impl RangeBounds<usize>) -> Self::FSlice<'_>;

	/// Borrows a subslice of a mutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of [`Self::MIN_SLICE_LEN`]
	fn slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl RangeBounds<usize>,
	) -> Self::FSliceMut<'a>;

	/// Splits a mutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of [`Self::MIN_SLICE_LEN`]
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

/// `SubfieldSlice` is a structure that represents a slice of elements stored in a compute memory,
/// along with an associated tower level. This structure is used to handle subfield operations
/// within a computational context, where the `slice` is an immutable reference to the data
/// and `tower_level` indicates the level of the field tower to which the elements belong.
///
/// # Type Parameters
/// - `'a`: The lifetime of the slice reference.
/// - `F`: The type of the field elements stored in the slice.
/// - `Mem`: A type that implements the `ComputeMemory` trait, which provides the necessary
///   operations for handling memory slices.
///
/// # Fields
/// - `slice`: An immutable slice of elements stored in compute memory, represented by
///   `Mem::FSlice<'a>`.
/// - `tower_level`: A `usize` value indicating the level of the field tower for the elements
///   in the slice.
///
/// # Usage
/// `SubfieldSlice` is typically used in scenarios where operations need to be performed on
/// specific subfields of a larger field structure, allowing for efficient computation and
/// manipulation of data within a hierarchical field system.
pub struct SubfieldSlice<'a, F, Mem: ComputeMemory<F>> {
	pub slice: Mem::FSlice<'a>,
	pub tower_level: usize,
}

impl<'a, F, Mem: ComputeMemory<F>> SubfieldSlice<'a, F, Mem> {
	pub fn new(slice: Mem::FSlice<'a>, tower_level: usize) -> Self {
		Self { slice, tower_level }
	}
}

/// `SubfieldSliceMut` represents a mutable slice of field elements with identical semantics to `SubfieldSlice`.
pub struct SubfieldSliceMut<'a, F, Mem: ComputeMemory<F>> {
	pub slice: Mem::FSliceMut<'a>,
	pub tower_level: usize,
}

impl<'a, F, Mem: ComputeMemory<F>> SubfieldSliceMut<'a, F, Mem> {
	pub fn new(slice: Mem::FSliceMut<'a>, tower_level: usize) -> Self {
		Self { slice, tower_level }
	}
}
