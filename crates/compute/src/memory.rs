// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

use binius_field::TowerField;
use binius_utils::checked_arithmetics::checked_int_div;

pub trait SizedSlice {
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	fn len(&self) -> usize;
}

impl<T> SizedSlice for &[T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

impl<T> SizedSlice for &mut [T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

/// A batch of slices of the same length.
pub struct SlicesBatch<Slice: SizedSlice> {
	rows: Vec<Slice>,
	row_len: usize,
}

impl<Slice: SizedSlice> SlicesBatch<Slice> {
	/// Creates a new batch of slices with the given length.
	///
	/// # Panics
	/// If any of the slices in `rows` does not have the specified `row_len`.
	pub fn new(rows: Vec<Slice>, row_len: usize) -> Self {
		for row in &rows {
			assert_eq!(row.len(), row_len);
		}

		Self { rows, row_len }
	}

	/// Number of memory slices
	pub fn n_rows(&self) -> usize {
		self.rows.len()
	}

	/// Length of each memory slice
	pub fn row_len(&self) -> usize {
		self.row_len
	}

	/// Returns a slice of the batch at the given index.
	pub fn row(&self, index: usize) -> &Slice {
		&self.rows[index]
	}

	/// Returns iterator over the slices in the batch.
	pub fn iter(&self) -> impl Iterator<Item = &Slice> {
		self.rows.iter()
	}
}

/// Interface for manipulating handles to memory in a compute device.
pub trait ComputeMemory<F> {
	const MIN_SLICE_LEN: usize;

	/// An opaque handle to an immutable slice of elements stored in a compute memory.
	type FSlice<'a>: Copy + SizedSlice + Sync;

	/// An opaque handle to a mutable slice of elements stored in a compute memory.
	type FSliceMut<'a>: SizedSlice;

	/// Borrows an immutable memory slice, narrowing the lifetime.
	fn narrow<'a>(data: &'a Self::FSlice<'_>) -> Self::FSlice<'a>;

	/// Borrows a mutable memory slice, narrowing the lifetime.
	fn narrow_mut<'a, 'b: 'a>(data: Self::FSliceMut<'b>) -> Self::FSliceMut<'a>;

	// Converts a reference to an FSliceMut to an FSliceMut
	fn to_owned_mut<'a>(data: &'a mut Self::FSliceMut<'_>) -> Self::FSliceMut<'a>;

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

	/// Splits an immutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of [`Self::MIN_SLICE_LEN`]
	fn split_at(data: Self::FSlice<'_>, mid: usize) -> (Self::FSlice<'_>, Self::FSlice<'_>) {
		let head = Self::slice(data, ..mid);
		let tail = Self::slice(data, mid..);
		(head, tail)
	}

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

	/// Splits slice into equal chunks.
	///
	/// ## Preconditions
	///
	/// - the length of the slice must be a multiple of `chunk_len`
	/// - `chunk_len` must be a multiple of [`Self::MIN_SLICE_LEN`]
	fn slice_chunks<'a>(
		data: Self::FSlice<'a>,
		chunk_len: usize,
	) -> impl Iterator<Item = Self::FSlice<'a>> {
		let n_chunks = checked_int_div(data.len(), chunk_len);
		(0..n_chunks).map(move |i| Self::slice(data, i * chunk_len..(i + 1) * chunk_len))
	}

	/// Splits a mutable slice into equal chunks.
	///
	/// ## Preconditions
	///
	/// - the length of the slice must be a multiple of `chunk_len`
	/// - `chunk_len` must be a multiple of [`Self::MIN_SLICE_LEN`]
	fn slice_chunks_mut<'a>(
		data: Self::FSliceMut<'a>,
		chunk_len: usize,
	) -> impl Iterator<Item = Self::FSliceMut<'a>>;
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
/// - `tower_level`: A `usize` value indicating the level of the field tower for the elements in the
///   slice.
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

	/// Returns the length of the slice in terms of the number of subfield elements it contains.
	pub fn len(&self) -> usize
	where
		F: TowerField,
	{
		self.slice.len() << (F::TOWER_LEVEL - self.tower_level)
	}

	pub fn is_empty(&self) -> bool
	where
		F: TowerField,
	{
		self.slice.is_empty()
	}
}

/// `SubfieldSliceMut` represents a mutable slice of field elements with identical semantics to
/// `SubfieldSlice`.
pub struct SubfieldSliceMut<'a, F, Mem: ComputeMemory<F>> {
	pub slice: Mem::FSliceMut<'a>,
	pub tower_level: usize,
}

impl<'a, F, Mem: ComputeMemory<F>> SubfieldSliceMut<'a, F, Mem> {
	pub fn new(slice: Mem::FSliceMut<'a>, tower_level: usize) -> Self {
		Self { slice, tower_level }
	}
}
