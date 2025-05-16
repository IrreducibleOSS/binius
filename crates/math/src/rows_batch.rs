// Copyright 2025 Irreducible Inc.

use std::ops::{Bound, RangeBounds};

/// This struct represents a batch of rows, each row having the same length equal to `row_len`.
pub struct RowsBatch<'a, T> {
	rows: Vec<&'a [T]>,
	row_len: usize,
}

impl<'a, T> RowsBatch<'a, T> {
	/// Create a new `RowsBatch` from a vector of rows and the given row length
	///
	/// # Panics
	/// In case if any of the rows has a length different from `row_len`.
	#[inline]
	pub fn new(rows: Vec<&'a [T]>, row_len: usize) -> Self {
		for row in &rows {
			assert_eq!(row.len(), row_len);
		}

		Self { rows, row_len }
	}

	/// Create a new `RowsBatch` from an iterator of rows and the given row length.
	///
	/// # Panics
	/// In case if any of the rows has a length less than `row_len`.
	#[inline]
	pub fn new_from_iter(rows: impl IntoIterator<Item = &'a [T]>, row_len: usize) -> Self {
		let rows = rows.into_iter().map(|x| &x[..row_len]).collect();
		Self { rows, row_len }
	}

	#[inline(always)]
	pub fn get_ref(&self) -> RowsBatchRef<'_, T> {
		RowsBatchRef {
			rows: self.rows.as_slice(),
			row_len: self.row_len,
			offset: 0,
		}
	}
}

/// This struct is similar to `RowsBatch`, but it holds a reference to the slice of rows
/// and an offset.
///
/// It is guaranteed that all rows have the length at least `row_len + offset`
///
/// Unfortunately due to liftime issues we can't have a single generic struct which is
/// parameterized by a container type.
pub struct RowsBatchRef<'a, T> {
	rows: &'a [&'a [T]],
	row_len: usize,
	offset: usize,
}

impl<'a, T> RowsBatchRef<'a, T> {
	/// Create a new `RowsBatchRef` from a slice of rows and the given row length.
	///
	/// # Panics
	/// In case if any of the rows has a length smaller than `row_len`.
	#[inline]
	pub fn new(rows: &'a [&'a [T]], row_len: usize) -> Self {
		Self::new_with_offset(rows, 0, row_len)
	}

	/// Create a new `RowsBatchRef` from a slice of rows and the given row length and offset.
	///
	/// # Panics
	/// In case if any of the rows has a length smaller than from `offset + row_len`.
	#[inline]
	pub fn new_with_offset(rows: &'a [&'a [T]], offset: usize, row_len: usize) -> Self {
		for row in rows {
			assert!(offset + row_len <= row.len());
		}

		Self {
			rows,
			row_len,
			offset,
		}
	}

	/// Create a new `RowsBatchRef` from a slice of rows and the given row length and offset.
	///
	/// # Safety
	/// This function is unsafe because it does not check if the rows have the enough length.
	/// It is the caller's responsibility to ensure that `row_len` is less than or equal
	/// to the length of each row.
	pub unsafe fn new_unchecked(rows: &'a [&'a [T]], row_len: usize) -> Self {
		unsafe { Self::new_with_offset_unchecked(rows, 0, row_len) }
	}

	/// Create a new `RowsBatchRef` from a slice of rows and the given row length and offset.
	///
	/// # Safety
	/// This function is unsafe because it does not check if the rows have the enough length.
	/// It is the caller's responsibility to ensure that `offset + row_len` is less than or equal
	/// to the length of each row.
	pub unsafe fn new_with_offset_unchecked(
		rows: &'a [&'a [T]],
		offset: usize,
		row_len: usize,
	) -> Self {
		Self {
			rows,
			row_len,
			offset,
		}
	}

	#[inline]
	pub fn iter(&self) -> impl Iterator<Item = &'a [T]> + '_ {
		self.rows.as_ref().iter().copied()
	}

	#[inline(always)]
	pub fn row(&self, index: usize) -> &'a [T] {
		&self.rows[index][self.offset..self.offset + self.row_len]
	}

	#[inline(always)]
	pub fn n_rows(&self) -> usize {
		self.rows.as_ref().len()
	}

	#[inline(always)]
	pub fn row_len(&self) -> usize {
		self.row_len
	}

	#[inline(always)]
	pub fn is_empty(&self) -> bool {
		self.rows.as_ref().is_empty()
	}

	/// Returns a new `RowsBatch` with the specified rows selected by the given indices.
	pub fn map(&self, indices: impl AsRef<[usize]>) -> RowsBatch<'a, T> {
		let rows = indices.as_ref().iter().map(|&i| self.rows[i]).collect();

		RowsBatch {
			rows,
			row_len: self.row_len,
		}
	}

	/// Returns a new `RowsBatchRef` with the specified columns selected by the given indices range.
	pub fn columns_subrange(&self, range: impl RangeBounds<usize>) -> Self {
		let start = match range.start_bound() {
			Bound::Included(&start) => start,
			Bound::Excluded(&start) => start + 1,
			Bound::Unbounded => 0,
		};
		let end = match range.end_bound() {
			Bound::Included(&end) => end + 1,
			Bound::Excluded(&end) => end,
			Bound::Unbounded => self.row_len,
		};

		assert!(start <= end);
		assert!(end <= self.row_len);

		Self {
			rows: self.rows,
			row_len: end - start,
			offset: self.offset + start,
		}
	}
}
