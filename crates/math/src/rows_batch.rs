// Copyright 2025 Irreducible Inc.

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
	pub const fn get_ref(&self) -> RowsBatchRef<'_, T> {
		RowsBatchRef {
			rows: self.rows.as_slice(),
			row_len: self.row_len,
		}
	}
}

/// This struct is similar to `RowsBatch`, but it holds a reference to the slice of rows.
/// Unfortunately due to liftime issues we can't have a single generic struct which is
/// parameterized by a container type.
pub struct RowsBatchRef<'a, T> {
	rows: &'a [&'a [T]],
	row_len: usize,
}

impl<'a, T> RowsBatchRef<'a, T> {
	/// Create a new `RowsBatchRef` from a slice of rows and the given row length.
	///
	/// # Panics
	/// In case if any of the rows has a length different from `n_cols`.
	#[inline]
	pub fn new(rows: &'a [&'a [T]], row_len: usize) -> Self {
		for row in rows {
			assert_eq!(row.len(), row_len);
		}

		Self { rows, row_len }
	}

	#[inline]
	pub fn iter(&self) -> impl Iterator<Item = &'a [T]> + '_ {
		self.rows.as_ref().iter().copied()
	}

	#[inline(always)]
	pub const fn rows(&self) -> &[&'a [T]] {
		self.rows
	}

	#[inline(always)]
	pub fn n_rows(&self) -> usize {
		self.rows.as_ref().len()
	}

	#[inline(always)]
	pub const fn row_len(&self) -> usize {
		self.row_len
	}

	#[inline(always)]
	pub fn is_empty(&self) -> bool {
		self.rows.as_ref().is_empty()
	}

	pub fn map(&self, indices: impl AsRef<[usize]>) -> RowsBatch<'a, T> {
		let rows = indices.as_ref().iter().map(|&i| self.rows[i]).collect();

		RowsBatch {
			rows,
			row_len: self.row_len,
		}
	}
}
