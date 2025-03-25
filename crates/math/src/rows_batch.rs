// Copyright 2025 Irreducible Inc.

/// This struct represents a batch of rows, each row having the same length equal to `row_len`.
pub struct RowsBatch<'a, T> {
	rows: Vec<&'a [T]>,
	row_len: usize,
}

impl<'a, T> RowsBatch<'a, T> {
	/// Create a new `RowsBatch` from a vector of rows and the given row length
	#[inline]
	pub fn new(rows: Vec<&'a [T]>, cols: usize) -> Self {
		for row in &rows {
			assert_eq!(row.len(), cols);
		}

		Self {
			rows,
			row_len: cols,
		}
	}

	#[inline]
	pub fn new_from_iter(rows: impl IntoIterator<Item = &'a [T]>, n_cols: usize) -> Self {
		let rows = rows.into_iter().map(|x| &x[..n_cols]).collect();
		Self {
			rows,
			row_len: n_cols,
		}
	}

	#[inline(always)]
	pub fn as_ref(&self) -> RowsBatchRef<'a, '_, T> {
		RowsBatchRef {
			rows: self.rows.as_slice(),
			row_len: self.row_len,
			_pd: std::marker::PhantomData,
		}
	}
}

/// This struct is similar to `RowsBatch`, but it holds a reference to the slice of rows.
/// Unfortunately due to liftime issues we can't have a single generic struct which is
/// parameterized by a container type.
pub struct RowsBatchRef<'a, 'b, T> {
	rows: &'a [&'b [T]],
	row_len: usize,
	_pd: std::marker::PhantomData<&'b [&'a [T]]>,
}

impl<'a, 'b, T> RowsBatchRef<'a, 'b, T> {
	/// Create a new `RowsBatchRef` from a slice of rows and the given row length.
	#[inline]
	pub fn new(rows: &'a [&'b [T]], n_cols: usize) -> Self {
		for row in rows {
			assert_eq!(row.len(), n_cols);
		}

		Self {
			rows,
			row_len: n_cols,
			_pd: std::marker::PhantomData,
		}
	}

	#[inline]
	pub fn iter(&self) -> impl Iterator<Item = &'a [T]> + '_ {
		self.rows.as_ref().iter().copied()
	}

	#[inline(always)]
	pub fn rows(&self) -> &[&'a [T]] {
		self.rows.as_ref()
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

	pub fn map(&self, indices: impl AsRef<[usize]>) -> RowsBatch<'a, T> {
		let rows = indices
			.as_ref()
			.iter()
			.map(|&i| self.rows.as_ref()[i])
			.collect();

		RowsBatch {
			rows,
			row_len: self.row_len,
		}
	}
}
