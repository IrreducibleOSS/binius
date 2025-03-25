// Copyright 2025 Irreducible Inc.

pub struct RowsBatch<'a, T> {
	rows: Vec<&'a [T]>,
	n_cols: usize,
}

impl<'a, T> RowsBatch<'a, T> {
	pub fn new(rows: Vec<&'a [T]>, cols: usize) -> Self {
		for row in &rows {
			assert_eq!(row.len(), cols);
		}

		Self { rows, n_cols: cols }
	}

	pub fn new_from_iter(rows: impl IntoIterator<Item = &'a [T]>, n_cols: usize) -> Self {
		let rows = rows.into_iter().map(|x| &x[..n_cols]).collect();
		Self { rows, n_cols }
	}

	pub fn as_ref(&self) -> RowsBatchRef<'a, '_, T> {
		RowsBatchRef {
			rows: self.rows.as_slice(),
			n_cols: self.n_cols,
			_pd: std::marker::PhantomData,
		}
	}
}

pub struct RowsBatchRef<'a, 'b, T> {
	rows: &'a [&'b [T]],
	n_cols: usize,
	_pd: std::marker::PhantomData<&'b [&'a [T]]>,
}

impl<'a, 'b, T> RowsBatchRef<'a, 'b, T> {
	pub fn new_from_data(rows: &'a [&'b [T]], n_cols: usize) -> Self {
		for row in rows {
			assert_eq!(row.len(), n_cols);
		}

		Self {
			rows,
			n_cols,
			_pd: std::marker::PhantomData,
		}
	}

	pub fn iter(&self) -> impl Iterator<Item = &'a [T]> + '_ {
		self.rows.as_ref().iter().copied()
	}

	pub fn rows(&self) -> &[&'a [T]] {
		self.rows.as_ref()
	}

	pub fn n_rows(&self) -> usize {
		self.rows.as_ref().len()
	}

	pub fn n_cols(&self) -> usize {
		self.n_cols
	}

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
			n_cols: self.n_cols,
		}
	}
}
