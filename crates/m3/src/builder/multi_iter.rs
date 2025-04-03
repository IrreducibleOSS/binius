// Copyright 2025 Irreducible Inc.

/// An iterator that iterates multiple inner iterators in parallel.
///
/// The iterator yields vectors of items from the inner iterators.
#[derive(Debug)]
pub struct MultiIterator<T> {
	entries: Vec<T>,
}

impl<I: Iterator> MultiIterator<I> {
	pub fn new(entries: Vec<I>) -> Self {
		Self { entries }
	}
}

impl<I: ExactSizeIterator> Iterator for MultiIterator<I> {
	type Item = Vec<I::Item>;

	fn next(&mut self) -> Option<Vec<I::Item>> {
		self.entries.iter_mut().map(Iterator::next).collect()
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = self.len();
		(len, Some(len))
	}
}

impl<I: ExactSizeIterator> ExactSizeIterator for MultiIterator<I> {
	fn len(&self) -> usize {
		self.entries
			.iter()
			.map(ExactSizeIterator::len)
			.min()
			.unwrap_or_default()
	}
}

impl<I: ExactSizeIterator + DoubleEndedIterator> DoubleEndedIterator for MultiIterator<I> {
	fn next_back(&mut self) -> Option<Vec<I::Item>> {
		self.entries
			.iter_mut()
			.map(DoubleEndedIterator::next_back)
			.collect()
	}
}
