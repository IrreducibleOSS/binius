// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon

use super::{
	parallel_iterator::ParallelIteratorInner, parallel_wrapper::ParallelWrapper,
	IntoParallelIterator, ParallelIterator,
};

/// The reason why we need this trait is because `IndexedParallelIterator` contains
/// methods that overlaps with `std::iterator::Iterator` trait. So we implement it separately for
/// different `std::iter::Iterator` types while `IndexedParallelIterator` is implemented for
/// `ParallelWrapper<I>` where `I` is `IndexedParallelIteratorInner`.
///
/// Currently only those methods are implemented that are used in the `binius` code base. All other methods
/// can be implemented upon request.
pub(crate) trait IndexedParallelIteratorInner: ParallelIteratorInner {
	#[inline(always)]
	fn with_min_len(self, _min: usize) -> Self
	where
		Self: Sized,
	{
		self
	}

	#[inline(always)]
	fn with_max_len(self, _max: usize) -> Self
	where
		Self: Sized,
	{
		self
	}

	#[inline]
	fn enumerate(self) -> impl IndexedParallelIteratorInner<Item = (usize, Self::Item)>
	where
		Self: Sized,
	{
		Iterator::enumerate(self.into_iter())
	}

	#[inline]
	fn zip<Z>(self, zip_op: Z) -> std::iter::Zip<Self, Z>
	where
		Z: IndexedParallelIteratorInner,
	{
		Iterator::zip(self, zip_op)
	}

	#[inline]
	fn step_by(self, step: usize) -> impl IndexedParallelIteratorInner<Item = Self::Item>
	where
		Self: Sized,
	{
		Iterator::step_by(self, step)
	}

	#[inline]
	fn chunks(self, chunk_size: usize) -> impl IndexedParallelIteratorInner<Item = Vec<Self::Item>>
	where
		Self: Sized,
	{
		Chunks {
			inner: self,
			chunk_size,
		}
	}
}

struct Chunks<I> {
	inner: I,
	chunk_size: usize,
}

impl<I: Iterator> Iterator for Chunks<I> {
	type Item = Vec<I::Item>;

	fn next(&mut self) -> Option<Self::Item> {
		let mut chunk = Vec::with_capacity(self.chunk_size);
		for _ in 0..self.chunk_size {
			match self.inner.next() {
				Some(item) => chunk.push(item),
				None => break,
			}
		}
		if chunk.is_empty() {
			None
		} else {
			Some(chunk)
		}
	}
}

impl<I: Iterator> IndexedParallelIteratorInner for Chunks<I> {}

// Implement `IndexedParallelIteratorInner` for different `std::iter::Iterator` types.
// Unfortunately, we can't implement it for all `std::iter::Iterator` types because of the collisions
// with generic implementation for tuples (see `multizip_impls!` macro in `parallel_iterator.rs`).
// If you need to implement it for some other type, please add implementation here.
impl<Idx> IndexedParallelIteratorInner for std::ops::Range<Idx> where Self: Iterator<Item = Idx> {}
impl<T> IndexedParallelIteratorInner for std::slice::IterMut<'_, T> {}
impl<L: IndexedParallelIteratorInner, R: IndexedParallelIteratorInner> IndexedParallelIteratorInner
	for std::iter::Zip<L, R>
{
}
impl<I: IndexedParallelIteratorInner> IndexedParallelIteratorInner for std::iter::Enumerate<I> {}
impl<I: IndexedParallelIteratorInner> IndexedParallelIteratorInner for std::iter::StepBy<I> {}
impl<I: Iterator, R, F: Fn(I::Item) -> R> IndexedParallelIteratorInner for std::iter::Map<I, F> {}
impl<T> IndexedParallelIteratorInner for std::vec::IntoIter<T> {}
impl<T, const N: usize> IndexedParallelIteratorInner for std::array::IntoIter<T, N> {}

#[allow(private_bounds)]
pub trait IndexedParallelIterator: ParallelIterator {
	type Inner: IndexedParallelIteratorInner<Item = Self::Item>;
	fn into_inner(self) -> <Self as IndexedParallelIterator>::Inner;

	#[inline(always)]
	fn with_min_len(self, min: usize) -> impl IndexedParallelIterator<Item = Self::Item>
	where
		Self: Sized,
	{
		ParallelWrapper::new(IndexedParallelIterator::into_inner(self).with_min_len(min))
	}

	#[inline(always)]
	fn with_max_len(self, max: usize) -> impl IndexedParallelIterator<Item = Self::Item>
	where
		Self: Sized,
	{
		ParallelWrapper::new(IndexedParallelIterator::into_inner(self).with_max_len(max))
	}

	#[inline]
	fn enumerate(self) -> impl IndexedParallelIterator<Item = (usize, Self::Item)>
	where
		Self: Sized,
	{
		ParallelWrapper::new(IndexedParallelIteratorInner::enumerate(
			IndexedParallelIterator::into_inner(self),
		))
	}

	#[inline]
	fn zip<Z>(
		self,
		zip_op: Z,
	) -> ParallelWrapper<
		std::iter::Zip<
			<Self as IndexedParallelIterator>::Inner,
			<<Z as IntoParallelIterator>::Iter as IndexedParallelIterator>::Inner,
		>,
	>
	where
		Z: IntoParallelIterator<Iter: IndexedParallelIterator>,
	{
		ParallelWrapper::new(IndexedParallelIteratorInner::zip(
			IndexedParallelIterator::into_inner(self),
			IndexedParallelIterator::into_inner(zip_op.into_par_iter()),
		))
	}

	#[inline]
	fn step_by(self, step: usize) -> impl IndexedParallelIterator<Item = Self::Item>
	where
		Self: Sized,
	{
		ParallelWrapper::new(IndexedParallelIteratorInner::step_by(
			IndexedParallelIterator::into_inner(self),
			step,
		))
	}

	#[inline]
	fn chunks(self, chunk_size: usize) -> impl IndexedParallelIterator<Item = Vec<Self::Item>> {
		assert!(chunk_size != 0, "chunk_size must not be zero");

		ParallelWrapper::new(IndexedParallelIterator::into_inner(self).chunks(chunk_size))
	}
}

impl<I: IndexedParallelIteratorInner> IndexedParallelIterator for ParallelWrapper<I> {
	type Inner = I;

	#[inline(always)]
	fn into_inner(self) -> I {
		self.0
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn check_zip() {
		let a = &[1, 2, 3];
		let b = &[4, 5, 6];

		let result = a.into_par_iter().zip(b.into_par_iter()).collect::<Vec<_>>();
		assert_eq!(result, vec![(1, 4), (2, 5), (3, 6)]);
	}

	#[test]
	fn check_step_by() {
		let a = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

		let result = a.into_par_iter().step_by(2).collect::<Vec<_>>();
		assert_eq!(result, vec![1, 3, 5, 7, 9]);
	}

	#[test]
	fn check_step_by_range() {
		let a = 1..10;

		let result = a.into_par_iter().step_by(2).collect::<Vec<_>>();
		assert_eq!(result, vec![1, 3, 5, 7, 9]);
	}

	#[test]
	fn check_map() {
		let a = &[1, 2, 3];

		let result = a.into_par_iter().map(|x| x * 2).collect::<Vec<_>>();
		assert_eq!(result, vec![2, 4, 6]);
	}
}
