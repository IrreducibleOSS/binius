// Copyright 2025 Irreducible Inc.

use crate::{
	packed::{get_packed_slice_unchecked, len_packed_slice, set_packed_slice_unchecked},
	PackedField,
};

/// A trait for collections of scalars, allowing for indexed access.
/// This trait is used to abstract over different types of collections - scalar slices,
/// slices of packed field elements including subranges of collections.
pub trait ScalarsCollection<F> {
	fn len(&self) -> usize;

	#[inline(always)]
	fn get(&self, index: usize) -> F {
		assert!(index < self.len(), "Index out of bounds");
		unsafe { self.get_unchecked(index) }
	}

	unsafe fn get_unchecked(&self, index: usize) -> F;
}

/// A trait for a mutable access to a collection of scalars.
pub trait ScalarsCollectionMut<F>: ScalarsCollection<F> {
	#[inline(always)]
	fn set(&mut self, index: usize, value: F) {
		assert!(index < self.len(), "Index out of bounds");
		unsafe { self.set_unchecked(index, value) }
	}

	unsafe fn set_unchecked(&mut self, index: usize, value: F);
}

impl<F: Copy> ScalarsCollection<F> for &[F] {
	#[inline(always)]
	fn len(&self) -> usize {
		<[F]>::len(self)
	}

	#[inline(always)]
	fn get(&self, index: usize) -> F {
		self[index]
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> F {
		*<[F]>::get_unchecked(self, index)
	}
}

impl<F: Copy> ScalarsCollection<F> for &mut [F] {
	#[inline(always)]
	fn len(&self) -> usize {
		<[F]>::len(self)
	}

	#[inline(always)]
	fn get(&self, index: usize) -> F {
		self[index]
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> F {
		*<[F]>::get_unchecked(self, index)
	}
}

impl<F: Copy> ScalarsCollectionMut<F> for &mut [F] {
	#[inline(always)]
	fn set(&mut self, index: usize, value: F) {
		self[index] = value;
	}

	#[inline(always)]
	unsafe fn set_unchecked(&mut self, index: usize, value: F) {
		*<[F]>::get_unchecked_mut(self, index) = value;
	}
}

/// A slice of packed field elements as a collection of scalars.
#[derive(Clone)]
pub struct PackedSlice<'a, P: PackedField> {
	slice: &'a [P],
	len: usize,
}

impl<'a, P: PackedField> PackedSlice<'a, P> {
	#[inline(always)]
	pub fn new(slice: &'a [P]) -> Self {
		Self {
			slice,
			len: len_packed_slice(slice),
		}
	}

	#[inline(always)]
	pub fn new_with_len(slice: &'a [P], len: usize) -> Self {
		assert!(len <= len_packed_slice(slice));

		Self { slice, len }
	}
}

impl<'a, P: PackedField> ScalarsCollection<P::Scalar> for PackedSlice<'a, P> {
	#[inline(always)]
	fn len(&self) -> usize {
		self.len
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> P::Scalar {
		get_packed_slice_unchecked(self.slice, index)
	}
}

/// A mutable slice of packed field elements as a collection of scalars.
pub struct PackedSliceMut<'a, P: PackedField> {
	slice: &'a mut [P],
	len: usize,
}

impl<'a, P: PackedField> PackedSliceMut<'a, P> {
	#[inline(always)]
	pub fn new(slice: &'a mut [P]) -> Self {
		let len = len_packed_slice(slice);
		Self { slice, len }
	}

	#[inline(always)]
	pub fn new_with_len(slice: &'a mut [P], len: usize) -> Self {
		assert!(len <= len_packed_slice(slice));

		Self { slice, len }
	}
}

impl<'a, P: PackedField> ScalarsCollection<P::Scalar> for PackedSliceMut<'a, P> {
	#[inline(always)]
	fn len(&self) -> usize {
		self.len
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> P::Scalar {
		get_packed_slice_unchecked(self.slice, index)
	}
}
impl<'a, P: PackedField> ScalarsCollectionMut<P::Scalar> for PackedSliceMut<'a, P> {
	#[inline(always)]
	unsafe fn set_unchecked(&mut self, index: usize, value: P::Scalar) {
		set_packed_slice_unchecked(self.slice, index, value);
	}
}

/// A subrange adapter of a collection of scalars.
#[derive(Clone)]
pub struct CollectionSubrange<'a, F, Inner: ScalarsCollection<F>> {
	inner: &'a Inner,
	offset: usize,
	len: usize,
	_marker: std::marker::PhantomData<F>,
}

impl<'a, F, Inner: ScalarsCollection<F>> CollectionSubrange<'a, F, Inner> {
	#[inline(always)]
	pub fn new(inner: &'a Inner, offset: usize, len: usize) -> Self {
		assert!(offset + len <= inner.len(), "subrange out of bounds");

		Self {
			inner,
			offset,
			len,
			_marker: std::marker::PhantomData,
		}
	}
}

impl<'a, F, Inner: ScalarsCollection<F>> ScalarsCollection<F> for CollectionSubrange<'a, F, Inner> {
	#[inline(always)]
	fn len(&self) -> usize {
		self.len
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> F {
		self.inner.get_unchecked(index + self.offset)
	}
}

/// A subrange adapter of a mutable collection of scalars.
pub struct CollectionSubrangeMut<'a, F, Inner: ScalarsCollectionMut<F>> {
	inner: &'a mut Inner,
	offset: usize,
	len: usize,
	_marker: std::marker::PhantomData<&'a F>,
}

impl<'a, F, Inner: ScalarsCollectionMut<F>> CollectionSubrangeMut<'a, F, Inner> {
	#[inline(always)]
	pub fn new(inner: &'a mut Inner, offset: usize, len: usize) -> Self {
		assert!(offset + len <= inner.len(), "subrange out of bounds");

		Self {
			inner,
			offset,
			len,
			_marker: std::marker::PhantomData,
		}
	}
}
impl<'a, F, Inner: ScalarsCollectionMut<F>> ScalarsCollection<F>
	for CollectionSubrangeMut<'a, F, Inner>
{
	#[inline(always)]
	fn len(&self) -> usize {
		self.len
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> F {
		self.inner.get_unchecked(index + self.offset)
	}
}
impl<'a, F, Inner: ScalarsCollectionMut<F>> ScalarsCollectionMut<F>
	for CollectionSubrangeMut<'a, F, Inner>
{
	#[inline(always)]
	unsafe fn set_unchecked(&mut self, index: usize, value: F) {
		self.inner.set_unchecked(index + self.offset, value);
	}
}

#[cfg(test)]
mod tests {
	use std::fmt::Debug;

	use itertools::Itertools;
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::*;
	use crate::{BinaryField8b, PackedBinaryField16x8b};

	fn check_collection<F: Eq + Debug, C: ScalarsCollection<F>>(
		collection: &impl ScalarsCollection<F>,
		expected: &[F],
	) {
		assert_eq!(collection.len(), expected.len());

		for (i, v) in expected.iter().enumerate() {
			assert_eq!(&collection.get(i), v);
			assert_eq!(&unsafe { collection.get_unchecked(i) }, v);
		}
	}

	fn check_collection_get_set<F: Eq + Copy + Debug, C: ScalarsCollectionMut<F>>(
		collection: &mut impl ScalarsCollectionMut<F>,
		gen: &mut impl FnMut() -> F,
	) {
		for i in 0..collection.len() {
			let value = gen();
			collection.set(i, value);
			assert_eq!(collection.get(i), value);
			assert_eq!(unsafe { collection.get_unchecked(i) }, value);
		}
	}

	#[test]
	fn check_slice() {
		let slice: &[usize] = &[];
		check_collection::<usize, &[usize]>(&slice, slice);

		let slice: &[usize] = &[1usize, 2, 3];
		check_collection::<usize, &[usize]>(&slice, slice);
	}

	#[test]
	fn check_slice_mut() {
		let mut rng = StdRng::seed_from_u64(0);
		let mut gen = || -> usize { rng.gen() };

		let mut slice: &mut [usize] = &mut [];

		check_collection::<usize, &mut [usize]>(&slice, &slice.to_vec());
		check_collection_get_set::<usize, &mut [usize]>(&mut slice, &mut gen);

		let mut slice: &mut [usize] = &mut [1, 2, 3];
		check_collection::<usize, &mut [usize]>(&slice, &slice.to_vec());
		check_collection_get_set::<usize, &mut [usize]>(&mut slice, &mut gen);
	}

	#[test]
	fn check_packed_slice() {
		let slice: &[PackedBinaryField16x8b] = &[];
		let packed_slice = PackedSlice::new(slice);
		check_collection::<_, PackedSlice<PackedBinaryField16x8b>>(&packed_slice, &[]);
		let packed_slice = PackedSlice::new_with_len(slice, 0);
		check_collection::<_, PackedSlice<PackedBinaryField16x8b>>(&packed_slice, &[]);

		let mut rng = StdRng::seed_from_u64(0);
		let slice: &[PackedBinaryField16x8b] = &[
			PackedBinaryField16x8b::random(&mut rng),
			PackedBinaryField16x8b::random(&mut rng),
		];
		let packed_slice = PackedSlice::new(slice);
		check_collection::<_, PackedSlice<PackedBinaryField16x8b>>(
			&packed_slice,
			&PackedField::iter_slice(&slice).collect_vec(),
		);

		let packed_slice = PackedSlice::new_with_len(slice, 3);
		check_collection::<_, PackedSlice<PackedBinaryField16x8b>>(
			&packed_slice,
			&PackedField::iter_slice(&slice).take(3).collect_vec(),
		);
	}

	#[test]
	fn check_packed_slice_mut() {
		let mut rng = StdRng::seed_from_u64(0);
		let mut gen = || BinaryField8b::random(&mut rng);

		let slice: &mut [PackedBinaryField16x8b] = &mut [];
		let packed_slice = PackedSliceMut::new(slice);
		check_collection::<_, PackedSliceMut<PackedBinaryField16x8b>>(&packed_slice, &[]);
		let packed_slice = PackedSliceMut::new_with_len(slice, 0);
		check_collection::<_, PackedSliceMut<PackedBinaryField16x8b>>(&packed_slice, &[]);

		let mut rng = StdRng::seed_from_u64(0);
		let slice: &mut [PackedBinaryField16x8b] = &mut [
			PackedBinaryField16x8b::random(&mut rng),
			PackedBinaryField16x8b::random(&mut rng),
		];
		let values = PackedField::iter_slice(&slice).collect_vec();
		let mut packed_slice = PackedSliceMut::new(slice);
		check_collection::<_, PackedSliceMut<PackedBinaryField16x8b>>(&packed_slice, &values);
		check_collection_get_set::<_, PackedSliceMut<PackedBinaryField16x8b>>(
			&mut packed_slice,
			&mut gen,
		);

		let values = PackedField::iter_slice(&slice).collect_vec();
		let mut packed_slice = PackedSliceMut::new_with_len(slice, 3);
		check_collection::<_, PackedSliceMut<PackedBinaryField16x8b>>(&packed_slice, &values[..3]);
		check_collection_get_set::<_, PackedSliceMut<PackedBinaryField16x8b>>(
			&mut packed_slice,
			&mut gen,
		);
	}

	#[test]
	fn test_subrange() {
		let slice: &[usize] = &[1, 2, 3, 4, 5];
		let subrange = CollectionSubrange::new(&slice, 1, 3);
		check_collection::<usize, CollectionSubrange<usize, &[usize]>>(&subrange, &[2, 3, 4]);
	}

	#[test]
	fn test_subrange_mut() {
		let mut rng = StdRng::seed_from_u64(0);
		let mut gen = || -> usize { rng.gen() };

		let mut slice: &mut [usize] = &mut [1, 2, 3, 4, 5];
		let values = (&slice[1..4]).to_vec();
		let mut subrange = CollectionSubrangeMut::new(&mut slice, 1, 3);
		check_collection::<usize, CollectionSubrangeMut<usize, &mut [usize]>>(
			&mut subrange,
			&values,
		);
		check_collection_get_set::<usize, CollectionSubrangeMut<usize, &mut [usize]>>(
			&mut subrange,
			&mut gen,
		);
	}
}
