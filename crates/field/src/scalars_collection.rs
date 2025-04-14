// Copyright 2025 Irreducible Inc.

use crate::{
	packed::{
		get_packed_slice, get_packed_slice_unchecked, len_packed_slice, set_packed_slice,
		set_packed_slice_unchecked,
	},
	PackedField,
};

pub trait ScalarsCollection<F> {
	fn len(&self) -> usize;

	#[inline(always)]
	fn get(&self, index: usize) -> F {
		assert!(index < self.len(), "Index out of bounds");
		unsafe { self.get_unchecked(index) }
	}

	unsafe fn get_unchecked(&self, index: usize) -> F;
}

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
		get_packed_slice_unchecked(&self.slice, index)
	}
}

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
		len_packed_slice(&self.slice)
	}

	#[inline(always)]
	unsafe fn get_unchecked(&self, index: usize) -> P::Scalar {
		get_packed_slice_unchecked(&self.slice, index)
	}
}
impl<'a, P: PackedField> ScalarsCollectionMut<P::Scalar> for PackedSliceMut<'a, P> {
	#[inline(always)]
	unsafe fn set_unchecked(&mut self, index: usize, value: P::Scalar) {
		set_packed_slice_unchecked(self.slice, index, value);
	}
}

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
