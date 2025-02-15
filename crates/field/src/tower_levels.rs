// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	ops::{Add, AddAssign, Index, IndexMut},
};

/// Public API for recursive algorithms over data represented as an array of its limbs
/// E.g. an F_{2^128} element expressed as 8 chunks of 2 bytes
/// or a 256-bit integer represented as 32 individual bytes
///
/// Join and split can be used to combine and split underlying data into upper and lower halves
///
/// This is mostly useful for recursively implementing arithmetic operations
///
/// These separate implementations are necessary to overcome the limitations of const generics in Rust.
/// These implementations eliminate costly bounds checking that would otherwise be imposed by the compiler
/// and allow easy inlining of recursive functions.
pub trait TowerLevel<T>
where
	T: Default + Copy,
{
	// WIDTH is ALWAYS a power of 2
	const WIDTH: usize;

	// The underlying Data should ALWAYS be a fixed-width array of T's
	type Data: AsMut<[T]>
		+ AsRef<[T]>
		+ Sized
		+ Index<usize, Output = T>
		+ IndexMut<usize, Output = T>;
	type Base: TowerLevel<T>;

	// Split something of type Self::Data into two equal halves
	#[allow(clippy::type_complexity)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data);

	// Split something of type Self::Data into two equal mutable halves
	#[allow(clippy::type_complexity)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data);

	// Join two equal-length arrays (the reverse of split)
	#[allow(clippy::type_complexity)]
	fn join(
		first: &<Self::Base as TowerLevel<T>>::Data,
		second: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data;

	// Fills an array of T's containing WIDTH elements
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data;

	// Fills an array of T's containing WIDTH elements with T::default()
	fn default() -> Self::Data {
		Self::from_fn(|_| T::default())
	}
}

pub trait TowerLevelWithArithOps<T>: TowerLevel<T>
where
	T: Default + Add<Output = T> + AddAssign + Copy,
{
	#[inline(always)]
	fn add_into(field_element: &Self::Data, destination: &mut Self::Data) {
		for i in 0..Self::WIDTH {
			destination[i] += field_element[i];
		}
	}

	#[inline(always)]
	fn copy_into(field_element: &Self::Data, destination: &mut Self::Data) {
		for i in 0..Self::WIDTH {
			destination[i] = field_element[i];
		}
	}

	#[inline(always)]
	fn sum(field_element_a: &Self::Data, field_element_b: &Self::Data) -> Self::Data {
		Self::from_fn(|i| field_element_a[i] + field_element_b[i])
	}
}

impl<T, U: TowerLevel<T>> TowerLevelWithArithOps<T> for U where
	T: Default + Add<Output = T> + AddAssign + Copy
{
}

pub struct TowerLevel64;

impl<T> TowerLevel<T> for TowerLevel64
where
	T: Default + Copy,
{
	const WIDTH: usize = 64;

	type Data = [T; 64];
	type Base = TowerLevel32;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		let left = unsafe { &*(data.as_ptr() as *const [T; 32]) };
		let right = unsafe { &*(data.as_ptr().add(32) as *const [T; 32]) };
		(left, right)
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		let left = unsafe { &mut *(data.as_mut_ptr() as *mut [T; 32]) };
		let right = unsafe { &mut *(data.as_mut_ptr().add(32) as *mut [T; 32]) };
		(left, right)
	}

	#[inline(always)]
	fn join<'a>(
		left: &<Self::Base as TowerLevel<T>>::Data,
		right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		let mut result = [T::default(); 64];
		result[..32].copy_from_slice(left);
		result[32..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel32;

impl<T> TowerLevel<T> for TowerLevel32
where
	T: Default + Copy,
{
	const WIDTH: usize = 32;

	type Data = [T; 32];
	type Base = TowerLevel16;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		((data[0..16].try_into().unwrap()), (data[16..32].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(16);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<'a>(
		left: &<Self::Base as TowerLevel<T>>::Data,
		right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		let mut result = [T::default(); 32];
		result[..16].copy_from_slice(left);
		result[16..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel16;

impl<T> TowerLevel<T> for TowerLevel16
where
	T: Default + Copy,
{
	const WIDTH: usize = 16;

	type Data = [T; 16];
	type Base = TowerLevel8;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		((data[0..8].try_into().unwrap()), (data[8..16].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(8);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<'a>(
		left: &<Self::Base as TowerLevel<T>>::Data,
		right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		let mut result = [T::default(); 16];
		result[..8].copy_from_slice(left);
		result[8..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel8;

impl<T> TowerLevel<T> for TowerLevel8
where
	T: Default + Copy,
{
	const WIDTH: usize = 8;

	type Data = [T; 8];
	type Base = TowerLevel4;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		((data[0..4].try_into().unwrap()), (data[4..8].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(4);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<'a>(
		left: &<Self::Base as TowerLevel<T>>::Data,
		right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		let mut result = [T::default(); 8];
		result[..4].copy_from_slice(left);
		result[4..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel4;

impl<T> TowerLevel<T> for TowerLevel4
where
	T: Default + Copy,
{
	const WIDTH: usize = 4;

	type Data = [T; 4];
	type Base = TowerLevel2;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		((data[0..2].try_into().unwrap()), (data[2..4].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(2);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<'a>(
		left: &<Self::Base as TowerLevel<T>>::Data,
		right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		let mut result = [T::default(); 4];
		result[..2].copy_from_slice(left);
		result[2..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel2;

impl<T> TowerLevel<T> for TowerLevel2
where
	T: Default + Copy,
{
	const WIDTH: usize = 2;

	type Data = [T; 2];
	type Base = TowerLevel1;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		((data[0..1].try_into().unwrap()), (data[1..2].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(1);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<'a>(
		left: &<Self::Base as TowerLevel<T>>::Data,
		right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		let mut result = [T::default(); 2];
		result[..1].copy_from_slice(left);
		result[1..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel1;

impl<T> TowerLevel<T> for TowerLevel1
where
	T: Default + Copy,
{
	const WIDTH: usize = 1;

	type Data = [T; 1];
	type Base = Self;

	// Level 1 is the atomic unit of backing data and must not be split.

	#[inline(always)]
	fn split(
		_data: &Self::Data,
	) -> (&<Self::Base as TowerLevel<T>>::Data, &<Self::Base as TowerLevel<T>>::Data) {
		unreachable!()
	}

	#[inline(always)]
	fn split_mut(
		_data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel<T>>::Data, &mut <Self::Base as TowerLevel<T>>::Data) {
		unreachable!()
	}

	#[inline(always)]
	fn join<'a>(
		_left: &<Self::Base as TowerLevel<T>>::Data,
		_right: &<Self::Base as TowerLevel<T>>::Data,
	) -> Self::Data {
		unreachable!()
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> T) -> Self::Data {
		array::from_fn(f)
	}
}
