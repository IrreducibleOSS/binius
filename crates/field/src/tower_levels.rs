// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	ops::{Add, AddAssign, Index, IndexMut},
};

use binius_utils::checked_arithmetics::checked_log_2;

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
pub trait TowerLevel: 'static {
	// WIDTH is ALWAYS a power of 2
	const WIDTH: usize;
	const LOG_WIDTH: usize = checked_log_2(Self::WIDTH);

	// The underlying Data should ALWAYS be a fixed-width array of T's
	type Data<T>: AsMut<[T]>
		+ AsRef<[T]>
		+ Sized
		+ Index<usize, Output = T>
		+ IndexMut<usize, Output = T>;
	type Base: TowerLevel;

	// Split something of type Self::Data<T>into two equal halves
	#[allow(clippy::type_complexity)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>);

	// Split something of type Self::Data<T>into two equal mutable halves
	#[allow(clippy::type_complexity)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>);

	// Join two equal-length arrays (the reverse of split)
	#[allow(clippy::type_complexity)]
	fn join<T: Copy + Default>(
		first: &<Self::Base as TowerLevel>::Data<T>,
		second: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T>;

	// Fills an array of T's containing WIDTH elements
	fn from_fn<T: Copy>(f: impl FnMut(usize) -> T) -> Self::Data<T>;

	// Fills an array of T's containing WIDTH elements with T::default()
	fn default<T: Copy + Default>() -> Self::Data<T> {
		Self::from_fn(|_| T::default())
	}
}

pub trait TowerLevelWithArithOps: TowerLevel {
	#[inline(always)]
	fn add_into<T: AddAssign + Copy>(
		field_element: &Self::Data<T>,
		destination: &mut Self::Data<T>,
	) {
		for i in 0..Self::WIDTH {
			destination[i] += field_element[i];
		}
	}

	#[inline(always)]
	fn copy_into<T: Copy>(field_element: &Self::Data<T>, destination: &mut Self::Data<T>) {
		for i in 0..Self::WIDTH {
			destination[i] = field_element[i];
		}
	}

	#[inline(always)]
	fn sum<T: Copy + Add<Output = T>>(
		field_element_a: &Self::Data<T>,
		field_element_b: &Self::Data<T>,
	) -> Self::Data<T> {
		Self::from_fn(|i| field_element_a[i] + field_element_b[i])
	}
}

impl<T: TowerLevel> TowerLevelWithArithOps for T {}

pub struct TowerLevel64;

impl TowerLevel for TowerLevel64 {
	const WIDTH: usize = 64;

	type Data<T> = [T; 64];
	type Base = TowerLevel32;

	#[inline(always)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		((data[0..32].try_into().unwrap()), (data[32..64].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		let (chunk_1, chunk_2) = data.split_at_mut(32);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		left: &<Self::Base as TowerLevel>::Data<T>,
		right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		let mut result = [T::default(); 64];
		result[..32].copy_from_slice(left);
		result[32..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}

pub struct TowerLevel32;

impl TowerLevel for TowerLevel32 {
	const WIDTH: usize = 32;

	type Data<T> = [T; 32];
	type Base = TowerLevel16;

	#[inline(always)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		((data[0..16].try_into().unwrap()), (data[16..32].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		let (chunk_1, chunk_2) = data.split_at_mut(16);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		left: &<Self::Base as TowerLevel>::Data<T>,
		right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		let mut result = [T::default(); 32];
		result[..16].copy_from_slice(left);
		result[16..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}

pub struct TowerLevel16;

impl TowerLevel for TowerLevel16 {
	const WIDTH: usize = 16;

	type Data<T> = [T; 16];
	type Base = TowerLevel8;

	#[inline(always)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		((data[0..8].try_into().unwrap()), (data[8..16].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		let (chunk_1, chunk_2) = data.split_at_mut(8);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		left: &<Self::Base as TowerLevel>::Data<T>,
		right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		let mut result = [T::default(); 16];
		result[..8].copy_from_slice(left);
		result[8..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}

pub struct TowerLevel8;

impl TowerLevel for TowerLevel8 {
	const WIDTH: usize = 8;

	type Data<T> = [T; 8];
	type Base = TowerLevel4;

	#[inline(always)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		((data[0..4].try_into().unwrap()), (data[4..8].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		let (chunk_1, chunk_2) = data.split_at_mut(4);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		left: &<Self::Base as TowerLevel>::Data<T>,
		right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		let mut result = [T::default(); 8];
		result[..4].copy_from_slice(left);
		result[4..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}

pub struct TowerLevel4;

impl TowerLevel for TowerLevel4 {
	const WIDTH: usize = 4;

	type Data<T> = [T; 4];
	type Base = TowerLevel2;

	#[inline(always)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		((data[0..2].try_into().unwrap()), (data[2..4].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		let (chunk_1, chunk_2) = data.split_at_mut(2);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		left: &<Self::Base as TowerLevel>::Data<T>,
		right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		let mut result = [T::default(); 4];
		result[..2].copy_from_slice(left);
		result[2..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}

pub struct TowerLevel2;

impl TowerLevel for TowerLevel2 {
	const WIDTH: usize = 2;

	type Data<T> = [T; 2];
	type Base = TowerLevel1;

	#[inline(always)]
	fn split<T>(
		data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		((data[0..1].try_into().unwrap()), (data[1..2].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut<T>(
		data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		let (chunk_1, chunk_2) = data.split_at_mut(1);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		left: &<Self::Base as TowerLevel>::Data<T>,
		right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		let mut result = [T::default(); 2];
		result[..1].copy_from_slice(left);
		result[1..].copy_from_slice(right);
		result
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}

pub struct TowerLevel1;

impl TowerLevel for TowerLevel1 {
	const WIDTH: usize = 1;

	type Data<T> = [T; 1];
	type Base = Self;

	// Level 1 is the atomic unit of backing data and must not be split.

	#[inline(always)]
	fn split<T>(
		_data: &Self::Data<T>,
	) -> (&<Self::Base as TowerLevel>::Data<T>, &<Self::Base as TowerLevel>::Data<T>) {
		unreachable!()
	}

	#[inline(always)]
	fn split_mut<T>(
		_data: &mut Self::Data<T>,
	) -> (&mut <Self::Base as TowerLevel>::Data<T>, &mut <Self::Base as TowerLevel>::Data<T>) {
		unreachable!()
	}

	#[inline(always)]
	fn join<T: Copy + Default>(
		_left: &<Self::Base as TowerLevel>::Data<T>,
		_right: &<Self::Base as TowerLevel>::Data<T>,
	) -> Self::Data<T> {
		unreachable!()
	}

	#[inline(always)]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Data<T> {
		array::from_fn(f)
	}
}
