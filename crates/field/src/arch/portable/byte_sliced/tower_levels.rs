// Copyright 2024 Irreducible Inc.

use crate::PackedAESBinaryField32x8b;
use std::array;

// These separate implementations are necessary to overcome the limitations of const generics in Rust.

pub trait TowerLevel {
	const WIDTH: usize;

	type Data: AsMut<[PackedAESBinaryField32x8b]> + AsRef<[PackedAESBinaryField32x8b]> + Default;
	type Base: TowerLevel;

	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel>::Data, &<Self::Base as TowerLevel>::Data);
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel>::Data, &mut <Self::Base as TowerLevel>::Data);

	fn from_fn(f: impl Fn(usize) -> PackedAESBinaryField32x8b) -> Self::Data;

	#[inline(always)]
	fn add_into(field_element: &Self::Data, destination: &mut Self::Data) {
		for i in 0..Self::WIDTH {
			destination.as_mut()[i] += field_element.as_ref()[i];
		}
	}

	#[inline(always)]
	fn copy_into(field_element: &Self::Data, destination: &mut Self::Data) {
		for i in 0..Self::WIDTH {
			destination.as_mut()[i] = field_element.as_ref()[i];
		}
	}

	#[inline(always)]
	fn sum(field_element_a: &Self::Data, field_element_b: &Self::Data) -> Self::Data {
		Self::from_fn(|i| field_element_a.as_ref()[i] + field_element_b.as_ref()[i])
	}
}

pub struct TowerLevel16;

impl TowerLevel for TowerLevel16 {
	const WIDTH: usize = 16;

	type Data = [PackedAESBinaryField32x8b; 16];
	type Base = TowerLevel8;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel>::Data, &<Self::Base as TowerLevel>::Data) {
		((data[0..8].try_into().unwrap()), (data[8..16].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel>::Data, &mut <Self::Base as TowerLevel>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(8);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> PackedAESBinaryField32x8b) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel8;

impl TowerLevel for TowerLevel8 {
	const WIDTH: usize = 8;

	type Data = [PackedAESBinaryField32x8b; 8];
	type Base = TowerLevel4;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel>::Data, &<Self::Base as TowerLevel>::Data) {
		((data[0..4].try_into().unwrap()), (data[4..8].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel>::Data, &mut <Self::Base as TowerLevel>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(4);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> PackedAESBinaryField32x8b) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel4;

impl TowerLevel for TowerLevel4 {
	const WIDTH: usize = 4;

	type Data = [PackedAESBinaryField32x8b; 4];
	type Base = TowerLevel2;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel>::Data, &<Self::Base as TowerLevel>::Data) {
		((data[0..2].try_into().unwrap()), (data[2..4].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel>::Data, &mut <Self::Base as TowerLevel>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(2);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> PackedAESBinaryField32x8b) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel2;

impl TowerLevel for TowerLevel2 {
	const WIDTH: usize = 2;

	type Data = [PackedAESBinaryField32x8b; 2];
	type Base = TowerLevel1;

	#[inline(always)]
	fn split(
		data: &Self::Data,
	) -> (&<Self::Base as TowerLevel>::Data, &<Self::Base as TowerLevel>::Data) {
		((data[0..1].try_into().unwrap()), (data[1..2].try_into().unwrap()))
	}

	#[inline(always)]
	fn split_mut(
		data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel>::Data, &mut <Self::Base as TowerLevel>::Data) {
		let (chunk_1, chunk_2) = data.split_at_mut(1);

		((chunk_1.try_into().unwrap()), (chunk_2.try_into().unwrap()))
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> PackedAESBinaryField32x8b) -> Self::Data {
		array::from_fn(f)
	}
}

pub struct TowerLevel1;

impl TowerLevel for TowerLevel1 {
	const WIDTH: usize = 1;

	type Data = [PackedAESBinaryField32x8b; 1];
	type Base = TowerLevel1;

	// Level 1 is the atomic unit of backing data and must not be split.

	#[inline(always)]
	fn split(
		_data: &Self::Data,
	) -> (&<Self::Base as TowerLevel>::Data, &<Self::Base as TowerLevel>::Data) {
		unreachable!()
	}

	#[inline(always)]
	fn split_mut(
		_data: &mut Self::Data,
	) -> (&mut <Self::Base as TowerLevel>::Data, &mut <Self::Base as TowerLevel>::Data) {
		unreachable!()
	}

	#[inline(always)]
	fn from_fn(f: impl Fn(usize) -> PackedAESBinaryField32x8b) -> Self::Data {
		array::from_fn(f)
	}
}
