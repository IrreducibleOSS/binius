// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

use binius_field::BinaryField;

struct TwiddleAccess<F: BinaryField> {}

pub enum Error {
	DeviceError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

pub trait DevSlice<T>: Copy {
	// This doesn't work for ranges too small or unaligned
	fn try_slice(&self, range: impl RangeBounds<usize>) -> Self;
}

pub trait DevSliceMut<T>: Into<Self::ConstSlice> {
	type ConstSlice: DevSlice<T>;

	fn try_slice(&self, range: impl RangeBounds<usize>) -> Self::ConstSlice;
	fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Self;
}

#[trait_variant::make(HAL: Send)]
pub trait LocalHAL<F: BinaryField> {
	type FSlice: DevSlice<F>;
	type FSliceMut: DevSliceMut<F, ConstSlice = Self::FSlice>; // Needs indexing

	fn alloc(&self, size: usize) -> Result<Self::FSliceMut, Error>;

	async fn copy_h2d(&self, src: &[F], dst: &mut Self::FSliceMut) -> Result<(), Error>;
	async fn copy_d2h(&self, src: Self::FSlice, dst: &mut [F]) -> Result<(), Error>;
	async fn copy_d2d(&self, src: Self::FSlice, dst: &mut Self::FSliceMut) -> Result<(), Error>;

	// Separate trait
	async fn rs_encode(
		&self,
		log_n: usize,
		log_batch_size: usize,
		log_inv_rate: usize,
		twiddles: &TwiddleAccess<F>,
		input: Self::FSlice,
		output: &mut Self::FSliceMut,
	) -> Result<(), Error>;
}
