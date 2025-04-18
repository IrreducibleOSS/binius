// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

use crate::alloc::Error;

/// A trait for types that behave like slices but may have different underlying representations.
///
/// This trait provides basic slice-like operations that work across different memory types
/// and compute devices. It abstracts over the specific slice implementation while maintaining
/// core slice functionality like length checks.
///
/// The trait is used as a bound on memory types to ensure they provide basic slice operations
/// regardless of their internal representation and accessibility from instructions executing
/// on the host.
pub trait OpaqueSlice<T> {
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	fn len(&self) -> usize;
}

impl<T> OpaqueSlice<T> for &[T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

impl<T> OpaqueSlice<T> for &mut [T] {
	fn len(&self) -> usize {
		(**self).len()
	}
}

/// A trait for memory types that can be used in compute operations.
///
/// This trait defines the core memory operations needed to work with slices of data across
/// different compute devices and memory types. It provides:
///
/// - Associated slice types that may have different representations on different devices
/// - Methods for borrowing and splitting slices safely
/// - A minimum slice length that implementations must respect
///
/// The trait is parameterized over the element type `F` to allow for different data types
/// while maintaining type safety.
///
/// # Safety
///
/// Implementations must ensure that:
/// - Slices do not outlive their parent memory
/// - Mutable slices provide exclusive access
/// - All operations respect the `MIN_SLICE_LEN` constant
/// - Split operations produce non-overlapping slices
pub trait ComputeMemory<F> {
	type FSlice<'a>: OpaqueSlice<F>;
	type FSliceMut<'a>: OpaqueSlice<F>;
	const MIN_SLICE_LEN: usize;

	/// Borrows a mutable memory slice as immutable.
	///
	/// This allows the immutable reference to be copied.
	fn as_const<'a>(data: &'a Self::FSliceMut<'_>) -> Self::FSlice<'a>;

	/// Borrows a subslice of an immutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of `MIN_SLICE_LEN`
	fn slice(data: Self::FSlice<'_>, range: impl RangeBounds<usize>) -> Self::FSlice<'_>;

	/// Borrows a subslice of a mutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of `Self::MIN_SLICE_LEN`
	fn slice_mut<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		range: impl RangeBounds<usize>,
	) -> Self::FSliceMut<'a>;

	/// Splits a mutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of `MIN_SLICE_LEN`
	fn split_at_mut(
		data: Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'_>, Self::FSliceMut<'_>);

	fn split_at_mut_borrowed<'a>(
		data: &'a mut Self::FSliceMut<'_>,
		mid: usize,
	) -> (Self::FSliceMut<'a>, Self::FSliceMut<'a>) {
		let borrowed = Self::slice_mut(data, ..);
		Self::split_at_mut(borrowed, mid)
	}
}

/// A trait for host-accessible memory that can be accessed as standard Rust slices.
///
/// This trait extends `ComputeMemory` by requiring that its slice types implement
/// `AsRef<[F]>` and `AsMut<[F]>`, allowing direct access to the underlying memory
/// as standard Rust slices. This enables efficient host-side operations and
/// interoperability with standard Rust code.
///
/// The trait is automatically implemented for any `ComputeMemory` type whose slices
/// implement the required conversion traits.
pub trait ComputeMemoryHost<F>: ComputeMemory<F>
where
	for<'a> Self::FSlice<'a>: AsRef<[F]>,
	for<'a> Self::FSliceMut<'a>: AsMut<[F]>,
{
}

/// A trait for managing memory allocation and data transfer between host and device memory.
///
/// This trait provides a unified interface for allocating memory on both host and device,
/// and copying data between them. It is designed to work with any pair of memory types
/// that implement `ComputeMemory`, where the host memory type additionally implements
/// `ComputeMemoryHost`.
///
/// The trait is parameterized by:
/// - `'a`: The lifetime of host allocated memory
/// - `'b`: The lifetime of device allocated memory
/// - `F`: The element type stored in memory, which must implement `Copy`
///
/// The associated types specify:
/// - `MemHost`: The host memory type, which must implement `ComputeMemoryHost<F>`
/// - `MemDevice`: The device memory type, which must implement `ComputeMemory<F>`
///
/// The trait provides methods for:
/// - Allocating memory on both host and device
/// - Copying data from host to device memory
/// - Copying data from device to host memory
pub trait ComputeMemorySuite<'a, 'b, F: Copy>
where
	for<'c> <<Self as ComputeMemorySuite<'a, 'b, F>>::MemHost as ComputeMemory<F>>::FSlice<'c>:
		AsRef<[F]>,
	for<'c> <<Self as ComputeMemorySuite<'a, 'b, F>>::MemHost as ComputeMemory<F>>::FSliceMut<'c>:
		AsMut<[F]>,
{
	type MemHost: ComputeMemoryHost<F>;
	type MemDevice: ComputeMemory<F>;

	fn alloc_host(
		&self,
		n: usize,
	) -> Result<<Self::MemHost as ComputeMemory<F>>::FSliceMut<'a>, Error>;
	fn alloc_device(
		&self,
		n: usize,
	) -> Result<<Self::MemDevice as ComputeMemory<F>>::FSliceMut<'b>, Error>;

	fn copy_host_to_device(
		host_slice: <Self::MemHost as ComputeMemory<F>>::FSlice<'_>,
		device_slice: &mut <Self::MemDevice as ComputeMemory<F>>::FSliceMut<'_>,
	);
	fn copy_device_to_host(
		device_slice: <Self::MemDevice as ComputeMemory<F>>::FSlice<'_>,
		host_slice: &mut <Self::MemHost as ComputeMemory<F>>::FSliceMut<'_>,
	);
}
