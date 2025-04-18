// Copyright 2025 Irreducible Inc.

use std::ops::RangeBounds;

use crate::alloc::ComputeBufferAllocator;

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

/// A trait defining the memory types and slice representations for a compute device.
///
/// This trait specifies the slice types and minimum slice length requirements for memory
/// that can be used with a particular compute device (e.g. CPU, GPU).
///
/// The generic parameter `F` represents the element type stored in the memory.
///
/// # Type Parameters
///
/// - `FSlice<'a>`: The immutable slice type for this memory, must implement `OpaqueSlice<F>`
/// - `FSliceMut<'a>`: The mutable slice type for this memory, must implement `OpaqueSlice<F>`
///
/// # Associated Constants
///
/// - `MIN_SLICE_LEN`: The minimum allowed length for slices of this memory type. Memory
///   operations must use slice lengths that are multiples of this value.
pub trait ComputeMemoryTypes<F> {
	type FSlice<'a>: OpaqueSlice<F>;
	type FSliceMut<'a>: OpaqueSlice<F>;
	const MIN_SLICE_LEN: usize;
}

/// A trait defining memory operations that can be performed on compute device memory.
///
/// This trait provides a common interface for performing slice operations on memory,
/// regardless of whether it is host-accessible or device-accessible. The operations
/// include converting between mutable and immutable views, taking subslices, and
/// splitting slices.
///
/// # Type Parameters
///
/// - `F`: The element type stored in memory
/// - `MemTypes`: The memory types trait defining the slice representations
///
/// # Safety
///
/// Implementations must ensure:
/// - All slice operations maintain proper alignment and size requirements
/// - Mutable slices have exclusive access to their memory regions
/// - Range bounds in slice operations are valid for the underlying memory
/// - Split operations produce non-overlapping slices
pub trait ComputeMemoryOperations<F, MemTypes: ComputeMemoryTypes<F>> {
	/// Borrows a mutable memory slice as immutable.
	///
	/// This allows the immutable reference to be copied.
	fn as_const<'a>(data: &'a MemTypes::FSliceMut<'_>) -> MemTypes::FSlice<'a>;

	/// Borrows a subslice of an immutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of `MemTypes::MIN_SLICE_LEN`
	fn slice(data: MemTypes::FSlice<'_>, range: impl RangeBounds<usize>) -> MemTypes::FSlice<'_>;

	/// Borrows a subslice of a mutable memory slice.
	///
	/// ## Preconditions
	///
	/// - the range bounds must be multiples of `MemTypes::MIN_SLICE_LEN`
	fn slice_mut<'a>(
		data: &'a mut MemTypes::FSliceMut<'_>,
		range: impl RangeBounds<usize>,
	) -> MemTypes::FSliceMut<'a>;

	/// Splits a mutable slice into two disjoint subslices.
	///
	/// ## Preconditions
	///
	/// - `mid` must be a multiple of `MemTypes::MIN_SLICE_LEN`
	fn split_at_mut(
		data: MemTypes::FSliceMut<'_>,
		mid: usize,
	) -> (MemTypes::FSliceMut<'_>, MemTypes::FSliceMut<'_>);

	fn split_at_mut_borrowed<'a>(
		data: &'a mut MemTypes::FSliceMut<'_>,
		mid: usize,
	) -> (MemTypes::FSliceMut<'a>, MemTypes::FSliceMut<'a>) {
		let borrowed = Self::slice_mut(data, ..);
		Self::split_at_mut(borrowed, mid)
	}
}

/// A trait for host-side memory types that can be accessed as Rust slices.
///
/// This trait extends [`ComputeMemoryTypes`] to require that the slice types can be converted
/// to standard Rust slices via [`AsRef`] and [`AsMut`]. This enables direct access to the
/// underlying memory on the host side.
///
/// Host memory types are used for data that needs to be accessed directly by the CPU, as opposed
/// to device memory which may live in a separate address space (e.g. GPU memory).
pub trait ComputeMemoryTypesHost<F>: ComputeMemoryTypes<F>
where
	for<'a> Self::FSlice<'a>: AsRef<[F]>,
	for<'a> Self::FSliceMut<'a>: AsMut<[F]>,
{
}

/// A trait that defines a complete memory management suite for a compute device.
///
/// This trait combines host and device memory types, memory operations, and allocators into a
/// single abstraction that can be used to manage memory across the host-device boundary.
///
/// The type parameters are:
/// - `'a`: lifetime of host memory allocations
/// - `'b`: lifetime of device memory allocations  
/// - `F`: the field element type being stored
///
/// The associated types define:
/// - `MemTypesHost`: Memory types for host-accessible memory
/// - `MemTypesDevice`: Memory types for device-accessible memory
/// - `HostComputeMemoryOperations`: Operations on host memory
/// - `HostAllocator`: Allocator for host memory
/// - `DeviceComputeMemoryOperations`: Operations on device memory  
/// - `DeviceAllocator`: Allocator for device memory
///
/// The trait provides methods to copy data between host and device memory.
pub trait ComputeMemorySuite<'a, 'b, F: Copy>
where
	for<'c> <<Self as ComputeMemorySuite<'a, 'b, F>>::MemTypesHost as ComputeMemoryTypes<F>>::FSlice<'c>:
		AsRef<[F]>,
	for<'c> <<Self as ComputeMemorySuite<'a, 'b, F>>::MemTypesHost as ComputeMemoryTypes<F>>::FSliceMut<'c>:
		AsMut<[F]>,
{
	type MemTypesHost: ComputeMemoryTypesHost<F>;
	type MemTypesDevice: ComputeMemoryTypes<F>;
	type HostComputeMemoryOperations: ComputeMemoryOperations<F, Self::MemTypesHost>;
	type HostAllocator: ComputeBufferAllocator<
		'a,
		F,
		Self::MemTypesHost,
		Self::HostComputeMemoryOperations,
	>;
	type DeviceComputeMemoryOperations: ComputeMemoryOperations<F, Self::MemTypesDevice>;
	type DeviceAllocator: ComputeBufferAllocator<
		'b,
		F,
		Self::MemTypesDevice,
		Self::DeviceComputeMemoryOperations,
	>;

	fn copy_host_to_device(
		host_slice: <Self::MemTypesHost as ComputeMemoryTypes<F>>::FSlice<'_>,
		device_slice: &mut <Self::MemTypesDevice as ComputeMemoryTypes<F>>::FSliceMut<'_>,
	);
	fn copy_device_to_host(
		device_slice: <Self::MemTypesDevice as ComputeMemoryTypes<F>>::FSlice<'_>,
		host_slice: &mut <Self::MemTypesHost as ComputeMemoryTypes<F>>::FSliceMut<'_>,
	);
}
