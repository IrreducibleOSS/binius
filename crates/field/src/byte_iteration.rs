// Copyright 2023-2025 Irreducible Inc.

use std::any::TypeId;

use bytemuck::Pod;

use crate::{
	packed::get_packed_slice, AESTowerField128b, AESTowerField16b, AESTowerField32b,
	AESTowerField64b, AESTowerField8b, BinaryField128b, BinaryField128bPolyval, BinaryField16b,
	BinaryField32b, BinaryField64b, BinaryField8b, ByteSlicedAES32x128b, ByteSlicedAES32x16b,
	ByteSlicedAES32x32b, ByteSlicedAES32x64b, ByteSlicedAES32x8b, Field,
	PackedAESBinaryField16x16b, PackedAESBinaryField16x32b, PackedAESBinaryField16x8b,
	PackedAESBinaryField1x128b, PackedAESBinaryField1x16b, PackedAESBinaryField1x32b,
	PackedAESBinaryField1x64b, PackedAESBinaryField1x8b, PackedAESBinaryField2x128b,
	PackedAESBinaryField2x16b, PackedAESBinaryField2x32b, PackedAESBinaryField2x64b,
	PackedAESBinaryField2x8b, PackedAESBinaryField32x16b, PackedAESBinaryField32x8b,
	PackedAESBinaryField4x128b, PackedAESBinaryField4x16b, PackedAESBinaryField4x32b,
	PackedAESBinaryField4x64b, PackedAESBinaryField4x8b, PackedAESBinaryField64x8b,
	PackedAESBinaryField8x16b, PackedAESBinaryField8x64b, PackedAESBinaryField8x8b,
	PackedBinaryField128x1b, PackedBinaryField128x2b, PackedBinaryField128x4b,
	PackedBinaryField16x16b, PackedBinaryField16x1b, PackedBinaryField16x2b,
	PackedBinaryField16x32b, PackedBinaryField16x4b, PackedBinaryField16x8b,
	PackedBinaryField1x128b, PackedBinaryField1x16b, PackedBinaryField1x32b,
	PackedBinaryField1x64b, PackedBinaryField1x8b, PackedBinaryField256x1b,
	PackedBinaryField256x2b, PackedBinaryField2x128b, PackedBinaryField2x16b,
	PackedBinaryField2x32b, PackedBinaryField2x4b, PackedBinaryField2x64b, PackedBinaryField2x8b,
	PackedBinaryField32x16b, PackedBinaryField32x1b, PackedBinaryField32x2b,
	PackedBinaryField32x4b, PackedBinaryField32x8b, PackedBinaryField4x128b,
	PackedBinaryField4x16b, PackedBinaryField4x2b, PackedBinaryField4x32b, PackedBinaryField4x4b,
	PackedBinaryField4x64b, PackedBinaryField4x8b, PackedBinaryField512x1b, PackedBinaryField64x1b,
	PackedBinaryField64x2b, PackedBinaryField64x4b, PackedBinaryField64x8b, PackedBinaryField8x16b,
	PackedBinaryField8x1b, PackedBinaryField8x2b, PackedBinaryField8x32b, PackedBinaryField8x4b,
	PackedBinaryField8x64b, PackedBinaryField8x8b, PackedBinaryPolyval1x128b,
	PackedBinaryPolyval2x128b, PackedBinaryPolyval4x128b, PackedField,
};

/// A marker trait that the slice of packed values can be iterated as a sequence of bytes.
/// The order of the iteration by BinaryField1b subfield elements and bits within iterated bytes must
/// be the same.
///
/// # Safety
/// The implementor must ensure that the cast of the slice of packed values to the slice of bytes
/// is safe and preserves the order of the 1-bit elements.
#[allow(unused)]
unsafe trait SequentialBytes: Pod {}

unsafe impl SequentialBytes for BinaryField8b {}
unsafe impl SequentialBytes for BinaryField16b {}
unsafe impl SequentialBytes for BinaryField32b {}
unsafe impl SequentialBytes for BinaryField64b {}
unsafe impl SequentialBytes for BinaryField128b {}

unsafe impl SequentialBytes for PackedBinaryField8x1b {}
unsafe impl SequentialBytes for PackedBinaryField16x1b {}
unsafe impl SequentialBytes for PackedBinaryField32x1b {}
unsafe impl SequentialBytes for PackedBinaryField64x1b {}
unsafe impl SequentialBytes for PackedBinaryField128x1b {}
unsafe impl SequentialBytes for PackedBinaryField256x1b {}
unsafe impl SequentialBytes for PackedBinaryField512x1b {}

unsafe impl SequentialBytes for PackedBinaryField4x2b {}
unsafe impl SequentialBytes for PackedBinaryField8x2b {}
unsafe impl SequentialBytes for PackedBinaryField16x2b {}
unsafe impl SequentialBytes for PackedBinaryField32x2b {}
unsafe impl SequentialBytes for PackedBinaryField64x2b {}
unsafe impl SequentialBytes for PackedBinaryField128x2b {}
unsafe impl SequentialBytes for PackedBinaryField256x2b {}

unsafe impl SequentialBytes for PackedBinaryField2x4b {}
unsafe impl SequentialBytes for PackedBinaryField4x4b {}
unsafe impl SequentialBytes for PackedBinaryField8x4b {}
unsafe impl SequentialBytes for PackedBinaryField16x4b {}
unsafe impl SequentialBytes for PackedBinaryField32x4b {}
unsafe impl SequentialBytes for PackedBinaryField64x4b {}
unsafe impl SequentialBytes for PackedBinaryField128x4b {}

unsafe impl SequentialBytes for PackedBinaryField1x8b {}
unsafe impl SequentialBytes for PackedBinaryField2x8b {}
unsafe impl SequentialBytes for PackedBinaryField4x8b {}
unsafe impl SequentialBytes for PackedBinaryField8x8b {}
unsafe impl SequentialBytes for PackedBinaryField16x8b {}
unsafe impl SequentialBytes for PackedBinaryField32x8b {}
unsafe impl SequentialBytes for PackedBinaryField64x8b {}

unsafe impl SequentialBytes for PackedBinaryField1x16b {}
unsafe impl SequentialBytes for PackedBinaryField2x16b {}
unsafe impl SequentialBytes for PackedBinaryField4x16b {}
unsafe impl SequentialBytes for PackedBinaryField8x16b {}
unsafe impl SequentialBytes for PackedBinaryField16x16b {}
unsafe impl SequentialBytes for PackedBinaryField32x16b {}

unsafe impl SequentialBytes for PackedBinaryField1x32b {}
unsafe impl SequentialBytes for PackedBinaryField2x32b {}
unsafe impl SequentialBytes for PackedBinaryField4x32b {}
unsafe impl SequentialBytes for PackedBinaryField8x32b {}
unsafe impl SequentialBytes for PackedBinaryField16x32b {}

unsafe impl SequentialBytes for PackedBinaryField1x64b {}
unsafe impl SequentialBytes for PackedBinaryField2x64b {}
unsafe impl SequentialBytes for PackedBinaryField4x64b {}
unsafe impl SequentialBytes for PackedBinaryField8x64b {}

unsafe impl SequentialBytes for PackedBinaryField1x128b {}
unsafe impl SequentialBytes for PackedBinaryField2x128b {}
unsafe impl SequentialBytes for PackedBinaryField4x128b {}

unsafe impl SequentialBytes for AESTowerField8b {}
unsafe impl SequentialBytes for AESTowerField16b {}
unsafe impl SequentialBytes for AESTowerField32b {}
unsafe impl SequentialBytes for AESTowerField64b {}
unsafe impl SequentialBytes for AESTowerField128b {}

unsafe impl SequentialBytes for PackedAESBinaryField1x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField2x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField4x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField8x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField16x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField32x8b {}
unsafe impl SequentialBytes for PackedAESBinaryField64x8b {}

unsafe impl SequentialBytes for PackedAESBinaryField1x16b {}
unsafe impl SequentialBytes for PackedAESBinaryField2x16b {}
unsafe impl SequentialBytes for PackedAESBinaryField4x16b {}
unsafe impl SequentialBytes for PackedAESBinaryField8x16b {}
unsafe impl SequentialBytes for PackedAESBinaryField16x16b {}
unsafe impl SequentialBytes for PackedAESBinaryField32x16b {}

unsafe impl SequentialBytes for PackedAESBinaryField1x32b {}
unsafe impl SequentialBytes for PackedAESBinaryField2x32b {}
unsafe impl SequentialBytes for PackedAESBinaryField4x32b {}
unsafe impl SequentialBytes for PackedAESBinaryField16x32b {}

unsafe impl SequentialBytes for PackedAESBinaryField1x64b {}
unsafe impl SequentialBytes for PackedAESBinaryField2x64b {}
unsafe impl SequentialBytes for PackedAESBinaryField4x64b {}
unsafe impl SequentialBytes for PackedAESBinaryField8x64b {}

unsafe impl SequentialBytes for PackedAESBinaryField1x128b {}
unsafe impl SequentialBytes for PackedAESBinaryField2x128b {}
unsafe impl SequentialBytes for PackedAESBinaryField4x128b {}

unsafe impl SequentialBytes for BinaryField128bPolyval {}

unsafe impl SequentialBytes for PackedBinaryPolyval1x128b {}
unsafe impl SequentialBytes for PackedBinaryPolyval2x128b {}
unsafe impl SequentialBytes for PackedBinaryPolyval4x128b {}

/// Returns true if T implements `SequentialBytes` trait.
/// Use a hack that exploits that array copying is optimized for the `Copy` types.
/// Unfortunately there is no more proper way to perform this check this in Rust at runtime.
#[inline(always)]
#[allow(clippy::redundant_clone)] // this is intentional in this method
pub fn is_sequential_bytes<T>() -> bool {
	struct X<U>(bool, std::marker::PhantomData<U>);

	impl<U> Clone for X<U> {
		fn clone(&self) -> Self {
			Self(false, std::marker::PhantomData)
		}
	}

	impl<U: SequentialBytes> Copy for X<U> {}

	let value = [X::<T>(true, std::marker::PhantomData)];
	let cloned = value.clone();

	cloned[0].0
}

/// Returns if we can iterate over bytes, each representing 8 1-bit values.
pub fn can_iterate_bytes<P: PackedField>() -> bool {
	// Packed fields with sequential byte order
	if is_sequential_bytes::<P>() {
		return true;
	}

	// Byte-sliced fields
	// Note: add more byte sliced types here as soon as they are added
	match TypeId::of::<P>() {
		x if x == TypeId::of::<ByteSlicedAES32x128b>() => true,
		x if x == TypeId::of::<ByteSlicedAES32x64b>() => true,
		x if x == TypeId::of::<ByteSlicedAES32x32b>() => true,
		x if x == TypeId::of::<ByteSlicedAES32x16b>() => true,
		x if x == TypeId::of::<ByteSlicedAES32x8b>() => true,
		_ => false,
	}
}

/// Helper macro to generate the iteration over bytes for byte-sliced types.
macro_rules! iterate_byte_sliced {
	($packed_type:ty, $data:ident, $callback:ident) => {
		assert_eq!(TypeId::of::<$packed_type>(), TypeId::of::<P>());

		// Safety: the cast is safe because the type is checked by arm statement
		let data = unsafe {
			std::slice::from_raw_parts($data.as_ptr() as *const $packed_type, $data.len())
		};
		let iter = data.iter().flat_map(|value| {
			(0..<$packed_type>::BYTES).map(move |i| unsafe { value.get_byte_unchecked(i) })
		});

		$callback.call(iter);
	};
}

/// Callback for byte iteration.
/// We can't return different types from the `iterate_bytes` and Fn traits don't support associated types
/// that's why we use a callback with a generic function.
pub trait ByteIteratorCallback {
	fn call(&mut self, iter: impl Iterator<Item = u8>);
}

/// Iterate over bytes of a slice of the packed values.
/// The method panics if the packed field doesn't support byte iteration, so use `can_iterate_bytes` to check it.
#[inline(always)]
pub fn iterate_bytes<P: PackedField>(data: &[P], callback: &mut impl ByteIteratorCallback) {
	if is_sequential_bytes::<P>() {
		// Safety: `P` implements `SequentialBytes` trait, so the following cast is safe
		// and preserves the order.
		let bytes = unsafe {
			std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
		};
		callback.call(bytes.iter().copied());
	} else {
		// Note: add more byte sliced types here as soon as they are added
		match TypeId::of::<P>() {
			x if x == TypeId::of::<ByteSlicedAES32x128b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x128b, data, callback);
			}
			x if x == TypeId::of::<ByteSlicedAES32x64b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x64b, data, callback);
			}
			x if x == TypeId::of::<ByteSlicedAES32x32b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x32b, data, callback);
			}
			x if x == TypeId::of::<ByteSlicedAES32x16b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x16b, data, callback);
			}
			x if x == TypeId::of::<ByteSlicedAES32x8b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x8b, data, callback);
			}
			_ => unreachable!("packed field doesn't support byte iteration"),
		}
	}
}

/// Scalars collection abstraction.
/// This trait is used to abstract over different types of collections of field elements.
pub trait ScalarsCollection<T> {
	fn len(&self) -> usize;
	fn get(&self, i: usize) -> T;
	fn is_empty(&self) -> bool {
		self.len() == 0
	}
}

impl<F: Field> ScalarsCollection<F> for &[F] {
	#[inline(always)]
	fn len(&self) -> usize {
		<[F]>::len(self)
	}

	#[inline(always)]
	fn get(&self, i: usize) -> F {
		self[i]
	}
}

pub struct PackedSlice<'a, P: PackedField> {
	slice: &'a [P],
	len: usize,
}

impl<'a, P: PackedField> PackedSlice<'a, P> {
	#[inline(always)]
	pub const fn new(slice: &'a [P], len: usize) -> Self {
		Self { slice, len }
	}
}

impl<P: PackedField> ScalarsCollection<P::Scalar> for PackedSlice<'_, P> {
	#[inline(always)]
	fn len(&self) -> usize {
		self.len
	}

	#[inline(always)]
	fn get(&self, i: usize) -> P::Scalar {
		get_packed_slice(self.slice, i)
	}
}

// /// Create a lookup table for partial sums of 8 consequent elements with coefficients corresponding to bits in a byte.
// /// The lookup table has the following structure:
// /// [
// ///     partial_sum_chunk_0_7_byte_0, partial_sum_chunk_0_7_byte_1, ..., partial_sum_chunk_0_7_byte_255,
// ///     partial_sum_chunk_8_15_byte_0, partial_sum_chunk_8_15_byte_1, ..., partial_sum_chunk_8_15_byte_255,
// ///    ...
// /// ]
// pub fn create_partial_sums_lookup_tables<P: PackedField>(
// 	values: impl ScalarsCollection<P>,
// ) -> Vec<P> {
// 	let len = values.len();
// 	assert!(len % 8 == 0);

// 	let mut result = Vec::with_capacity(len * 32); // 256 / 8 = 32 per chunk

// 	for chunk_start in (0..len).step_by(8) {
// 		let mut sums = [P::zero(); 256];

// 		for j in 0..8 {
// 			let value = values.get(chunk_start + j);
// 			let mask = 1 << j;
// 			for i in (mask..256).step_by(mask * 2) {
// 				for k in 0..mask {
// 					sums[i + k] += value;
// 				}
// 			}
// 		}

// 		result.extend_from_slice(&sums);
// 	}

// 	result
// }

/// Create a lookup table for partial sums of 8 consequent elements with coefficients corresponding to bits in a byte.
/// The lookup table has the following structure:
/// [
///     partial_sum_chunk_0_7_byte_0, partial_sum_chunk_0_7_byte_1, ..., partial_sum_chunk_0_7_byte_255,
///     partial_sum_chunk_8_15_byte_0, partial_sum_chunk_8_15_byte_1, ..., partial_sum_chunk_8_15_byte_255,
///    ...
/// ]
pub fn create_partial_sums_lookup_tables<P: PackedField>(
	values: impl ScalarsCollection<P>,
) -> Vec<P> {
	let len = values.len();
	assert!(len % 8 == 0);

	let mut result = Vec::with_capacity(len * 32);
	result.resize(result.capacity(), P::zero());

	for (chunk_idx, chunk_start) in (0..len).step_by(8).enumerate() {
		let sums = &mut result[chunk_idx * 256..(chunk_idx + 1) * 256];

		for j in 0..8 {
			let value = values.get(chunk_start + j);
			let mask = 1 << j;
			for i in (mask..256).step_by(mask * 2) {
				for k in 0..mask {
					sums[i + k] += value;
				}
			}
		}
	}

	result
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{PackedBinaryField1x1b, PackedBinaryField2x1b, PackedBinaryField4x1b};

	#[test]
	fn test_sequential_bits() {
		assert!(is_sequential_bytes::<BinaryField8b>());
		assert!(is_sequential_bytes::<BinaryField16b>());
		assert!(is_sequential_bytes::<BinaryField32b>());
		assert!(is_sequential_bytes::<BinaryField64b>());
		assert!(is_sequential_bytes::<BinaryField128b>());

		assert!(is_sequential_bytes::<PackedBinaryField8x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField64x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField128x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField256x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField512x1b>());

		assert!(is_sequential_bytes::<PackedBinaryField4x2b>());
		assert!(is_sequential_bytes::<PackedBinaryField8x2b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x2b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x2b>());
		assert!(is_sequential_bytes::<PackedBinaryField64x2b>());
		assert!(is_sequential_bytes::<PackedBinaryField128x2b>());
		assert!(is_sequential_bytes::<PackedBinaryField256x2b>());

		assert!(is_sequential_bytes::<PackedBinaryField2x4b>());
		assert!(is_sequential_bytes::<PackedBinaryField4x4b>());
		assert!(is_sequential_bytes::<PackedBinaryField8x4b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x4b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x4b>());
		assert!(is_sequential_bytes::<PackedBinaryField64x4b>());
		assert!(is_sequential_bytes::<PackedBinaryField128x4b>());

		assert!(is_sequential_bytes::<PackedBinaryField1x8b>());
		assert!(is_sequential_bytes::<PackedBinaryField2x8b>());
		assert!(is_sequential_bytes::<PackedBinaryField4x8b>());
		assert!(is_sequential_bytes::<PackedBinaryField8x8b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x8b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x8b>());
		assert!(is_sequential_bytes::<PackedBinaryField64x8b>());

		assert!(is_sequential_bytes::<PackedBinaryField1x16b>());
		assert!(is_sequential_bytes::<PackedBinaryField2x16b>());
		assert!(is_sequential_bytes::<PackedBinaryField4x16b>());
		assert!(is_sequential_bytes::<PackedBinaryField8x16b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x16b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x16b>());

		assert!(is_sequential_bytes::<PackedBinaryField1x32b>());
		assert!(is_sequential_bytes::<PackedBinaryField2x32b>());
		assert!(is_sequential_bytes::<PackedBinaryField4x32b>());
		assert!(is_sequential_bytes::<PackedBinaryField8x32b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x32b>());

		assert!(is_sequential_bytes::<PackedBinaryField1x64b>());
		assert!(is_sequential_bytes::<PackedBinaryField2x64b>());
		assert!(is_sequential_bytes::<PackedBinaryField4x64b>());
		assert!(is_sequential_bytes::<PackedBinaryField8x64b>());

		assert!(is_sequential_bytes::<PackedBinaryField1x128b>());
		assert!(is_sequential_bytes::<PackedBinaryField2x128b>());
		assert!(is_sequential_bytes::<PackedBinaryField4x128b>());

		assert!(is_sequential_bytes::<AESTowerField8b>());
		assert!(is_sequential_bytes::<AESTowerField16b>());
		assert!(is_sequential_bytes::<AESTowerField32b>());
		assert!(is_sequential_bytes::<AESTowerField64b>());
		assert!(is_sequential_bytes::<AESTowerField128b>());

		assert!(is_sequential_bytes::<PackedAESBinaryField1x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField2x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField4x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField8x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField16x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField32x8b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField64x8b>());

		assert!(is_sequential_bytes::<PackedAESBinaryField1x16b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField2x16b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField4x16b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField8x16b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField16x16b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField32x16b>());

		assert!(is_sequential_bytes::<PackedAESBinaryField1x32b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField2x32b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField4x32b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField16x32b>());

		assert!(is_sequential_bytes::<PackedAESBinaryField1x64b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField2x64b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField4x64b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField8x64b>());

		assert!(is_sequential_bytes::<PackedAESBinaryField1x128b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField2x128b>());
		assert!(is_sequential_bytes::<PackedAESBinaryField4x128b>());

		assert!(is_sequential_bytes::<BinaryField128bPolyval>());

		assert!(is_sequential_bytes::<PackedBinaryPolyval1x128b>());
		assert!(is_sequential_bytes::<PackedBinaryPolyval2x128b>());
		assert!(is_sequential_bytes::<PackedBinaryPolyval4x128b>());

		assert!(!is_sequential_bytes::<PackedBinaryField1x1b>());
		assert!(!is_sequential_bytes::<PackedBinaryField2x1b>());
		assert!(!is_sequential_bytes::<PackedBinaryField4x1b>());
		assert!(!is_sequential_bytes::<ByteSlicedAES32x128b>());
		assert!(!is_sequential_bytes::<ByteSlicedAES32x64b>());
		assert!(!is_sequential_bytes::<ByteSlicedAES32x32b>());
		assert!(!is_sequential_bytes::<ByteSlicedAES32x16b>());
		assert!(!is_sequential_bytes::<ByteSlicedAES32x8b>());
	}

	#[test]
	fn test_partial_sums_basic() {
		let v1 = BinaryField32b::from(1);
		let v2 = BinaryField32b::from(2);
		let v3 = BinaryField32b::from(3);
		let v4 = BinaryField32b::from(4);
		let v5 = BinaryField32b::from(5);
		let v6 = BinaryField32b::from(6);
		let v7 = BinaryField32b::from(7);
		let v8 = BinaryField32b::from(8);

		let values = vec![v1, v2, v3, v4, v5, v6, v7, v8];

		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		// Check specific precomputed sums
		assert_eq!(lookup_table[0b0000_0000], BinaryField32b::from(0));
		assert_eq!(lookup_table[0b0000_0001], v1);
		assert_eq!(lookup_table[0b0000_0011], v1 + v2);
		assert_eq!(lookup_table[0b0000_0111], v1 + v2 + v3);
		assert_eq!(lookup_table[0b0000_1111], v1 + v2 + v3 + v4);
		assert_eq!(lookup_table[0b0001_1111], v1 + v2 + v3 + v4 + v5);
		assert_eq!(lookup_table[0b0011_1111], v1 + v2 + v3 + v4 + v5 + v6);
		assert_eq!(lookup_table[0b0111_1111], v1 + v2 + v3 + v4 + v5 + v6 + v7);
		assert_eq!(lookup_table[0b1111_1111], v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8);
	}

	#[test]
	fn test_partial_sums_all_zeros() {
		let values = vec![BinaryField32b::from(0); 8];
		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		for &l in lookup_table.iter().take(256) {
			assert_eq!(l, BinaryField32b::from(0));
		}
	}

	#[test]
	fn test_partial_sums_single_element() {
		let mut values = vec![BinaryField32b::from(0); 8];
		// Set only the fourth element (index 3)
		values[3] = BinaryField32b::from(10);

		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		// Only cases where the 4th bit is set should have non-zero sums
		assert_eq!(lookup_table[0b0000_0000], BinaryField32b::from(0));
		assert_eq!(lookup_table[0b0000_1000], BinaryField32b::from(10));
		assert_eq!(lookup_table[0b0000_1100], BinaryField32b::from(10));
		assert_eq!(lookup_table[0b0001_1000], BinaryField32b::from(10));
		assert_eq!(lookup_table[0b1111_1111], BinaryField32b::from(10));
	}

	#[test]
	fn test_partial_sums_alternating_values() {
		let v1 = BinaryField32b::from(10);
		let v2 = BinaryField32b::from(20);
		let v3 = BinaryField32b::from(30);
		let v4 = BinaryField32b::from(40);

		let zero = BinaryField32b::from(0);

		let values = vec![v1, zero, v2, zero, v3, zero, v4, zero];

		let lookup_table = create_partial_sums_lookup_tables(values.as_slice());

		assert_eq!(lookup_table.len(), 256);

		// Expect only the even indexed elements to contribute to the sum
		assert_eq!(lookup_table[0b0000_0000], zero);
		assert_eq!(lookup_table[0b0000_0001], v1);
		assert_eq!(lookup_table[0b0000_0101], v1 + v2);
		assert_eq!(lookup_table[0b0000_1111], v1 + v2);
		assert_eq!(lookup_table[0b1111_1111], v1 + v2 + v3 + v4);
	}
}
