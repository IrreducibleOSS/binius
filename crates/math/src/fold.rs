// Copyright 2024-2025 Irreducible Inc.

use core::slice;
use std::{any::TypeId, cmp::min};

use binius_field::{
	arch::byte_sliced::ByteSlicedAES32x128b,
	packed::{get_packed_slice, set_packed_slice_unchecked},
	BinaryField1b, ByteSlicedAES32x16b, ByteSlicedAES32x32b, ByteSlicedAES32x64b,
	ByteSlicedAES32x8b, ExtensionField, Field, PackedBinaryField128x1b, PackedBinaryField16x1b,
	PackedBinaryField256x1b, PackedBinaryField32x1b, PackedBinaryField512x1b,
	PackedBinaryField64x1b, PackedBinaryField8x1b, PackedField,
};
use binius_maybe_rayon::{
	iter::{IndexedParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use binius_utils::bail;
use bytemuck::Pod;
use itertools::max;

use crate::Error;

/// Execute the fold operation.
///
/// Every consequent `1 << log_query_size` scalar values are dot-producted with the corresponding
/// query elements. The result is stored in the `output` slice of packed values.
pub fn fold<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [PE],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if log_evals_size < log_query_size {
		bail!(Error::IncorrectQuerySize {
			expected: log_query_size
		});
	}
	if out.len() != 1 << ((log_evals_size - log_query_size).saturating_sub(PE::LOG_WIDTH)) {
		bail!(Error::IncorrectOutputPolynomialSize {
			expected: log_evals_size - log_query_size,
		});
	}

	// Try execute the optimized version for 1-bit values if possible
	if TypeId::of::<P::Scalar>() == TypeId::of::<BinaryField1b>()
		&& fold_1bit_evals(evals, log_evals_size, query, log_query_size, out)
	{
		return Ok(());
	}

	fold_fallback(evals, log_evals_size, query, log_query_size, out);

	Ok(())
}

/// A marker trait that the slice of packed values can be iterated as a sequence of bytes.
/// The order of the iteration by BinaryField1b subfield elements and bits within iterated bytes must
/// be the same.
///
/// # Safety
/// The implementor must ensure that the cast of the slice of packed values to the slice of bytes
/// is safe and preserves the order of the 1-bit elements.
#[allow(unused)]
unsafe trait SequentialBytes: Pod {}

unsafe impl SequentialBytes for PackedBinaryField8x1b {}
unsafe impl SequentialBytes for PackedBinaryField16x1b {}
unsafe impl SequentialBytes for PackedBinaryField32x1b {}
unsafe impl SequentialBytes for PackedBinaryField64x1b {}
unsafe impl SequentialBytes for PackedBinaryField128x1b {}
unsafe impl SequentialBytes for PackedBinaryField256x1b {}
unsafe impl SequentialBytes for PackedBinaryField512x1b {}

/// Returns true if T implements `SequentialBytes` trait.
/// Use a hack that exploits that array copying is optimized for the `Copy` types.
/// Unfortunately there is no more proper way to perform this check this in Rust at runtime.
fn is_sequential_bytes<T>() -> bool {
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
fn can_iterate_bytes<P: PackedField>() -> bool {
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
	($packed_type:ty, $data:ident, $f:ident) => {
		assert_eq!(TypeId::of::<$packed_type>(), TypeId::of::<P>());

		// Safety: the cast is safe because the type is checked by arm statement
		let data =
			unsafe { slice::from_raw_parts($data.as_ptr() as *const $packed_type, $data.len()) };
		for value in data.iter() {
			for i in 0..<$packed_type>::BYTES {
				// Safety: j is less than `ByteSlicedAES32x128b::BYTES`
				$f(unsafe { value.get_byte_unchecked(i) });
			}
		}
	};
}

/// Iterate over bytes of a slice of the packed values.
fn iterate_bytes<P: PackedField>(data: &[P], mut f: impl FnMut(u8)) {
	if is_sequential_bytes::<P>() {
		// Safety: `P` implements `SequentialBytes` trait, so the following cast is safe
		// and preserves the order.
		let bytes = unsafe {
			std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
		};
		for byte in bytes.iter() {
			f(*byte);
		}
	} else {
		// Note: add more byte sliced types here as soon as they are added
		match TypeId::of::<P>() {
			x if x == TypeId::of::<ByteSlicedAES32x128b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x128b, data, f);
			}
			x if x == TypeId::of::<ByteSlicedAES32x64b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x64b, data, f);
			}
			x if x == TypeId::of::<ByteSlicedAES32x32b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x32b, data, f);
			}
			x if x == TypeId::of::<ByteSlicedAES32x16b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x16b, data, f);
			}
			x if x == TypeId::of::<ByteSlicedAES32x8b>() => {
				iterate_byte_sliced!(ByteSlicedAES32x8b, data, f);
			}
			_ => unreachable!("packed field doesn't support byte iteration"),
		}
	}
}

/// Optimized version for 1-bit values with query size 0-2
fn fold_1bit_evals_small_query<P, PE, const LOG_QUERY_SIZE: usize>(
	evals: &[P],
	query: &[PE],
	out: &mut [PE],
) -> bool
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if LOG_QUERY_SIZE >= 3 {
		return false;
	}
	let chunk_size = 1
		<< max(&[
			10,
			(P::LOG_WIDTH + LOG_QUERY_SIZE).saturating_sub(PE::LOG_WIDTH),
			PE::LOG_WIDTH,
		])
		.unwrap();
	if out.len() % chunk_size != 0 {
		return false;
	}

	if P::WIDTH << LOG_QUERY_SIZE > chunk_size << PE::LOG_WIDTH {
		return false;
	}

	// Cache the table for a single evaluation
	let cached_table = (0..1 << (1 << LOG_QUERY_SIZE))
		.map(|i| {
			let mut result = PE::Scalar::ZERO;
			for j in 0..1 << LOG_QUERY_SIZE {
				if i >> j & 1 == 1 {
					result += get_packed_slice(query, j);
				}
			}
			result
		})
		.collect::<Vec<_>>();

	out.par_chunks_mut(chunk_size)
		.enumerate()
		.for_each(|(index, chunk)| {
			let input_offset =
				((index * chunk_size) << (LOG_QUERY_SIZE + PE::LOG_WIDTH)) / P::WIDTH;
			let input_end =
				(((index + 1) * chunk_size) << (LOG_QUERY_SIZE + PE::LOG_WIDTH)) / P::WIDTH;

			let mut current_index = 0;
			iterate_bytes(&evals[input_offset..input_end], |byte| {
				let mask = (1 << (1 << LOG_QUERY_SIZE)) - 1;
				let values_in_byte = 1 << (3 - LOG_QUERY_SIZE);
				for k in 0..values_in_byte {
					let index = (byte >> (k * (1 << LOG_QUERY_SIZE))) & mask;
					// Safety: `i` is less than `chunk_size`
					unsafe {
						set_packed_slice_unchecked(
							chunk,
							current_index + k,
							cached_table[index as usize],
						);
					}
				}

				current_index += values_in_byte;
			});
		});

	true
}

/// Optimized version for 1-bit values with medium log query size (3-6)
fn fold_1bit_evals_medium_query<P, PE, const LOG_QUERY_SIZE: usize>(
	evals: &[P],
	query: &[PE],
	out: &mut [PE],
) -> bool
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if LOG_QUERY_SIZE < 3 {
		return false;
	}
	let chunk_size = 1
		<< max(&[
			10,
			(P::LOG_WIDTH + LOG_QUERY_SIZE).saturating_sub(PE::LOG_WIDTH),
			PE::LOG_WIDTH,
		])
		.unwrap();
	if out.len() % chunk_size != 0 {
		return false;
	}

	let log_tables_count = LOG_QUERY_SIZE - 3;
	let tables_count = 1 << log_tables_count;
	let cached_tables = (0..tables_count)
		.map(|i| {
			(0..256)
				.map(|j| {
					let mut result = PE::Scalar::ZERO;
					for k in 0..8 {
						if j >> k & 1 == 1 {
							result += get_packed_slice(query, (i << 3) | k);
						}
					}
					result
				})
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();

	out.par_chunks_mut(chunk_size)
		.enumerate()
		.for_each(|(index, chunk)| {
			let input_offset =
				((index * chunk_size) << (LOG_QUERY_SIZE + PE::LOG_WIDTH)) / P::WIDTH;
			let input_end =
				(((index + 1) * chunk_size) << (LOG_QUERY_SIZE + PE::LOG_WIDTH)) / P::WIDTH;

			let mut current_value = PE::Scalar::ZERO;
			let mut current_table = 0;
			let mut current_index = 0;
			iterate_bytes(&evals[input_offset..input_end], |byte| {
				current_value += cached_tables[current_table][byte as usize];
				current_table += 1;

				if current_table == tables_count {
					// Safety: `i` is less than `chunk_size`
					unsafe {
						set_packed_slice_unchecked(chunk, current_index, current_value);
					}
					current_table = 0;
					current_index += 1;
					current_value = PE::Scalar::ZERO;
				}
			});
		});

	true
}

/// Try run optimized version for 1-bit values.
/// Returns true in case when the optimized calculation was performed.
/// Otherwise, returns false and the fallback should be used.
fn fold_1bit_evals<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [PE],
) -> bool
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if log_evals_size < P::LOG_WIDTH {
		return false;
	}

	if !can_iterate_bytes::<P>() {
		return false;
	}

	// We pass log_query_size_ as a generic parameter because that allows a compiler producing
	// more efficient code in the tight loops inside the functions.
	match log_query_size {
		0 => fold_1bit_evals_small_query::<P, PE, 0>(evals, query, out),
		1 => fold_1bit_evals_small_query::<P, PE, 1>(evals, query, out),
		2 => fold_1bit_evals_small_query::<P, PE, 2>(evals, query, out),
		3 => fold_1bit_evals_medium_query::<P, PE, 3>(evals, query, out),
		4 => fold_1bit_evals_medium_query::<P, PE, 4>(evals, query, out),
		5 => fold_1bit_evals_medium_query::<P, PE, 5>(evals, query, out),
		6 => fold_1bit_evals_medium_query::<P, PE, 6>(evals, query, out),
		7 => fold_1bit_evals_medium_query::<P, PE, 7>(evals, query, out),
		_ => false,
	}
}

/// Fallback implementation for fold that can be executed for any field types and sizes.
fn fold_fallback<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [PE],
) where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	const CHUNK_SIZE: usize = 1 << 10;
	let packed_result_evals = out;
	packed_result_evals
		.par_chunks_mut(CHUNK_SIZE)
		.enumerate()
		.for_each(|(i, packed_result_evals)| {
			for (k, packed_result_eval) in packed_result_evals.iter_mut().enumerate() {
				let offset = i * CHUNK_SIZE;
				for j in 0..min(PE::WIDTH, 1 << (log_evals_size - log_query_size)) {
					let index = ((offset + k) << PE::LOG_WIDTH) | j;

					let offset = index << log_query_size;

					let mut result_eval = PE::Scalar::ZERO;
					for (t, query_expansion) in PackedField::iter_slice(query)
						.take(1 << log_query_size)
						.enumerate()
					{
						result_eval += query_expansion * get_packed_slice(evals, t + offset);
					}

					// Safety: `j` < `PE::WIDTH`
					unsafe {
						packed_result_eval.set_unchecked(j, result_eval);
					}
				}
			}
		});
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{
		packed::set_packed_slice, PackedBinaryField16x32b, PackedBinaryField16x8b,
		PackedBinaryField4x1b, PackedBinaryField512x1b,
	};
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;

	#[test]
	fn test_sequential_bits() {
		assert!(is_sequential_bytes::<PackedBinaryField8x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField16x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField32x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField64x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField128x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField256x1b>());
		assert!(is_sequential_bytes::<PackedBinaryField512x1b>());

		assert!(!is_sequential_bytes::<PackedBinaryField4x1b>());
		assert!(!is_sequential_bytes::<PackedBinaryField16x8b>());
	}

	fn fold_reference<P, PE>(
		evals: &[P],
		log_evals_size: usize,
		query: &[PE],
		log_query_size: usize,
		out: &mut [PE],
	) where
		P: PackedField,
		PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
	{
		for i in 0..1 << (log_evals_size - log_query_size) {
			let mut result = PE::Scalar::ZERO;
			for j in 0..1 << log_query_size {
				result +=
					get_packed_slice(query, j) * get_packed_slice(evals, (i << log_query_size) | j);
			}

			set_packed_slice(out, i, result);
		}
	}

	fn check_fold<P, PE>(evals: &[P], log_evals_size: usize, query: &[PE], log_query_size: usize)
	where
		P: PackedField,
		PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
	{
		let mut reference_out =
			vec![PE::zero(); (1usize << (log_evals_size - log_query_size)).div_ceil(PE::WIDTH)];
		let mut out = reference_out.clone();

		fold(evals, log_evals_size, query, log_query_size, &mut out).unwrap();
		fold_reference(evals, log_evals_size, query, log_query_size, &mut reference_out);

		for i in 0..1 << (log_evals_size - log_query_size) {
			assert_eq!(get_packed_slice(&out, i), get_packed_slice(&reference_out, i));
		}
	}

	#[test]
	fn test_1b_small_poly_query_log_size_0() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold(&evals, 0, &query, 0);
	}

	#[test]
	fn test_1b_small_poly_query_log_size_1() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold(&evals, 2, &query, 1);
	}

	#[test]
	fn test_1b_small_poly_query_log_size_7() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold(&evals, 7, &query, 7);
	}

	#[test]
	fn test_1b_many_evals() {
		const LOG_EVALS_SIZE: usize = 14;
		let mut rng = StdRng::seed_from_u64(1);
		let evals = repeat_with(|| PackedBinaryField128x1b::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE)
			.collect::<Vec<_>>();
		let query = vec![PackedBinaryField512x1b::random(&mut rng)];

		for log_query_size in 0..10 {
			check_fold(
				&evals,
				LOG_EVALS_SIZE + PackedBinaryField128x1b::LOG_WIDTH,
				&query,
				log_query_size,
			);
		}
	}

	#[test]
	fn test_8b_small_poly() {
		const LOG_EVALS_SIZE: usize = 5;
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| PackedBinaryField16x8b::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE)
			.collect::<Vec<_>>();
		let query = repeat_with(|| PackedBinaryField16x32b::random(&mut rng))
			.take(1 << 8)
			.collect::<Vec<_>>();

		for log_query_size in 0..8 {
			check_fold(
				&evals,
				LOG_EVALS_SIZE + PackedBinaryField16x8b::LOG_WIDTH,
				&query,
				log_query_size,
			);
		}
	}

	#[test]
	fn test_8b_many_evals() {
		const LOG_EVALS_SIZE: usize = 13;
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| PackedBinaryField16x8b::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE)
			.collect::<Vec<_>>();
		let query = repeat_with(|| PackedBinaryField16x32b::random(&mut rng))
			.take(1 << 8)
			.collect::<Vec<_>>();

		for log_query_size in 0..8 {
			check_fold(
				&evals,
				LOG_EVALS_SIZE + PackedBinaryField16x8b::LOG_WIDTH,
				&query,
				log_query_size,
			);
		}
	}
}
