// Copyright 2024-2025 Irreducible Inc.

use core::slice;
use std::{any::TypeId, cmp::min, mem::MaybeUninit};

use binius_field::{
	arch::{byte_sliced::ByteSlicedAES32x128b, ArchOptimal, OptimalUnderlier},
	packed::{get_packed_slice, set_packed_slice_unchecked},
	underlier::{UnderlierWithBitOps, WithUnderlier},
	AESTowerField128b, BinaryField128b, BinaryField128bPolyval, BinaryField1b, ByteSlicedAES32x16b,
	ByteSlicedAES32x32b, ByteSlicedAES32x64b, ByteSlicedAES32x8b, ExtensionField, Field,
	PackedBinaryField128x1b, PackedBinaryField16x1b, PackedBinaryField256x1b,
	PackedBinaryField32x1b, PackedBinaryField512x1b, PackedBinaryField64x1b, PackedBinaryField8x1b,
	PackedField,
};
use binius_maybe_rayon::{
	iter::{IndexedParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use binius_utils::bail;
use bytemuck::{fill_zeroes, Pod};
use itertools::max;
use lazy_static::lazy_static;
use stackalloc::helpers::slice_assume_init_mut;

use crate::Error;

/// Execute the right fold operation.
///
/// Every consequent `1 << log_query_size` scalar values are dot-producted with the corresponding
/// query elements. The result is stored in the `output` slice of packed values.
pub fn fold_right<P, PE>(
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
	check_fold_arguments(evals, log_evals_size, query, log_query_size, out)?;

	// Try execute the optimized version for 1-bit values if possible
	if TypeId::of::<P::Scalar>() == TypeId::of::<BinaryField1b>()
		&& fold_right_1bit_evals(evals, log_evals_size, query, log_query_size, out)
	{
		return Ok(());
	}

	fold_right_fallback(evals, log_evals_size, query, log_query_size, out);

	Ok(())
}

/// Execute the left fold operation.
///
/// evals is treated as a matrix with `1 << log_query_size` rows and each column is dot-producted
/// with the corresponding query element. The results is written to the `output` slice of packed values.
/// If the function returns `Ok(())`, then `out` can be safely interpreted as initialized.
///
/// Please note that unlike `fold_right`, this method is single threaded. Currently we always have some
/// parallelism above this level, so it's not a problem. Having no parallelism inside allows us to
/// use more efficient optimizations for special cases. If we ever need a parallel version of this
/// function, we can implement it separately.
pub fn fold_left<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [MaybeUninit<PE>],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	check_fold_arguments(evals, log_evals_size, query, log_query_size, out)?;

	if TypeId::of::<P::Scalar>() == TypeId::of::<BinaryField1b>()
		&& fold_left_1b_128b(evals, log_evals_size, query, log_query_size, out)
	{
		return Ok(());
	}

	fold_left_fallback(evals, log_evals_size, query, log_query_size, out);

	Ok(())
}

#[inline]
fn check_fold_arguments<P, PE, POut>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &[POut],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if log_evals_size < log_query_size {
		bail!(Error::IncorrectQuerySize {
			expected: log_evals_size
		});
	}

	if P::LOG_WIDTH + evals.len() < log_evals_size {
		bail!(Error::IncorrectArgumentLength {
			arg: "evals".into(),
			expected: log_evals_size
		});
	}

	if PE::LOG_WIDTH + query.len() < log_query_size {
		bail!(Error::IncorrectArgumentLength {
			arg: "query".into(),
			expected: log_query_size
		});
	}

	if PE::LOG_WIDTH + out.len() < log_evals_size - log_query_size {
		bail!(Error::IncorrectOutputPolynomialSize {
			expected: log_evals_size - log_query_size
		});
	}

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
#[allow(clippy::redundant_clone)]
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
		for byte in bytes {
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
fn fold_right_1bit_evals_small_query<P, PE, const LOG_QUERY_SIZE: usize>(
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
fn fold_right_1bit_evals_medium_query<P, PE, const LOG_QUERY_SIZE: usize>(
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
fn fold_right_1bit_evals<P, PE>(
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
		0 => fold_right_1bit_evals_small_query::<P, PE, 0>(evals, query, out),
		1 => fold_right_1bit_evals_small_query::<P, PE, 1>(evals, query, out),
		2 => fold_right_1bit_evals_small_query::<P, PE, 2>(evals, query, out),
		3 => fold_right_1bit_evals_medium_query::<P, PE, 3>(evals, query, out),
		4 => fold_right_1bit_evals_medium_query::<P, PE, 4>(evals, query, out),
		5 => fold_right_1bit_evals_medium_query::<P, PE, 5>(evals, query, out),
		6 => fold_right_1bit_evals_medium_query::<P, PE, 6>(evals, query, out),
		7 => fold_right_1bit_evals_medium_query::<P, PE, 7>(evals, query, out),
		_ => false,
	}
}

/// Fallback implementation for fold that can be executed for any field types and sizes.
fn fold_right_fallback<P, PE>(
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

type ArchOptimaType<F> = <F as ArchOptimal>::OptimalThroughputPacked;

#[inline(always)]
fn get_arch_optimal_packed_type_id<F: ArchOptimal>() -> TypeId {
	TypeId::of::<ArchOptimaType<F>>()
}

/// Use optimized algorithm for 1-bit evaluations with 128-bit query values packed with the optimal underlier.
/// Returns true if the optimized algorithm was used. In this case `out` can be safely interpreted
/// as initialized.
///
/// We could potentially this of specializing for other query fields, but that would require
/// separate implementations.
fn fold_left_1b_128b<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [MaybeUninit<PE>],
) -> bool
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if log_evals_size < P::LOG_WIDTH || !is_sequential_bytes::<P>() {
		return false;
	}

	let log_row_size = log_evals_size - log_query_size;
	if log_row_size < 3 {
		return false;
	}

	if PE::LOG_WIDTH > 3 {
		return false;
	}

	// Safety: the cast is safe because the type is checked by the previous if statement
	let evals_u8: &[u8] = unsafe {
		std::slice::from_raw_parts(evals.as_ptr() as *const u8, std::mem::size_of_val(evals))
	};

	// Try to run the optimized version for the specific 128-bit field type.
	// This is a workaround for the lack of specialization in Rust.
	#[inline]
	fn try_run_specialization<PE, F>(
		lookup_table: &[OptimalUnderlier],
		evals_u8: &[u8],
		log_evals_size: usize,
		query: &[PE],
		log_query_size: usize,
		out: &mut [MaybeUninit<PE>],
	) -> bool
	where
		PE: PackedField,
		F: ArchOptimal,
	{
		if TypeId::of::<PE>() == get_arch_optimal_packed_type_id::<F>() {
			let query = cast_same_type_slice::<_, ArchOptimaType<F>>(query);
			let out = cast_same_type_slice_mut::<_, MaybeUninit<ArchOptimaType<F>>>(out);

			fold_left_1b_128b_impl(
				lookup_table,
				evals_u8,
				log_evals_size,
				query,
				log_query_size,
				out,
			);
			true
		} else {
			false
		}
	}

	let lookup_table = &*LOOKUP_TABLE;
	try_run_specialization::<_, BinaryField128b>(
		lookup_table,
		evals_u8,
		log_evals_size,
		query,
		log_query_size,
		out,
	) || try_run_specialization::<_, AESTowerField128b>(
		lookup_table,
		evals_u8,
		log_evals_size,
		query,
		log_query_size,
		out,
	) || try_run_specialization::<_, BinaryField128bPolyval>(
		lookup_table,
		evals_u8,
		log_evals_size,
		query,
		log_query_size,
		out,
	)
}

/// Cast slice from unknown type to the known one assuming that the types are the same.
#[inline(always)]
fn cast_same_type_slice_mut<T: Sized + 'static, U: Sized + 'static>(slice: &mut [T]) -> &mut [U] {
	assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
	// Safety: the cast is safe because the type is checked by the previous if statement
	unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut U, slice.len()) }
}

/// Cast slice from unknown type to the known one assuming that the types are the same.
#[inline(always)]
fn cast_same_type_slice<T: Sized + 'static, U: Sized + 'static>(slice: &[T]) -> &[U] {
	assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
	// Safety: the cast is safe because the type is checked by the previous if statement
	unsafe { slice::from_raw_parts(slice.as_ptr() as *const U, slice.len()) }
}

/// Initialize the lookup table u8 -> [U; 8 / <number of 128-bit elements in U>] where
/// each bit in `u8` corresponds to a 128-bit element in `U` filled with ones or zeros
/// depending on the bit value. We use the values as bit masks for fast multiplication
/// by packed BinaryField1b values.
fn init_lookup_table_width<U>() -> Vec<U>
where
	U: UnderlierWithBitOps + From<u128>,
{
	let items_b128 = U::BITS / u128::BITS as usize;
	assert!(items_b128 <= 8);
	let items_in_byte = 8 / items_b128;

	let mut result = Vec::with_capacity(256 * items_in_byte);
	for i in 0..256 {
		for j in 0..items_in_byte {
			let bits = (i >> (j * items_b128)) & ((1 << items_b128) - 1);
			let mut value = U::ZERO;
			for k in 0..items_b128 {
				if (bits >> k) & 1 == 1 {
					unsafe {
						value.set_subvalue(k, u128::ONES);
					}
				}
			}
			result.push(value);
		}
	}

	result
}

lazy_static! {
	static ref LOOKUP_TABLE: Vec<OptimalUnderlier> = init_lookup_table_width::<OptimalUnderlier>();
}

#[inline]
fn fold_left_1b_128b_impl<PE, U>(
	lookup_table: &[U],
	evals: &[u8],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [MaybeUninit<PE>],
) where
	PE: PackedField + WithUnderlier<Underlier = U>,
	U: UnderlierWithBitOps,
{
	let out = unsafe { slice_assume_init_mut(out) };
	fill_zeroes(out);

	let items_in_byte = 8 / PE::WIDTH;
	let row_size_bytes = 1 << (log_evals_size - log_query_size - 3);
	for (query_val, row_bytes) in PE::iter_slice(query).zip(evals.chunks(row_size_bytes)) {
		let query_val = PE::broadcast(query_val).to_underlier();
		for (byte_index, byte) in row_bytes.iter().enumerate() {
			let mask_offset = *byte as usize * items_in_byte;
			let out_offset = byte_index * items_in_byte;
			for i in 0..items_in_byte {
				let mask = unsafe { lookup_table.get_unchecked(mask_offset + i) };
				let multiplied = query_val & *mask;
				let out = unsafe { out.get_unchecked_mut(out_offset + i) };
				*out += PE::from_underlier(multiplied);
			}
		}
	}
}

fn fold_left_fallback<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	out: &mut [MaybeUninit<PE>],
) where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	let new_n_vars = log_evals_size - log_query_size;

	out.iter_mut()
		.enumerate()
		.for_each(|(outer_index, out_val)| {
			let mut res = PE::default();
			for inner_index in 0..min(PE::WIDTH, 1 << new_n_vars) {
				res.set(
					inner_index,
					PackedField::iter_slice(query)
						.take(1 << log_query_size)
						.enumerate()
						.map(|(query_index, basis_eval)| {
							let eval_index = (query_index << new_n_vars)
								| (outer_index << PE::LOG_WIDTH)
								| inner_index;
							let subpoly_eval_i = get_packed_slice(evals, eval_index);
							basis_eval * subpoly_eval_i
						})
						.sum(),
				);
			}

			out_val.write(res);
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

	fn fold_right_reference<P, PE>(
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

	fn check_fold_right<P, PE>(
		evals: &[P],
		log_evals_size: usize,
		query: &[PE],
		log_query_size: usize,
	) where
		P: PackedField,
		PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
	{
		let mut reference_out =
			vec![PE::zero(); (1usize << (log_evals_size - log_query_size)).div_ceil(PE::WIDTH)];
		let mut out = reference_out.clone();

		fold_right(evals, log_evals_size, query, log_query_size, &mut out).unwrap();
		fold_right_reference(evals, log_evals_size, query, log_query_size, &mut reference_out);

		for i in 0..1 << (log_evals_size - log_query_size) {
			assert_eq!(get_packed_slice(&out, i), get_packed_slice(&reference_out, i));
		}
	}

	#[test]
	fn test_1b_small_poly_query_log_size_0() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold_right(&evals, 0, &query, 0);
	}

	#[test]
	fn test_1b_small_poly_query_log_size_1() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold_right(&evals, 2, &query, 1);
	}

	#[test]
	fn test_1b_small_poly_query_log_size_7() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold_right(&evals, 7, &query, 7);
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
			check_fold_right(
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
			check_fold_right(
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
			check_fold_right(
				&evals,
				LOG_EVALS_SIZE + PackedBinaryField16x8b::LOG_WIDTH,
				&query,
				log_query_size,
			);
		}
	}

	fn fold_left_reference<P, PE>(
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
				result += get_packed_slice(query, j)
					* get_packed_slice(evals, i | (j << (log_evals_size - log_query_size)));
			}

			set_packed_slice(out, i, result);
		}
	}

	fn check_fold_left<P, PE>(
		evals: &[P],
		log_evals_size: usize,
		query: &[PE],
		log_query_size: usize,
	) where
		P: PackedField,
		PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
	{
		let mut reference_out =
			vec![PE::zero(); (1usize << (log_evals_size - log_query_size)).div_ceil(PE::WIDTH)];

		let mut out = reference_out.clone();
		out.clear();
		fold_left(evals, log_evals_size, query, log_query_size, out.spare_capacity_mut()).unwrap();
		unsafe {
			out.set_len(out.capacity());
		}

		fold_left_reference(evals, log_evals_size, query, log_query_size, &mut reference_out);

		for i in 0..1 << (log_evals_size - log_query_size) {
			assert_eq!(get_packed_slice(&out, i), get_packed_slice(&reference_out, i));
		}
	}

	#[test]
	fn test_fold_left_1b_small_poly_query_log_size_0() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold_left(&evals, 0, &query, 0);
	}

	#[test]
	fn test_fold_left_1b_small_poly_query_log_size_1() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold_left(&evals, 2, &query, 1);
	}

	#[test]
	fn test_fold_left_1b_small_poly_query_log_size_7() {
		let mut rng = StdRng::seed_from_u64(0);
		let evals = vec![PackedBinaryField128x1b::random(&mut rng)];
		let query = vec![PackedBinaryField128x1b::random(&mut rng)];

		check_fold_left(&evals, 7, &query, 7);
	}

	#[test]
	fn test_fold_left_1b_many_evals() {
		const LOG_EVALS_SIZE: usize = 14;
		let mut rng = StdRng::seed_from_u64(1);
		let evals = repeat_with(|| PackedBinaryField128x1b::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE)
			.collect::<Vec<_>>();
		let query = vec![PackedBinaryField512x1b::random(&mut rng)];

		for log_query_size in 0..10 {
			check_fold_left(
				&evals,
				LOG_EVALS_SIZE + PackedBinaryField128x1b::LOG_WIDTH,
				&query,
				log_query_size,
			);
		}
	}

	type B128bOptimal = ArchOptimaType<BinaryField128b>;

	#[test]
	fn test_fold_left_1b_128b_optimal() {
		const LOG_EVALS_SIZE: usize = 14;
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| PackedBinaryField128x1b::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE)
			.collect::<Vec<_>>();
		let query = repeat_with(|| B128bOptimal::random(&mut rng))
			.take(1 << (10 - B128bOptimal::LOG_WIDTH))
			.collect::<Vec<_>>();

		for log_query_size in 0..10 {
			check_fold_left(
				&evals,
				LOG_EVALS_SIZE + PackedBinaryField128x1b::LOG_WIDTH,
				&query,
				log_query_size,
			);
		}
	}

	#[test]
	fn test_fold_left_128b_128b() {
		const LOG_EVALS_SIZE: usize = 14;
		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| B128bOptimal::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE)
			.collect::<Vec<_>>();
		let query = repeat_with(|| B128bOptimal::random(&mut rng))
			.take(1 << 10)
			.collect::<Vec<_>>();

		for log_query_size in 0..10 {
			check_fold_left(&evals, LOG_EVALS_SIZE, &query, log_query_size);
		}
	}
}
