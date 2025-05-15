// Copyright 2024-2025 Irreducible Inc.

use core::slice;
use std::{any::TypeId, cmp::min, mem::MaybeUninit};

use binius_field::{
	AESTowerField128b, BinaryField1b, BinaryField128b, BinaryField128bPolyval, ExtensionField,
	Field, PackedField,
	arch::{ArchOptimal, OptimalUnderlier},
	byte_iteration::{
		ByteIteratorCallback, can_iterate_bytes, create_partial_sums_lookup_tables,
		is_sequential_bytes, iterate_bytes,
	},
	packed::{
		PackedSlice, get_packed_slice, get_packed_slice_unchecked, set_packed_slice,
		set_packed_slice_unchecked,
	},
	underlier::{UnderlierWithBitOps, WithUnderlier},
};
use binius_utils::bail;
use bytemuck::fill_zeroes;
use itertools::izip;
use lazy_static::lazy_static;
use stackalloc::helpers::slice_assume_init_mut;

use crate::Error;

pub fn zero_pad<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	log_new_vals: usize,
	start_index: usize,
	nonzero_index: usize,
	out: &mut [PE],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if start_index > log_evals_size {
		bail!(Error::IncorrectStartIndex {
			expected: log_evals_size
		});
	}

	let extra_vars = log_new_vals - log_evals_size;
	if nonzero_index > 1 << extra_vars {
		bail!(Error::IncorrectNonZeroIndex {
			expected: 1 << extra_vars
		});
	}

	let block_size = 1 << start_index;
	let nb_elts_per_row = 1 << (start_index + extra_vars);
	let length = 1 << start_index;

	let start_index_in_row = nonzero_index * block_size;
	for (outer_index, packed_result_eval) in out.iter_mut().enumerate() {
		// Index of the Scalar at the start of the current Packed element in `out`.
		let outer_word_start = outer_index << PE::LOG_WIDTH;
		for inner_index in 0..min(PE::WIDTH, 1 << log_evals_size) {
			// Index of the current Scalar in `out`.
			let outer_scalar_index = outer_word_start + inner_index;
			// Column index, within the reshaped `evals`, where we start the dot-product with the
			// query elements.
			let inner_col = outer_scalar_index % nb_elts_per_row;
			// Row index.
			let inner_row = outer_scalar_index / nb_elts_per_row;

			if inner_col >= start_index_in_row && inner_col < start_index_in_row + block_size {
				let eval_index = inner_row * length + inner_col - start_index_in_row;
				let result_eval = get_packed_slice(evals, eval_index).into();
				unsafe {
					packed_result_eval.set_unchecked(inner_index, result_eval);
				}
			}
		}
	}
	Ok(())
}

/// Execute the right fold operation.
///
/// Every consequent `1 << log_query_size` scalar values are dot-producted with the corresponding
/// query elements. The result is stored in the `output` slice of packed values.
///
/// Please note that this method is single threaded. Currently we always have some
/// parallelism above this level, so it's not a problem.
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

	// Use linear interpolation for single variable multilinear queries.
	let is_lerp = log_query_size == 1
		&& get_packed_slice(query, 0) + get_packed_slice(query, 1) == PE::Scalar::ONE;

	if is_lerp {
		let lerp_query = get_packed_slice(query, 1);
		fold_right_lerp(evals, 1 << log_evals_size, lerp_query, P::Scalar::ZERO, out)?;
	} else {
		fold_right_fallback(evals, log_evals_size, query, log_query_size, out);
	}

	Ok(())
}

/// Execute the left fold operation.
///
/// evals is treated as a matrix with `1 << log_query_size` rows and each column is dot-producted
/// with the corresponding query element. The results is written to the `output` slice of packed
/// values. If the function returns `Ok(())`, then `out` can be safely interpreted as initialized.
///
/// Please note that this method is single threaded. Currently we always have some
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

	// Use linear interpolation for single variable multilinear queries.
	// Unlike right folds, left folds are often used with packed fields which are not indexable
	// and have quite slow scalar indexing methods. For now, we specialize the practically important
	// case of single variable fold when PE == P.
	let is_lerp = log_query_size == 1
		&& get_packed_slice(query, 0) + get_packed_slice(query, 1) == PE::Scalar::ONE
		&& TypeId::of::<P>() == TypeId::of::<PE>();

	if is_lerp {
		let lerp_query = get_packed_slice(query, 1);
		// Safety: P == PE checked above.
		let out_p =
			unsafe { std::mem::transmute::<&mut [MaybeUninit<PE>], &mut [MaybeUninit<P>]>(out) };

		let lerp_query_p = lerp_query.try_into().ok().expect("P == PE");
		fold_left_lerp(
			evals,
			1 << log_evals_size,
			P::Scalar::ZERO,
			log_evals_size,
			lerp_query_p,
			out_p,
		)?;
	} else {
		fold_left_fallback(evals, log_evals_size, query, log_query_size, out);
	}

	Ok(())
}

/// Execute the middle fold operation.
///
/// Every consequent `1 << start_index` scalar values are considered as a row. Then each column
/// is dot-producted in chunks (of size `1 << log_query_size`) with the `query` slice.
/// The results are written to the `output`.
///
/// Please note that this method is single threaded. Currently we always have some
/// parallelism above this level, so it's not a problem.
pub fn fold_middle<P, PE>(
	evals: &[P],
	log_evals_size: usize,
	query: &[PE],
	log_query_size: usize,
	start_index: usize,
	out: &mut [MaybeUninit<PE>],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	check_fold_arguments(evals, log_evals_size, query, log_query_size, out)?;

	if log_evals_size < log_query_size + start_index {
		bail!(Error::IncorrectStartIndex {
			expected: log_evals_size
		});
	}

	let lower_indices_size = 1 << start_index;
	let new_n_vars = log_evals_size - log_query_size;

	out.iter_mut()
		.enumerate()
		.for_each(|(outer_index, out_val)| {
			let mut res = PE::default();
			// Index of the Scalar at the start of the current Packed element in `out`.
			let outer_word_start = outer_index << PE::LOG_WIDTH;
			for inner_index in 0..min(PE::WIDTH, 1 << new_n_vars) {
				// Index of the current Scalar in `out`.
				let outer_scalar_index = outer_word_start + inner_index;
				// Column index, within the reshaped `evals`, where we start the dot-product with
				// the query elements.
				let inner_i = outer_scalar_index % lower_indices_size;
				// Row index.
				let inner_j = outer_scalar_index / lower_indices_size;
				res.set(
					inner_index,
					PackedField::iter_slice(query)
						.take(1 << log_query_size)
						.enumerate()
						.map(|(query_index, basis_eval)| {
							// The reshaped `evals` "matrix" is subdivided in chunks of size
							// `1 << (start_index + log_query_size)` for the dot-product.
							let offset = inner_j << (start_index + log_query_size) | inner_i;
							let eval_index = offset + (query_index << start_index);
							let subpoly_eval_i = get_packed_slice(evals, eval_index);
							basis_eval * subpoly_eval_i
						})
						.sum(),
				);
			}
			out_val.write(res);
		});

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

	if P::WIDTH * evals.len() < 1 << log_evals_size {
		bail!(Error::IncorrectArgumentLength {
			arg: "evals".into(),
			expected: 1 << log_evals_size
		});
	}

	if PE::WIDTH * query.len() < 1 << log_query_size {
		bail!(Error::IncorrectArgumentLength {
			arg: "query".into(),
			expected: 1 << log_query_size
		});
	}

	if PE::WIDTH * out.len() < 1 << (log_evals_size - log_query_size) {
		bail!(Error::IncorrectOutputPolynomialSize {
			expected: 1 << (log_evals_size - log_query_size)
		});
	}

	Ok(())
}

#[inline]
fn check_right_lerp_fold_arguments<P, PE, POut>(
	evals: &[P],
	evals_size: usize,
	out: &[POut],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if P::WIDTH * evals.len() < evals_size {
		bail!(Error::IncorrectArgumentLength {
			arg: "evals".into(),
			expected: evals_size
		});
	}

	if PE::WIDTH * out.len() * 2 < evals_size {
		bail!(Error::IncorrectOutputPolynomialSize {
			expected: evals_size.div_ceil(2)
		});
	}

	Ok(())
}

#[inline]
fn check_left_lerp_fold_arguments<P, PE, POut>(
	evals: &[P],
	non_const_prefix: usize,
	log_evals_size: usize,
	out: &[POut],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	if log_evals_size == 0 {
		bail!(Error::IncorrectQuerySize { expected: 1 });
	}

	if non_const_prefix > 1 << log_evals_size {
		bail!(Error::IncorrectNonzeroScalarPrefix {
			expected: 1 << log_evals_size,
		});
	}

	if P::WIDTH * evals.len() < non_const_prefix {
		bail!(Error::IncorrectArgumentLength {
			arg: "evals".into(),
			expected: non_const_prefix,
		});
	}

	let folded_non_const_prefix = non_const_prefix.min(1 << (log_evals_size - 1));

	if PE::WIDTH * out.len() < folded_non_const_prefix {
		bail!(Error::IncorrectOutputPolynomialSize {
			expected: folded_non_const_prefix,
		});
	}

	Ok(())
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
	if LOG_QUERY_SIZE >= 3 || (P::LOG_WIDTH + LOG_QUERY_SIZE > PE::LOG_WIDTH) {
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

	struct Callback<'a, PE: PackedField, const LOG_QUERY_SIZE: usize> {
		out: &'a mut [PE],
		cached_table: &'a [PE::Scalar],
	}

	impl<PE: PackedField, const LOG_QUERY_SIZE: usize> ByteIteratorCallback
		for Callback<'_, PE, LOG_QUERY_SIZE>
	{
		#[inline(always)]
		fn call(&mut self, iterator: impl Iterator<Item = u8>) {
			let mask = (1 << (1 << LOG_QUERY_SIZE)) - 1;
			let values_in_byte = 1 << (3 - LOG_QUERY_SIZE);
			let mut current_index = 0;
			for byte in iterator {
				for k in 0..values_in_byte {
					let index = (byte >> (k * (1 << LOG_QUERY_SIZE))) & mask;
					// Safety: `i` is less than `chunk_size`
					unsafe {
						set_packed_slice_unchecked(
							self.out,
							current_index + k,
							self.cached_table[index as usize],
						);
					}
				}

				current_index += values_in_byte;
			}
		}
	}

	let mut callback = Callback::<'_, PE, LOG_QUERY_SIZE> {
		out,
		cached_table: &cached_table,
	};

	iterate_bytes(evals, &mut callback);

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

	if P::LOG_WIDTH + LOG_QUERY_SIZE > PE::LOG_WIDTH {
		return false;
	}

	let cached_tables =
		create_partial_sums_lookup_tables(PackedSlice::new_with_len(query, 1 << LOG_QUERY_SIZE));

	struct Callback<'a, PE: PackedField, const LOG_QUERY_SIZE: usize> {
		out: &'a mut [PE],
		cached_tables: &'a [PE::Scalar],
	}

	impl<PE: PackedField, const LOG_QUERY_SIZE: usize> ByteIteratorCallback
		for Callback<'_, PE, LOG_QUERY_SIZE>
	{
		#[inline(always)]
		fn call(&mut self, iterator: impl Iterator<Item = u8>) {
			let log_tables_count = LOG_QUERY_SIZE - 3;
			let tables_count = 1 << log_tables_count;
			let mut current_index = 0;
			let mut current_table = 0;
			let mut current_value = PE::Scalar::ZERO;
			for byte in iterator {
				current_value += self.cached_tables[(current_table << 8) + byte as usize];
				current_table += 1;

				if current_table == tables_count {
					// Safety: `i` is less than `chunk_size`
					unsafe {
						set_packed_slice_unchecked(self.out, current_index, current_value);
					}
					current_index += 1;
					current_table = 0;
					current_value = PE::Scalar::ZERO;
				}
			}
		}
	}

	let mut callback = Callback::<'_, _, LOG_QUERY_SIZE> {
		out,
		cached_tables: &cached_tables,
	};

	iterate_bytes(evals, &mut callback);

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
	if log_evals_size < P::LOG_WIDTH || !can_iterate_bytes::<P>() {
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

/// Specialized implementation for a single parameter right fold using linear interpolation
/// instead of tensor expansion resulting in  a single multiplication instead of two:
///   f(r||w) = r * (f(1||w) - f(0||w)) + f(0||w).
///
/// The same approach may be generalized to higher variable counts, with diminishing returns.
///
/// Please note that this method is single threaded. Currently we always have some
/// parallelism above this level, so it's not a problem. Having no parallelism inside allows us to
/// use more efficient optimizations for special cases. If we ever need a parallel version of this
/// function, we can implement it separately.
pub fn fold_right_lerp<P, PE>(
	evals: &[P],
	evals_size: usize,
	lerp_query: PE::Scalar,
	suffix_eval: P::Scalar,
	out: &mut [PE],
) -> Result<(), Error>
where
	P: PackedField,
	PE: PackedField<Scalar: ExtensionField<P::Scalar>>,
{
	check_right_lerp_fold_arguments::<_, PE, _>(evals, evals_size, out)?;

	let folded_evals_size = evals_size >> 1;
	out[..folded_evals_size.div_ceil(PE::WIDTH)]
		.iter_mut()
		.enumerate()
		.for_each(|(i, packed_result_eval)| {
			for j in 0..min(PE::WIDTH, folded_evals_size - (i << PE::LOG_WIDTH)) {
				let index = (i << PE::LOG_WIDTH) | j;

				let (eval0, eval1) = unsafe {
					(
						get_packed_slice_unchecked(evals, index << 1),
						get_packed_slice_unchecked(evals, (index << 1) | 1),
					)
				};

				let result_eval =
					PE::Scalar::from(eval1 - eval0) * lerp_query + PE::Scalar::from(eval0);

				// Safety: `j` < `PE::WIDTH`
				unsafe {
					packed_result_eval.set_unchecked(j, result_eval);
				}
			}
		});

	if evals_size % 2 == 1 {
		let eval0 = get_packed_slice(evals, folded_evals_size << 1);
		set_packed_slice(
			out,
			folded_evals_size,
			PE::Scalar::from(suffix_eval - eval0) * lerp_query + PE::Scalar::from(eval0),
		);
	}

	Ok(())
}

/// Left linear interpolation (lerp, single variable) fold
///
/// Please note that this method is single threaded. Currently we always have some
/// parallelism above this level, so it's not a problem. Having no parallelism inside allows us to
/// use more efficient optimizations for special cases. If we ever need a parallel version of this
/// function, we can implement it separately.
///
/// Also note that left folds are often intended to be used with non-indexable packed fields that
/// have inefficient scalar access; fully generic handling of all interesting cases that can
/// leverage spread multiplication requires dynamically checking the `PackedExtension` relations, so
/// for now we just handle the simplest yet important case of a single variable left fold in packed
/// field P with a lerp query of its scalar (and not a nontrivial extension field!).
pub fn fold_left_lerp<P>(
	evals: &[P],
	non_const_prefix: usize,
	suffix_eval: P::Scalar,
	log_evals_size: usize,
	lerp_query: P::Scalar,
	out: &mut [MaybeUninit<P>],
) -> Result<(), Error>
where
	P: PackedField,
{
	check_left_lerp_fold_arguments::<_, P, _>(evals, non_const_prefix, log_evals_size, out)?;

	if log_evals_size > P::LOG_WIDTH {
		let pivot = non_const_prefix
			.saturating_sub(1 << (log_evals_size - 1))
			.div_ceil(P::WIDTH);

		let packed_len = 1 << (log_evals_size - 1 - P::LOG_WIDTH);
		let upper_bound = non_const_prefix.div_ceil(P::WIDTH).min(packed_len);

		if pivot > 0 {
			let (evals_0, evals_1) = evals.split_at(packed_len);
			for (out, eval_0, eval_1) in izip!(&mut out[..pivot], evals_0, evals_1) {
				out.write(*eval_0 + (*eval_1 - *eval_0) * lerp_query);
			}
		}

		let broadcast_suffix_eval = P::broadcast(suffix_eval);
		for (out, eval) in izip!(&mut out[pivot..upper_bound], &evals[pivot..]) {
			out.write(*eval + (broadcast_suffix_eval - *eval) * lerp_query);
		}

		for out in &mut out[upper_bound..] {
			out.write(P::zero());
		}
	} else if non_const_prefix > 0 {
		let only_packed = *evals.first().expect("log_evals_size > 0");
		let mut folded = P::zero();

		for i in 0..1 << (log_evals_size - 1) {
			let eval_0 = only_packed.get(i);
			let eval_1 = only_packed.get(i | 1 << (log_evals_size - 1));
			folded.set(i, eval_0 + lerp_query * (eval_1 - eval_0));
		}

		out.first_mut().expect("log_evals_size > 0").write(folded);
	}

	Ok(())
}

/// Inplace left linear interpolation (lerp, single variable) fold
///
/// Please note that this method is single threaded. Currently we always have some
/// parallelism above this level, so it's not a problem. Having no parallelism inside allows us to
/// use more efficient optimizations for special cases. If we ever need a parallel version of this
/// function, we can implement it separately.
pub fn fold_left_lerp_inplace<P>(
	evals: &mut Vec<P>,
	non_const_prefix: usize,
	suffix_eval: P::Scalar,
	log_evals_size: usize,
	lerp_query: P::Scalar,
) -> Result<(), Error>
where
	P: PackedField,
{
	check_left_lerp_fold_arguments::<_, P, _>(evals, non_const_prefix, log_evals_size, evals)?;

	if log_evals_size > P::LOG_WIDTH {
		let pivot = non_const_prefix
			.saturating_sub(1 << (log_evals_size - 1))
			.div_ceil(P::WIDTH);

		let packed_len = 1 << (log_evals_size - 1 - P::LOG_WIDTH);
		let upper_bound = non_const_prefix.div_ceil(P::WIDTH).min(packed_len);

		if pivot > 0 {
			let (evals_0, evals_1) = evals.split_at_mut(packed_len);
			for (eval_0, eval_1) in izip!(&mut evals_0[..pivot], evals_1) {
				*eval_0 += (*eval_1 - *eval_0) * lerp_query;
			}
		}

		let broadcast_suffix_eval = P::broadcast(suffix_eval);
		for eval in &mut evals[pivot..upper_bound] {
			*eval += (broadcast_suffix_eval - *eval) * lerp_query;
		}

		evals.truncate(upper_bound);
	} else if non_const_prefix > 0 {
		let only_packed = evals.first_mut().expect("log_evals_size > 0");
		let mut folded = P::zero();
		let half_size = 1 << (log_evals_size - 1);

		for i in 0..half_size {
			let eval_0 = only_packed.get(i);
			let eval_1 = only_packed.get(i | half_size);
			folded.set(i, eval_0 + lerp_query * (eval_1 - eval_0));
		}

		*only_packed = folded;
	}

	Ok(())
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
	for (k, packed_result_eval) in out.iter_mut().enumerate() {
		for j in 0..min(PE::WIDTH, 1 << (log_evals_size - log_query_size)) {
			let index = (k << PE::LOG_WIDTH) | j;

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
}

type ArchOptimalType<F> = <F as ArchOptimal>::OptimalThroughputPacked;

#[inline(always)]
fn get_arch_optimal_packed_type_id<F: ArchOptimal>() -> TypeId {
	TypeId::of::<ArchOptimalType<F>>()
}

/// Use optimized algorithm for 1-bit evaluations with 128-bit query values packed with the optimal
/// underlier. Returns true if the optimized algorithm was used. In this case `out` can be safely
/// interpreted as initialized.
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
			let query = cast_same_type_slice::<_, ArchOptimalType<F>>(query);
			let out = cast_same_type_slice_mut::<_, MaybeUninit<ArchOptimalType<F>>>(out);

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
		PackedBinaryField8x1b, PackedBinaryField16x8b, PackedBinaryField16x32b,
		PackedBinaryField32x1b, PackedBinaryField64x8b, PackedBinaryField128x1b,
		PackedBinaryField512x1b, packed::set_packed_slice,
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

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
		let query = repeat_with(|| PackedBinaryField64x8b::random(&mut rng))
			.take(8)
			.collect::<Vec<_>>();

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

	#[test]
	fn test_lower_higher_bits() {
		const LOG_EVALS_SIZE: usize = 7;
		let mut rng = StdRng::seed_from_u64(0);
		// We have four elements of 32 bits each. Overall, that gives us 128 bits.
		let evals = [
			PackedBinaryField32x1b::random(&mut rng),
			PackedBinaryField32x1b::random(&mut rng),
			PackedBinaryField32x1b::random(&mut rng),
			PackedBinaryField32x1b::random(&mut rng),
		];

		// We get the lower bits of all elements (`log_query_size` = 1, so only the first
		// two bits of `query` are used).
		let log_query_size = 1;
		let query = [PackedBinaryField32x1b::from(
			0b00000000_00000000_00000000_00000001u32,
		)];
		let mut out = vec![
			PackedBinaryField32x1b::zero();
			(1 << (LOG_EVALS_SIZE - log_query_size)) / PackedBinaryField32x1b::WIDTH
		];
		out.clear();
		fold_middle(&evals, LOG_EVALS_SIZE, &query, log_query_size, 4, out.spare_capacity_mut())
			.unwrap();
		unsafe {
			out.set_len(out.capacity());
		}

		// Every 16 consecutive bits in `out` should be equal to the lower bits of an element.
		for (i, out_val) in out.iter().enumerate() {
			let first_lower = (evals[2 * i].0 as u16) as u32;
			let second_lower = (evals[2 * i + 1].0 as u16) as u32;
			let expected_out = first_lower + (second_lower << 16);
			assert!(out_val.0 == expected_out);
		}

		// We get the higher bits of all elements (`log_query_size` = 1, so only the first
		// two bits of `query` are used).
		let query = [PackedBinaryField32x1b::from(
			0b00000000_00000000_00000000_00000010u32,
		)];

		out.clear();
		fold_middle(&evals, LOG_EVALS_SIZE, &query, log_query_size, 4, out.spare_capacity_mut())
			.unwrap();
		unsafe {
			out.set_len(out.capacity());
		}

		// Every 16 consecutive bits in `out` should be equal to the higher bits of an element.
		for (i, out_val) in out.iter().enumerate() {
			let first_higher = evals[2 * i].0 >> 16;
			let second_higher = evals[2 * i + 1].0 >> 16;
			let expected_out = first_higher + (second_higher << 16);
			assert!(out_val.0 == expected_out);
		}
	}

	#[test]
	fn test_zeropad_bits() {
		const LOG_EVALS_SIZE: usize = 5;
		let mut rng = StdRng::seed_from_u64(0);
		// We have four elements of 32 bits each. Overall, that gives us 128 bits.
		let evals = [
			PackedBinaryField8x1b::random(&mut rng),
			PackedBinaryField8x1b::random(&mut rng),
			PackedBinaryField8x1b::random(&mut rng),
			PackedBinaryField8x1b::random(&mut rng),
		];

		// We double the size of each element by padding each element to the left.
		let num_extra_vars = 2;
		let start_index = 3;

		for nonzero_index in 0..1 << num_extra_vars {
			let mut out =
				vec![
					PackedBinaryField8x1b::zero();
					1usize << (LOG_EVALS_SIZE + num_extra_vars - PackedBinaryField8x1b::LOG_WIDTH)
				];
			zero_pad(
				&evals,
				LOG_EVALS_SIZE,
				LOG_EVALS_SIZE + num_extra_vars,
				start_index,
				nonzero_index,
				&mut out,
			)
			.unwrap();

			// Every `1 << start_index` consecutive bits from the original `evals` are placed at
			// index `nonzero_index` within a block of size `1 << (LOGS_EVALS_SIZE +
			// num_extra_vals)`.
			for (i, out_val) in out.iter().enumerate() {
				let expansion = 1 << num_extra_vars;
				if i % expansion == nonzero_index {
					assert!(out_val.0 == evals[i / expansion].0);
				} else {
					assert!(out_val.0 == 0);
				}
			}
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

	type B128bOptimal = ArchOptimalType<BinaryField128b>;

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

	#[test]
	fn test_fold_left_lerp_inplace_conforms_reference() {
		const LOG_EVALS_SIZE: usize = 14;
		let mut rng = StdRng::seed_from_u64(0);
		let mut evals = repeat_with(|| B128bOptimal::random(&mut rng))
			.take(1 << LOG_EVALS_SIZE.saturating_sub(B128bOptimal::LOG_WIDTH))
			.collect::<Vec<_>>();
		let lerp_query = <BinaryField128b as Field>::random(&mut rng);
		let mut query =
			vec![B128bOptimal::default(); 1 << 1usize.saturating_sub(B128bOptimal::LOG_WIDTH)];
		set_packed_slice(&mut query, 0, BinaryField128b::ONE - lerp_query);
		set_packed_slice(&mut query, 1, lerp_query);

		for log_evals_size in (1..=LOG_EVALS_SIZE).rev() {
			let mut out = vec![
				MaybeUninit::uninit();
				1 << log_evals_size.saturating_sub(B128bOptimal::LOG_WIDTH + 1)
			];
			fold_left(&evals, log_evals_size, &query, 1, &mut out).unwrap();
			fold_left_lerp_inplace(
				&mut evals,
				1 << log_evals_size,
				Field::ZERO,
				log_evals_size,
				lerp_query,
			)
			.unwrap();

			for (out, &inplace) in izip!(&out, &evals) {
				unsafe {
					assert_eq!(out.assume_init(), inplace);
				}
			}
		}
	}

	#[test]
	fn test_check_fold_arguments_valid() {
		let evals = vec![PackedBinaryField128x1b::default(); 8];
		let query = vec![PackedBinaryField128x1b::default(); 4];
		let out = vec![PackedBinaryField128x1b::default(); 4];

		// Should pass as query and output sizes are valid
		let result = check_fold_arguments(&evals, 3, &query, 2, &out);
		assert!(result.is_ok());
	}

	#[test]
	fn test_check_fold_arguments_invalid_query_size() {
		let evals = vec![PackedBinaryField128x1b::default(); 8];
		let query = vec![PackedBinaryField128x1b::default(); 4];
		let out = vec![PackedBinaryField128x1b::default(); 4];

		// Should fail as log_query_size > log_evals_size
		let result = check_fold_arguments(&evals, 2, &query, 3, &out);
		assert!(matches!(result, Err(Error::IncorrectQuerySize { .. })));
	}
}
