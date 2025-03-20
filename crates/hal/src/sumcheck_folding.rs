// Copyright 2025 Irreducible Inc.

use binius_field::PackedField;
use binius_math::{
	fold_left_lerp_inplace, fold_right_lerp, EvaluationOrder, MultilinearPoly, MultilinearQueryRef,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bytemuck::zeroed_vec;

use crate::{
	common::{subcube_vars_for_bits, MAX_SRC_SUBCUBE_LOG_BITS},
	Error, SumcheckMultilinear,
};

pub(crate) fn fold_multilinears<P, M>(
	evaluation_order: EvaluationOrder,
	n_vars: usize,
	multilinears: &mut [SumcheckMultilinear<P, M>],
	challenge: P::Scalar,
	tensor_query: Option<MultilinearQueryRef<P>>,
) -> Result<bool, Error>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	match evaluation_order {
		EvaluationOrder::LowToHigh => {
			fold_multilinears_low_to_high(n_vars, multilinears, challenge, tensor_query)
		}
		EvaluationOrder::HighToLow => {
			fold_multilinears_high_to_low(n_vars, multilinears, challenge, tensor_query)
		}
	}
}

fn fold_multilinears_low_to_high<P, M>(
	n_vars: usize,
	multilinears: &mut [SumcheckMultilinear<P, M>],
	challenge: P::Scalar,
	tensor_query: Option<MultilinearQueryRef<P>>,
) -> Result<bool, Error>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	assert!(n_vars > 0);
	parallel_map(multilinears, |sumcheck_multilinear| -> Result<_, Error> {
		match sumcheck_multilinear {
			SumcheckMultilinear::Transparent {
				multilinear,
				switchover_round,
				nonzero_scalars_prefix,
			} => {
				if *switchover_round == 0 {
					// At switchover we partially evaluate the multilinear at an expanded tensor query.
					let tensor_query = tensor_query
						.as_ref()
						.expect("guaranteed to be Some while there is still a transparent");

					assert!(tensor_query.n_vars() > 0);

					let nonzero_scalars_prefix = nonzero_scalars_prefix.unwrap_or(1 << n_vars);

					let large_field_folded_evals = if nonzero_scalars_prefix < 1 << n_vars {
						let (subcube_vars, packed_len) = subcube_vars_for_bits::<P>(
							MAX_SRC_SUBCUBE_LOG_BITS,
							log2_ceil_usize(nonzero_scalars_prefix),
							tensor_query.n_vars(),
							n_vars - 1,
						);

						let folded_scalars =
							nonzero_scalars_prefix.div_ceil(1 << tensor_query.n_vars());

						let mut folded =
							zeroed_vec(folded_scalars.div_ceil(1 << subcube_vars) * packed_len);

						// REVIEW: no lerp optimization in subcube_partial_low_evals currently
						for (subcube_index, subcube_evals) in
							folded.chunks_exact_mut(packed_len).enumerate()
						{
							multilinear.subcube_partial_low_evals(
								*tensor_query,
								subcube_vars,
								subcube_index,
								subcube_evals,
							)?;
						}

						folded.truncate(folded_scalars.div_ceil(P::WIDTH));
						folded
					} else {
						multilinear
							.evaluate_partial_low(*tensor_query)?
							.into_evals()
					};

					*sumcheck_multilinear = SumcheckMultilinear::Folded {
						large_field_folded_evals,
					};

					Ok(false)
				} else {
					*switchover_round -= 1;
					Ok(true)
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_evals,
			} => {
				// Post-switchover, we perform single variable folding (linear interpolation).
				// NB: Lerp folding in low-to-high evaluation order can be made inplace, but not
				// easily so if multithreading is desired.
				let mut new_large_field_folded_evals =
					zeroed_vec(1 << n_vars.saturating_sub(1 + P::LOG_WIDTH));

				fold_right_lerp(
					large_field_folded_evals.as_slice(),
					// TODO comment
					large_field_folded_evals.len() * P::WIDTH,
					challenge,
					&mut new_large_field_folded_evals,
				)?;

				*large_field_folded_evals = new_large_field_folded_evals;
				Ok(false)
			}
		}
	})
}

fn fold_multilinears_high_to_low<P, M>(
	n_vars: usize,
	multilinears: &mut [SumcheckMultilinear<P, M>],
	challenge: P::Scalar,
	tensor_query: Option<MultilinearQueryRef<P>>,
) -> Result<bool, Error>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	parallel_map(multilinears, |sumcheck_multilinear| -> Result<_, Error> {
		match sumcheck_multilinear {
			SumcheckMultilinear::Transparent {
				multilinear,
				switchover_round,
				nonzero_scalars_prefix,
			} => {
				if *switchover_round == 0 {
					// At switchover we partially evaluate the multilinear at an expanded tensor query.
					let tensor_query = tensor_query
						.as_ref()
						.expect("guaranteed to be Some while there is still a transparent");

					let nonzero_scalars_prefix = nonzero_scalars_prefix.unwrap_or(1 << n_vars);

					let large_field_folded_evals = if nonzero_scalars_prefix < 1 << n_vars {
						let (subcube_vars, packed_len) = subcube_vars_for_bits::<P>(
							MAX_SRC_SUBCUBE_LOG_BITS,
							log2_ceil_usize(nonzero_scalars_prefix),
							tensor_query.n_vars(),
							n_vars - 1,
						);

						let folded_scalars =
							nonzero_scalars_prefix.min(1 << (n_vars - tensor_query.n_vars()));

						let mut folded =
							zeroed_vec(folded_scalars.div_ceil(1 << subcube_vars) * packed_len);

						// REVIEW: no lerp optimization in subcube_partial_high_evals currently
						for (subcube_index, subcube_evals) in
							folded.chunks_exact_mut(packed_len).enumerate()
						{
							multilinear.subcube_partial_low_evals(
								*tensor_query,
								subcube_vars,
								subcube_index,
								subcube_evals,
							)?;
						}

						folded.truncate(folded_scalars.div_ceil(P::WIDTH));
						folded
					} else {
						multilinear
							.evaluate_partial_high(*tensor_query)?
							.into_evals()
					};

					*sumcheck_multilinear = SumcheckMultilinear::Folded {
						large_field_folded_evals,
					};

					Ok(false)
				} else {
					*switchover_round -= 1;
					Ok(true)
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_evals,
			} => {
				// REVIEW: note that this method is currently _not_ multithreaded, as
				//         traces are usually sufficiently wide
				fold_left_lerp_inplace(
					large_field_folded_evals,
					(large_field_folded_evals.len() * P::WIDTH).min(1 << n_vars),
					n_vars,
					challenge,
				)?;
				Ok(false)
			}
		}
	})
}

fn parallel_map<P, M>(
	multilinears: &mut [SumcheckMultilinear<P, M>],
	map_multilinear: impl Fn(&mut SumcheckMultilinear<P, M>) -> Result<bool, Error> + Sync,
) -> Result<bool, Error>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	let any_transparent_left = multilinears
		.par_iter_mut()
		.try_fold(
			|| false,
			|any_transparent_left, sumcheck_multilinear| -> Result<bool, Error> {
				let is_still_transparent = map_multilinear(sumcheck_multilinear)?;
				Ok(any_transparent_left || is_still_transparent)
			},
		)
		.try_reduce(|| false, |lhs, rhs| Ok(lhs || rhs))?;

	Ok(any_transparent_left)
}
