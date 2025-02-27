// Copyright 2024-2025 Irreducible Inc.

use binius_field::{packed::get_packed_slice, PackedField};
use binius_hal::ComputationBackend;
use binius_math::EvaluationOrder;
use binius_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::protocols::utils::packed_from_fn_with_offset;

#[instrument(skip_all, level = "debug")]
pub fn fold_partial_eq_ind<P, Backend>(
	evaluation_order: EvaluationOrder,
	n_vars: usize,
	partial_eq_ind_evals: &mut Backend::Vec<P>,
) where
	P: PackedField,
	Backend: ComputationBackend,
{
	debug_assert_eq!(1 << n_vars.saturating_sub(P::LOG_WIDTH), partial_eq_ind_evals.len());

	if n_vars == 0 {
		return;
	}

	if partial_eq_ind_evals.len() == 1 {
		let only_packed = partial_eq_ind_evals.first().expect("len == 1");

		let mut folded = P::zero();
		for i in 0..1 << (n_vars - 1) {
			folded.set(
				i,
				match evaluation_order {
					EvaluationOrder::LowToHigh => {
						only_packed.get(i << 1) + only_packed.get(i << 1 | 1)
					}
					EvaluationOrder::HighToLow => {
						only_packed.get(i) + only_packed.get(i | 1 << (n_vars - 1))
					}
				},
			);
		}

		*partial_eq_ind_evals.first_mut().expect("len == 1") = folded;
	} else {
		let new_packed_len = partial_eq_ind_evals.len() >> 1;
		let updated_evals = match evaluation_order {
			EvaluationOrder::LowToHigh => (0..new_packed_len)
				.into_par_iter()
				.map(|i| {
					packed_from_fn_with_offset(i, |index| {
						let eval0 = get_packed_slice(&*partial_eq_ind_evals, index << 1);
						let eval1 = get_packed_slice(&*partial_eq_ind_evals, index << 1 | 1);
						eval0 + eval1
					})
				})
				.collect(),

			EvaluationOrder::HighToLow => {
				// REVIEW: make this inplace, by enabling truncation in Backend::Vec
				let (evals_0, evals_1) = partial_eq_ind_evals.split_at(new_packed_len);

				(evals_0, evals_1)
					.into_par_iter()
					.map(|(&eval_0, &eval_1)| eval_0 + eval_1)
					.collect()
			}
		};

		*partial_eq_ind_evals = Backend::to_hal_slice(updated_evals);
	}
}
