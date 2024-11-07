// Copyright 2024 Irreducible Inc.

use crate::protocols::utils::packed_from_fn_with_offset;
use binius_field::{packed::get_packed_slice, PackedFieldIndexable};
use binius_hal::ComputationBackend;
use rayon::prelude::*;
use tracing::instrument;

#[instrument(skip_all, level = "debug")]
pub fn fold_partial_eq_ind<P, Backend>(n_vars: usize, partial_eq_ind_evals: &mut Backend::Vec<P>)
where
	P: PackedFieldIndexable,
	Backend: ComputationBackend,
{
	debug_assert_eq!(1 << n_vars.saturating_sub(P::LOG_WIDTH), partial_eq_ind_evals.len());

	if n_vars == 0 {
		return;
	}

	if partial_eq_ind_evals.len() == 1 {
		let unpacked = P::unpack_scalars_mut(partial_eq_ind_evals);
		for i in 0..1 << (n_vars - 1) {
			unpacked[i] = unpacked[2 * i] + unpacked[2 * i + 1];
		}
	} else {
		let current_evals = &*partial_eq_ind_evals;
		let updated_evals = (0..current_evals.len() / 2)
			.into_par_iter()
			.map(|i| {
				packed_from_fn_with_offset(i, |index| {
					let eval0 = get_packed_slice(current_evals, index << 1);
					let eval1 = get_packed_slice(current_evals, (index << 1) + 1);
					eval0 + eval1
				})
			})
			.collect();

		*partial_eq_ind_evals = Backend::to_hal_slice(updated_evals);
	}
}
