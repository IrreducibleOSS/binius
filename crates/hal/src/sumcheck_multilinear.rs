// Copyright 2024-2025 Irreducible Inc.

use binius_field::PackedField;
use binius_math::MultilinearPoly;

/// An individual multilinear polynomial in a multivariate composite.
#[derive(Debug, Clone)]
pub enum SumcheckMultilinear<P, M>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	/// Small field multilinear - to be folded into large field at `switchover` round
	Transparent {
		multilinear: M,
		switchover_round: usize,
		zero_scalars_suffix: usize,
	},
	/// Large field multilinear - halved in size each round
	Folded { large_field_folded_evals: Vec<P> },
}

impl<P: PackedField, M: MultilinearPoly<P>> SumcheckMultilinear<P, M> {
	pub fn transparent(multilinear: M, switchover_fn: &impl Fn(usize) -> usize) -> Self {
		let switchover_round = (*switchover_fn)(1 << multilinear.log_extension_degree());

		Self::Transparent {
			multilinear,
			switchover_round,
			zero_scalars_suffix: 0,
		}
	}

	pub fn folded(large_field_folded_evals: Vec<P>) -> Self {
		Self::Folded {
			large_field_folded_evals,
		}
	}

	pub fn zero_scalars_suffix(&self, n_vars: usize) -> usize {
		match self {
			Self::Transparent {
				zero_scalars_suffix,
				..
			} => *zero_scalars_suffix,
			Self::Folded {
				large_field_folded_evals,
			} => (1usize << n_vars).saturating_sub(large_field_folded_evals.len() << P::LOG_WIDTH),
		}
	}

	pub fn update_zero_scalars_suffix(&mut self, n_vars: usize, new_zero_scalars_suffix: usize) {
		match self {
			Self::Transparent {
				zero_scalars_suffix,
				..
			} => {
				*zero_scalars_suffix = new_zero_scalars_suffix;
			}

			Self::Folded {
				large_field_folded_evals,
			} => {
				large_field_folded_evals
					.truncate(((1 << n_vars) - new_zero_scalars_suffix).div_ceil(P::WIDTH));
			}
		}
	}
}
