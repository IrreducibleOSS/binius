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
