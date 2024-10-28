// Copyright 2024 Irreducible Inc.

use crate::{MLEDirectAdapter, MultilinearPoly};
use binius_field::PackedField;

/// An individual multilinear polynomial in a multivariate composite.
#[derive(Debug, Clone)]
pub enum SumcheckMultilinear<P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	/// Small field polynomial - to be folded into large field at `switchover` round
	Transparent {
		multilinear: M,
		switchover_round: usize,
	},
	/// Large field polynomial - halved in size each round
	Folded {
		large_field_folded_multilinear: MLEDirectAdapter<P>,
	},
}
