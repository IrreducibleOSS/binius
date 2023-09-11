// Copyright 2023 Ulvetanna Inc.

pub struct SumcheckRound<F> {
	pub coeffs: Vec<F>,
}

pub struct SumcheckProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
}
