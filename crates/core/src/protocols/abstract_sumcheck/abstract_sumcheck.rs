// Copyright 2024 Ulvetanna Inc.

use binius_field::Field;

use crate::{oracle::CompositePolyOracle, protocols::evalcheck::EvalcheckClaim};

#[derive(Debug, Clone)]
pub struct AbstractSumcheckRound<F> {
	/// Monomial-Basis Coefficients of a round polynomial sent by the prover
	///
	/// For proof-size optimization, this vector is
	/// trimmed as much as possible such that the verifier
	/// can recover the missing coefficients. Which specific
	/// coefficients are missing depends on context.
	pub coeffs: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct AbstractSumcheckProof<F> {
	pub rounds: Vec<AbstractSumcheckRound<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbstractSumcheckRoundClaim<F: Field> {
	pub partial_point: Vec<F>,
	pub current_round_sum: F,
}

pub trait AbstractSumcheckReductor<F: Field> {
	type Error: std::error::Error;

	/// Reduce a round claim to a round claim for the next round
	///
	/// Arguments:
	/// * `round`: The current round number
	/// * `claim`: The current round claim
	/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round
	/// * `round_proof`: The current round's round proof
	fn reduce_intermediate_round_claim(
		&self,
		round: usize,
		claim: AbstractSumcheckRoundClaim<F>,
		challenge: F,
		round_proof: AbstractSumcheckRound<F>,
	) -> Result<AbstractSumcheckRoundClaim<F>, Self::Error>;

	/// Reduce the final round claim to an evalcheck claim
	///
	/// Arguments:
	/// * `poly_oracle`: The original polynomial oracle
	/// * `round_claim`: The final round claim
	fn reduce_final_round_claim(
		&self,
		poly_oracle: &CompositePolyOracle<F>,
		round_claim: AbstractSumcheckRoundClaim<F>,
	) -> Result<EvalcheckClaim<F>, Self::Error>;
}

pub trait AbstractSumcheckProver<F: Field> {
	type Error: std::error::Error;

	fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<AbstractSumcheckRound<F>, Self::Error>;
	fn finalize(self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Self::Error>;
}
