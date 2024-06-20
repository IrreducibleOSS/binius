// Copyright 2024 Ulvetanna Inc.

use binius_field::Field;

use crate::{
	oracle::CompositePolyOracle, polynomial::EvaluationDomain, protocols::evalcheck::EvalcheckClaim,
};

use super::{Error, VerificationError};

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

#[derive(Debug, Clone)]
pub struct AbstractSumcheckClaim<F: Field> {
	pub poly: CompositePolyOracle<F>,
	pub sum: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbstractSumcheckRoundClaim<F: Field> {
	pub partial_point: Vec<F>,
	pub current_round_sum: F,
}

pub trait AbstractSumcheckReductor<F: Field> {
	type Error: std::error::Error + From<Error>;

	/// Reduce a round claim to a round claim for the next round
	///
	/// Arguments:
	/// * `round`: The current round number
	/// * `claim`: The current round claim
	/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round
	/// * `round_proof`: The current round's round proof
	fn reduce_round_claim(
		&self,
		round: usize,
		claim: AbstractSumcheckRoundClaim<F>,
		challenge: F,
		round_proof: AbstractSumcheckRound<F>,
	) -> Result<AbstractSumcheckRoundClaim<F>, Self::Error>;
}

pub trait AbstractSumcheckProver<F: Field> {
	type Error: std::error::Error + From<Error>;

	fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<AbstractSumcheckRound<F>, Self::Error>;

	fn finalize(self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Self::Error>;

	/// Returns whether two provers may be used together in a batch proof
	///
	/// REQUIRES:
	/// * The relation between batch-consistent provers is an equivalence relation
	/// * self.n_vars() >= other.n_vars()
	fn batch_proving_consistent(&self, other: &Self) -> bool;

	fn n_vars(&self) -> usize;
}

pub fn reduce_final_round_claim<F: Field>(
	poly_oracle: &CompositePolyOracle<F>,
	round_claim: AbstractSumcheckRoundClaim<F>,
) -> Result<EvalcheckClaim<F>, Error> {
	let AbstractSumcheckRoundClaim {
		partial_point: eval_point,
		current_round_sum: eval,
	} = round_claim;

	if eval_point.len() != poly_oracle.n_vars() {
		return Err(VerificationError::NumberOfRounds.into());
	}

	let evalcheck_claim = EvalcheckClaim {
		poly: poly_oracle.clone(),
		eval_point,
		eval,
		is_random_point: true,
	};
	Ok(evalcheck_claim)
}

/// Validate that evaluation domain starts with 0 & 1 and the size is exactly one greater than the
/// maximum individual degree of the polynomial.
pub fn check_evaluation_domain<F: Field>(
	max_individual_degree: usize,
	domain: &EvaluationDomain<F>,
) -> Result<(), Error> {
	if max_individual_degree == 0
		|| domain.size() != max_individual_degree + 1
		|| domain.points()[0] != F::ZERO
		|| domain.points()[1] != F::ONE
	{
		return Err(Error::EvaluationDomainMismatch);
	}
	Ok(())
}

/// Ensures that previous round challenge is present if and only if not in the first round.
pub fn validate_rd_challenge<F: Field>(
	prev_rd_challenge: Option<F>,
	round: usize,
) -> Result<(), Error> {
	if round == 0 && prev_rd_challenge.is_some() {
		return Err(Error::PreviousRoundChallengePresent);
	} else if round > 0 && prev_rd_challenge.is_none() {
		return Err(Error::PreviousRoundChallengeAbsent);
	}

	Ok(())
}
