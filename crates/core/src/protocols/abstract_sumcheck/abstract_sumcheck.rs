// Copyright 2024 Ulvetanna Inc.

use super::{Error, VerificationError};
use crate::{
	oracle::{CompositePolyOracle, OracleId},
	polynomial::{CompositionPoly, MultilinearComposite, MultilinearPoly},
	protocols::evalcheck::EvalcheckClaim,
};
use auto_impl::auto_impl;
use binius_field::{Field, PackedField};
use binius_math::EvaluationDomain;
use binius_utils::bail;
use std::hash::Hash;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
}

impl<F: Field> From<AbstractSumcheckRoundClaim<F>> for ReducedClaim<F> {
	fn from(claim: AbstractSumcheckRoundClaim<F>) -> Self {
		Self {
			eval_point: claim.partial_point,
			eval: claim.current_round_sum,
		}
	}
}

pub trait AbstractSumcheckReductor<F: Field> {
	type Error: std::error::Error + From<Error>;

	/// Verify that the round proof contains the correct amount of information.
	fn validate_round_proof_shape(
		&self,
		round: usize,
		proof: &AbstractSumcheckRound<F>,
	) -> Result<(), Self::Error>;

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

/// A sumcheck protocol claim.
///
/// A claim implicitly refers to a multivariate polynomial with a number of variables $\nu$, where
/// the degree of each individual variable is bounded by $d$.
pub trait AbstractSumcheckClaim<F: Field> {
	/// Returns the number of variables $\nu$ of the multivariate polynomial.
	fn n_vars(&self) -> usize;

	/// Returns the maximum individual degree $d$ of all variables.
	fn max_individual_degree(&self) -> usize;

	/// Returns the claimed sum of the polynomial values over the $\nu$-dimensional boolean
	/// hypercube.
	fn sum(&self) -> F;
}

/// Polynomial must be representable as a composition of multilinear polynomials
#[auto_impl(&)]
pub trait AbstractSumcheckWitness<PW: PackedField> {
	/// Some identifier of a multilinear witness that is used to deduplicate the witness index when folding.
	type MultilinearId: Clone + Hash + Eq + Sync;
	type Composition: CompositionPoly<PW>;
	type Multilinear: MultilinearPoly<PW> + Send + Sync;

	fn composition(&self) -> &Self::Composition;

	/// Extract multilinear witnesses out of composite sumcheck witness.
	///
	/// Arguments:
	/// * `seq_id`: Sequential id of the sumcheck instance in a batch (treat as if assigned arbitrarily)
	/// * `claim_multilinear_ids`: Multilinear identifiers extracted from a claim.
	fn multilinears(
		&self,
		seq_id: usize,
		claim_multilinear_ids: &[Self::MultilinearId],
	) -> Result<impl IntoIterator<Item = (Self::MultilinearId, Self::Multilinear)>, Error>;
}

/// A trait that oversees the batched sumcheck execution
///
/// Implementations are expected to be used to:
/// * Manage common witness state (typically by perusing [`super::CommonProversState`])
/// * Create new prover instances at the beginning of a each round (via `new_prover`)
/// * Perform common pre-round update steps (via `pre_execute_rounds`)
/// * Advance the state of individual prover instances (via `prover_execute_round` and `prover_finalize`)
///
/// See the implementation [`super::batch_prove`] for more details.
pub trait AbstractSumcheckProversState<F: Field> {
	type Error: std::error::Error + From<Error>;

	type PackedWitnessField: PackedField<Scalar: From<F> + Into<F>>;

	type Claim: AbstractSumcheckClaim<F>;
	type Witness: AbstractSumcheckWitness<Self::PackedWitnessField>;

	type Prover;

	fn pre_execute_rounds(&mut self, prev_rd_challenge: Option<F>) -> Result<(), Self::Error>;

	fn new_prover(
		&mut self,
		claim: Self::Claim,
		witness: Self::Witness,
		seq_id: usize,
	) -> Result<Self::Prover, Self::Error>;

	fn prover_execute_round(
		&self,
		prover: &mut Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<AbstractSumcheckRound<F>, Self::Error>;

	fn prover_finalize(
		prover: Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<ReducedClaim<F>, Self::Error>;
}

impl<P, C, M> AbstractSumcheckWitness<P> for MultilinearComposite<P, C, M>
where
	P: PackedField,
	C: CompositionPoly<P>,
	M: MultilinearPoly<P> + Clone + Send + Sync,
{
	type MultilinearId = OracleId;
	type Composition = C;
	type Multilinear = M;

	fn composition(&self) -> &C {
		&self.composition
	}

	fn multilinears(
		&self,
		_seq_id: usize,
		claim_multilinear_ids: &[OracleId],
	) -> Result<impl IntoIterator<Item = (OracleId, M)>, Error> {
		if claim_multilinear_ids.len() != self.multilinears.len() {
			bail!(Error::ProverClaimWitnessMismatch);
		}

		Ok(claim_multilinear_ids
			.iter()
			.copied()
			.zip(self.multilinears.iter().cloned()))
	}
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
		bail!(Error::EvaluationDomainMismatch);
	}
	Ok(())
}

/// Ensures that previous round challenge is present if and only if not in the first round.
pub fn validate_rd_challenge<F: Field>(
	prev_rd_challenge: Option<F>,
	round: usize,
) -> Result<(), Error> {
	if round == 0 && prev_rd_challenge.is_some() {
		bail!(Error::PreviousRoundChallengePresent);
	} else if round > 0 && prev_rd_challenge.is_none() {
		bail!(Error::PreviousRoundChallengeAbsent);
	}

	Ok(())
}

pub fn finalize_evalcheck_claim<F: Field>(
	poly_oracle: &CompositePolyOracle<F>,
	reduced_claim: ReducedClaim<F>,
) -> Result<EvalcheckClaim<F>, Error> {
	let ReducedClaim { eval_point, eval } = reduced_claim;

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

/// Constructs a switchover function thaw returns the round number where folded multilinear is at
/// least 2^k times smaller (in bytes) than the original, or 1 when not applicable.
pub fn standard_switchover_heuristic(k: isize) -> impl Fn(usize) -> usize + Copy {
	move |extension_degree: usize| {
		let switchover_round = extension_degree.ilog2() as isize + k;
		switchover_round.max(1) as usize
	}
}
