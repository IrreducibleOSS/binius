// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{ExtensionField, Field},
	iopoly::MultivariatePolyOracle,
	polynomial::{EvaluationDomain, MultilinearComposite},
	protocols::evalcheck::evalcheck::{EvalcheckClaim, EvalcheckWitness},
};

use super::{Error, VerificationError};

#[derive(Debug, Clone)]
pub struct SumcheckRound<F> {
	pub coeffs: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
}

#[derive(Debug)]
pub struct SumcheckProveOutput<'a, F: Field, FE: ExtensionField<F>> {
	pub evalcheck_claim: EvalcheckClaim<F, FE>,
	pub evalcheck_witness: EvalcheckWitness<'a, F, FE>,
	pub sumcheck_proof: SumcheckProof<FE>,
}

#[derive(Debug, Clone)]
pub struct SumcheckClaim<F: Field> {
	/// Virtual Polynomial Oracle of the function whose sum is claimed on hypercube domain
	pub poly: MultivariatePolyOracle<F>,
	/// Claimed Sum over the Boolean Hypercube
	pub sum: F,
}

/// SumCheckWitness Struct
#[derive(Debug, Clone)]
pub struct SumcheckWitness<'a, F: Field, FE: ExtensionField<F>> {
	/// Polynomial must be representable as a composition of multilinear polynomials
	pub polynomial: MultilinearComposite<'a, F, FE>,
}

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

#[derive(Clone, Debug, PartialEq)]
pub struct SumcheckRoundClaim<F: Field> {
	pub partial_point: Vec<F>,
	pub current_round_sum: F,
}

pub fn setup_first_round_claim<F: Field, FE: ExtensionField<F>>(
	claim: &SumcheckClaim<F>,
) -> SumcheckRoundClaim<FE> {
	SumcheckRoundClaim {
		partial_point: vec![],
		current_round_sum: claim.sum.into(),
	}
}

/// Reduce a sumcheck round claim to a claim for the next round.
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round
pub fn reduce_sumcheck_claim_round<F, FE>(
	poly_oracle: &MultivariatePolyOracle<F>,
	domain: &EvaluationDomain<FE>,
	round: SumcheckRound<FE>,
	round_claim: SumcheckRoundClaim<FE>,
	challenge: FE,
) -> Result<SumcheckRoundClaim<FE>, Error>
where
	F: Field,
	FE: ExtensionField<F>,
{
	let SumcheckRoundClaim {
		mut partial_point,
		current_round_sum,
	} = round_claim;
	let max_individual_degree = poly_oracle.max_individual_degree();
	check_evaluation_domain(max_individual_degree, domain)?;

	if round.coeffs.len() != max_individual_degree {
		return Err(Error::Verification(VerificationError::NumberOfCoefficients {
			round: partial_point.len(),
		}));
	}

	let SumcheckRound { mut coeffs } = round;
	coeffs.insert(0, current_round_sum - coeffs[0]);

	partial_point.push(challenge);
	let new_round_sum = domain.extrapolate(&coeffs, challenge)?;
	Ok(SumcheckRoundClaim {
		partial_point,
		current_round_sum: new_round_sum,
	})
}

pub fn reduce_sumcheck_claim_final<F: Field, FE: ExtensionField<F>>(
	poly_oracle: &MultivariatePolyOracle<F>,
	round_claim: SumcheckRoundClaim<FE>,
) -> Result<EvalcheckClaim<F, FE>, Error> {
	let SumcheckRoundClaim {
		partial_point: eval_point,
		current_round_sum: eval,
	} = round_claim;
	if eval_point.len() != poly_oracle.n_vars() {
		return Err(Error::Verification(VerificationError::NumberOfCoefficients {
			round: eval_point.len(),
		}));
	}

	let evalcheck_claim = EvalcheckClaim {
		poly: poly_oracle.clone(),
		eval_point,
		eval,
	};
	Ok(evalcheck_claim)
}
