// Copyright 2023 Ulvetanna Inc.

use super::{Error, VerificationError};
use crate::{
	oracle::{CompositePolyOracle, OracleId},
	polynomial::{evaluate_univariate, MultilinearComposite},
	protocols::{
		abstract_sumcheck::{
			AbstractSumcheckClaim, AbstractSumcheckProof, AbstractSumcheckReductor,
			AbstractSumcheckRound, AbstractSumcheckRoundClaim, AbstractSumcheckWitness,
		},
		evalcheck::EvalcheckClaim,
	},
};
use binius_field::{Field, PackedField};

pub type SumcheckRound<F> = AbstractSumcheckRound<F>;
pub type SumcheckProof<F> = AbstractSumcheckProof<F>;

#[derive(Debug)]
pub struct SumcheckProveOutput<F: Field> {
	pub evalcheck_claim: EvalcheckClaim<F>,
	pub sumcheck_proof: SumcheckProof<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckClaim<F: Field> {
	pub poly: CompositePolyOracle<F>,
	pub sum: F,
}

impl<F: Field> AbstractSumcheckClaim<F> for SumcheckClaim<F> {
	fn n_vars(&self) -> usize {
		self.poly.n_vars()
	}

	fn sum(&self) -> F {
		self.sum
	}
}

// Default sumcheck witness type is just multilinear composite
pub type SumcheckWitness<P, C, M> = MultilinearComposite<P, C, M>;

pub type SumcheckRoundClaim<F> = AbstractSumcheckRoundClaim<F>;

pub struct SumcheckReductor;

impl<F: Field> AbstractSumcheckReductor<F> for SumcheckReductor {
	type Error = Error;

	fn reduce_round_claim(
		&self,
		_round: usize,
		claim: AbstractSumcheckRoundClaim<F>,
		challenge: F,
		round_proof: AbstractSumcheckRound<F>,
	) -> Result<AbstractSumcheckRoundClaim<F>, Self::Error> {
		reduce_intermediate_round_claim_helper(claim, challenge, round_proof)
	}
}

fn reduce_intermediate_round_claim_helper<F: Field>(
	claim: SumcheckRoundClaim<F>,
	challenge: F,
	proof: SumcheckRound<F>,
) -> Result<SumcheckRoundClaim<F>, Error> {
	let SumcheckRoundClaim {
		mut partial_point,
		current_round_sum,
	} = claim;

	let SumcheckRound { mut coeffs } = proof;
	if coeffs.is_empty() {
		return Err(VerificationError::NumberOfCoefficients.into());
	}

	// The prover has sent coefficients for the purported ith round polynomial
	// * $r_i(X) = \sum_{j=0}^d a_j * X^j$
	// However, the prover has not sent the highest degree coefficient $a_d$.
	// The verifier will need to recover this missing coefficient.
	//
	// Let $s$ denote the current round's claimed sum.
	// The verifier expects the round polynomial $r_i$ to satisfy the identity
	// * $s = r_i(0) + r_i(1)$
	// Using
	//     $r_i(0) = a_0$
	//     $r_i(1) = \sum_{j=0}^d a_j$
	// There is a unique $a_d$ that allows $r_i$ to satisfy the above identity.
	// Specifically
	//     $a_d = s - a_0 - \sum_{j=0}^{d-1} a_j$
	//
	// Not sending the whole round polynomial is an optimization.
	// In the unoptimized version of the protocol, the verifier will halt and reject
	// if given a round polynomial that does not satisfy the above identity.
	let last_coeff = current_round_sum - coeffs[0] - coeffs.iter().sum::<F>();
	coeffs.push(last_coeff);
	let new_round_sum = evaluate_univariate(&coeffs, challenge);

	partial_point.push(challenge);

	Ok(SumcheckRoundClaim {
		partial_point,
		current_round_sum: new_round_sum,
	})
}

pub fn validate_witness<F, PW, W>(claim: &SumcheckClaim<F>, witness: W) -> Result<(), Error>
where
	F: Field,
	PW: PackedField<Scalar: From<F> + Into<F>>,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
{
	let log_size = claim.n_vars();
	let oracle_ids = claim.poly.inner_polys_oracle_ids().collect::<Vec<_>>();
	let multilinears = witness
		.multilinears(0, oracle_ids.as_slice())?
		.into_iter()
		.map(|(_, multilinear)| multilinear)
		.collect::<Vec<_>>();

	let witness = MultilinearComposite::new(log_size, witness.composition(), multilinears)?;

	let sum = (0..(1 << log_size))
		.try_fold(PW::Scalar::ZERO, |acc, i| witness.evaluate_on_hypercube(i).map(|res| res + acc));

	if sum? == claim.sum().into() {
		Ok(())
	} else {
		Err(Error::NaiveValidation)
	}
}
