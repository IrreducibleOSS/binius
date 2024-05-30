// Copyright 2023 Ulvetanna Inc.

use crate::{
	oracle::CompositePolyOracle,
	polynomial::{evaluate_univariate, MultilinearComposite},
	protocols::{
		abstract_sumcheck::{
			AbstractSumcheckProof, AbstractSumcheckReductor, AbstractSumcheckRound,
			AbstractSumcheckRoundClaim,
		},
		evalcheck::EvalcheckClaim,
	},
	witness::MultilinearWitness,
};
use binius_field::Field;
use std::fmt::Debug;

use super::{Error, VerificationError};

#[derive(Debug, Clone)]
pub struct ZerocheckClaim<F: Field> {
	/// Virtual Polynomial Oracle of the function claimed to be zero on hypercube
	pub poly: CompositePolyOracle<F>,
}

impl<F: Field> ZerocheckClaim<F> {
	pub fn n_vars(&self) -> usize {
		self.poly.n_vars()
	}
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type ZerocheckWitness<'a, P, C> = MultilinearComposite<P, C, MultilinearWitness<'a, P>>;

pub type ZerocheckRound<F> = AbstractSumcheckRound<F>;
pub type ZerocheckProof<F> = AbstractSumcheckProof<F>;
pub type ZerocheckRoundClaim<F> = AbstractSumcheckRoundClaim<F>;

#[derive(Debug)]
pub struct ZerocheckProveOutput<F: Field> {
	pub evalcheck_claim: EvalcheckClaim<F>,
	pub zerocheck_proof: ZerocheckProof<F>,
}

pub struct ZerocheckReductor<'a, F> {
	pub alphas: &'a [F],
}

impl<'a, F: Field> AbstractSumcheckReductor<F> for ZerocheckReductor<'a, F> {
	type Error = Error;

	fn reduce_intermediate_round_claim(
		&self,
		round: usize,
		claim: AbstractSumcheckRoundClaim<F>,
		challenge: F,
		round_proof: AbstractSumcheckRound<F>,
	) -> Result<AbstractSumcheckRoundClaim<F>, Self::Error> {
		let alpha_i = if round == 0 {
			None
		} else {
			Some(self.alphas[round - 1])
		};

		reduce_intermediate_round_claim_helper(claim, challenge, round_proof, alpha_i)
	}

	fn reduce_final_round_claim(
		&self,
		poly_oracle: &CompositePolyOracle<F>,
		round_claim: AbstractSumcheckRoundClaim<F>,
	) -> Result<EvalcheckClaim<F>, Self::Error> {
		reduce_final_round_claim_helper(poly_oracle, round_claim)
	}
}

/// Reduce a zerocheck round claim to a claim for the next round
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round.
/// * `alpha_i`: The zerocheck challenge for round i sampled by the verifier during the zerocheck to sumcheck reduction.
fn reduce_intermediate_round_claim_helper<F: Field>(
	claim: ZerocheckRoundClaim<F>,
	challenge: F,
	proof: ZerocheckRound<F>,
	alpha_i: Option<F>,
) -> Result<ZerocheckRoundClaim<F>, Error> {
	let ZerocheckRoundClaim {
		mut partial_point,
		current_round_sum,
	} = claim;

	let ZerocheckRound { mut coeffs } = proof;
	let which_round = partial_point.len();

	// The prover has sent some coefficients for the purported ith round polynomial
	// * $r_i(X) = \sum_{j=0}^d a_j * X^j$
	// The verifier will need to recover the missing coefficient(s).
	//
	// Let $s$ denote the current round's claimed sum. There are two cases
	// Case 1: This is the first round
	// The verifier expects two identities of $r$ to hold
	// * $r_i(0) = 0$ and $r(1) = 0$.
	// Case 2: This is a subsequent round
	// The verifier expects one identity of $r$ to hold
	// * $s = (1 - \alpha_i) r(0) + \alpha_i r(1)$.
	//
	// In both cases, the prover has sent just enough information that there
	// will be a unique way to recover missing coefficients that satisfy
	// the required identities for the round polynomial.
	//
	// Not sending the whole round polynomial is an optimization.
	// In the unoptimized version of the protocol, the verifier will halt and reject
	// if given a round polynomial that does not satisfy the required identities.
	// For more information, see Section 3 of https://eprint.iacr.org/2024/108
	if which_round == 0 {
		if current_round_sum != F::ZERO {
			return Err(VerificationError::ExpectedClaimedSumToBeZero.into());
		}
		if alpha_i.is_some() {
			return Err(VerificationError::UnexpectedZerocheckChallengeFound.into());
		}
		// In case 1, the verifier has not been given $a_0$ or $a_1$
		// The identities that must hold are that $f(0) = 0$ and $f(1) = 0$.
		// Therefore
		//     $a_0 = f(0) = 0$
		// This implies
		//     $r_i(1) = \sum_{j=1}^d a_j$
		// Therefore
		//     $a_1 = r_i(1) - \sum_{j=2}^d a_j$
		let constant_term = F::ZERO;
		let linear_term = F::ZERO - coeffs.iter().sum::<F>();
		coeffs.insert(0, linear_term);
		coeffs.insert(0, constant_term);
	} else {
		if coeffs.is_empty() {
			return Err(VerificationError::NumberOfCoefficients.into());
		}
		let alpha_i = alpha_i.ok_or(VerificationError::ExpectedZerocheckChallengeNotFound)?;

		// In case 2, the verifier has not been given $a_0$.
		// The identity that must hold is:
		//     $s = (1 - \alpha_i) r_i(0) + \alpha_i r_i(1)$
		// Or equivalently
		//     $s = a_0 + \alpha_i * \sum_{j=1}^d a_j$
		// Therefore
		//     $a_0 = s - \alpha_i * \sum_{j=1}^d a_j$
		let constant_term = current_round_sum - alpha_i * coeffs.iter().sum::<F>();
		coeffs.insert(0, constant_term);
	}

	let new_round_sum = evaluate_univariate(&coeffs, challenge);

	partial_point.push(challenge);

	Ok(ZerocheckRoundClaim {
		partial_point,
		current_round_sum: new_round_sum,
	})
}

fn reduce_final_round_claim_helper<F: Field>(
	poly_oracle: &CompositePolyOracle<F>,
	round_claim: ZerocheckRoundClaim<F>,
) -> Result<EvalcheckClaim<F>, Error> {
	let ZerocheckRoundClaim {
		partial_point: eval_point,
		current_round_sum: eval,
	} = round_claim;
	if eval_point.len() != poly_oracle.n_vars() {
		return Err(VerificationError::NumberOfCoefficients.into());
	}

	let evalcheck_claim = EvalcheckClaim {
		poly: poly_oracle.clone(),
		eval_point,
		eval,
		is_random_point: true,
	};
	Ok(evalcheck_claim)
}
