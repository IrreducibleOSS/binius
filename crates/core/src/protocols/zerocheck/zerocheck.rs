// Copyright 2023 Ulvetanna Inc.

use crate::{
	oracle::CompositePolyOracle,
	polynomial::{evaluate_univariate, CompositionPoly, MultilinearComposite},
	protocols::{
		abstract_sumcheck::{
			AbstractSumcheckClaim, AbstractSumcheckProof, AbstractSumcheckReductor,
			AbstractSumcheckRound, AbstractSumcheckRoundClaim,
		},
		evalcheck::EvalcheckClaim,
	},
	witness::MultilinearWitness,
};
use binius_field::{Field, PackedField};
use std::fmt::Debug;

use super::{Error, VerificationError};

/// A claim for the zerocheck interactive reduction.
///
/// The claim is that a multilinear composite polynomial, that the verifier has oracle access to,
/// evaluates to zero on the boolean hypercube.
#[derive(Debug, Clone)]
pub struct ZerocheckClaim<F: Field> {
	/// Virtual Polynomial Oracle of the function claimed to be zero on hypercube
	pub poly: CompositePolyOracle<F>,
}

impl<F: Field> From<ZerocheckClaim<F>> for AbstractSumcheckClaim<F> {
	fn from(value: ZerocheckClaim<F>) -> Self {
		Self {
			n_vars: value.poly.n_vars(),
			sum: F::ZERO,
		}
	}
}

impl<F: Field> ZerocheckClaim<F> {
	/// The number of variables of the composite polynomial the claim is about.
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

	fn reduce_round_claim(
		&self,
		round: usize,
		claim: AbstractSumcheckRoundClaim<F>,
		challenge: F,
		round_proof: AbstractSumcheckRound<F>,
	) -> Result<AbstractSumcheckRoundClaim<F>, Self::Error> {
		if round != claim.partial_point.len() {
			return Err(Error::RoundArgumentRoundClaimMismatch);
		}
		let alpha_i = if round == 0 {
			None
		} else {
			Some(self.alphas[round - 1])
		};

		reduce_intermediate_round_claim_helper(claim, challenge, round_proof, alpha_i)
	}
}

/// Reduce a zerocheck round claim to a claim for the next round
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round.
/// * `alpha_i`: The zerocheck challenge for round i.
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
	let round = partial_point.len();

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
	if round == 0 {
		if coeffs.is_empty() {
			return Err(VerificationError::NumberOfCoefficients.into());
		}
		if alpha_i.is_some() {
			return Err(VerificationError::UnexpectedZerocheckChallengeFound.into());
		}
		// In case 1, the verifier has not been given $a_0$
		// However, the verifier knows that $f(0) = f(1) = 0$
		// Therefore
		//     $a_0 = f(0) = 0$
		// This implies
		//     $r_i(1) = \sum_{j=1}^d a_j$
		// Therefore
		//     $a_1 = r_i(1) - \sum_{j=2}^d a_j$
		let constant_term = F::ZERO;
		let expected_linear_term = F::ZERO - coeffs.iter().skip(1).sum::<F>();
		let actual_linear_term = coeffs[0];
		if expected_linear_term != actual_linear_term {
			return Err(Error::RoundPolynomialCheckFailed);
		}
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

pub fn validate_witness<P, C>(witness: &ZerocheckWitness<P, C>) -> Result<(), Error>
where
	P: PackedField,
	C: CompositionPoly<P>,
{
	let log_size = witness.n_vars();

	for index in 0..(1 << log_size) {
		if witness.evaluate_on_hypercube(index)? != P::Scalar::zero() {
			return Err(Error::NaiveValidation { index });
		}
	}
	Ok(())
}
