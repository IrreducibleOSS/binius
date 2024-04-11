// Copyright 2023 Ulvetanna Inc.

use super::{Error, VerificationError};
use crate::{
	oracle::CompositePolyOracle,
	polynomial::{evaluate_univariate, MultilinearComposite},
	protocols::evalcheck::EvalcheckClaim,
};
use binius_field::Field;

#[derive(Debug, Clone)]
pub struct SumcheckRound<F> {
	/// Monomial-Basis Coefficients of a round polynomial sent by the prover
	///
	/// For proof-size optimization, this vector is
	/// trimmed as much as possible such that the verifier
	/// can recover the missing coefficients. Which specific
	/// coefficients are missing depends on context.
	/// Specifically:
	/// * If zerocheck first round: two lowest degree coefficients
	/// * If zerocheck subsequent round: constant term coefficient
	/// * If non-zerocheck: highest degree coefficient
	pub coeffs: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckProof<F> {
	pub rounds: Vec<SumcheckRound<F>>,
}

#[derive(Debug)]
pub struct SumcheckProveOutput<F: Field, C> {
	pub evalcheck_claim: EvalcheckClaim<F, C>,
	pub sumcheck_proof: SumcheckProof<F>,
}

#[derive(Debug, Clone)]
pub struct SumcheckClaim<F: Field, C> {
	/// Virtual Polynomial Oracle of the function whose sum is claimed on hypercube domain
	pub poly: CompositePolyOracle<F, C>,
	/// Claimed Sum over the Boolean Hypercube
	pub sum: F,
	/// The zerocheck challenges if the sumcheck claim came directly from a zerocheck reduction
	pub zerocheck_challenges: Option<Vec<F>>,
}

impl<F: Field, C> SumcheckClaim<F, C> {
	pub fn n_vars(&self) -> usize {
		self.poly.n_vars()
	}
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type SumcheckWitness<P, C, M> = MultilinearComposite<P, C, M>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRoundClaim<F: Field> {
	pub partial_point: Vec<F>,
	pub current_round_sum: F,
}

/// Reduce a sumcheck round claim to a claim for the next round (when sumcheck is not from a zerocheck reduction)
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round
pub fn reduce_sumcheck_claim_round<F: Field>(
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

/// Reduce a sumcheck round claim to a claim for the next round (when sumcheck came from a zerocheck reduction)
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round.
/// * `alpha_i`: The zerocheck challenge for round i sampled by the verifier during the zerocheck to sumcheck reduction.
pub fn reduce_zerocheck_claim_round<F: Field>(
	claim: SumcheckRoundClaim<F>,
	challenge: F,
	proof: SumcheckRound<F>,
	alpha_i: Option<F>,
) -> Result<SumcheckRoundClaim<F>, Error> {
	let SumcheckRoundClaim {
		mut partial_point,
		current_round_sum,
	} = claim;

	let SumcheckRound { mut coeffs } = proof;
	let which_round = partial_point.len();

	// The prover has sent some coefficients for the purported ith round polynomial
	// * $r_i(X) = \sum_{j=0}^d a_j * X^j$
	// The verifier will need to recover the missing coefficient(s).
	//
	// Let $s$ denote the current round's claimed sum. There are two cases
	// Case 1: This is the first round of sumcheck
	// The verifier expects two identities of $r$ to hold
	// * $r_i(0) = 0$ and $r(1) = 0$.
	// Case 2: This is a subsequent round of sumcheck
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

	Ok(SumcheckRoundClaim {
		partial_point,
		current_round_sum: new_round_sum,
	})
}

pub fn reduce_sumcheck_claim_final<F: Field, C: Clone>(
	poly_oracle: &CompositePolyOracle<F, C>,
	round_claim: SumcheckRoundClaim<F>,
) -> Result<EvalcheckClaim<F, C>, Error> {
	let SumcheckRoundClaim {
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
