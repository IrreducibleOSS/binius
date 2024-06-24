// Copyright 2024 Ulvetanna Inc.

use binius_field::{Field, PackedField};

use crate::{
	oracle::CompositePolyOracle,
	polynomial::{evaluate_univariate, MultilinearComposite, MultilinearExtension},
	protocols::abstract_sumcheck::{
		AbstractSumcheckClaim, AbstractSumcheckReductor, AbstractSumcheckRound,
		AbstractSumcheckRoundClaim,
	},
	witness::MultilinearWitness,
};

use super::{Error, VerificationError};

/// A claim for the GKR Sumcheck protocol
///
/// Specifically the claim is that for
/// * an $n$-variate polynomial $f$,
/// * a random vector $r$, called the GKR challenge,
/// * a claimed sum $s$,
///
/// that $\sum_{x \in \{0, 1\}^n} f(x) * \mathsf{eq}(x, r) = s$.
/// where $\mathsf{eq}(x, y) = \prod_{i=0}^{n-1} x_i y_i + (1-x_i)(1-y_i)$
/// is the multilinear extension of the equality indicator partially evaluated
/// at $r$.
#[derive(Debug, Clone)]
pub struct GkrSumcheckClaim<F: Field> {
	pub poly: CompositePolyOracle<F>,
	pub sum: F,
	pub r: Vec<F>,
}

/// Witness for the GKR Sumcheck protocol
///
/// The prover will prove a claim of the following flavor
/// * $\sum_{x \in \{0, 1\}^n} f(x) * \mathsf{eq}(x, r) = s$.
#[derive(Debug, Clone)]
pub struct GkrSumcheckWitness<'a, P, C>
where
	P: PackedField,
{
	/// The $n$-variate multilinear composite polynomial $f(x)$
	pub poly: MultilinearComposite<P, C, MultilinearWitness<'a, P>>,
	/// The $n$-variate multilinear extension $R_0(x)$ of the values
	/// of the evaluated GKR circuit at the current layer.
	/// This is useful advice to the honest prover as it will equal the
	/// multilinear extension of the boolean hypercube evaluations of $f(x)$
	/// This fact allows for less computation in round 0.
	///
	/// Specifically $\forall x \in \{0, 1\}^n, f(x) = R_0(x)$
	pub current_layer: MultilinearExtension<P::Scalar>,
}

pub type GkrSumcheckRound<F> = AbstractSumcheckRound<F>;
pub type GkrSumcheckRoundClaim<F> = AbstractSumcheckRoundClaim<F>;

impl<F: Field> GkrSumcheckClaim<F> {
	pub fn n_vars(&self) -> usize {
		self.poly.n_vars()
	}
}

impl<F: Field> From<GkrSumcheckClaim<F>> for AbstractSumcheckClaim<F> {
	fn from(value: GkrSumcheckClaim<F>) -> Self {
		Self {
			poly: value.poly,
			sum: value.sum,
		}
	}
}

pub struct GkrSumcheckReductor<'a, F> {
	pub gkr_challenge_point: &'a [F],
}

impl<'a, F: Field> AbstractSumcheckReductor<F> for GkrSumcheckReductor<'a, F> {
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
		let alpha_i = self.gkr_challenge_point[round];
		reduce_round_claim_helper(claim, challenge, round_proof, alpha_i)
	}
}

/// Reduce a Gkr Sumcheck round claim to a claim for the next round
///
/// Arguments:
/// * `challenge`: The random challenge sampled by the verifier at the beginning of the round.
/// * `alpha_i`: The Gkr Challenge for round i
fn reduce_round_claim_helper<F: Field>(
	claim: GkrSumcheckRoundClaim<F>,
	challenge: F,
	proof: GkrSumcheckRound<F>,
	alpha_i: F,
) -> Result<GkrSumcheckRoundClaim<F>, Error> {
	let GkrSumcheckRoundClaim {
		mut partial_point,
		current_round_sum,
	} = claim;

	let GkrSumcheckRound { mut coeffs } = proof;

	// The prover has sent coefficients for the purported ith round polynomial
	// sans the constant coefficient.
	// * $r_i(X) = \sum_{j=0}^d a_j * X^j$
	// The verifier will need to recover the missing coefficient.
	//
	// Let $s$ denote the current round's claimed sum.
	// The verifier expects one identity of $r$ to hold
	// * $s = (1 - \alpha_i) r(0) + \alpha_i r(1)$.
	//
	// Not sending the whole round polynomial is an optimization.
	// In the unoptimized version of the protocol, the verifier will halt and reject
	// if given a round polynomial that does not satisfy the required identities.
	// For more information, see Section 3 of https://eprint.iacr.org/2024/108

	if coeffs.is_empty() {
		return Err(VerificationError::NumberOfCoefficients.into());
	}

	// The verifier has not been given $a_0$.
	// The identity that must hold is:
	//     $s = (1 - \alpha_i) r_i(0) + \alpha_i r_i(1)$
	// Or equivalently
	//     $s = a_0 + \alpha_i * \sum_{j=1}^d a_j$
	// Therefore
	//     $a_0 = s - \alpha_i * \sum_{j=1}^d a_j$
	let constant_term = current_round_sum - alpha_i * coeffs.iter().sum::<F>();
	coeffs.insert(0, constant_term);

	let new_round_sum = evaluate_univariate(&coeffs, challenge);

	partial_point.push(challenge);

	Ok(GkrSumcheckRoundClaim {
		partial_point,
		current_round_sum: new_round_sum,
	})
}
