// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{
		reduce_sumcheck_claim_final, reduce_sumcheck_claim_round, reduce_zerocheck_claim_round,
		SumcheckClaim, SumcheckRound, SumcheckRoundClaim, SumcheckWitness,
	},
};
use crate::{
	oracle::CompositePolyOracle,
	polynomial::{
		extrapolate_line, transparent::eq_ind::EqIndPartialEval, CompositionPoly, EvaluationDomain,
		MultilinearExtension, MultilinearPoly,
	},
	protocols::{
		evalcheck::EvalcheckClaim,
		sumcheck::prove_general::{ProverState, SumcheckEvaluator},
	},
};
use binius_field::{Field, PackedField};
use either::Either;
use getset::Getters;
use rayon::prelude::*;
use std::fmt::Debug;
use tracing::instrument;

#[derive(Debug)]
struct ZerocheckAuxiliaryState<F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	challenges: Vec<F>,
	round_eq_ind: MultilinearExtension<'static, PW::Scalar>,
}

impl<F, PW> ZerocheckAuxiliaryState<F, PW>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	// Update the round_eq_ind for the next sumcheck round
	//
	// Let
	//  * $n$ be the number of variables in the sumcheck claim
	//  * $eq_k(X, Y)$ denote the equality indicator polynomial on $2 * k$ variables.
	//  * $\alpha_1, \ldots, \alpha_{n-1}$ be the $n-1$ zerocheck challenges
	// In round $i$, before computing the round polynomial, we seek the invariant that
	// * round_eq_ind is MLE of $eq_{n-i-1}(X, Y)$ partially evaluated at $Y = (\alpha_{i+1}, \ldots, \alpha_{n-1})$.
	//
	// To update the round_eq_ind, from $eq_{n-i}(X, \alpha_i, \ldots, \alpha_{n-1})$
	// to $eq_{n-i-1}(X, \alpha_{i+1}, \ldots, \alpha_{n-1})$, we sum consecutive hypercube evaluations.
	//
	// For example consider the hypercube evaluations of $eq_2(X, $\alpha_1, \alpha_2)$
	// * [$(1-\alpha_1) * (1-\alpha_2)$, $\alpha_1 * (1-\alpha_2)$, $(1-\alpha_1) * \alpha_2$, $\alpha_1 * \alpha_2$]
	// and consider the hypercube evaluations of $eq_1(X, \alpha_2)$
	// * [$(1-\alpha_2)$, $\alpha_2$]
	// We obtain the ith hypercube evaluation of $eq_1(X, \alpha_2)$ by summing the $(2*i)$ and $(2*i+1)$
	// hypercube evaluations of $eq_2(X, \alpha_1, \alpha_2)$.
	fn update_round_eq_ind(&mut self) -> Result<(), Error> {
		let current_evals = self.round_eq_ind.evals();
		let mut new_evals = vec![PW::Scalar::default(); current_evals.len() >> 1];
		new_evals.par_iter_mut().enumerate().for_each(|(i, e)| {
			*e = current_evals[i << 1] + current_evals[(i << 1) + 1];
		});
		let new_multilin = MultilinearExtension::from_values(new_evals)?;
		self.round_eq_ind = new_multilin;

		Ok(())
	}
}

/// A sumcheck protocol prover.
///
/// To prove a sumcheck claim, supply a multivariate composite witness. In
/// some cases it makes sense to do so in an different yet isomorphic field PW (witness packed
/// field) which may preferable due to superior performance. One example of such operating field
/// would be `BinaryField128bPolyval`, which tends to be much faster than 128-bit tower field on x86
/// CPUs. The only constraint is that constituent MLEs should have MultilinearPoly impls for PW -
/// something which is trivially satisfied for MLEs with tower field scalars for claims in tower
/// field as well.
///
/// Prover state is instantiated via `new` method, followed by exactly $n\\_vars$ `execute_round` invocations.
/// Each of those takes in an optional challenge (None on first round and Some on following rounds) and
/// evaluation domain. Proof and Evalcheck claim are obtained via `finalize` call at the end.
#[derive(Debug, Getters)]
pub struct SumcheckProver<'a, F, PW, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync,
{
	oracle: CompositePolyOracle<F>,
	composition: CW,
	domain: &'a EvaluationDomain<PW::Scalar>,
	#[getset(get = "pub")]
	round_claim: SumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<SumcheckRound<F>>,
	zerocheck_aux_state: Option<ZerocheckAuxiliaryState<F, PW>>,
	state: ProverState<PW, M>,
}

impl<'a, F, PW, CW, M> SumcheckProver<'a, F, PW, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Sync,
{
	/// Start a new sumcheck instance with claim in field `F`. Witness may be given in
	/// a different (but isomorphic) packed field PW. `switchover_fn` closure specifies
	/// switchover round number per multilinear polynomial as a function of its
	/// [`MultilinearPoly::extension_degree`] value.
	pub fn new(
		domain: &'a EvaluationDomain<PW::Scalar>,
		sumcheck_claim: SumcheckClaim<F>,
		sumcheck_witness: SumcheckWitness<PW, CW, M>,
		switchover_fn: impl Fn(usize) -> usize,
	) -> Result<Self, Error> {
		let n_vars = sumcheck_claim.n_vars();

		if sumcheck_claim.poly.max_individual_degree() == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}

		if sumcheck_witness.n_vars() != n_vars {
			let err_str = format!(
				"Claim and Witness n_vars mismatch in sumcheck. Claim: {}, Witness: {}",
				sumcheck_claim.n_vars(),
				sumcheck_witness.n_vars(),
			);

			return Err(Error::ProverClaimWitnessMismatch(err_str));
		}

		check_evaluation_domain(sumcheck_claim.poly.max_individual_degree(), domain)?;

		let state = ProverState::new(n_vars, sumcheck_witness.multilinears, switchover_fn)?;

		let composition = sumcheck_witness.composition;

		let round_claim = SumcheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: sumcheck_claim.sum,
		};

		let zerocheck_aux_state =
			if let Some(zc_challenges) = sumcheck_claim.zerocheck_challenges.clone() {
				let pw_challenges = zc_challenges
					.iter()
					.map(|&f| f.into())
					.collect::<Vec<PW::Scalar>>();

				let round_eq_multilin =
					EqIndPartialEval::new(n_vars - 1, pw_challenges)?.multilinear_extension()?;
				Some(ZerocheckAuxiliaryState {
					challenges: zc_challenges,
					round_eq_ind: round_eq_multilin,
				})
			} else {
				None
			};

		let prover_state = SumcheckProver {
			oracle: sumcheck_claim.poly,
			composition,
			domain,
			round_claim,
			round: 0,
			last_round_proof: None,
			zerocheck_aux_state,
			state,
		};

		Ok(prover_state)
	}

	pub fn n_vars(&self) -> usize {
		self.oracle.n_vars()
	}

	/// Generic parameters allow to pass a different witness type to the inner Evalcheck claim.
	#[instrument(skip_all, name = "sumcheck::finalize")]
	pub fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		self.validate_rd_challenge(prev_rd_challenge)?;

		if self.round != self.n_vars() {
			return Err(Error::ImproperInput(format!(
				"finalize() called on round {} while n_vars={}",
				self.round,
				self.n_vars()
			)));
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(prev_rd_challenge)?;
		}

		reduce_sumcheck_claim_final(&self.oracle, self.round_claim)
	}

	fn is_zerocheck(&self) -> bool {
		self.zerocheck_aux_state.is_some()
	}

	fn zerocheck_challenges(&self) -> Option<&Vec<F>> {
		self.zerocheck_aux_state.as_ref().map(|aux| &aux.challenges)
	}

	fn evals_to_coeffs(&self, mut evals: Vec<PW::Scalar>) -> Result<Vec<PW::Scalar>, Error> {
		// We have partial information about the degree $d$ univariate round polynomial $r(X)$,
		// but in each of the following cases, we can complete the picture and attain evaluations
		// at $r(0), \ldots, r(d+1)$.
		if let Some(zc_challenges) = self.zerocheck_challenges() {
			if self.round == 0 {
				// This is the case where we are processing the first round of a sumcheck that came from zerocheck.
				// We are given $r(2), \ldots, r(d+1)$.
				// From context, we infer that $r(0) = r(1) = 0$.
				evals.insert(0, PW::Scalar::ZERO);
				evals.insert(0, PW::Scalar::ZERO);
			} else {
				let current_round_sum = PW::Scalar::from(self.round_claim().current_round_sum);
				// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d+1)$
				// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
				// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
				// which allows us to compute the value of $r(0)$
				let alpha = PW::Scalar::from(zc_challenges[self.round - 1]);
				let alpha_bar = PW::Scalar::ONE - alpha;
				let one_evaluation = evals[0];
				let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
				let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
				let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

				evals.insert(0, zero_evaluation);
			}
		} else {
			let current_round_sum = PW::Scalar::from(self.round_claim().current_round_sum);
			// In the case where this is a sumcheck that did not come from zerocheck,
			// Given $r(1), \ldots, r(d+1)$, letting $s$ be the current round's claimed sum,
			// we can compute $r(0)$ using the identity $r(0) = s - r(1)$
			evals.insert(0, current_round_sum - evals[0]);
		}

		assert_eq!(evals.len(), self.domain.size());
		let coeffs = self.domain.interpolate(&evals)?;
		Ok(coeffs)
	}

	// NB: Can omit some coefficients to reduce proof size because verifier can compute
	// these missing coefficients.
	// This is documented in detail in the common prover-verifier logic for
	// reducing a sumcheck round claim.
	fn trim_coeffs<T: Clone>(&self, coeffs: Vec<T>) -> Vec<T> {
		if self.is_zerocheck() && self.round == 0 {
			coeffs[2..].to_vec()
		} else if self.is_zerocheck() {
			coeffs[1..].to_vec()
		} else {
			coeffs[..coeffs.len() - 1].to_vec()
		}
	}

	#[instrument(skip_all, name = "sumcheck::execute_round")]
	pub fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckRound<F>, Error> {
		// First round has no challenge, other rounds should have it
		self.validate_rd_challenge(prev_rd_challenge)?;

		if self.round >= self.n_vars() {
			return Err(Error::ImproperInput("too many execute_round calls".to_string()));
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// If zerocheck, update round_eq_ind
			if self.is_zerocheck() {
				self.zerocheck_aux_state
					.as_mut()
					.unwrap()
					.update_round_eq_ind()?;
			}

			// Process switchovers of small field multilinears and folding of large field ones
			self.state.fold(prev_rd_challenge.into())?;

			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		let evaluator = if let Some(ref zc_aux_state) = self.zerocheck_aux_state {
			Either::Left(if self.round == 0 {
				Either::Left(ZerocheckFirstRoundEvaluator {
					composition: &self.composition,
					domain: self.domain.points(),
					eq_ind: zc_aux_state.round_eq_ind.to_ref(),
				})
			} else {
				Either::Right(ZerocheckLaterRoundEvaluator {
					composition: &self.composition,
					domain: self.domain.points(),
					eq_ind: zc_aux_state.round_eq_ind.to_ref(),
				})
			})
		} else {
			Either::Right(RegularSumcheckEvaluator {
				composition: &self.composition,
				domain: self.domain.points(),
			})
		};

		let evals = self.state.sum_round_evals(evaluator);

		let untrimmed_coeffs = self.evals_to_coeffs(evals)?;
		let trimmed_coeffs = self.trim_coeffs(untrimmed_coeffs);
		let proof_round = SumcheckRound {
			coeffs: trimmed_coeffs.into_iter().map(Into::into).collect(),
		};
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	fn validate_rd_challenge(&self, prev_rd_challenge: Option<F>) -> Result<(), Error> {
		if prev_rd_challenge.is_none() != (self.round == 0) {
			return Err(Error::ImproperInput(format!(
				"incorrect optional challenge: is_some()={:?} at round {}",
				prev_rd_challenge.is_some(),
				self.round
			)));
		}

		Ok(())
	}

	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let round_claim = self.round_claim.clone();
		let round_proof = self
			.last_round_proof
			.as_ref()
			.expect("round is at least 1 by invariant")
			.clone();

		let new_round_claim = if let Some(zc_challenges) = self.zerocheck_challenges() {
			let alpha = if self.round == 1 {
				None
			} else {
				Some(zc_challenges[self.round - 2])
			};
			reduce_zerocheck_claim_round(round_claim, prev_rd_challenge, round_proof, alpha)
		} else {
			reduce_sumcheck_claim_round(round_claim, prev_rd_challenge, round_proof)
		}?;

		self.round_claim = new_round_claim;

		Ok(())
	}
}

/// Validate that evaluation domain starts with 0 & 1 and the size is exactly one greater than the
/// maximum individual degree of the polynomial.
fn check_evaluation_domain<F: Field>(
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

/// Evaluator for the regular (non-zerocheck) sumcheck protocol.
#[derive(Debug)]
struct RegularSumcheckEvaluator<'a, P, C>
where
	P: PackedField,
	C: CompositionPoly<P>,
{
	composition: &'a C,
	domain: &'a [P::Scalar],
}

impl<'a, F, P, C> SumcheckEvaluator<F> for RegularSumcheckEvaluator<'a, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	C: CompositionPoly<P>,
{
	fn n_round_evals(&self) -> usize {
		// NB: We skip evaluation of $r(X)$ at $X = 0$ as it is derivable from the
		// current_round_sum - $r(1)$.
		self.domain.len() - 1
	}

	fn process_vertex(
		&self,
		_index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		// Sumcheck evaluation at a specific point - given an array of 0 & 1 evaluations at some
		// index, use them to linearly interpolate each MLE value at domain point, and then
		// evaluate multivariate composite over those.

		round_evals[0] += self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

		// The rest require interpolation.
		for d in 2..self.domain.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain[d]);
				});

			round_evals[d - 1] += self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");
		}
	}
}

/// Evaluator for the first round of the zerocheck protocol.
///
/// In the first round, we do not need to evaluate at the point F::ONE, because the value is known
/// to be zero. This version of the zerocheck protocol uses the optimizations from section 3 of
/// [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
struct ZerocheckFirstRoundEvaluator<'a, P, C>
where
	P: PackedField,
	C: CompositionPoly<P>,
{
	composition: &'a C,
	domain: &'a [P::Scalar],
	eq_ind: MultilinearExtension<'a, P::Scalar>,
}

impl<'a, F, P, C> SumcheckEvaluator<F> for ZerocheckFirstRoundEvaluator<'a, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	C: CompositionPoly<P>,
{
	fn n_round_evals(&self) -> usize {
		// In the very first round of a sumcheck that comes from zerocheck, we can uniquely
		// determine the degree d univariate round polynomial r with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		self.domain.len() - 2
	}

	fn process_vertex(
		&self,
		index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		debug_assert!(index < self.eq_ind.size());

		let eq_ind_factor = self.eq_ind.evaluate_on_hypercube(index).unwrap_or(F::ZERO);

		// The rest require interpolation.
		for d in 2..self.domain.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 2] += composite_value * eq_ind_factor;
		}
	}
}

/// Evaluator for the later rounds of the zerocheck protocol.
///
/// This version of the zerocheck protocol uses the optimizations from section 3 of [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
struct ZerocheckLaterRoundEvaluator<'a, P, C>
where
	P: PackedField,
	C: CompositionPoly<P>,
{
	composition: &'a C,
	domain: &'a [P::Scalar],
	eq_ind: MultilinearExtension<'a, P::Scalar>,
}

impl<'a, F, P, C> SumcheckEvaluator<F> for ZerocheckLaterRoundEvaluator<'a, P, C>
where
	F: Field,
	P: PackedField<Scalar = F>,
	C: CompositionPoly<P>,
{
	fn n_round_evals(&self) -> usize {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		self.domain.len() - 1
	}

	fn process_vertex(
		&self,
		index: usize,
		evals_0: &[F],
		evals_1: &[F],
		evals_z: &mut [F],
		round_evals: &mut [F],
	) {
		debug_assert!(index < self.eq_ind.size());

		let eq_ind_factor = self.eq_ind.evaluate_on_hypercube(index).unwrap_or(F::ZERO);

		let composite_value = self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");
		round_evals[0] += composite_value * eq_ind_factor;

		// The rest require interpolation.
		for d in 2..self.domain.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;
		}
	}
}
