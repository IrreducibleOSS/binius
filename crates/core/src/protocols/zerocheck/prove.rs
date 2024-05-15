// Copyright 2023 Ulvetanna Inc.

use super::{
	error::Error,
	zerocheck::{
		ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckReductor, ZerocheckRound,
		ZerocheckRoundClaim, ZerocheckWitness,
	},
};
use crate::{
	oracle::CompositePolyOracle,
	polynomial::{
		extrapolate_line, transparent::eq_ind::EqIndPartialEval, CompositionPoly, EvaluationDomain,
		MultilinearExtension,
	},
	protocols::{
		abstract_sumcheck::{
			self, AbstractSumcheckEvaluator, AbstractSumcheckProver, AbstractSumcheckReductor,
			Error as AbstractSumcheckError, ProverState,
		},
		evalcheck::EvalcheckClaim,
	},
	witness::MultilinearWitness,
};
use binius_field::{Field, PackedField, TowerField};
use either::Either;
use getset::Getters;
use p3_challenger::{CanObserve, CanSample};
use rayon::prelude::*;
use tracing::instrument;

/// Prove a zerocheck to evalcheck reduction.
#[instrument(skip_all, name = "zerocheck::prove")]
pub fn prove<'a, F, PW, CW, CH>(
	claim: &ZerocheckClaim<F>,
	witness: ZerocheckWitness<'a, PW, CW>,
	domain: &EvaluationDomain<PW::Scalar>,
	mut challenger: CH,
	switchover_fn: impl Fn(usize) -> usize,
) -> Result<ZerocheckProveOutput<F>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: TowerField + From<F>,
	CW: CompositionPoly<PW::Scalar>,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = witness.n_vars();
	let zerocheck_challenges = challenger.sample_vec(n_vars - 1);

	let zerocheck_prover =
		ZerocheckProver::new(domain, claim.clone(), witness, zerocheck_challenges, switchover_fn)?;

	let (evalcheck_claim, rounds) =
		abstract_sumcheck::prove(claim.poly.n_vars(), zerocheck_prover, challenger)?;

	let zerocheck_proof = ZerocheckProof { rounds };
	let output = ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	};
	Ok(output)
}

/// A zerocheck protocol prover.
///
/// To prove a zerocheck claim, supply a multivariate composite witness. In
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
pub struct ZerocheckProver<'a, F, PW, CW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW::Scalar>,
{
	oracle: CompositePolyOracle<F>,
	composition: CW,
	domain: &'a EvaluationDomain<PW::Scalar>,
	#[getset(get = "pub")]
	round_claim: ZerocheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<ZerocheckRound<F>>,
	state: ProverState<PW, MultilinearWitness<'a, PW>>,

	zerocheck_challenges: Vec<F>,
	round_eq_ind: MultilinearExtension<'static, PW::Scalar>,
}

impl<'a, F, PW, CW> ZerocheckProver<'a, F, PW, CW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW::Scalar>,
{
	/// Start a new sumcheck instance with claim in field `F`. Witness may be given in
	/// a different (but isomorphic) packed field PW. `switchover_fn` closure specifies
	/// switchover round number per multilinear polynomial as a function of its
	/// [`MultilinearPoly::extension_degree`] value.
	pub fn new(
		domain: &'a EvaluationDomain<PW::Scalar>,
		claim: ZerocheckClaim<F>,
		witness: ZerocheckWitness<'a, PW, CW>,
		zerocheck_challenges: Vec<F>,
		switchover_fn: impl Fn(usize) -> usize,
	) -> Result<Self, Error> {
		let n_vars = claim.poly.n_vars();

		if claim.poly.max_individual_degree() == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}

		if witness.n_vars() != n_vars {
			let err_str = format!(
				"Claim and Witness n_vars mismatch in sumcheck. Claim: {}, Witness: {}",
				n_vars,
				witness.n_vars(),
			);

			return Err(Error::ProverClaimWitnessMismatch(err_str));
		}

		check_evaluation_domain(claim.poly.max_individual_degree(), domain)?;

		let state = ProverState::new(n_vars, witness.multilinears, switchover_fn)?;

		let composition = witness.composition;

		let round_claim = ZerocheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: F::ZERO,
		};

		let pw_challenges = zerocheck_challenges
			.iter()
			.map(|&f| f.into())
			.collect::<Vec<PW::Scalar>>();
		let round_eq_ind =
			EqIndPartialEval::new(n_vars - 1, pw_challenges)?.multilinear_extension()?;

		let zerocheck_prover = ZerocheckProver {
			oracle: claim.poly,
			composition,
			domain,
			round_claim,
			round: 0,
			last_round_proof: None,
			zerocheck_challenges,
			round_eq_ind,
			state,
		};

		Ok(zerocheck_prover)
	}

	pub fn n_vars(&self) -> usize {
		self.oracle.n_vars()
	}

	#[instrument(skip_all, name = "zerocheck::finalize")]
	pub fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

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

		let reductor = ZerocheckReductor {
			alphas: &self.zerocheck_challenges,
		};
		let evalcheck_claim = reductor.reduce_final_round_claim(&self.oracle, self.round_claim)?;
		Ok(evalcheck_claim)
	}

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

	#[instrument(skip_all, name = "sumcheck::execute_round")]
	pub fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<ZerocheckRound<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.n_vars() {
			return Err(Error::ImproperInput("too many execute_round calls".to_string()));
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// Process switchovers of small field multilinears and folding of large field ones
			self.state.fold(prev_rd_challenge.into())?;

			// update the round eq indicator multilinear
			self.update_round_eq_ind()?;

			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		let degree = self.oracle.max_individual_degree();
		let evaluator = if self.round == 0 {
			Either::Left(ZerocheckFirstRoundEvaluator {
				degree,
				eq_ind: self.round_eq_ind.to_ref(),
				evaluation_domain: self.domain,
				domain_points: self.domain.points(),
				composition: &self.composition,
			})
		} else {
			Either::Right(ZerocheckLaterRoundEvaluator {
				degree,
				eq_ind: self.round_eq_ind.to_ref(),
				round_zerocheck_challenge: self.zerocheck_challenges[self.round - 1].into(),
				evaluation_domain: self.domain,
				domain_points: self.domain.points(),
				composition: &self.composition,
			})
		};

		let round_coeffs = self
			.state
			.calculate_round_coeffs(evaluator, self.round_claim.current_round_sum.into())?;

		let coeffs = round_coeffs
			.clone()
			.into_iter()
			.map(Into::into)
			.collect::<Vec<F>>();

		let proof_round = ZerocheckRound { coeffs };
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let reductor = ZerocheckReductor {
			alphas: &self.zerocheck_challenges,
		};
		let round_claim = self.round_claim.clone();
		let round_proof = self
			.last_round_proof
			.as_ref()
			.expect("round is at least 1 by invariant")
			.clone();

		let new_round_claim = reductor.reduce_intermediate_round_claim(
			self.round - 1,
			round_claim,
			prev_rd_challenge,
			round_proof,
		)?;

		self.round_claim = new_round_claim;

		Ok(())
	}
}

impl<'a, F, PW, CW> AbstractSumcheckProver<F> for ZerocheckProver<'a, F, PW, CW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW::Scalar>,
{
	fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<ZerocheckRound<F>, AbstractSumcheckError> {
		let round = ZerocheckProver::execute_round(self, prev_rd_challenge)?;
		Ok(round)
	}

	fn finalize(
		self,
		prev_rd_challenge: Option<F>,
	) -> Result<EvalcheckClaim<F>, AbstractSumcheckError> {
		let evalcheck_claim = ZerocheckProver::finalize(self, prev_rd_challenge)?;
		Ok(evalcheck_claim)
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
pub struct ZerocheckFirstRoundEvaluator<'a, F: Field, C: CompositionPoly<F>> {
	pub composition: &'a C,
	pub domain_points: &'a [F],
	pub evaluation_domain: &'a EvaluationDomain<F>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<'a, F>,
}

impl<'a, F: Field, C: CompositionPoly<F>> AbstractSumcheckEvaluator<F>
	for ZerocheckFirstRoundEvaluator<'a, F, C>
{
	fn n_round_evals(&self) -> usize {
		// In the very first round of a sumcheck that comes from zerocheck, we can uniquely
		// determine the degree d univariate round polynomial r with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		self.degree - 1
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
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 2] += composite_value * eq_ind_factor;
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, AbstractSumcheckError> {
		debug_assert_eq!(current_round_sum, F::ZERO);
		// We are given $r(2), \ldots, r(d+1)$.
		// From context, we infer that $r(0) = r(1) = 0$.
		round_evals.insert(0, F::ZERO);
		round_evals.insert(0, F::ZERO);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;

		Ok(coeffs[2..].to_vec())
	}
}

/// Evaluator for the later rounds of the zerocheck protocol.
///
/// This version of the zerocheck protocol uses the optimizations from section 3 of [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckLaterRoundEvaluator<'a, F: Field, C: CompositionPoly<F>> {
	pub composition: &'a C,
	pub domain_points: &'a [F],
	pub evaluation_domain: &'a EvaluationDomain<F>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<'a, F>,
	pub round_zerocheck_challenge: F,
}

impl<'a, F: Field, C: CompositionPoly<F>> AbstractSumcheckEvaluator<F>
	for ZerocheckLaterRoundEvaluator<'a, F, C>
{
	fn n_round_evals(&self) -> usize {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		self.degree
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
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			let composite_value = self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, AbstractSumcheckError> {
		// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d+1)$
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.round_zerocheck_challenge;
		let alpha_bar = F::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;

		Ok(coeffs[1..].to_vec())
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

/// Ensures that previous round challenge is present if and only if not in the first round.
fn validate_rd_challenge<F: Field>(
	prev_rd_challenge: Option<F>,
	round: usize,
) -> Result<(), Error> {
	if prev_rd_challenge.is_none() != (round == 0) {
		return Err(Error::ImproperInput(format!(
			"incorrect optional challenge: is_some()={:?} at round {}",
			prev_rd_challenge.is_some(),
			round
		)));
	}

	Ok(())
}
