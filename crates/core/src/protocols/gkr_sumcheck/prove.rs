// Copyright 2024 Ulvetanna Inc.

use super::{
	gkr_sumcheck::{
		GkrSumcheckClaim, GkrSumcheckReductor, GkrSumcheckRound, GkrSumcheckRoundClaim,
		GkrSumcheckWitness,
	},
	Error,
};
use crate::{
	polynomial::{
		extrapolate_line, transparent::eq_ind::EqIndPartialEval, CompositionPoly,
		Error as PolynomialError, EvaluationDomain, MultilinearExtension,
	},
	protocols::abstract_sumcheck::{
		check_evaluation_domain, validate_rd_challenge, AbstractSumcheckEvaluator,
		AbstractSumcheckProver, AbstractSumcheckReductor, ProverState, ReducedClaim,
	},
	witness::MultilinearWitness,
};
use binius_field::{ExtensionField, Field, PackedField};
use getset::Getters;
use rayon::prelude::*;
use tracing::instrument;

/// A GKR Sumcheck protocol prover.
#[derive(Debug, Getters)]
pub struct GkrSumcheckProver<'a, F, PW, FS, CW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F> + ExtensionField<FS>,
	FS: Field,
	CW: CompositionPoly<PW::Scalar>,
{
	n_vars: usize,
	degree: usize,
	composition: CW,
	domain: &'a EvaluationDomain<FS>,
	#[getset(get = "pub")]
	round_claim: GkrSumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<GkrSumcheckRound<F>>,
	state: ProverState<PW, MultilinearWitness<'a, PW>>,

	gkr_challenge_point: Vec<F>,
	round_eq_ind: MultilinearExtension<PW::Scalar>,

	poly_mle: Option<MultilinearExtension<PW::Scalar>>,
}

impl<'a, F, PW, FS, CW> GkrSumcheckProver<'a, F, PW, FS, CW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F> + ExtensionField<FS>,
	FS: Field,
	CW: CompositionPoly<PW::Scalar>,
{
	pub fn new(
		domain: &'a EvaluationDomain<FS>,
		claim: GkrSumcheckClaim<F>,
		witness: GkrSumcheckWitness<'a, PW, CW>,
		gkr_round_challenge: &[F],
		switchover_fn: impl Fn(usize) -> usize,
	) -> Result<Self, Error> {
		let n_vars = claim.n_vars;
		let degree = claim.degree;

		if degree == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}
		check_evaluation_domain(degree, domain)?;

		if witness.poly.n_vars() != n_vars || n_vars != gkr_round_challenge.len() {
			return Err(Error::ProverClaimWitnessMismatch);
		}

		let state = ProverState::new(n_vars, witness.poly.multilinears, switchover_fn)?;

		let composition = witness.poly.composition;

		let round_claim = GkrSumcheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: claim.sum,
		};

		let pw_challenges = gkr_round_challenge
			.iter()
			.skip(1)
			.map(|&f| f.into())
			.collect::<Vec<PW::Scalar>>();
		let round_eq_ind =
			EqIndPartialEval::new(n_vars - 1, pw_challenges)?.multilinear_extension()?;

		let gkr_round_challenge = gkr_round_challenge.to_vec();

		let gkr_sumcheck_prover = GkrSumcheckProver {
			n_vars,
			degree,
			composition,
			domain,
			round_claim,
			round: 0,
			last_round_proof: None,
			state,
			gkr_challenge_point: gkr_round_challenge,
			round_eq_ind,
			poly_mle: Some(witness.current_layer),
		};

		Ok(gkr_sumcheck_prover)
	}

	#[instrument(skip_all, name = "gkr_sumcheck::finalize")]
	fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<ReducedClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round != self.n_vars() {
			return Err(Error::PrematureFinalizeCall);
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(prev_rd_challenge)?;
		}

		Ok(self.round_claim.into())
	}

	fn update_round_eq_ind(&mut self) -> Result<(), Error> {
		let current_evals = self.round_eq_ind.evals();
		let new_evals = (0..current_evals.len() >> 1)
			.into_par_iter()
			.map(|i| current_evals[i << 1] + current_evals[(i << 1) + 1])
			.collect();
		let new_multilin = MultilinearExtension::from_values(new_evals)?;
		self.round_eq_ind = new_multilin;

		Ok(())
	}

	fn compute_round_coeffs(&mut self) -> Result<Vec<PW::Scalar>, Error> {
		if self.degree == 1 {
			return Ok(vec![PW::Scalar::default()]);
		}

		let rd_vars = self.n_vars - self.round;
		let vertex_state_iterator = (0..1 << (rd_vars - 1)).into_par_iter().map(|_i| ());

		let round_coeffs = if self.round == 0 {
			let poly_mle = self.poly_mle.take().expect("poly_mle is initialized");
			let evaluator = GkrSumcheckFirstRoundEvaluator {
				degree: self.degree,
				eq_ind: &self.round_eq_ind,
				evaluation_domain: self.domain,
				domain_points: self.domain.points(),
				composition: &self.composition,
				poly_mle,
				gkr_challenge: self.gkr_challenge_point[0].into(),
			};
			self.state.calculate_round_coeffs(
				evaluator,
				self.round_claim.current_round_sum.into(),
				vertex_state_iterator,
			)
		} else {
			let evaluator = GkrSumcheckLaterRoundEvaluator {
				degree: self.degree,
				eq_ind: &self.round_eq_ind,
				evaluation_domain: self.domain,
				domain_points: self.domain.points(),
				composition: &self.composition,
				gkr_challenge: self.gkr_challenge_point[self.round].into(),
			};
			self.state.calculate_round_coeffs(
				evaluator,
				self.round_claim.current_round_sum.into(),
				vertex_state_iterator,
			)
		}?;

		Ok(round_coeffs)
	}

	#[instrument(skip_all, name = "gkr_sumcheck::execute_round")]
	fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<GkrSumcheckRound<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.n_vars {
			return Err(Error::TooManyExecuteRoundCalls);
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			let isomorphic_prev_rd_challenge = prev_rd_challenge.into();

			// Process switchovers of small field multilinears and folding of large field ones
			self.state.fold(isomorphic_prev_rd_challenge)?;

			// update the round eq indicator multilinear
			self.update_round_eq_ind()?;

			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		// Compute Round Coeffs using the appropriate evaluator
		let round_coeffs = self.compute_round_coeffs()?;

		// Convert round_coeffs to F
		let coeffs = round_coeffs
			.clone()
			.into_iter()
			.map(Into::into)
			.collect::<Vec<F>>();

		let proof_round = GkrSumcheckRound { coeffs };
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let reductor = GkrSumcheckReductor {
			gkr_challenge_point: &self.gkr_challenge_point,
		};
		let round_claim = self.round_claim.clone();
		let round_proof = self
			.last_round_proof
			.as_ref()
			.expect("round is at least 1 by invariant")
			.clone();

		let new_round_claim = reductor.reduce_round_claim(
			self.round - 1,
			round_claim,
			prev_rd_challenge,
			round_proof,
		)?;

		self.round_claim = new_round_claim;

		Ok(())
	}
}

impl<'a, F, PW, DomainField, CW> AbstractSumcheckProver<F>
	for GkrSumcheckProver<'a, F, PW, DomainField, CW>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CW: CompositionPoly<PW::Scalar>,
{
	type Error = Error;

	fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<GkrSumcheckRound<F>, Self::Error> {
		GkrSumcheckProver::execute_round(self, prev_rd_challenge)
	}

	fn finalize(self, prev_rd_challenge: Option<F>) -> Result<ReducedClaim<F>, Self::Error> {
		GkrSumcheckProver::finalize(self, prev_rd_challenge)
	}

	fn batch_proving_consistent(&self, other: &Self) -> bool {
		self.gkr_challenge_point == other.gkr_challenge_point
	}

	fn n_vars(&self) -> usize {
		self.n_vars
	}
}

pub struct GkrSumcheckFirstRoundEvaluator<'a, F, FS, C>
where
	FS: Field,
	F: Field + ExtensionField<FS>,
	C: CompositionPoly<F>,
{
	pub composition: &'a C,
	pub domain_points: &'a [FS],
	pub evaluation_domain: &'a EvaluationDomain<FS>,
	pub degree: usize,
	pub eq_ind: &'a MultilinearExtension<F>,
	pub poly_mle: MultilinearExtension<F>,
	pub gkr_challenge: F,
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P>
	for GkrSumcheckFirstRoundEvaluator<'a, P::Scalar, FS, C>
where
	FS: Field,
	P: PackedField,
	P::Scalar: ExtensionField<FS>,
	C: CompositionPoly<P::Scalar>,
{
	type VertexState = ();

	fn n_round_evals(&self) -> usize {
		debug_assert_eq!(self.domain_points.len(), self.degree + 1);
		self.degree
	}

	fn process_vertex(
		&self,
		i: usize,
		_vertex_state: Self::VertexState,
		evals_0: &[P::Scalar],
		evals_1: &[P::Scalar],
		evals_z: &mut [P::Scalar],
		round_evals: &mut [P::Scalar],
	) {
		debug_assert!(i < self.eq_ind.size());
		let eq_ind_factor = self
			.eq_ind
			.evaluate_on_hypercube(i)
			.unwrap_or(P::Scalar::ZERO);
		let poly_mle_one_eval = self
			.poly_mle
			.evaluate_on_hypercube(i << 1 | 1)
			.unwrap_or(P::Scalar::ZERO);

		// For X = 1, we can replace evaluating poly(1, i) with evaluating poly_mle(1, i)
		round_evals[0] += eq_ind_factor * poly_mle_one_eval;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line::<P::Scalar, FS>(
						evals_0_j,
						evals_1_j,
						self.domain_points[d],
					);
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
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith gkr_challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.gkr_challenge;
		let alpha_bar = P::Scalar::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap_or(P::Scalar::ZERO);
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}
pub struct GkrSumcheckLaterRoundEvaluator<'a, F, FS, C>
where
	FS: Field,
	F: Field + ExtensionField<FS>,
	C: CompositionPoly<F>,
{
	pub composition: &'a C,
	pub domain_points: &'a [FS],
	pub evaluation_domain: &'a EvaluationDomain<FS>,
	pub degree: usize,
	pub eq_ind: &'a MultilinearExtension<F>,
	pub gkr_challenge: F,
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P>
	for GkrSumcheckLaterRoundEvaluator<'a, P::Scalar, FS, C>
where
	FS: Field,
	P: PackedField,
	P::Scalar: ExtensionField<FS>,
	C: CompositionPoly<P::Scalar>,
{
	type VertexState = ();

	fn n_round_evals(&self) -> usize {
		debug_assert_eq!(self.domain_points.len(), self.degree + 1);
		self.degree
	}

	fn process_vertex(
		&self,
		i: usize,
		_vertex_state: Self::VertexState,
		evals_0: &[P::Scalar],
		evals_1: &[P::Scalar],
		evals_z: &mut [P::Scalar],
		round_evals: &mut [P::Scalar],
	) {
		debug_assert!(i < self.eq_ind.size());
		let eq_ind_factor = self
			.eq_ind
			.evaluate_on_hypercube(i)
			.unwrap_or(P::Scalar::ZERO);

		// Process X = 1
		let composite_value = self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

		round_evals[0] += eq_ind_factor * composite_value;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line::<P::Scalar, FS>(
						evals_0_j,
						evals_1_j,
						self.domain_points[d],
					);
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
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith gkr_challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.gkr_challenge;
		let alpha_bar = P::Scalar::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap_or(P::Scalar::ZERO);
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}
