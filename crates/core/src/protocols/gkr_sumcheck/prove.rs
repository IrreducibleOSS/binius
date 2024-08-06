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
		Error as PolynomialError, EvaluationDomain, EvaluationDomainFactory, MultilinearExtension,
		MultilinearPoly,
	},
	protocols::abstract_sumcheck::{
		check_evaluation_domain, validate_rd_challenge, AbstractSumcheckEvaluator,
		AbstractSumcheckProversState, AbstractSumcheckReductor, AbstractSumcheckWitness,
		CommonProversState, ReducedClaim,
	},
};
use binius_field::{packed::get_packed_slice, ExtensionField, Field, PackedField};
use getset::Getters;
use rayon::prelude::*;
use std::marker::PhantomData;
use tracing::instrument;

pub struct GkrSumcheckProversState<'a, F, PW, DomainField, EDF, CW, M>
where
	F: Field,
	PW: PackedField,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Send + Sync,
{
	common: CommonProversState<(usize, usize), PW, M>,
	evaluation_domain_factory: EDF,
	gkr_round_challenge: &'a [F],
	round_eq_ind: MultilinearExtension<PW>,
	_domain_field_marker: PhantomData<DomainField>,
	_cw_marker: PhantomData<CW>,
}

impl<'a, F, PW, DomainField, EDF, CW, M> GkrSumcheckProversState<'a, F, PW, DomainField, EDF, CW, M>
where
	F: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Send + Sync,
{
	pub fn new(
		n_vars: usize,
		evaluation_domain_factory: EDF,
		gkr_round_challenge: &'a [F],
		switchover_fn: impl Fn(usize) -> usize + 'static,
	) -> Result<Self, Error> {
		let common = CommonProversState::new(n_vars, switchover_fn);

		let pw_scalar_challenges = gkr_round_challenge
			.iter()
			.skip(1)
			.map(|&f| f.into())
			.collect::<Vec<PW::Scalar>>();

		let round_eq_ind =
			EqIndPartialEval::new(n_vars - 1, pw_scalar_challenges)?.multilinear_extension()?;

		Ok(Self {
			common,
			evaluation_domain_factory,
			gkr_round_challenge,
			round_eq_ind,
			_domain_field_marker: PhantomData,
			_cw_marker: PhantomData,
		})
	}

	fn update_round_eq_ind(&mut self) -> Result<(), Error> {
		let current_evals = self.round_eq_ind.evals();
		let new_evals = (0..current_evals.len() >> 1)
			.into_par_iter()
			.map(|i| {
				PW::from_fn(|j| {
					let index = i * PW::WIDTH + j;
					let eval0 = get_packed_slice(current_evals, index << 1);
					let eval1 = get_packed_slice(current_evals, (index << 1) + 1);

					eval0 + eval1
				})
			})
			.collect();

		self.round_eq_ind = MultilinearExtension::from_values(new_evals)?;
		Ok(())
	}
}

impl<'a, F, PW, DomainField, EDF, CW, M> AbstractSumcheckProversState<F>
	for GkrSumcheckProversState<'a, F, PW, DomainField, EDF, CW, M>
where
	F: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Send + Sync,
{
	type Error = Error;

	type PackedWitnessField = PW;

	type Claim = GkrSumcheckClaim<F>;
	type Witness = GkrSumcheckWitness<PW, CW, M>;
	type Prover = GkrSumcheckProver<'a, F, PW, DomainField, CW, M>;

	fn new_prover(
		&mut self,
		claim: GkrSumcheckClaim<F>,
		witness: GkrSumcheckWitness<PW, CW, M>,
		seq_id: usize,
	) -> Result<Self::Prover, Error> {
		if claim.r != self.gkr_round_challenge {
			return Err(Error::MismatchedGkrChallengeInClaimsBatch);
		}
		let multilinears = witness
			.multilinears(seq_id, &[])?
			.into_iter()
			.collect::<Vec<_>>();
		self.common.extend(multilinears.clone())?;
		let domain = self.evaluation_domain_factory.create(claim.degree + 1)?;
		let multilinear_ids = multilinears
			.into_iter()
			.map(|(id, _)| id)
			.collect::<Vec<_>>();
		let prover = GkrSumcheckProver::new(
			claim,
			witness,
			domain,
			multilinear_ids,
			self.gkr_round_challenge,
		)?;
		Ok(prover)
	}

	fn pre_execute_rounds(&mut self, prev_rd_challenge: Option<F>) -> Result<(), Error> {
		self.common
			.pre_execute_rounds(prev_rd_challenge.map(Into::into))?;

		if prev_rd_challenge.is_some() {
			self.update_round_eq_ind()?;
		}

		Ok(())
	}

	fn prover_execute_round(
		&self,
		prover: &mut Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<GkrSumcheckRound<F>, Error> {
		prover.execute_round(self, prev_rd_challenge)
	}

	fn prover_finalize(
		prover: Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<ReducedClaim<F>, Error> {
		prover.finalize(prev_rd_challenge)
	}
}

/// A GKR Sumcheck protocol prover.
#[derive(Debug, Getters)]
pub struct GkrSumcheckProver<'a, F, PW, DomainField, CW, M>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CW: CompositionPoly<PW>,
{
	n_vars: usize,
	degree: usize,
	composition: CW,
	domain: EvaluationDomain<DomainField>,
	multilinear_ids: Vec<(usize, usize)>,

	#[getset(get = "pub")]
	round_claim: GkrSumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<GkrSumcheckRound<F>>,

	gkr_round_challenge: &'a [F],

	poly_mle: Option<M>,
	_pw_marker: PhantomData<PW>,
}

impl<'a, F, PW, DomainField, CW, M> GkrSumcheckProver<'a, F, PW, DomainField, CW, M>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F> + ExtensionField<DomainField>,
	DomainField: Field,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Send + Sync,
{
	pub fn new(
		claim: GkrSumcheckClaim<F>,
		witness: GkrSumcheckWitness<PW, CW, M>,
		domain: EvaluationDomain<DomainField>,
		multilinear_ids: Vec<(usize, usize)>,
		gkr_round_challenge: &'a [F],
	) -> Result<Self, Error> {
		let n_vars = claim.n_vars;
		let degree = claim.degree;

		if degree == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}
		check_evaluation_domain(degree, &domain)?;

		if gkr_round_challenge.len() + 1 < n_vars {
			return Err(Error::NotEnoughGkrRoundChallenges);
		}

		if witness.poly.n_vars() != n_vars || n_vars != gkr_round_challenge.len() {
			return Err(Error::ProverClaimWitnessMismatch);
		}

		let composition = witness.poly.composition;

		let round_claim = GkrSumcheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: claim.sum,
		};

		let gkr_sumcheck_prover = GkrSumcheckProver {
			n_vars,
			degree,
			composition,
			domain,
			multilinear_ids,
			round_claim,
			round: 0,
			last_round_proof: None,
			gkr_round_challenge,
			poly_mle: Some(witness.current_layer),
			_pw_marker: PhantomData,
		};

		Ok(gkr_sumcheck_prover)
	}

	#[instrument(skip_all, name = "gkr_sumcheck::finalize", level = "debug")]
	fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<ReducedClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round != self.n_vars {
			return Err(Error::PrematureFinalizeCall);
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(prev_rd_challenge)?;
		}

		Ok(self.round_claim.into())
	}

	fn compute_round_coeffs<EDF>(
		&self,
		provers_state: &GkrSumcheckProversState<'a, F, PW, DomainField, EDF, CW, M>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
	{
		if self.degree == 1 {
			return Ok(vec![PW::Scalar::default()]);
		}

		let rd_vars = self.n_vars - self.round;
		let vertex_state_iterator = (0..1 << (rd_vars - 1)).into_par_iter().map(|_i| ());

		let round_coeffs = if self.round == 0 {
			let poly_mle = self.poly_mle.as_ref().expect("poly_mle is initialized");
			let evaluator = GkrSumcheckFirstRoundEvaluator {
				degree: self.degree,
				eq_ind: &provers_state.round_eq_ind,
				evaluation_domain: &self.domain,
				domain_points: self.domain.points(),
				composition: &self.composition,
				poly_mle,
				gkr_challenge: self.gkr_round_challenge[0].into(),
			};
			provers_state.common.calculate_round_coeffs(
				self.multilinear_ids.as_slice(),
				evaluator,
				self.round_claim.current_round_sum.into(),
				vertex_state_iterator,
			)
		} else {
			let evaluator = GkrSumcheckLaterRoundEvaluator {
				degree: self.degree,
				eq_ind: &provers_state.round_eq_ind,
				evaluation_domain: &self.domain,
				domain_points: self.domain.points(),
				composition: &self.composition,
				gkr_challenge: self.gkr_round_challenge[self.round].into(),
			};
			provers_state.common.calculate_round_coeffs(
				self.multilinear_ids.as_slice(),
				evaluator,
				self.round_claim.current_round_sum.into(),
				vertex_state_iterator,
			)
		}?;

		Ok(round_coeffs)
	}

	#[instrument(skip_all, name = "gkr_sumcheck::execute_round", level = "debug")]
	fn execute_round<EDF>(
		&mut self,
		provers_state: &GkrSumcheckProversState<'a, F, PW, DomainField, EDF, CW, M>,
		prev_rd_challenge: Option<F>,
	) -> Result<GkrSumcheckRound<F>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
	{
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.n_vars {
			return Err(Error::TooManyExecuteRoundCalls);
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		// Compute Round Coeffs using the appropriate evaluator
		let round_coeffs = self.compute_round_coeffs(provers_state)?;

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
			gkr_challenge_point: self.gkr_round_challenge,
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

// eligibility - gkr challenge point

pub struct GkrSumcheckFirstRoundEvaluator<'a, PW, DomainField, C, M>
where
	PW: PackedField<Scalar: ExtensionField<DomainField>>,
	DomainField: Field,
	C: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Send + Sync,
{
	pub composition: &'a C,
	pub domain_points: &'a [DomainField],
	pub evaluation_domain: &'a EvaluationDomain<DomainField>,
	pub degree: usize,
	pub eq_ind: &'a MultilinearExtension<PW>,
	pub poly_mle: &'a M,
	pub gkr_challenge: PW::Scalar,
}

impl<'a, PW, DomainField, C, M> AbstractSumcheckEvaluator<PW>
	for GkrSumcheckFirstRoundEvaluator<'a, PW, DomainField, C, M>
where
	DomainField: Field,
	PW: PackedField<Scalar: ExtensionField<DomainField>>,
	C: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Send + Sync,
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
		evals_0: &[PW::Scalar],
		evals_1: &[PW::Scalar],
		evals_z: &mut [PW::Scalar],
		round_evals: &mut [PW::Scalar],
	) {
		debug_assert!(i < self.eq_ind.size());
		let eq_ind_factor = self
			.eq_ind
			.evaluate_on_hypercube(i)
			.unwrap_or(PW::Scalar::ZERO);
		let poly_mle_one_eval = self
			.poly_mle
			.evaluate_on_hypercube(i << 1 | 1)
			.unwrap_or(PW::Scalar::ZERO);

		// For X = 1, we can replace evaluating poly(1, i) with evaluating poly_mle(1, i)
		round_evals[0] += eq_ind_factor * poly_mle_one_eval;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line::<PW::Scalar, DomainField>(
						evals_0_j,
						evals_1_j,
						self.domain_points[d],
					);
				});

			let composite_value = self
				.composition
				.evaluate_scalar(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: PW::Scalar,
		mut round_evals: Vec<PW::Scalar>,
	) -> Result<Vec<PW::Scalar>, PolynomialError> {
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith gkr_challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.gkr_challenge;
		let alpha_bar = PW::Scalar::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap_or(PW::Scalar::ZERO);
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}
pub struct GkrSumcheckLaterRoundEvaluator<'a, PW, DomainField, C>
where
	DomainField: Field,
	PW: PackedField<Scalar: ExtensionField<DomainField>>,
	C: CompositionPoly<PW>,
{
	pub composition: &'a C,
	pub domain_points: &'a [DomainField],
	pub evaluation_domain: &'a EvaluationDomain<DomainField>,
	pub degree: usize,
	pub eq_ind: &'a MultilinearExtension<PW>,
	pub gkr_challenge: PW::Scalar,
}

impl<'a, PW, DomainField, C> AbstractSumcheckEvaluator<PW>
	for GkrSumcheckLaterRoundEvaluator<'a, PW, DomainField, C>
where
	DomainField: Field,
	PW: PackedField<Scalar: ExtensionField<DomainField>>,
	C: CompositionPoly<PW>,
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
		evals_0: &[PW::Scalar],
		evals_1: &[PW::Scalar],
		evals_z: &mut [PW::Scalar],
		round_evals: &mut [PW::Scalar],
	) {
		debug_assert!(i < self.eq_ind.size());
		let eq_ind_factor = self
			.eq_ind
			.evaluate_on_hypercube(i)
			.unwrap_or(PW::Scalar::ZERO);

		// Process X = 1
		let composite_value = self
			.composition
			.evaluate_scalar(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

		round_evals[0] += eq_ind_factor * composite_value;

		// The rest require interpolation.
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line::<PW::Scalar, DomainField>(
						evals_0_j,
						evals_1_j,
						self.domain_points[d],
					);
				});

			let composite_value = self
				.composition
				.evaluate_scalar(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: PW::Scalar,
		mut round_evals: Vec<PW::Scalar>,
	) -> Result<Vec<PW::Scalar>, PolynomialError> {
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith gkr_challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.gkr_challenge;
		let alpha_bar = PW::Scalar::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap_or(PW::Scalar::ZERO);
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}
