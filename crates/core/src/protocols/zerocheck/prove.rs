// Copyright 2023 Ulvetanna Inc.

use super::{
	batch::batch_prove,
	error::Error,
	zerocheck::{
		ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput, ZerocheckReductor, ZerocheckRound,
		ZerocheckRoundClaim,
	},
};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::OracleId,
	polynomial::{
		extrapolate_line, transparent::eq_ind::EqIndPartialEval, CompositionPoly,
		Error as PolynomialError, EvaluationDomain, EvaluationDomainFactory, MultilinearExtension,
		MultilinearQuery,
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

#[cfg(feature = "debug_validate_sumcheck")]
use super::zerocheck::validate_witness;

/// Prove a zerocheck to evalcheck reduction.
/// FS is the domain type.
#[instrument(skip_all, name = "zerocheck::prove")]
pub fn prove<F, PW, DomainField, CH>(
	claim: &ZerocheckClaim<F>,
	witness: impl AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
	evaluation_domain_factory: impl EvaluationDomainFactory<DomainField>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	challenger: CH,
) -> Result<ZerocheckProveOutput<F>, Error>
where
	F: Field,
	DomainField: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	CH: CanSample<F> + CanObserve<F>,
{
	let batch_proof = batch_prove::<F, PW, DomainField, CH>(
		[(claim.clone(), witness)],
		evaluation_domain_factory,
		switchover_fn,
		challenger,
	)?;

	Ok(ZerocheckProveOutput {
		evalcheck_claim: batch_proof
			.evalcheck_claims
			.first()
			.expect("exactly one")
			.clone(),
		zerocheck_proof: ZerocheckProof {
			rounds: batch_proof.proof.rounds,
		},
	})
}

pub struct ZerocheckProversState<'a, F, PW, DomainField, EDF, W>
where
	F: Field,
	PW: PackedField,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW>,
{
	common: CommonProversState<OracleId, PW, W::Multilinear>,
	evaluation_domain_factory: EDF,
	zerocheck_challenges: &'a [F],
	round_eq_ind: MultilinearExtension<PW>,
	_f_marker: PhantomData<F>,
	_domain_field_marker: PhantomData<DomainField>,
	_w_marker: PhantomData<W>,
}

impl<'a, F, PW, DomainField, EDF, W> ZerocheckProversState<'a, F, PW, DomainField, EDF, W>
where
	F: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW>,
{
	pub fn new(
		n_vars: usize,
		evaluation_domain_factory: EDF,
		zerocheck_challenges: &'a [F],
		switchover_fn: impl Fn(usize) -> usize + 'static,
	) -> Result<Self, Error> {
		let common = CommonProversState::new(n_vars, switchover_fn);

		if zerocheck_challenges.len() + 1 < n_vars {
			return Err(Error::NotEnoughZerocheckChallenges);
		}

		let pw_scalar_challenges = zerocheck_challenges
			.iter()
			.map(|&f| f.into())
			.collect::<Vec<PW::Scalar>>();
		let round_eq_ind =
			EqIndPartialEval::new(n_vars - 1, pw_scalar_challenges)?.multilinear_extension()?;

		Ok(Self {
			common,
			evaluation_domain_factory,
			zerocheck_challenges,
			round_eq_ind,
			_f_marker: PhantomData,
			_domain_field_marker: PhantomData,
			_w_marker: PhantomData,
		})
	}

	// Update the round_eq_ind for the next zerocheck round
	//
	// Let
	//  * $n$ be the number of variables in the zerocheck claim
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

impl<'a, F, PW, DomainField, EDF, W> AbstractSumcheckProversState<F>
	for ZerocheckProversState<'a, F, PW, DomainField, EDF, W>
where
	F: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
{
	type Error = Error;

	type PackedWitnessField = PW;

	type Claim = ZerocheckClaim<F>;
	type Witness = W;
	type Prover = ZerocheckProver<'a, F, PW, DomainField, W>;

	fn new_prover(
		&mut self,
		claim: ZerocheckClaim<F>,
		witness: W,
		seq_id: usize,
	) -> Result<Self::Prover, Error> {
		let ids = claim.poly.inner_polys_oracle_ids().collect::<Vec<_>>();
		self.common
			.extend(witness.multilinears(seq_id, ids.as_slice())?)?;
		let domain = self
			.evaluation_domain_factory
			.create(claim.poly.max_individual_degree() + 1)?;
		let prover = ZerocheckProver::new(claim, witness, domain, self.zerocheck_challenges)?;
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
	) -> Result<ZerocheckRound<F>, Error> {
		prover.execute_round(self, prev_rd_challenge)
	}

	fn prover_finalize(
		prover: Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<ReducedClaim<F>, Error> {
		prover.finalize(prev_rd_challenge)
	}
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
pub struct ZerocheckProver<'a, F, PW, DomainField, W>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F> + ExtensionField<DomainField>,
	DomainField: Field,
	W: AbstractSumcheckWitness<PW>,
{
	#[getset(get = "pub")]
	claim: ZerocheckClaim<F>,
	witness: W,
	domain: EvaluationDomain<DomainField>,
	oracle_ids: Vec<OracleId>,

	#[getset(get = "pub")]
	round_claim: ZerocheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<ZerocheckRound<F>>,

	zerocheck_challenges: &'a [F],

	// Junk (scratch space) at the start of each round
	// After prover computes ith round polynomial, represents evaluations of
	// * $Q_i(r_0, \ldots, r_{i-1}, X, x_{i+1}, \ldots, x_{n-1})$
	// for $X \in \{2, \ldots, d\}$, and $x_{i+1}, \ldots, x_{n-1} \in \{0, 1\}$
	// where
	// * $d$ is degree of polynomial in zerocheck claim
	// * $r_i$ is the verifier challenge at the end of round $i$.
	// with sorting that is lexicographically-inspired (lowest index = least significant).
	round_q: Vec<PW::Scalar>,
	// None initially
	// After ith round, represents the $\bar{Q}_i$ multilinear polynomial partially evaluated
	// at lowest $i$ variables with the $i$ verifier challenges received so far.
	round_q_bar: Option<MultilinearExtension<PW::Scalar>>,
	// inverse of domain[i] * domain[i] - 1 for i in 0, ..., d-2
	smaller_denom_inv: Vec<DomainField>,
	// Subdomain of domain, but without 0 and 1 (can be used for degree d-2 polynomials)
	smaller_domain: EvaluationDomain<DomainField>,
}

impl<'a, F, PW, DomainField, W> ZerocheckProver<'a, F, PW, DomainField, W>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F> + ExtensionField<DomainField>,
	DomainField: Field,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
{
	/// Start a new zerocheck instance with claim in field `F`. Witness may be given in
	/// a different (but isomorphic) packed field PW. `switchover_fn` closure specifies
	/// switchover round number per multilinear polynomial as a function of its
	/// [`crate::polynomial::MultilinearPoly::extension_degree`] value.
	pub fn new(
		claim: ZerocheckClaim<F>,
		witness: W,
		domain: EvaluationDomain<DomainField>,
		zerocheck_challenges: &'a [F],
	) -> Result<Self, Error> {
		#[cfg(feature = "debug_validate_sumcheck")]
		validate_witness(&claim, &witness)?;

		let n_vars = claim.n_vars();
		let degree = claim.poly.max_individual_degree();

		if degree == 0 {
			return Err(Error::PolynomialDegreeIsZero);
		}
		check_evaluation_domain(degree, &domain)?;

		let oracle_ids = claim.poly.inner_polys_oracle_ids().collect::<Vec<_>>();

		if zerocheck_challenges.len() + 1 < n_vars {
			return Err(Error::NotEnoughZerocheckChallenges);
		}
		let zerocheck_challenges = &zerocheck_challenges[zerocheck_challenges.len() + 1 - n_vars..];

		let round_claim = ZerocheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: F::ZERO,
		};

		let round_q = vec![PW::Scalar::default(); (1 << (n_vars - 1)) * (degree - 1)];
		let smaller_domain_points = domain.points()[2..].to_vec();
		let smaller_denom_inv = domain.points()[2..]
			.iter()
			.map(|&x| (x * (x - DomainField::ONE)).invert().unwrap())
			.collect::<Vec<_>>();
		let smaller_domain = EvaluationDomain::from_points(smaller_domain_points)?;

		let zerocheck_prover = ZerocheckProver {
			claim,
			witness,
			domain,
			oracle_ids,
			round_claim,
			round: 0,
			last_round_proof: None,
			zerocheck_challenges,
			round_q,
			round_q_bar: None,
			smaller_denom_inv,
			smaller_domain,
		};

		Ok(zerocheck_prover)
	}

	#[instrument(skip_all, name = "zerocheck::finalize")]
	fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<ReducedClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round != self.claim.n_vars() {
			return Err(Error::PrematureFinalizeCall);
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(prev_rd_challenge)?;
		}

		Ok(self.round_claim.into())
	}

	// Updates auxiliary state that we store corresponding to Section 4 Optimizations in [Gruen24].
	// [Gruen24]: https://eprint.iacr.org/2024/108
	fn update_round_q(&mut self, prev_rd_challenge: PW::Scalar) -> Result<(), Error> {
		let rd_vars = self.claim.n_vars() - self.round;
		let new_q_len = self.round_q.len() / 2;

		// Let d be the degree of the polynomial in the zerocheck claim.
		let specialized_q_values = if self.smaller_domain.size() == 0 {
			// Special handling for the d = 1 (multilinear) case
			vec![PW::Scalar::default(); 1 << rd_vars]
		} else if self.smaller_domain.size() == 1 {
			// We do not need to interpolate in this special d = 2 case
			std::mem::replace(&mut self.round_q, vec![PW::Scalar::default(); new_q_len])
		} else {
			// This is for the d >= 3 case
			//
			// Let r_i be the prev_rd_challenge
			// * Each d-1 sized chunk of round_q is, for some fixed x_{i+1}, ..., x_{n-1}, evaluations of
			// Q_i(r_0, ..., r_{i-1}, X, x_{i+1}, ..., x_{n-1}) at X = 2, ..., d
			// * Q_i is degree d-2 in X, so we can extrapolate each chunk into the evaluation
			// of Q_i(r_0, ..., r_{i-1}, X, x_{i+1}, ..., x_{n-1}) at X = r_i
			self.round_q
				.chunks(self.smaller_domain.size())
				.map(|chunk| self.smaller_domain.extrapolate(chunk, prev_rd_challenge))
				.collect::<Result<Vec<_>, _>>()?
		};

		// We currently have the evaluations of $\bar{Q}_i(r_0, \ldots, r_{i-1}, x_i, x_{i+1}, \ldots, x_{n-1})$
		// at $x_i, x_{i+1}, \ldots, x_{n-1} \in \{0, 1\}$. It is linear in $x_i$.
		// We first partially evaluate $\bar{Q}_i$ at  $x_i = r_i$.
		// Then, we update these evaluations at a fixed $x_{i+1}, \ldots, x_{n-1}$ by
		// adding $r_i * (r_i - 1) * Q_i(r_0, \ldots, r_{i-1}, 2, x_{i+1}, \ldots, x_{n-1})$ to obtain
		// evaluations of $\bar{Q}_{i+1}(r_0, \ldots, r_{i-1}, r_i, x_{i+1}, \ldots, x_{n-1})$.
		let coeff = prev_rd_challenge * (prev_rd_challenge - PW::Scalar::ONE);

		let mut new_q_bar_values = specialized_q_values;
		new_q_bar_values.par_iter_mut().for_each(|e| *e *= coeff);

		if let Some(prev_q_bar) = self.round_q_bar.as_mut() {
			let query = MultilinearQuery::<PW::Scalar>::with_full_query(&[prev_rd_challenge])?;
			let specialized_prev_q_bar = prev_q_bar.evaluate_partial_low(&query)?;
			let specialized_prev_q_bar_evals = specialized_prev_q_bar.evals();

			const CHUNK_SIZE: usize = 64;

			new_q_bar_values
				.par_chunks_mut(CHUNK_SIZE)
				.enumerate()
				.for_each(|(chunk_index, chunks)| {
					for (k, e) in chunks.iter_mut().enumerate() {
						*e += specialized_prev_q_bar_evals[chunk_index * CHUNK_SIZE + k];
					}
				});
		}
		self.round_q_bar = Some(MultilinearExtension::from_values(new_q_bar_values)?);

		// Step 3:
		// Truncate round_q
		self.round_q.truncate(new_q_len);

		Ok(())
	}

	fn compute_round_coeffs<EDF>(
		&mut self,
		provers_state: &ZerocheckProversState<'a, F, PW, DomainField, EDF, W>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
	{
		let degree = self.claim.poly.max_individual_degree();
		if degree == 1 {
			return Ok(vec![PW::Scalar::default()]);
		}

		let vertex_state_iterator = self.round_q.par_chunks_exact_mut(degree - 1);

		let round_coeffs = if self.round == 0 {
			let evaluator = ZerocheckFirstRoundEvaluator {
				degree,
				eq_ind: provers_state.round_eq_ind.to_ref(),
				evaluation_domain: &self.domain,
				domain_points: self.domain.points(),
				composition: self.witness.composition(),
				denom_inv: &self.smaller_denom_inv,
			};
			provers_state.common.calculate_round_coeffs(
				self.oracle_ids.as_slice(),
				evaluator,
				self.round_claim.current_round_sum.into(),
				vertex_state_iterator,
			)
		} else {
			let evaluator = ZerocheckLaterRoundEvaluator {
				degree,
				eq_ind: provers_state.round_eq_ind.to_ref(),
				round_zerocheck_challenge: self.zerocheck_challenges[self.round - 1].into(),
				evaluation_domain: &self.domain,
				domain_points: self.domain.points(),
				composition: self.witness.composition(),
				denom_inv: &self.smaller_denom_inv,
				round_q_bar: self
					.round_q_bar
					.as_ref()
					.expect("round_q_bar is Some after round 0")
					.to_ref(),
			};
			provers_state.common.calculate_round_coeffs(
				self.oracle_ids.as_slice(),
				evaluator,
				self.round_claim.current_round_sum.into(),
				vertex_state_iterator,
			)
		}?;

		Ok(round_coeffs)
	}

	#[instrument(skip_all, name = "zerocheck::execute_round")]
	fn execute_round<EDF>(
		&mut self,
		provers_state: &ZerocheckProversState<'a, F, PW, DomainField, EDF, W>,
		prev_rd_challenge: Option<F>,
	) -> Result<ZerocheckRound<F>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
	{
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.claim.n_vars() {
			return Err(Error::TooManyExecuteRoundCalls);
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			let isomorphic_prev_rd_challenge = PW::Scalar::from(prev_rd_challenge);

			// update round_q and round_q_bar
			self.update_round_q(isomorphic_prev_rd_challenge)?;

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

		let proof_round = ZerocheckRound { coeffs };
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let reductor = ZerocheckReductor {
			alphas: self.zerocheck_challenges,
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

/// Evaluator for the first round of the zerocheck protocol.
///
/// In the first round, we do not need to evaluate at the point F::ONE, because the value is known
/// to be zero. This version of the zerocheck protocol uses the optimizations from section 3 of
/// [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckFirstRoundEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	pub composition: &'a C,
	pub domain_points: &'a [FS],
	pub evaluation_domain: &'a EvaluationDomain<FS>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<P, &'a [P]>,
	pub denom_inv: &'a [FS],
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P> for ZerocheckFirstRoundEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	type VertexState = &'a mut [P::Scalar];
	fn n_round_evals(&self) -> usize {
		// In the first round of zerocheck we can uniquely determine the degree d
		// univariate round polynomial $R(X)$ with evaluations at X = 2, ..., d
		// because we know r(0) = r(1) = 0
		self.degree - 1
	}

	fn process_vertex(
		&self,
		i: usize,
		round_q_chunk: Self::VertexState,
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
				.evaluate_scalar(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 2] += composite_value * eq_ind_factor;
			round_q_chunk[d - 2] = composite_value * self.denom_inv[d - 2];
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		debug_assert_eq!(current_round_sum, P::Scalar::ZERO);
		// We are given $r(2), \ldots, r(d)$.
		// From context, we infer that $r(0) = r(1) = 0$.
		round_evals.insert(0, P::Scalar::ZERO);
		round_evals.insert(0, P::Scalar::ZERO);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();
		Ok(coeffs)
	}
}

/// Evaluator for the later rounds of the zerocheck protocol.
///
/// This version of the zerocheck protocol uses the optimizations from section 3 and 4 of [Gruen24].
///
/// [Gruen24]: https://eprint.iacr.org/2024/108
#[derive(Debug)]
pub struct ZerocheckLaterRoundEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	pub composition: &'a C,
	pub domain_points: &'a [FS],
	pub evaluation_domain: &'a EvaluationDomain<FS>,
	pub degree: usize,
	pub eq_ind: MultilinearExtension<P, &'a [P]>,
	pub round_zerocheck_challenge: P::Scalar,
	pub denom_inv: &'a [FS],
	pub round_q_bar: MultilinearExtension<P::Scalar, &'a [P::Scalar]>,
}

impl<'a, P, FS, C> AbstractSumcheckEvaluator<P> for ZerocheckLaterRoundEvaluator<'a, P, FS, C>
where
	P: PackedField<Scalar: ExtensionField<FS>>,
	FS: Field,
	C: CompositionPoly<P>,
{
	type VertexState = &'a mut [P::Scalar];
	fn n_round_evals(&self) -> usize {
		// We can uniquely derive the degree d univariate round polynomial r from evaluations at
		// X = 1, ..., d because we have an identity that relates r(0), r(1), and the current
		// round's claimed sum
		self.degree
	}

	fn process_vertex(
		&self,
		i: usize,
		round_q_chunk: Self::VertexState,
		evals_0: &[P::Scalar],
		evals_1: &[P::Scalar],
		evals_z: &mut [P::Scalar],
		round_evals: &mut [P::Scalar],
	) {
		let q_bar_zero = self
			.round_q_bar
			.evaluate_on_hypercube(i << 1)
			.unwrap_or(P::Scalar::ZERO);
		let q_bar_one = self
			.round_q_bar
			.evaluate_on_hypercube((i << 1) + 1)
			.unwrap_or(P::Scalar::ZERO);
		debug_assert!(i < self.eq_ind.size());

		let eq_ind_factor = self
			.eq_ind
			.evaluate_on_hypercube(i)
			.unwrap_or(P::Scalar::ZERO);

		// We can replace constraint polynomial evaluations at C(r, 1, x) with Q_i_bar(r, 1, x)
		// See section 4 of [https://eprint.iacr.org/2024/108] for details
		round_evals[0] += q_bar_one * eq_ind_factor;

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
				.evaluate_scalar(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");

			round_evals[d - 1] += composite_value * eq_ind_factor;

			// We compute Q_i(r, domain[d], x) values with minimal additional work (linear extrapolation, multiplication, and inversion)
			// and cache these values for later use. These values will help us update Q_i_bar into Q_{i+1}_bar, which will in turn
			// help us avoid next round's constraint polynomial evaluations at X = 1.
			// For more details, see section 4 of [https://eprint.iacr.org/2024/108]
			let specialized_qbar_eval =
				extrapolate_line(q_bar_zero, q_bar_one, self.domain_points[d]);
			round_q_chunk[d - 2] =
				(composite_value - specialized_qbar_eval) * self.denom_inv[d - 2];
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		// This is a subsequent round of a sumcheck that came from zerocheck, given $r(1), \ldots, r(d)$
		// Letting $s$ be the current round's claimed sum, and $\alpha_i$ the ith zerocheck challenge
		// we have the identity $r(0) = \frac{1}{1 - \alpha_i} * (s - \alpha_i * r(1))$
		// which allows us to compute the value of $r(0)$

		let alpha = self.round_zerocheck_challenge;
		let alpha_bar = P::Scalar::ONE - alpha;
		let one_evaluation = round_evals[0];
		let zero_evaluation_numerator = current_round_sum - one_evaluation * alpha;
		let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
		let zero_evaluation = zero_evaluation_numerator * zero_evaluation_denominator_inv;

		round_evals.insert(0, zero_evaluation);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;
		// We can omit the constant term safely
		let coeffs = coeffs[1..].to_vec();

		Ok(coeffs)
	}
}
