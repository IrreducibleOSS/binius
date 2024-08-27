// Copyright 2024 Ulvetanna Inc.

use super::{
	batch::batch_prove,
	error::Error,
	sumcheck::{
		SumcheckClaim, SumcheckProveOutput, SumcheckReductor, SumcheckRound, SumcheckRoundClaim,
	},
};
use crate::{
	challenger::{CanObserve, CanSample},
	oracle::OracleId,
	protocols::{
		abstract_sumcheck::{
			check_evaluation_domain, validate_rd_challenge, AbstractSumcheckClaim,
			AbstractSumcheckEvaluator, AbstractSumcheckProversState, AbstractSumcheckReductor,
			AbstractSumcheckWitness, CommonProversState, ReducedClaim,
		},
		sumcheck::SumcheckProof,
	},
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
use binius_math::polynomial::{
	extrapolate_line, CompositionPoly, Error as PolynomialError, EvaluationDomain,
	EvaluationDomainFactory,
};
use binius_utils::bail;
use getset::Getters;
use rayon::prelude::*;
use std::{fmt::Debug, marker::PhantomData};
use tracing::instrument;

#[cfg(feature = "debug_validate_sumcheck")]
use super::sumcheck::validate_witness;

/// Prove a sumcheck to evalcheck reduction.
pub fn prove<F, PW, DomainField, CH>(
	claim: &SumcheckClaim<F>,
	witness: impl AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
	evaluation_domain_factory: impl EvaluationDomainFactory<DomainField>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	challenger: CH,
) -> Result<SumcheckProveOutput<F>, Error>
where
	F: Field,
	DomainField: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	CH: CanSample<F> + CanObserve<F>,
{
	let batch_proof = batch_prove::<F, PW, DomainField, CH>(
		[(claim.clone(), witness)],
		evaluation_domain_factory,
		switchover_fn,
		challenger,
	)?;

	Ok(SumcheckProveOutput {
		evalcheck_claim: batch_proof
			.evalcheck_claims
			.first()
			.expect("exactly one")
			.clone(),
		sumcheck_proof: SumcheckProof {
			rounds: batch_proof.proof.rounds,
		},
	})
}

pub struct SumcheckProversState<F, PW, DomainField, EDF, W>
where
	F: Field,
	PW: PackedField,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW>,
{
	common: CommonProversState<OracleId, PW, W::Multilinear>,
	evaluation_domain_factory: EDF,
	_f_marker: PhantomData<F>,
	_domain_field_marker: PhantomData<DomainField>,
	_w_marker: PhantomData<W>,
}

impl<F, PW, DomainField, EDF, W> SumcheckProversState<F, PW, DomainField, EDF, W>
where
	F: Field,
	PW: PackedField,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW>,
{
	pub fn new(
		n_vars: usize,
		evaluation_domain_factory: EDF,
		switchover_fn: impl Fn(usize) -> usize + 'static,
	) -> Self {
		let common = CommonProversState::new(n_vars, switchover_fn);
		Self {
			common,
			evaluation_domain_factory,
			_f_marker: PhantomData,
			_domain_field_marker: PhantomData,
			_w_marker: PhantomData,
		}
	}
}

impl<F, PW, DomainField, EDF, W> AbstractSumcheckProversState<F>
	for SumcheckProversState<F, PW, DomainField, EDF, W>
where
	F: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
{
	type Error = Error;

	type PackedWitnessField = PW;

	type Claim = SumcheckClaim<F>;
	type Witness = W;
	type Prover = SumcheckProver<F, PW, DomainField, W>;

	fn new_prover(
		&mut self,
		claim: SumcheckClaim<F>,
		witness: W,
		seq_id: usize,
	) -> Result<Self::Prover, Error> {
		let ids = claim.poly.inner_polys_oracle_ids().collect::<Vec<_>>();
		self.common
			.extend(witness.multilinears(seq_id, ids.as_slice())?)?;
		let domain = self
			.evaluation_domain_factory
			.create(claim.poly.max_individual_degree() + 1)?;
		let prover = SumcheckProver::new(claim, witness, domain)?;
		Ok(prover)
	}

	fn pre_execute_rounds(&mut self, prev_rd_challenge: Option<F>) -> Result<(), Error> {
		self.common
			.pre_execute_rounds(prev_rd_challenge.map(Into::into))?;

		Ok(())
	}

	fn prover_execute_round(
		&self,
		prover: &mut Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckRound<F>, Error> {
		prover.execute_round(self, prev_rd_challenge)
	}

	fn prover_finalize(
		prover: Self::Prover,
		prev_rd_challenge: Option<F>,
	) -> Result<ReducedClaim<F>, Error> {
		prover.finalize(prev_rd_challenge)
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
#[derive(Getters)]
pub struct SumcheckProver<F, PW, DomainField, W>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
	DomainField: Field,
	W: AbstractSumcheckWitness<PW>,
{
	#[getset(get = "pub")]
	claim: SumcheckClaim<F>,
	witness: W,
	domain: EvaluationDomain<DomainField>,
	oracle_ids: Vec<OracleId>,

	#[getset(get = "pub")]
	round_claim: SumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<SumcheckRound<F>>,

	_pw_marker: PhantomData<PW>,
}

impl<F, PW, DomainField, W> SumcheckProver<F, PW, DomainField, W>
where
	F: Field,
	DomainField: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
{
	/// Start a new sumcheck instance with claim in field `F`. Witness may be given in
	/// a different (but isomorphic) packed field PW. `switchover_fn` closure specifies
	/// switchover round number per multilinear polynomial as a function of its
	/// [`MultilinearPoly::extension_degree`] value.
	fn new(
		claim: SumcheckClaim<F>,
		witness: W,
		domain: EvaluationDomain<DomainField>,
	) -> Result<Self, Error> {
		#[cfg(feature = "debug_validate_sumcheck")]
		validate_witness(&claim, &witness)?;

		if claim.poly.max_individual_degree() == 0 {
			bail!(Error::PolynomialDegreeIsZero);
		}

		check_evaluation_domain(claim.poly.max_individual_degree(), &domain)?;

		let oracle_ids = claim.poly.inner_polys_oracle_ids().collect::<Vec<_>>();

		let round_claim = SumcheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: claim.sum,
		};

		let sumcheck_prover = SumcheckProver {
			claim,
			witness,
			domain,
			oracle_ids,
			round_claim,
			round: 0,
			last_round_proof: None,
			_pw_marker: PhantomData,
		};

		Ok(sumcheck_prover)
	}

	/// Generic parameters allow to pass a different witness type to the inner Evalcheck claim.
	#[instrument(skip_all, name = "sumcheck::finalize", level = "debug")]
	fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<ReducedClaim<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round != self.claim.n_vars() {
			bail!(Error::PrematureFinalizeCall);
		}

		// Last reduction to obtain eval value at eval_point
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			self.reduce_claim(prev_rd_challenge)?;
		}

		Ok(self.round_claim.into())
	}

	#[instrument(skip_all, name = "sumcheck::execute_round", level = "debug")]
	fn execute_round<EDF>(
		&mut self,
		provers_state: &SumcheckProversState<F, PW, DomainField, EDF, W>,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckRound<F>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
	{
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.claim.n_vars() {
			bail!(Error::TooManyExecuteRoundCalls);
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		let degree = self.claim.poly.max_individual_degree();
		let evaluator = SumcheckEvaluator {
			degree,
			composition: self.witness.composition(),
			evaluation_domain: &self.domain,
			domain_points: self.domain.points(),
			_p: PhantomData,
		};

		let rd_vars = (self.claim.n_vars() - self.round).saturating_sub(PW::LOG_WIDTH);
		let vertex_state_iterator = (0..1 << (rd_vars.saturating_sub(1)))
			.into_par_iter()
			.map(|_i| ());

		let round_coeffs = provers_state.common.calculate_round_coeffs(
			self.oracle_ids.as_slice(),
			evaluator,
			self.round_claim.current_round_sum.into(),
			vertex_state_iterator,
		)?;
		let coeffs = round_coeffs.into_iter().map(Into::into).collect::<Vec<F>>();

		let proof_round = SumcheckRound { coeffs };
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let sumcheck_reductor = SumcheckReductor {
			max_individual_degree: self.claim.max_individual_degree(),
		};

		let round_claim = self.round_claim.clone();
		let round_proof = self
			.last_round_proof
			.as_ref()
			.expect("round is at least 1 by invariant")
			.clone();

		let new_round_claim = sumcheck_reductor.reduce_round_claim(
			self.round,
			round_claim,
			prev_rd_challenge,
			round_proof,
		)?;

		self.round_claim = new_round_claim;

		Ok(())
	}
}

/// Evaluator for the sumcheck protocol.
#[derive(Debug)]
struct SumcheckEvaluator<'a, P, DomainField, C>
where
	P: PackedField<Scalar: ExtensionField<DomainField>>,
	DomainField: Field,
	C: CompositionPoly<P>,
{
	pub degree: usize,
	composition: &'a C,
	evaluation_domain: &'a EvaluationDomain<DomainField>,
	domain_points: &'a [DomainField],

	_p: PhantomData<P>,
}

impl<'a, P, DomainField, C> AbstractSumcheckEvaluator<P>
	for SumcheckEvaluator<'a, P, DomainField, C>
where
	P: PackedExtension<DomainField, Scalar: ExtensionField<DomainField>>,
	DomainField: Field,
	C: CompositionPoly<P>,
{
	type VertexState = ();

	fn n_round_evals(&self) -> usize {
		// NB: We skip evaluation of $r(X)$ at $X = 0$ as it is derivable from the
		// current_round_sum - $r(1)$.
		self.degree
	}

	fn process_vertex(
		&self,
		_i: usize,
		_vertex_state: Self::VertexState,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	) {
		// Sumcheck evaluation at a specific point - given an array of 0 & 1 evaluations at some
		// index, use them to linearly interpolate each MLE value at domain point, and then
		// evaluate multivariate composite over those.

		round_evals[0] += self
			.composition
			.evaluate(evals_1)
			.expect("evals_1 is initialized with a length of poly.composition.n_vars()");

		// The rest require interpolation.
		for d in 2..=self.degree {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					*evals_z_j = extrapolate_line::<P, DomainField>(
						evals_0_j,
						evals_1_j,
						self.domain_points[d],
					);
				});

			round_evals[d - 1] += self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: P::Scalar,
		mut round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError> {
		// Given $r(1), \ldots, r(d+1)$, letting $s$ be the current round's claimed sum,
		// we can compute $r(0)$ using the identity $r(0) = s - r(1)$
		round_evals.insert(0, current_round_sum - round_evals[0]);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;

		// Trimming highest degree coefficient as it can be recovered by the verifier
		Ok(coeffs[..coeffs.len() - 1].to_vec())
	}
}
