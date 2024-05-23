// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{
		SumcheckClaim, SumcheckProveOutput, SumcheckReductor, SumcheckRound, SumcheckRoundClaim,
		SumcheckWitness,
	},
};
use crate::{
	oracle::CompositePolyOracle,
	polynomial::{
		extrapolate_line, CompositionPoly, Error as PolynomialError, EvaluationDomain,
		MultilinearPoly,
	},
	protocols::{
		abstract_sumcheck::{
			self, AbstractSumcheckEvaluator, AbstractSumcheckProver, AbstractSumcheckReductor,
			ProverState,
		},
		evalcheck::EvalcheckClaim,
		sumcheck::SumcheckProof,
	},
};
use binius_field::{Field, PackedField};
use getset::Getters;
use p3_challenger::{CanObserve, CanSample};
use rayon::prelude::*;
use std::fmt::Debug;
use tracing::instrument;

/// Prove a sumcheck to evalcheck reduction.
#[instrument(skip_all, name = "sumcheck::prove")]
pub fn prove<F, PW, CW, M, CH>(
	claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<PW, CW, M>,
	domain: &EvaluationDomain<PW::Scalar>,
	challenger: CH,
	switchover_fn: impl Fn(usize) -> usize,
) -> Result<SumcheckProveOutput<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW::Scalar>,
	M: MultilinearPoly<PW> + Clone + Sync,
	CH: CanSample<F> + CanObserve<F>,
{
	let sumcheck_prover = SumcheckProver::new(domain, claim.clone(), witness, switchover_fn)?;

	let (evalcheck_claim, rounds) =
		abstract_sumcheck::prove(claim.n_vars(), sumcheck_prover, challenger)?;

	let sumcheck_proof = SumcheckProof { rounds };
	let output = SumcheckProveOutput {
		evalcheck_claim,
		sumcheck_proof,
	};
	Ok(output)
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
	CW: CompositionPoly<PW::Scalar>,
	M: MultilinearPoly<PW> + Sync,
{
	oracle: CompositePolyOracle<F>,
	composition: CW,
	domain: &'a EvaluationDomain<PW::Scalar>,
	#[getset(get = "pub")]
	round_claim: SumcheckRoundClaim<F>,

	round: usize,
	last_round_proof: Option<SumcheckRound<F>>,
	state: ProverState<PW, M>,
}

impl<'a, F, PW, CW, M> SumcheckProver<'a, F, PW, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW::Scalar>,
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
				n_vars,
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

		let sumcheck_prover = SumcheckProver {
			oracle: sumcheck_claim.poly,
			composition,
			domain,
			round_claim,
			round: 0,
			last_round_proof: None,
			state,
		};

		Ok(sumcheck_prover)
	}

	pub fn n_vars(&self) -> usize {
		self.oracle.n_vars()
	}

	/// Generic parameters allow to pass a different witness type to the inner Evalcheck claim.
	#[instrument(skip_all, name = "sumcheck::finalize")]
	fn finalize(mut self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Error> {
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

		let sumcheck_reductor = SumcheckReductor;
		let evalcheck_claim =
			sumcheck_reductor.reduce_final_round_claim(&self.oracle, self.round_claim)?;
		Ok(evalcheck_claim)
	}

	#[instrument(skip_all, name = "sumcheck::execute_round")]
	fn execute_round(&mut self, prev_rd_challenge: Option<F>) -> Result<SumcheckRound<F>, Error> {
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.n_vars() {
			return Err(Error::ImproperInput("too many execute_round calls".to_string()));
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			// Process switchovers of small field multilinears and folding of large field ones
			self.state.fold(prev_rd_challenge.into())?;

			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		let degree = self.oracle.max_individual_degree();
		let evaluator = SumcheckEvaluator {
			degree,
			composition: &self.composition,
			evaluation_domain: self.domain,
			domain_points: self.domain.points(),
		};

		let rd_vars = self.n_vars() - self.round;
		let vertex_state_iterator = (0..1 << (rd_vars - 1)).into_par_iter().map(|_i| ());

		let round_coeffs = self.state.calculate_round_coeffs(
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
		let sumcheck_reductor = SumcheckReductor;
		let round_claim = self.round_claim.clone();
		let round_proof = self
			.last_round_proof
			.as_ref()
			.expect("round is at least 1 by invariant")
			.clone();

		let new_round_claim = sumcheck_reductor.reduce_intermediate_round_claim(
			self.round,
			round_claim,
			prev_rd_challenge,
			round_proof,
		)?;

		self.round_claim = new_round_claim;

		Ok(())
	}
}

impl<'a, F, PW, CW, M> AbstractSumcheckProver<F> for SumcheckProver<'a, F, PW, CW, M>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW::Scalar>,
	M: MultilinearPoly<PW> + Sync,
{
	type Error = Error;

	fn execute_round(
		&mut self,
		prev_rd_challenge: Option<F>,
	) -> Result<SumcheckRound<F>, Self::Error> {
		SumcheckProver::execute_round(self, prev_rd_challenge)
	}

	fn finalize(self, prev_rd_challenge: Option<F>) -> Result<EvalcheckClaim<F>, Self::Error> {
		SumcheckProver::finalize(self, prev_rd_challenge)
	}
}

/// Evaluator for the sumcheck protocol.
#[derive(Debug)]
struct SumcheckEvaluator<'a, F, C>
where
	F: Field,
	C: CompositionPoly<F>,
{
	pub degree: usize,
	composition: &'a C,
	evaluation_domain: &'a EvaluationDomain<F>,
	domain_points: &'a [F],
}

impl<'a, F: Field, C: CompositionPoly<F>> AbstractSumcheckEvaluator<F>
	for SumcheckEvaluator<'a, F, C>
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
		for d in 2..self.domain_points.len() {
			evals_0
				.iter()
				.zip(evals_1.iter())
				.zip(evals_z.iter_mut())
				.for_each(|((&evals_0_j, &evals_1_j), evals_z_j)| {
					// TODO: Enable small field multiplication.
					*evals_z_j =
						extrapolate_line::<F, F>(evals_0_j, evals_1_j, self.domain_points[d]);
				});

			round_evals[d - 1] += self
				.composition
				.evaluate(evals_z)
				.expect("evals_z is initialized with a length of poly.composition.n_vars()");
		}
	}

	fn round_evals_to_coeffs(
		&self,
		current_round_sum: F,
		mut round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError> {
		// Given $r(1), \ldots, r(d+1)$, letting $s$ be the current round's claimed sum,
		// we can compute $r(0)$ using the identity $r(0) = s - r(1)$
		round_evals.insert(0, current_round_sum - round_evals[0]);

		let coeffs = self.evaluation_domain.interpolate(&round_evals)?;

		// Trimming highest degree coefficient as it can be recovered by the verifier
		Ok(coeffs[..coeffs.len() - 1].to_vec())
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
