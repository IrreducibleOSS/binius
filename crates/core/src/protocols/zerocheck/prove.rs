// Copyright 2023 Ulvetanna Inc.

#[cfg(feature = "debug_validate_sumcheck")]
use super::zerocheck::validate_witness;
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
	polynomial::{MultilinearExtension, MultilinearPoly, MultilinearQuery},
	protocols::{
		abstract_sumcheck::{
			check_evaluation_domain, validate_rd_challenge, AbstractSumcheckClaim,
			AbstractSumcheckProversState, AbstractSumcheckReductor, AbstractSumcheckWitness,
			CommonProversState, ReducedClaim,
		},
		utils::packed_from_fn_with_offset,
		zerocheck::backend::ZerocheckProverBackendWrapper,
	},
	transparent::eq_ind::EqIndPartialEval,
};
use binius_field::{packed::get_packed_slice, ExtensionField, Field, PackedExtension, PackedField};
use binius_hal::{
	zerocheck::{ZerocheckRoundInput, ZerocheckRoundParameters},
	ComputationBackend, VecOrImmutableSlice,
};
use binius_math::{EvaluationDomain, EvaluationDomainFactory};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use getset::Getters;
use itertools::Itertools;
use rayon::prelude::*;
use std::{cmp::max, marker::PhantomData, mem::size_of};
use tracing::{instrument, trace};

/// Prove a zerocheck to evalcheck reduction.
/// FS is the domain type.
#[instrument(skip_all, name = "zerocheck::prove", level = "debug")]
pub fn prove<F, PW, DomainField, CH, Backend>(
	claim: &ZerocheckClaim<F>,
	witness: impl AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
	evaluation_domain_factory: impl EvaluationDomainFactory<DomainField>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	mixing_challenge: F,
	challenger: CH,
	backend: Backend,
) -> Result<ZerocheckProveOutput<F>, Error>
where
	F: Field,
	DomainField: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	CH: CanSample<F> + CanObserve<F>,
	Backend: ComputationBackend,
{
	let batch_proof = batch_prove::<F, PW, DomainField, CH, Backend>(
		[(claim.clone(), witness)],
		evaluation_domain_factory,
		switchover_fn,
		mixing_challenge,
		challenger,
		backend,
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

pub struct ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>
where
	F: Field,
	PW: PackedField,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW>,
	Backend: ComputationBackend,
{
	pub(crate) common: CommonProversState<OracleId, PW, W::Multilinear, Backend>,
	evaluation_domain_factory: EDF,
	zerocheck_challenges: &'a [F],
	pub(crate) round_eq_ind: MultilinearExtension<PW, VecOrImmutableSlice<PW>>,
	mixing_challenge: F,
	backend: Backend,
	_marker: PhantomData<(F, DomainField, W)>,
}

impl<'a, F, PW, DomainField, EDF, W, Backend>
	ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>
where
	F: Field,
	PW: PackedField<Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW>,
	Backend: ComputationBackend,
{
	pub fn new(
		n_vars: usize,
		evaluation_domain_factory: EDF,
		zerocheck_challenges: &'a [F],
		switchover_fn: impl Fn(usize) -> usize + 'static,
		mixing_challenge: F,
		backend: Backend,
	) -> Result<Self, Error> {
		let common = CommonProversState::new(n_vars, switchover_fn, backend.clone());

		if zerocheck_challenges.len() + 1 < n_vars {
			bail!(Error::NotEnoughZerocheckChallenges);
		}

		let pw_scalar_challenges = zerocheck_challenges
			.iter()
			.map(|&f| f.into())
			.collect::<Vec<PW::Scalar>>();
		assert_eq!(zerocheck_challenges.len(), n_vars - 1);
		let round_eq_ind = EqIndPartialEval::new(n_vars - 1, pw_scalar_challenges)?
			.multilinear_extension(backend.clone())?;

		Ok(Self {
			common,
			evaluation_domain_factory,
			zerocheck_challenges,
			round_eq_ind,
			mixing_challenge,
			backend: backend.clone(),
			_marker: PhantomData,
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

		let get_value_at = |index: usize| {
			let eval0 = get_packed_slice(current_evals, index << 1);
			let eval1 = get_packed_slice(current_evals, (index << 1) + 1);

			eval0 + eval1
		};

		let new_evals = if PW::LOG_WIDTH < self.round_eq_ind.n_vars() {
			(0..current_evals.len() >> 1)
				.into_par_iter()
				.map(|i| packed_from_fn_with_offset(i, get_value_at))
				.collect()
		} else {
			vec![PW::from_fn(|j| {
				if 2 * j < self.round_eq_ind.size() {
					get_value_at(j)
				} else {
					PW::Scalar::default()
				}
			})]
		};

		self.round_eq_ind =
			MultilinearExtension::from_values_generic(VecOrImmutableSlice::V(new_evals))?;
		Ok(())
	}
}

impl<'a, F, PW, DomainField, EDF, W, Backend> AbstractSumcheckProversState<F>
	for ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>
where
	F: Field,
	PW: PackedExtension<DomainField, Scalar: From<F> + Into<F> + ExtensionField<DomainField>>,
	DomainField: Field,
	EDF: EvaluationDomainFactory<DomainField>,
	W: AbstractSumcheckWitness<PW, MultilinearId = OracleId>,
	Backend: ComputationBackend,
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
			.create(claim.poly.max_individual_degree() + 1)
			.map_err(Error::MathError)?;
		let prover =
			ZerocheckProver::new(claim, witness, domain, self.zerocheck_challenges, seq_id)?;
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

	smaller_domain_optimization: Option<SmallerDomainOptimization<PW, DomainField>>,

	seq_id: usize,
}

#[derive(Debug)]
pub(crate) struct SmallerDomainOptimization<PW, DomainField>
where
	PW: PackedField,
	PW::Scalar: ExtensionField<DomainField>,
	DomainField: Field,
{
	// Junk (scratch space) at the start of each round
	// After prover computes ith round polynomial, represents evaluations of
	// * $Q_i(r_0, \ldots, r_{i-1}, X, x_{i+1}, \ldots, x_{n-1})$
	// for $X \in \{2, \ldots, d\}$, and $x_{i+1}, \ldots, x_{n-1} \in \{0, 1\}$
	// where
	// * $d$ is degree of polynomial in zerocheck claim
	// * $r_i$ is the verifier challenge at the end of round $i$.
	// with sorting that is lexicographically-inspired (lowest index = least significant).
	pub(crate) round_q: Vec<PW>,
	// None initially
	// After ith round, represents the $\bar{Q}_i$ multilinear polynomial partially evaluated
	// at lowest $i$ variables with the $i$ verifier challenges received so far.
	pub(crate) round_q_bar: Option<MultilinearExtension<PW>>,
	// inverse of domain[i] * domain[i] - 1 for i in 0, ..., d-2
	pub(crate) smaller_denom_inv: Vec<DomainField>,
	// Subdomain of domain, but without 0 and 1 (can be used for degree d-2 polynomials)
	smaller_domain: EvaluationDomain<DomainField>,
}

impl<PW, DomainField> SmallerDomainOptimization<PW, DomainField>
where
	PW: PackedField,
	PW::Scalar: ExtensionField<DomainField>,
	DomainField: Field,
{
	fn new(
		n_vars: usize,
		degree: usize,
		domain: &EvaluationDomain<DomainField>,
	) -> Result<Self, Error> {
		let round_q = zeroed_vec((1 << (n_vars - 1)) * (degree - 1) / PW::WIDTH);
		let smaller_domain_points = domain.points()[2..].to_vec();
		let smaller_denom_inv = domain.points()[2..]
			.iter()
			.map(|&x| (x * (x - DomainField::ONE)).invert().unwrap())
			.collect::<Vec<_>>();
		let smaller_domain = EvaluationDomain::from_points(smaller_domain_points)?;
		Ok(Self {
			round_q,
			round_q_bar: None,
			smaller_denom_inv,
			smaller_domain,
		})
	}
}

impl<'a, F, PW, DomainField, W> ZerocheckProver<'a, F, PW, DomainField, W>
where
	F: Field,
	PW: PackedExtension<DomainField>,
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
		seq_id: usize,
	) -> Result<Self, Error> {
		#[cfg(feature = "debug_validate_sumcheck")]
		validate_witness(&claim, &witness)?;

		let n_vars = claim.n_vars();
		let degree = claim.poly.max_individual_degree();

		if degree == 0 {
			bail!(Error::PolynomialDegreeIsZero);
		}
		check_evaluation_domain(degree, &domain)?;

		let oracle_ids = claim.poly.inner_polys_oracle_ids().collect::<Vec<_>>();

		if zerocheck_challenges.len() + 1 < n_vars {
			bail!(Error::NotEnoughZerocheckChallenges);
		}
		let zerocheck_challenges = &zerocheck_challenges[zerocheck_challenges.len() + 1 - n_vars..];
		assert_eq!(zerocheck_challenges.len(), n_vars - 1);

		let round_claim = ZerocheckRoundClaim {
			partial_point: Vec::new(),
			current_round_sum: F::ZERO,
		};

		let smaller_domain_optimization = SmallerDomainOptimization::new(n_vars, degree, &domain)?;

		let zerocheck_prover = ZerocheckProver {
			claim,
			witness,
			domain,
			oracle_ids,
			round_claim,
			round: 0,
			last_round_proof: None,
			zerocheck_challenges,
			smaller_domain_optimization: Some(smaller_domain_optimization),
			seq_id,
		};

		Ok(zerocheck_prover)
	}

	#[instrument(skip_all, name = "zerocheck::finalize", level = "debug")]
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

	// Updates auxiliary state that we store corresponding to Section 4 Optimizations in [Gruen24].
	// [Gruen24]: https://eprint.iacr.org/2024/108
	fn update_round_q<Backend: ComputationBackend>(
		&mut self,
		prev_rd_challenge: PW::Scalar,
		p: &mut SmallerDomainOptimization<PW, DomainField>,
		backend: Backend,
	) -> Result<(), Error> {
		let rd_vars = self.claim.n_vars() - self.round;
		let new_q_len = max(p.round_q.len() / 2, p.smaller_domain.size());

		// Let d be the degree of the polynomial in the zerocheck claim.
		let specialized_q_values = if p.smaller_domain.size() == 0 {
			// Special handling for the d = 1 (multilinear) case
			zeroed_vec((1usize << rd_vars).div_ceil(PW::WIDTH))
		} else if p.smaller_domain.size() == 1 {
			// We do not need to interpolate in this special d = 2 case
			std::mem::replace(&mut p.round_q, zeroed_vec(new_q_len))
		} else {
			// This is for the d >= 3 case
			//
			// Let r_i be the prev_rd_challenge
			// * Each d-1 sized chunk of round_q is, for some fixed x_{i+1}, ..., x_{n-1}, evaluations of
			// Q_i(r_0, ..., r_{i-1}, X, x_{i+1}, ..., x_{n-1}) at X = 2, ..., d
			// * Q_i is degree d-2 in X, so we can extrapolate each chunk into the evaluation
			// of Q_i(r_0, ..., r_{i-1}, X, x_{i+1}, ..., x_{n-1}) at X = r_i
			p.round_q
				.chunks(p.smaller_domain.size())
				.map(|chunk| p.smaller_domain.extrapolate(chunk, prev_rd_challenge))
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

		if let Some(prev_q_bar) = p.round_q_bar.as_mut() {
			let query =
				MultilinearQuery::<PW::Scalar>::with_full_query(&[prev_rd_challenge], backend)?;
			let specialized_prev_q_bar = prev_q_bar.evaluate_partial_low(&query)?;
			let specialized_prev_q_bar_evals = specialized_prev_q_bar.evals();

			const CHUNK_SIZE: usize = 64;

			new_q_bar_values
				.par_chunks_mut(CHUNK_SIZE)
				.enumerate()
				.for_each(|(chunk_index, chunks)| {
					for (k, e) in chunks.iter_mut().enumerate() {
						*e += packed_from_fn_with_offset::<PW>(chunk_index * CHUNK_SIZE + k, |i| {
							specialized_prev_q_bar_evals
								.get(i)
								.copied()
								.unwrap_or(PW::Scalar::ZERO)
						});
					}
				});
		}
		p.round_q_bar = Some(MultilinearExtension::from_values(new_q_bar_values)?);

		// Step 3:
		// Truncate round_q
		p.round_q.truncate(new_q_len);

		Ok(())
	}

	#[instrument(skip_all)]
	fn compute_round_coeffs<EDF, Backend>(
		&mut self,
		provers_state: &ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
		Backend: ComputationBackend,
	{
		let mut wrapper = ZerocheckProverBackendWrapper {
			claim: &self.claim,
			// Pass `smaller_domain_optimization` to the wrapper.
			smaller_domain_optimization: self.smaller_domain_optimization.take(),
			provers_state,
			domain: &self.domain,
			oracle_ids: &self.oracle_ids,
			witness: &self.witness,
		};
		let (params, input) = self.descriptor(provers_state);
		let result = provers_state
			.backend
			.zerocheck_compute_round_coeffs::<F, PW, DomainField>(&params, &input, &mut wrapper)?;
		// Take the `smaller_domain_optimization` back.
		// Note that `smaller_domain_optimization` may get changed to None.
		self.smaller_domain_optimization = wrapper.smaller_domain_optimization.take();
		Ok(result)
	}

	#[instrument(skip_all, name = "zerocheck::execute_round", level = "debug")]
	fn execute_round<EDF, Backend>(
		&mut self,
		provers_state: &ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>,
		prev_rd_challenge: Option<F>,
	) -> Result<ZerocheckRound<F>, Error>
	where
		EDF: EvaluationDomainFactory<DomainField>,
		Backend: ComputationBackend,
	{
		trace!(?self.round, "execute_round");
		// First round has no challenge, other rounds should have it
		validate_rd_challenge(prev_rd_challenge, self.round)?;

		if self.round >= self.claim.n_vars() {
			bail!(Error::TooManyExecuteRoundCalls);
		}

		// Rounds 1..n_vars-1 - Some(..) challenge is given
		if let Some(prev_rd_challenge) = prev_rd_challenge {
			let isomorphic_prev_rd_challenge = PW::Scalar::from(prev_rd_challenge);

			// update round_q and round_q_bar
			if let Some(mut p) = self.smaller_domain_optimization.take() {
				self.update_round_q(
					isomorphic_prev_rd_challenge,
					&mut p,
					provers_state.backend.clone(),
				)?;
				self.smaller_domain_optimization = Some(p);
			}

			// Reduce Evalcheck claim
			self.reduce_claim(prev_rd_challenge)?;
		}

		// Compute Round Coeffs using the appropriate evaluator
		let round_coeffs = self.compute_round_coeffs(provers_state)?;
		trace!(?round_coeffs);

		// Convert round_coeffs to F
		let coeffs = round_coeffs
			.iter()
			.cloned()
			.map(Into::into)
			.collect::<Vec<F>>();

		let proof_round = ZerocheckRound { coeffs };
		self.last_round_proof = Some(proof_round.clone());

		self.round += 1;

		Ok(proof_round)
	}

	#[instrument(skip_all)]
	fn reduce_claim(&mut self, prev_rd_challenge: F) -> Result<(), Error> {
		let reductor = ZerocheckReductor {
			max_individual_degree: self.claim.max_individual_degree(),
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

	/// Describes the shape of the computation and input data for the current zerocheck round.
	#[instrument(skip_all)]
	fn descriptor<EDF, Backend>(
		&'a self,
		state: &'a ZerocheckProversState<'a, F, PW, DomainField, EDF, W, Backend>,
	) -> (ZerocheckRoundParameters, ZerocheckRoundInput<F, PW, DomainField>)
	where
		EDF: EvaluationDomainFactory<DomainField>,
		Backend: ComputationBackend,
	{
		// query is expected to be present until the switchover round.
		let (small_field_width, underlier_data, query) = if let Some(query) =
			state.common.get_subset_query(&self.oracle_ids)
		{
			let query = query.1.expansion();

			let oracle_ids = self.claim.poly.inner_polys_oracle_ids().collect_vec();
			let multilinears = self.witness.multilinears(self.seq_id, &oracle_ids).unwrap();
			let (small_field_width, underlier_data) = multilinears
				.into_iter()
				.map(|(_m_id, m)| {
					assert_eq!(0, (size_of::<PW>() * 8) % m.extension_degree());
					assert_eq!(0, ((size_of::<PW>() * 8) / m.extension_degree()) % PW::WIDTH);
					((size_of::<PW>() * 8) / m.extension_degree() / PW::WIDTH, m.underlier_data())
				})
				.unzip::<_, _, Vec<_>, Vec<_>>();
			assert!(small_field_width.iter().all_equal());
			trace!(n_oracle_ids = oracle_ids.len(), tower_level = ?small_field_width[0]);

			// Assume tower_level is the same for all multilinears.
			// This holds true only for a small set of expressions.
			(Some(small_field_width[0]), Some(underlier_data), Some(query))
		} else {
			(None, None, None)
		};

		let mixing_challenge = state.mixing_challenge;
		let eq_ind = state.round_eq_ind.evals();
		(
			ZerocheckRoundParameters {
				round: self.round,
				n_vars: self.claim.poly.n_vars(),
				cols: self.claim.poly.n_multilinears(),
				degree: self.claim.poly.max_individual_degree(),
				small_field_width,
			},
			ZerocheckRoundInput {
				zc_challenges: self.zerocheck_challenges,
				eq_ind,
				query,
				current_round_sum: self.round_claim.current_round_sum,
				mixing_challenge,
				domain: &self.domain,
				underlier_data,
			},
		)
	}
}
