// Copyright 2023 Ulvetanna Inc.

use crate::{
	oracle::MultilinearOracleSet,
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearExtension,
		MultilinearPoly, MultivariatePoly,
	},
	protocols::{
		evalcheck::{
			subclaims::{
				non_same_query_pcs_sumcheck_claim, non_same_query_pcs_sumcheck_metas,
				non_same_query_pcs_sumcheck_witness, BivariateSumcheck, MemoizedQueries,
			},
			BatchCommittedEvalClaims, CommittedEvalClaim, Error as EvalcheckError, EvalcheckClaim,
		},
		sumcheck::{
			batch_prove, batch_verify, setup_first_round_claim, verify_final, verify_round,
			verify_zerocheck_round, Error as SumcheckError, SumcheckBatchProof, SumcheckClaim,
			SumcheckProof, SumcheckProveOutput, SumcheckProverState, SumcheckRoundClaim,
			SumcheckWitness,
		},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{packed::set_packed_slice, BinaryField1b, Field, PackedField, TowerField};
use p3_challenger::{CanObserve, CanSample};
use std::iter::{repeat, Step};
use tracing::instrument;

// If the macro is not used in the same module, rustc thinks it is unused for some reason
#[allow(unused_macros, unused_imports)]
pub mod macros {
	macro_rules! felts {
		($f:ident[$($elem:expr),* $(,)?]) => { vec![$($f::new($elem)),*] };
	}
	pub(crate) use felts;
}

pub fn hypercube_evals_from_oracle<F: Field>(oracle: &dyn MultivariatePoly<F>) -> Vec<F> {
	(0..(1 << oracle.n_vars()))
		.map(|i| {
			oracle
				.evaluate(&decompose_index_to_hypercube_point(oracle.n_vars(), i))
				.unwrap()
		})
		.collect()
}

pub fn decompose_index_to_hypercube_point<F: Field>(n_vars: usize, index: usize) -> Vec<F> {
	(0..n_vars)
		.map(|k| {
			if (index >> k) % 2 == 1 {
				F::ONE
			} else {
				F::ZERO
			}
		})
		.collect::<Vec<_>>()
}

pub fn packed_slice<P>(assignments: &[(std::ops::Range<usize>, u8)]) -> Vec<P>
where
	P: PackedField<Scalar = BinaryField1b>,
{
	assert_eq!(assignments[0].0.start, 0, "First assignment must start at index 0");
	assert_eq!(
		assignments[assignments.len() - 1].0.end % P::WIDTH,
		0,
		"Last assignment must end at an index divisible by packing width"
	);
	for i in 1..assignments.len() {
		assert_eq!(
			assignments[i].0.start,
			assignments[i - 1].0.end,
			"2 assignments following each other can't be overlapping or have holes in between"
		);
	}
	assignments
		.iter()
		.for_each(|(r, _)| assert!(r.end > r.start, "Range must have positive size"));
	let packed_len = (P::WIDTH - 1
		+ (assignments.iter().map(|(range, _)| range.end))
			.max()
			.unwrap_or(0))
		/ P::WIDTH;
	let mut result: Vec<P> = vec![P::default(); packed_len];
	for (range, value) in assignments.iter() {
		for i in range.clone() {
			set_packed_slice(&mut result, i, P::Scalar::new(*value));
		}
	}
	result
}

#[derive(Clone, Debug)]
pub struct TestProductComposition {
	arity: usize,
}

impl TestProductComposition {
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<F> CompositionPoly<F> for TestProductComposition
where
	F: Field,
{
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

pub fn transform_poly<F, OF>(
	multilin: MultilinearExtension<F>,
) -> Result<MultilinearExtension<'static, OF>, PolynomialError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let values = multilin
		.evals()
		.iter()
		.cloned()
		.map(OF::from)
		.collect::<Vec<_>>();

	MultilinearExtension::from_values(values)
}

#[instrument(skip_all, name = "test_utils::full_verify")]
pub fn full_verify<F, CH>(
	claim: &SumcheckClaim<F>,
	proof: SumcheckProof<F>,
	mut challenger: CH,
) -> (Vec<SumcheckRoundClaim<F>>, EvalcheckClaim<F>)
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = claim.poly.n_vars();
	assert!(n_vars > 0);
	let is_zerocheck = claim.zerocheck_challenges.is_some();

	let n_rounds = proof.rounds.len();
	assert_eq!(n_rounds, n_vars);

	let mut rd_claim = setup_first_round_claim(claim);
	let mut rd_claims = Vec::with_capacity(n_rounds);
	for (i, round_proof) in proof.rounds.into_iter().enumerate() {
		rd_claims.push(rd_claim.clone());

		challenger.observe_slice(round_proof.coeffs.as_slice());
		rd_claim = if is_zerocheck {
			let alpha = if i == 0 {
				None
			} else {
				Some(claim.zerocheck_challenges.clone().unwrap()[i - 1])
			};
			verify_zerocheck_round(rd_claim, challenger.sample(), round_proof.clone(), alpha)
				.unwrap()
		} else {
			verify_round(rd_claim, challenger.sample(), round_proof.clone()).unwrap()
		};
	}

	let final_claim = verify_final(&claim.poly, rd_claim).unwrap();
	(rd_claims, final_claim)
}

fn full_prove_with_switchover_impl<F, PW, PCW, M, CH>(
	n_vars: usize,
	mut prover_state: SumcheckProverState<F, PW, PCW, M>,
	mut challenger: CH,
) -> (Vec<SumcheckRoundClaim<F>>, SumcheckProveOutput<F>)
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	PCW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Sync,
	CH: CanSample<F> + CanObserve<F>,
{
	let mut prev_rd_challenge = None;
	let mut rd_claims = Vec::with_capacity(n_vars);
	let mut rd_proofs = Vec::with_capacity(n_vars);

	for _round in 0..n_vars {
		let proof_round = prover_state.execute_round(prev_rd_challenge).unwrap();

		challenger.observe_slice(&proof_round.coeffs);

		prev_rd_challenge = Some(challenger.sample());

		rd_claims.push(prover_state.round_claim().clone());
		rd_proofs.push(proof_round);
	}

	let evalcheck_claim = prover_state.finalize(prev_rd_challenge).unwrap();
	let prove_output = SumcheckProveOutput {
		evalcheck_claim,
		sumcheck_proof: SumcheckProof { rounds: rd_proofs },
	};

	(rd_claims, prove_output)
}

#[instrument(skip_all, name = "test_utils::full_prove_with_switchover")]
pub fn full_prove_with_switchover<F, PW, CW, M, CH>(
	claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<PW, CW, M>,
	domain: &EvaluationDomain<F>,
	challenger: CH,
	switchover: usize,
) -> (Vec<SumcheckRoundClaim<F>>, SumcheckProveOutput<F>)
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CW: CompositionPoly<PW>,
	M: MultilinearPoly<PW> + Clone + Sync,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = claim.poly.n_vars();

	assert!(switchover > 0);
	assert_eq!(witness.n_vars(), n_vars);

	let switchovers = repeat(switchover)
		.take(witness.n_multilinears())
		.collect::<Vec<_>>();

	let prover_state =
		SumcheckProverState::new(domain, claim.clone(), witness, &switchovers).unwrap();
	full_prove_with_switchover_impl(n_vars, prover_state, challenger)
}

#[instrument(
	skip_all,
	name = "test_utils::prove_bivariate_sumchecks_with_switchover"
)]
pub fn prove_bivariate_sumchecks_with_switchover<'a, F, PW, CH>(
	sumchecks: impl IntoIterator<Item = BivariateSumcheck<'a, F, PW>>,
	challenger: &mut CH,
	switchover: usize,
) -> Result<(SumcheckBatchProof<F>, impl IntoIterator<Item = EvalcheckClaim<F>>), SumcheckError>
where
	F: Field + Step + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	CH: CanObserve<F> + CanSample<F> + Clone,
{
	let bivariate_domain = EvaluationDomain::new(3)?;

	let (claims, witnesses) = sumchecks.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

	assert!(switchover > 0);
	let prover_states = witnesses
		.into_iter()
		.zip(&claims)
		.map(|(witness, claim)| {
			// FIXME temporary hack until a better way to specify switchovers is implemented
			SumcheckProverState::new(&bivariate_domain, claim.clone(), witness, &[switchover; 2])
		})
		.collect::<Result<Vec<_>, _>>()?;

	let challenger_snapshot = challenger.clone();
	let batch_proof = batch_prove(prover_states, challenger)?;
	let evalcheck_claims = batch_verify(claims, batch_proof.clone(), challenger_snapshot)?;

	Ok((batch_proof, evalcheck_claims))
}

#[instrument(skip_all, name = "test_utils::make_non_same_query_pcs_sumcheck_claims")]
pub fn make_non_same_query_pcs_sumcheck_claims<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
	new_batch_committed_eval_claims: &mut BatchCommittedEvalClaims<F>,
) -> Result<Vec<SumcheckClaim<F>>, EvalcheckError> {
	let metas = non_same_query_pcs_sumcheck_metas(
		oracles,
		committed_eval_claims,
		new_batch_committed_eval_claims,
	)?;

	let claims = metas
		.into_iter()
		.map(|meta| non_same_query_pcs_sumcheck_claim(oracles, meta))
		.collect::<Result<Vec<_>, EvalcheckError>>()?;

	Ok(claims)
}

#[instrument(skip_all, name = "test_utils::make_non_same_query_pcs_sumchecks")]
pub fn make_non_same_query_pcs_sumchecks<'a, F, PW>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
	new_batch_committed_eval_claims: &mut BatchCommittedEvalClaims<F>,
) -> Result<Vec<BivariateSumcheck<'a, F, PW>>, EvalcheckError>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	let metas = non_same_query_pcs_sumcheck_metas(
		oracles,
		committed_eval_claims,
		new_batch_committed_eval_claims,
	)?;

	let mut memoized_queries = MemoizedQueries::new();

	let sumchecks = metas
		.into_iter()
		.map(|meta| {
			let claim = non_same_query_pcs_sumcheck_claim(oracles, meta.clone())?;
			let witness =
				non_same_query_pcs_sumcheck_witness(witness_index, &mut memoized_queries, meta)?;
			Ok((claim, witness))
		})
		.collect::<Result<Vec<_>, EvalcheckError>>()?;

	Ok(sumchecks)
}
