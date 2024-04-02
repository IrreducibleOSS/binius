// Copyright 2023 Ulvetanna Inc.

use crate::{
	field::{packed::set_packed_slice, BinaryField1b, Field, PackedField},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearExtension,
		MultilinearPoly, MultivariatePoly,
	},
	protocols::{
		evalcheck::EvalcheckClaim,
		sumcheck::{
			setup_first_round_claim, verify_final, verify_round, SumcheckClaim, SumcheckProof,
			SumcheckProveOutput, SumcheckProverState, SumcheckRoundClaim, SumcheckWitness,
		},
	},
};
use p3_challenger::{CanObserve, CanSample};
use std::iter::repeat;
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
pub fn full_verify<F, C, CH>(
	claim: &SumcheckClaim<F, C>,
	proof: SumcheckProof<F>,
	mut challenger: CH,
) -> (Vec<SumcheckRoundClaim<F>>, EvalcheckClaim<F, C>)
where
	F: Field,
	C: CompositionPoly<F>,
	CH: CanSample<F> + CanObserve<F>,
{
	let n_vars = claim.poly.n_vars();
	assert!(n_vars > 0);

	// Make initial round claim
	let mut rd_claim = setup_first_round_claim(claim);
	let mut rd_claims = vec![rd_claim.clone()];

	let n_rounds = proof.rounds.len();
	for round_proof in proof.rounds[..n_rounds - 1].iter() {
		challenger.observe_slice(round_proof.coeffs.as_slice());
		rd_claim = verify_round(rd_claim, challenger.sample(), round_proof.clone()).unwrap();
		rd_claims.push(rd_claim.clone());
	}

	let last_round_proof = &proof.rounds[n_rounds - 1];
	challenger.observe_slice(last_round_proof.coeffs.as_slice());

	let final_claim =
		verify_final(&claim.poly, rd_claim, challenger.sample(), last_round_proof.clone()).unwrap();

	(rd_claims, final_claim)
}

#[instrument(skip_all, name = "test_utils::full_prove_with_switchover")]
pub fn full_prove_with_switchover<F, PW, C, CW, M, CH>(
	claim: &SumcheckClaim<F, C>,
	witness: SumcheckWitness<PW, CW, M>,
	domain: &EvaluationDomain<F>,
	mut challenger: CH,
	switchover: usize,
) -> (Vec<SumcheckRoundClaim<F>>, SumcheckProveOutput<F, C>)
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
	C: Clone,
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

	let mut prover_state = SumcheckProverState::new(claim, witness, &switchovers).unwrap();

	let mut prev_rd_challenge = None;
	let mut rd_claims = Vec::new();

	for _round in 0..n_vars {
		let proof_round = prover_state
			.execute_round(domain, prev_rd_challenge)
			.unwrap();

		challenger.observe_slice(&proof_round.coeffs);

		prev_rd_challenge = Some(challenger.sample());

		rd_claims.push(prover_state.get_claim());
	}

	let prove_output = prover_state
		.finalize(&claim.poly, prev_rd_challenge)
		.unwrap();

	(rd_claims, prove_output)
}
