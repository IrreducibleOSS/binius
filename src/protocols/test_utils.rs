// Copyright 2023 Ulvetanna Inc.

use p3_challenger::{CanObserve, CanSample};
use std::{borrow::Borrow, sync::Arc};

use crate::field::Field;

use crate::{
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearExtension, MultilinearPoly,
	},
	protocols::{
		evalcheck::evalcheck::EvalcheckClaim,
		sumcheck::{
			prove::{
				prove_at_switchover, prove_before_switchover, prove_final, prove_first_round,
				prove_first_round_with_operating_field, prove_later_round_with_operating_field,
				prove_post_switchover,
			},
			setup_first_round_claim,
			verify::{verify_final, verify_round},
			SumcheckClaim, SumcheckProof, SumcheckProveOutput, SumcheckRoundClaim, SumcheckWitness,
		},
	},
};

#[derive(Debug)]
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

pub fn transform_poly<'a, F, OF>(
	poly: &MultilinearComposite<F, MultilinearExtension<'a, F>, MultilinearExtension<'a, F>>,
	replacement_composition: Arc<dyn CompositionPoly<OF>>,
) -> Result<
	MultilinearComposite<OF, MultilinearExtension<'static, OF>, MultilinearExtension<'static, OF>>,
	PolynomialError,
>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let multilinears = poly
		.iter_multilinear_polys()
		.map(|multilin| {
			let values = multilin
				.evals()
				.iter()
				.cloned()
				.map(OF::from)
				.collect::<Vec<_>>();
			MultilinearExtension::from_values(values)
		})
		.collect::<Result<Vec<_>, _>>()?;
	let ret = MultilinearComposite::new(poly.n_vars(), replacement_composition, multilinears)?;
	Ok(ret)
}

pub fn full_verify<F, CH>(
	claim: &SumcheckClaim<F>,
	proof: SumcheckProof<F>,
	domain: &EvaluationDomain<F>,
	mut challenger: CH,
) -> (Vec<SumcheckRoundClaim<F>>, EvalcheckClaim<F>)
where
	F: Field,
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
		rd_claim =
			verify_round(&claim.poly, round_proof.clone(), rd_claim, challenger.sample(), domain)
				.unwrap();
		rd_claims.push(rd_claim.clone());
	}

	let last_round_proof = &proof.rounds[n_rounds - 1];
	let final_claim =
		verify_final(&claim.poly, last_round_proof.clone(), rd_claim, challenger.sample(), domain)
			.unwrap();

	(rd_claims, final_claim)
}

pub fn full_prove_with_switchover<F, M, BM, CH>(
	claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<F, M, BM>,
	domain: &EvaluationDomain<F>,
	mut challenger: CH,
	switchover: usize,
) -> (Vec<SumcheckRoundClaim<F>>, SumcheckProveOutput<F, M, BM>)
where
	F: Field,
	M: MultilinearPoly<F> + Send + Sync + ?Sized,
	BM: Borrow<M> + Clone + Sync,
	CH: CanSample<F> + CanObserve<F>,
{
	let current_witness = witness.clone();
	let n_vars = claim.poly.n_vars();

	assert!(switchover > 0);
	assert_eq!(witness.n_vars(), n_vars);
	assert!(n_vars > 0);

	// Make initial round claim
	let mut rd_claims = vec![];

	// FIRST ROUND
	let mut rd_output = prove_first_round(claim, current_witness, domain, switchover).unwrap();
	rd_claims.push(rd_output.claim.clone());

	// BEFORE SWITCHOVER
	for i in 0..switchover - 1 {
		challenger.observe_slice(&rd_output.current_proof.rounds[i].coeffs);
		rd_output = prove_before_switchover(claim, challenger.sample(), rd_output, domain).unwrap();
		rd_claims.push(rd_output.claim.clone());
	}

	// AT SWITCHOVER
	challenger.observe_slice(&rd_output.current_proof.rounds[switchover - 1].coeffs);
	let mut rd_output = prove_at_switchover(claim, challenger.sample(), rd_output, domain).unwrap();
	rd_claims.push(rd_output.claim.clone());

	// AFTER SWITCHOVER
	for i in switchover..n_vars - 1 {
		challenger.observe_slice(&rd_output.current_proof.rounds[i].coeffs);
		rd_output = prove_post_switchover(claim, challenger.sample(), rd_output, domain).unwrap();
		rd_claims.push(rd_output.claim.clone());
	}

	let final_output = prove_final(
		claim,
		witness,
		rd_output.current_proof,
		rd_output.claim,
		challenger.sample(),
		domain,
	)
	.unwrap();

	(rd_claims, final_output)
}

pub fn full_prove_with_operating_field<F, OF, M, BM, OM, BOM, CH>(
	claim: &SumcheckClaim<F>,
	witness: SumcheckWitness<F, M, BM>,
	operating_witness: SumcheckWitness<OF, OM, BOM>,
	domain: &EvaluationDomain<F>,
	mut challenger: CH,
) -> (Vec<SumcheckRoundClaim<F>>, SumcheckProveOutput<F, M, BM>)
where
	F: Field,
	OF: Field + From<F> + Into<F>,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
	OM: MultilinearPoly<OF> + Sync + ?Sized,
	BOM: Borrow<OM> + Sync,
	CH: CanObserve<F> + CanSample<F>,
{
	let n_vars = claim.poly.n_vars();

	assert_eq!(operating_witness.n_vars(), n_vars);
	assert!(n_vars > 0);

	// Setup Round Claim
	let mut rd_claims = vec![];

	// FIRST ROUND
	let rd_output =
		prove_first_round_with_operating_field(claim, operating_witness, domain).unwrap();
	rd_claims.push(rd_output.claim.clone());

	let (final_round_proof, final_round_claim) = match n_vars {
		1 => (rd_output.current_proof, rd_output.claim),
		_ => {
			// Run second round outside the loop because this one takes an OM witness and the subsequent rounds take a
			// concrete MultilinearExtension witness.
			challenger.observe_slice(&rd_output.current_proof.rounds[0].coeffs);
			let mut rd_output = prove_later_round_with_operating_field(
				claim,
				challenger.sample(),
				rd_output,
				domain,
			)
			.unwrap();
			rd_claims.push(rd_output.claim.clone());

			for i in 1..n_vars - 1 {
				challenger.observe_slice(&rd_output.current_proof.rounds[i].coeffs);
				rd_output = prove_later_round_with_operating_field(
					claim,
					challenger.sample(),
					rd_output,
					domain,
				)
				.unwrap();
				rd_claims.push(rd_output.claim.clone());
			}

			(rd_output.current_proof, rd_output.claim)
		}
	};

	let final_output = prove_final(
		claim,
		witness,
		final_round_proof,
		final_round_claim,
		challenger.sample(),
		domain,
	)
	.unwrap();

	(rd_claims, final_output)
}
