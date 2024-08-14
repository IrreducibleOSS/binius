// Copyright 2024 Ulvetanna Inc.

use super::{
	gkr_gpa::{BatchLayerProof, GrandProductBatchProof, LayerClaim},
	Error, GrandProductClaim, VerificationError,
};
use crate::{
	polynomial::{composition::BivariateProduct, extrapolate_line_scalar},
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		gkr_sumcheck::{self, GkrSumcheckClaim},
	},
};
use binius_field::{Field, TowerField};
use binius_utils::sorting::{stable_sort, unsort};
use itertools::izip;
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

/// Verifies batch reduction turning each GrandProductClaim into an EvalcheckMultilinearClaim
#[instrument(skip_all, name = "gkr_gpa::batch_verify", level = "debug")]
pub fn batch_verify<F, Challenger>(
	claims: impl IntoIterator<Item = GrandProductClaim<F>>,
	proof: GrandProductBatchProof<F>,
	mut challenger: Challenger,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error>
where
	F: TowerField,
	Challenger: CanSample<F> + CanObserve<F>,
{
	let GrandProductBatchProof { batch_layer_proofs } = proof;

	let (original_indices, mut sorted_claims) =
		stable_sort(claims, |claim| claim.poly.n_vars(), true);
	let max_n_vars = sorted_claims
		.first()
		.map(|claim| claim.poly.n_vars())
		.ok_or(Error::EmptyClaimsArray)?;

	if max_n_vars != batch_layer_proofs.len() {
		return Err(Error::MismatchedClaimsAndProofs);
	}

	// Create LayerClaims for each of the claims
	let mut layer_claims = sorted_claims
		.iter()
		.map(|claim| LayerClaim {
			eval_point: vec![],
			eval: claim.product,
		})
		.collect::<Vec<_>>();

	// Create a vector of evalchecks with the same length as the number of claims
	let n_claims = sorted_claims.len();
	let mut reverse_sorted_evalcheck_claims = Vec::with_capacity(n_claims);

	for (layer_no, batch_layer_proof) in batch_layer_proofs.into_iter().enumerate() {
		process_finished_claims(
			n_claims,
			layer_no,
			&mut layer_claims,
			&mut sorted_claims,
			&mut reverse_sorted_evalcheck_claims,
		);

		layer_claims = reduce_layer_claim_batch(layer_claims, batch_layer_proof, &mut challenger)?;
	}
	process_finished_claims(
		n_claims,
		max_n_vars,
		&mut layer_claims,
		&mut sorted_claims,
		&mut reverse_sorted_evalcheck_claims,
	);

	debug_assert!(layer_claims.is_empty());
	debug_assert_eq!(reverse_sorted_evalcheck_claims.len(), n_claims);

	reverse_sorted_evalcheck_claims.reverse();
	let sorted_evalcheck_claims = reverse_sorted_evalcheck_claims;

	let evalcheck_multilinear_claims = unsort(original_indices, sorted_evalcheck_claims);
	Ok(evalcheck_multilinear_claims)
}

fn process_finished_claims<F: Field>(
	n_claims: usize,
	layer_no: usize,
	layer_claims: &mut Vec<LayerClaim<F>>,
	sorted_claims: &mut Vec<GrandProductClaim<F>>,
	reverse_sorted_evalcheck_multilinear_claims: &mut Vec<EvalcheckMultilinearClaim<F>>,
) {
	while !sorted_claims.is_empty() && sorted_claims.last().unwrap().poly.n_vars() == layer_no {
		debug_assert!(layer_no > 0);
		debug_assert_eq!(sorted_claims.len(), layer_claims.len());
		let finished_layer_claim = layer_claims.pop().unwrap();
		let finished_original_claim = sorted_claims.pop().unwrap();
		let evalcheck_multilinear_claim = EvalcheckMultilinearClaim {
			poly: finished_original_claim.poly,
			eval: finished_layer_claim.eval,
			eval_point: finished_layer_claim.eval_point,
			is_random_point: true,
		};
		reverse_sorted_evalcheck_multilinear_claims.push(evalcheck_multilinear_claim);
		debug_assert_eq!(
			sorted_claims.len() + reverse_sorted_evalcheck_multilinear_claims.len(),
			n_claims
		);
	}
}

/// Reduces n kth LayerClaims to n (k+1)th LayerClaims
///
/// Arguments
/// * `claims` - The kth layer LayerClaims
/// * `proof` - The batch layer proof that reduces the kth layer claims of the product circuits to the (k+1)th
/// * `challenger` - The verifier challenger
fn reduce_layer_claim_batch<F, CH>(
	claims: Vec<LayerClaim<F>>,
	proof: BatchLayerProof<F>,
	mut challenger: CH,
) -> Result<Vec<LayerClaim<F>>, Error>
where
	F: Field,
	CH: CanSample<F> + CanObserve<F>,
{
	let BatchLayerProof {
		gkr_sumcheck_batch_proof,
		zero_evals,
		one_evals,
	} = proof;

	// Validation
	if claims.is_empty() {
		return Ok(vec![]);
	} else if zero_evals.len() != claims.len() {
		return Err(VerificationError::MismatchedZeroEvals.into());
	} else if one_evals.len() != claims.len() {
		return Err(VerificationError::MismatchedOneEvals.into());
	}

	let curr_layer_challenge = &claims[0].eval_point[..];
	if !claims
		.iter()
		.all(|claim| claim.eval_point == curr_layer_challenge)
	{
		return Err(Error::MismatchedEvalPointLength);
	}

	// Verify the gkr sumcheck batch proof and receive the corresponding reduced claims
	let gkr_sumcheck_claims = claims.iter().map(|claim| GkrSumcheckClaim {
		sum: claim.eval,
		r: claim.eval_point.clone(),
		n_vars: claim.eval_point.len(),
		degree: BivariateProduct.degree(),
	});
	let reduced_claims =
		gkr_sumcheck::batch_verify(gkr_sumcheck_claims, gkr_sumcheck_batch_proof, &mut challenger)?;

	debug_assert_eq!(reduced_claims.len(), claims.len());
	challenger.observe_slice(&zero_evals);
	challenger.observe_slice(&one_evals);

	// Validate the relationship between zero_evals, one_evals, and evals
	let evals = reduced_claims.iter().map(|claim| claim.eval);
	let is_zero_one_eval_advice_valid = izip!(zero_evals.iter(), one_evals.iter(), evals)
		.all(|(&zero_eval, &one_eval, eval)| zero_eval * one_eval == eval);

	if !is_zero_one_eval_advice_valid {
		return Err(Error::InvalidZeroOneEvalAdvice);
	}

	// Create the new (k+1)th layer LayerClaims for each grand product circuit
	let sumcheck_challenge = reduced_claims[0].eval_point.clone();
	let gkr_challenge = challenger.sample();
	let new_layer_challenge = sumcheck_challenge
		.into_iter()
		.chain(Some(gkr_challenge))
		.collect::<Vec<_>>();
	let new_layer_claims = zero_evals
		.into_iter()
		.zip(one_evals)
		.map(|(zero_eval, one_eval)| {
			let new_eval = extrapolate_line_scalar(zero_eval, one_eval, gkr_challenge);
			LayerClaim {
				eval_point: new_layer_challenge.clone(),
				eval: new_eval,
			}
		})
		.collect::<Vec<_>>();

	Ok(new_layer_claims)
}
