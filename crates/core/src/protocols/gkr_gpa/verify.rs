// Copyright 2024-2025 Irreducible Inc.

use binius_field::{Field, TowerField};
use binius_math::{extrapolate_line_scalar, EvaluationOrder};
use binius_utils::{
	bail,
	sorting::{stable_sort, unsort},
};
use tracing::instrument;

use super::{gkr_gpa::LayerClaim, Error, GrandProductClaim};
use crate::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::{CanSample, Challenger},
	polynomial::Error as PolynomialError,
	protocols::sumcheck::{
		self, eq_ind::ClaimsSortingOrder, CompositeSumClaim, EqIndSumcheckClaim,
	},
	transcript::VerifierTranscript,
};

/// Verifies batch reduction turning each GrandProductClaim into an EvalcheckMultilinearClaim
#[instrument(skip_all, name = "gkr_gpa::batch_verify", level = "debug")]
pub fn batch_verify<F, Challenger_>(
	evaluation_order: EvaluationOrder,
	claims: impl IntoIterator<Item = GrandProductClaim<F>>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<Vec<LayerClaim<F>>, Error>
where
	F: TowerField,
	Challenger_: Challenger,
{
	let (original_indices, mut sorted_claims) = stable_sort(claims, |claim| claim.n_vars, true);
	let max_n_vars = sorted_claims.first().map(|claim| claim.n_vars).unwrap_or(0);

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

	for layer_no in 0..max_n_vars {
		process_finished_claims(
			n_claims,
			layer_no,
			&mut layer_claims,
			&mut sorted_claims,
			&mut reverse_sorted_evalcheck_claims,
		);

		layer_claims = reduce_layer_claim_batch(evaluation_order, &layer_claims, transcript)?;
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

	let final_layer_claims = unsort(original_indices, sorted_evalcheck_claims);
	Ok(final_layer_claims)
}

fn process_finished_claims<F: Field>(
	n_claims: usize,
	layer_no: usize,
	layer_claims: &mut Vec<LayerClaim<F>>,
	sorted_claims: &mut Vec<GrandProductClaim<F>>,
	reverse_sorted_final_layer_claims: &mut Vec<LayerClaim<F>>,
) {
	while let Some(claim) = sorted_claims.last() {
		if claim.n_vars != layer_no {
			break;
		}

		debug_assert!(layer_no > 0);
		debug_assert_eq!(sorted_claims.len(), layer_claims.len());
		let finished_layer_claim = layer_claims.pop().expect("must exist");
		let _ = sorted_claims.pop().expect("must exist");
		reverse_sorted_final_layer_claims.push(finished_layer_claim);
		debug_assert_eq!(sorted_claims.len() + reverse_sorted_final_layer_claims.len(), n_claims);
	}
}

/// Reduces n kth LayerClaims to n (k+1)th LayerClaims
///
/// Arguments
/// * `claims` - The kth layer LayerClaims
/// * `proof` - The batch layer proof that reduces the kth layer claims of the product circuits to the (k+1)th
/// * `transcript` - The verifier transcript
fn reduce_layer_claim_batch<F, Challenger_>(
	evaluation_order: EvaluationOrder,
	claims: &[LayerClaim<F>],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<Vec<LayerClaim<F>>, Error>
where
	F: TowerField,
	Challenger_: Challenger,
{
	// Validation
	if claims.is_empty() {
		return Ok(vec![]);
	}

	let curr_layer_challenge = &claims[0].eval_point;
	if !claims
		.iter()
		.all(|claim| &claim.eval_point == curr_layer_challenge)
	{
		bail!(Error::MismatchedEvalPointLength);
	}

	let n_vars = curr_layer_challenge.len();
	let n_multilinears = 2 * claims.len();

	let composite_sums = claims
		.iter()
		.enumerate()
		.map(|(i, claim)| {
			let composition =
				IndexComposition::new(n_multilinears, [2 * i, 2 * i + 1], BivariateProduct {})?;

			let composite_sum_claim = CompositeSumClaim {
				composition,
				sum: claim.eval,
			};

			Ok(composite_sum_claim)
		})
		.collect::<Result<Vec<_>, PolynomialError>>()?;

	let eq_ind_sumcheck_claim = EqIndSumcheckClaim::new(n_vars, n_multilinears, composite_sums)?;

	let eq_ind_sumcheck_claims = [eq_ind_sumcheck_claim];

	let regular_sumcheck_claims =
		sumcheck::eq_ind::reduce_to_regular_sumchecks(&eq_ind_sumcheck_claims)?;

	let batch_sumcheck_output =
		sumcheck::batch_verify(evaluation_order, &regular_sumcheck_claims, transcript)?;

	let batch_sumcheck_output = sumcheck::eq_ind::verify_sumcheck_outputs(
		ClaimsSortingOrder::DescendingVars,
		&eq_ind_sumcheck_claims,
		curr_layer_challenge,
		batch_sumcheck_output,
	)?;

	// Create the new (k+1)th layer LayerClaims for each grand product circuit
	let sumcheck_challenge = batch_sumcheck_output.challenges.clone();
	let gpa_challenge = transcript.sample();
	let new_layer_challenge = sumcheck_challenge
		.into_iter()
		.chain(Some(gpa_challenge))
		.collect::<Vec<_>>();
	let new_layer_claims = batch_sumcheck_output.multilinear_evals[0]
		.chunks_exact(2)
		.map(|evals| {
			let new_eval = extrapolate_line_scalar::<_, F>(evals[0], evals[1], gpa_challenge);
			LayerClaim {
				eval_point: new_layer_challenge.clone(),
				eval: new_eval,
			}
		})
		.collect::<Vec<_>>();

	Ok(new_layer_claims)
}
