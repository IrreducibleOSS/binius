// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, Field, TowerField};
use binius_math::EvaluationOrder;
use binius_utils::{bail, sorting::is_sorted_ascending};

use super::{
	common::{BaseExpReductionOutput, ExpClaim, LayerClaim},
	compositions::IndexedExpComposition,
	error::{Error, VerificationError},
	verifiers::{DynamicExpVerifier, ExpVerifier, StaticBaseExpVerifier},
};
use crate::{
	fiat_shamir::Challenger,
	polynomial::MultivariatePoly,
	protocols::sumcheck::{self, BatchSumcheckOutput, EqIndSumcheckClaim},
	transcript::VerifierTranscript,
	transparent::eq_ind::EqIndPartialEval,
};

/// Verify a batched GKR exponentiation protocol execution.
///
/// The protocol can be batched over multiple instances by grouping consecutive verifiers over
/// eval_points in [ExpClaim] into [EqIndSumcheckClaim]s. To achieve this, we use
/// [crate::composition::IndexComposition], where eq indicator is always the last element. Since
/// exponents can have different bit sizes, resulting in a varying number of layers, we group
/// them starting from the first layer to maximize the opportunity to share the same evaluation point.
///
/// # Requirements
/// - Claims must be sorted in descending order by `n_vars`.
pub fn batch_verify<F, Challenger_>(
	evaluation_order: EvaluationOrder,
	claims: &[ExpClaim<F>],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BaseExpReductionOutput<F>, Error>
where
	F: TowerField,
	Challenger_: Challenger,
{
	let mut layers_claims = Vec::new();

	if claims.is_empty() {
		return Ok(BaseExpReductionOutput { layers_claims });
	}

	// Check that the witnesses are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let mut verifiers = make_verifiers(claims)?;

	let max_exponent_bit_number = verifiers
		.iter()
		.map(|p| p.exponent_bit_width())
		.max()
		.unwrap_or(0);

	for layer_no in 0..max_exponent_bit_number {
		let EqIndSumcheckClaimsWithEvalPoints {
			eq_ind_sumcheck_claims,
			eval_points,
		} = build_layer_eq_ind_sumcheck_claims(&verifiers, layer_no)?;

		let regular_sumcheck_claims =
			sumcheck::eq_ind::reduce_to_regular_sumchecks(&eq_ind_sumcheck_claims)?;

		let sumcheck_verification_output =
			sumcheck::batch_verify(evaluation_order, &regular_sumcheck_claims, transcript)?;

		let layer_exponent_claims = build_layer_exponent_bit_claims(
			evaluation_order,
			&mut verifiers,
			sumcheck_verification_output,
			eval_points,
			layer_no,
		)?;

		layers_claims.push(layer_exponent_claims);

		verifiers.retain(|verifier| !verifier.is_last_layer(layer_no));
	}

	Ok(BaseExpReductionOutput { layers_claims })
}

struct EqIndSumcheckClaimsWithEvalPoints<F: Field> {
	eq_ind_sumcheck_claims: Vec<EqIndSumcheckClaim<F, IndexedExpComposition<F>>>,
	eval_points: Vec<Vec<F>>,
}

/// Groups consecutive verifier by their `eval_point` and reduces them to sumcheck claims.
fn build_layer_eq_ind_sumcheck_claims<'a, F>(
	verifiers: &[Box<dyn ExpVerifier<F> + 'a>],
	layer_no: usize,
) -> Result<EqIndSumcheckClaimsWithEvalPoints<F>, Error>
where
	F: Field,
{
	let mut eq_ind_sumcheck_claims = Vec::new();

	let first_eval_point = verifiers[0].layer_claim_eval_point().to_vec();
	let mut eval_points = vec![first_eval_point];

	let mut active_index = 0;

	// group verifiers by evaluation points and build sumcheck claims.
	for i in 0..verifiers.len() {
		if verifiers[i].layer_claim_eval_point() != eval_points[eval_points.len() - 1] {
			let eq_ind_sumcheck_claim =
				build_eval_point_claims(&verifiers[active_index..i], layer_no)?;

			if let Some(eq_ind_sumcheck_claim) = eq_ind_sumcheck_claim {
				eq_ind_sumcheck_claims.push(eq_ind_sumcheck_claim);
			} else {
				// extract the last point because verifiers with this point will not participate in the sumcheck.
				eval_points.pop();
			}

			eval_points.push(verifiers[i].layer_claim_eval_point().to_vec());

			active_index = i;
		}

		if i == verifiers.len() - 1 {
			let eq_ind_sumcheck_claim =
				build_eval_point_claims(&verifiers[active_index..], layer_no)?;

			if let Some(eq_ind_sumcheck_claim) = eq_ind_sumcheck_claim {
				eq_ind_sumcheck_claims.push(eq_ind_sumcheck_claim);
			}
		}
	}

	Ok(EqIndSumcheckClaimsWithEvalPoints {
		eq_ind_sumcheck_claims,
		eval_points,
	})
}

/// Builds sumcheck claim for verifiers that share the same `eval_point` from their internal
/// [ExpClaim]s. The batched multilinears are structured as a single concatenated vector of all
/// multilinears used by the verifiers, with the eq indicator positioned at the end.
fn build_eval_point_claims<'a, F>(
	verifiers: &[Box<dyn ExpVerifier<F> + 'a>],
	layer_no: usize,
) -> Result<Option<EqIndSumcheckClaim<F, IndexedExpComposition<F>>>, Error>
where
	F: Field,
{
	let (composite_claims_n_multilinears, n_claims) =
		verifiers
			.iter()
			.fold((0, 0), |(n_multilinears, n_claims), verifier| {
				let layer_n_multilinears = verifier.layer_n_multilinears(layer_no);
				let layer_n_claims = verifier.layer_n_claims(layer_no);

				(n_multilinears + layer_n_multilinears, n_claims + layer_n_claims)
			});

	if composite_claims_n_multilinears == 0 {
		return Ok(None);
	}

	let n_vars = verifiers[0].layer_claim_eval_point().len();

	let mut multilinears_index = 0;

	let mut composite_sums = Vec::with_capacity(n_claims);

	for verifier in verifiers {
		let composite_sum_claim = verifier.layer_composite_sum_claim(
			layer_no,
			composite_claims_n_multilinears,
			multilinears_index,
		)?;

		if let Some(composite_sum_claim) = composite_sum_claim {
			composite_sums.push(composite_sum_claim);
		}

		multilinears_index += verifier.layer_n_multilinears(layer_no);
	}

	Ok(Some(EqIndSumcheckClaim::new(n_vars, composite_claims_n_multilinears, composite_sums)?))
}

/// Reduces the sumcheck output to [LayerClaim]s and updates the internal verifier [ExpClaim]s for the next layer.
pub fn build_layer_exponent_bit_claims<'a, F>(
	evaluation_order: EvaluationOrder,
	verifiers: &mut [Box<dyn ExpVerifier<F> + 'a>],
	mut sumcheck_output: BatchSumcheckOutput<F>,
	eval_points: Vec<Vec<F>>,
	layer_no: usize,
) -> Result<Vec<LayerClaim<F>>, Error>
where
	F: TowerField,
{
	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	for (multilinear_evals, current_eval_point) in sumcheck_output
		.multilinear_evals
		.iter_mut()
		.zip(eval_points.into_iter())
	{
		let n_vars = current_eval_point.len();

		let eval_point = match evaluation_order {
			EvaluationOrder::LowToHigh => {
				sumcheck_output.challenges[sumcheck_output.challenges.len() - n_vars..].to_vec()
			}
			EvaluationOrder::HighToLow => sumcheck_output.challenges[..n_vars].to_vec(),
		};

		let expected_eq_ind_eval =
			EqIndPartialEval::new(current_eval_point).evaluate(&eval_point)?;

		let eq_ind_eval = multilinear_evals
			.pop()
			.expect("multilinear_evals contains the evaluation of the equality indicator");

		if expected_eq_ind_eval != eq_ind_eval {
			return Err(VerificationError::IncorrectEqIndEvaluation.into());
		}
	}

	let mut multilinear_evals = sumcheck_output.multilinear_evals.into_iter().flatten();

	for verifier in verifiers {
		let this_verifier_n_multilinears = verifier.layer_n_multilinears(layer_no);

		let this_verifier_multilinear_evals = multilinear_evals
			.by_ref()
			.take(this_verifier_n_multilinears)
			.collect::<Vec<_>>();

		let layer_claims = verifier.finish_layer(
			evaluation_order,
			layer_no,
			&this_verifier_multilinear_evals,
			&sumcheck_output.challenges,
		);

		eval_claims_on_exponent_bit_columns.extend(layer_claims);
	}

	Ok(eval_claims_on_exponent_bit_columns)
}

/// Creates a vector of boxed [ExpVerifier]s from the given claims.
fn make_verifiers<'a, F>(claims: &[ExpClaim<F>]) -> Result<Vec<Box<dyn ExpVerifier<F> + 'a>>, Error>
where
	F: BinaryField,
{
	claims
		.iter()
		.map(|claim| {
			if claim.static_base.is_none() {
				DynamicExpVerifier::new(claim)
					.map(|verifier| Box::new(verifier) as Box<dyn ExpVerifier<F> + 'a>)
			} else {
				StaticBaseExpVerifier::<F>::new(claim)
					.map(|verifier| Box::new(verifier) as Box<dyn ExpVerifier<F> + 'a>)
			}
		})
		.collect::<Result<Vec<_>, Error>>()
}
