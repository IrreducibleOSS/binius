// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, Field, PackedField, TowerField};
use binius_utils::{bail, sorting::is_sorted_ascending};

use super::{
	super::error::Error,
	common::{GeneratorExponentClaim, GeneratorExponentReductionOutput},
	compositions::ExponentCompositions,
	utils::first_layer_inverse,
};
use crate::{
	composition::{ComplexIndexComposition, IndexComposition},
	fiat_shamir::Challenger,
	polynomial::MultivariatePoly,
	protocols::{
		gkr_gpa::LayerClaim,
		gkr_int_mul::error::VerificationError,
		sumcheck::{
			self, zerocheck::ExtraProduct, BatchSumcheckOutput, CompositeSumClaim, SumcheckClaim,
		},
	},
	transcript::VerifierTranscript,
	transparent::eq_ind::EqIndPartialEval,
};

/// This is the verification side of the following interactive protocol.
///
/// Consider the multilinears a_0, a_1, ..., a_{n - 1}.
/// At each point on the hypercube, we construct the n-bit integer a(X) as:
///
/// a(X) = 2^0 * a_0(X) + 2^1 * a_1(X) + 2^2 * a_2(X) + ... + 2^{n - 1} * a_{n - 1}(X)
///
/// The multilinear w has values at each point on the hypercube such that:
///
/// g^a(X) = w(X) for all X on the hypercube.
///
/// This interactive protocol reduces a vector of claimed evaluations of w  
/// to corresponding vector of claimed evaluations of the a_i's.
///
/// **Input:** A vector of evaluation claims on w.  
/// **Output:** n separate vectors of claims (at different points) on each of the a_i's.
pub fn batch_verify<FGenerator, F, Challenger_>(
	mut claims: Vec<GeneratorExponentClaim<F>>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<GeneratorExponentReductionOutput<F>, Error>
where
	FGenerator: TowerField,
	F: TowerField + ExtensionField<FGenerator>,
	Challenger_: Challenger,
{
	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	let max_exponent_bit_number = claims
		.first()
		.map(|claim| claim.exponent_bit_width)
		.unwrap_or(0);

	if claims.is_empty() {
		return Ok(GeneratorExponentReductionOutput {
			eval_claims_on_exponent_bit_columns,
		});
	}

	// Check that the witnesses are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	for layer_no in 0..max_exponent_bit_number {
		let gkr_sumcheck_claims =
			build_layer_gkr_sumcheck_claims::<_, FGenerator>(&claims, layer_no)?;

		let sumcheck_verification_output =
			sumcheck::batch_verify(&gkr_sumcheck_claims, transcript)?;

		let layer_exponent_claims = build_layer_exponent_claims::<_, FGenerator>(
			&mut claims,
			sumcheck_verification_output,
			layer_no,
		)?;

		eval_claims_on_exponent_bit_columns.push(layer_exponent_claims);

		claims.retain(|claim| !claim.is_last_layer(layer_no));
	}

	Ok(GeneratorExponentReductionOutput {
		eval_claims_on_exponent_bit_columns,
	})
}

#[allow(clippy::type_complexity)]
fn build_layer_gkr_sumcheck_claims<F, FGenerator>(
	claims: &[GeneratorExponentClaim<F>],
	layer_no: usize,
) -> Result<
	Vec<SumcheckClaim<F, ComplexIndexComposition<ExtraProduct<ExponentCompositions<F>>>>>,
	Error,
>
where
	FGenerator: BinaryField,
	F: ExtensionField<FGenerator>,
{
	let mut sumcheck_claims = Vec::new();

	let mut current_eval_point = claims[0].eval_point.clone();

	let mut active_index = 0;

	for i in 0..claims.len() {
		if current_eval_point != claims[i].eval_point {
			let this_round_sumcheck_claim =
				build_eval_point_claims(&claims[active_index..i], &current_eval_point, layer_no)?;

			if let Some(this_round_sumcheck_claim) = this_round_sumcheck_claim {
				sumcheck_claims.push(this_round_sumcheck_claim);
			}

			active_index = i;
			current_eval_point = claims[i].eval_point.clone();
		}

		if i == claims.len() - 1 {
			let this_round_sumcheck_claim =
				build_eval_point_claims(&claims[active_index..=i], &current_eval_point, layer_no)?;

			if let Some(this_round_sumcheck_claim) = this_round_sumcheck_claim {
				sumcheck_claims.push(this_round_sumcheck_claim);
			}
		}
	}
	Ok(sumcheck_claims)
}

#[allow(clippy::type_complexity)]
fn build_eval_point_claims<F, FGenerator>(
	claims: &[GeneratorExponentClaim<F>],
	eval_point: &[F],
	layer_no: usize,
) -> Result<
	Option<SumcheckClaim<F, ComplexIndexComposition<ExtraProduct<ExponentCompositions<F>>>>>,
	Error,
>
where
	F: Field + ExtensionField<FGenerator>,
	FGenerator: BinaryField,
{
	let (mut composite_claims_n_multilinears, n_claims) =
		claims
			.iter()
			.fold((0, 0), |(mut n_multilinears, mut n_claims), claim| {
				n_multilinears += claim.layer_n_multilinears(layer_no);
				if claim.with_dynamic_generator || !claim.is_last_layer(layer_no) {
					n_claims += 1;
				}
				(n_multilinears, n_claims)
			});

	if composite_claims_n_multilinears == 0 {
		return Ok(None);
	}

	let eq_ind_index = composite_claims_n_multilinears;

	// add eq_ind
	composite_claims_n_multilinears += 1;

	let mut ml_index = 0;

	let mut composite_sums = Vec::with_capacity(n_claims);

	for claim in claims {
		let composition = match (claim.with_dynamic_generator, claim.is_last_layer(layer_no)) {
			// Last internal layer with dynamic generator
			(true, true) => {
				let composition = IndexComposition::new(
					composite_claims_n_multilinears,
					[ml_index, ml_index + 1, eq_ind_index],
					ExtraProduct {
						inner: ExponentCompositions::DynamicGeneratorLastLayer,
					},
				)?;

				ComplexIndexComposition::Trivariate(composition)
			}
			// Non-last internal layer with a dynamic generator.
			(true, false) => {
				let composition = IndexComposition::new(
					composite_claims_n_multilinears,
					[ml_index, ml_index + 1, ml_index + 2, eq_ind_index],
					ExtraProduct {
						inner: ExponentCompositions::DynamicGenerator,
					},
				)?;

				ComplexIndexComposition::Quadrivariate(composition)
			}
			// Non-last internal layer with static generator
			(false, false) => {
				let internal_exponent_bit_number = claim.current_layer_exponent_bit_no(layer_no);

				let generator_power_constant = F::from(
					FGenerator::MULTIPLICATIVE_GENERATOR.pow(1 << internal_exponent_bit_number),
				);

				let composition = IndexComposition::new(
					composite_claims_n_multilinears,
					[ml_index, ml_index + 1, eq_ind_index],
					ExtraProduct {
						inner: ExponentCompositions::StaticGenerator {
							generator_power_constant,
						},
					},
				)?;

				ComplexIndexComposition::Trivariate(composition)
			}
			// Zero layer with static generator
			(false, true) => continue,
		};

		ml_index += claim.layer_n_multilinears(layer_no);

		composite_sums.push(CompositeSumClaim {
			composition,
			sum: claim.eval,
		});
	}

	SumcheckClaim::new(eval_point.len(), composite_claims_n_multilinears, composite_sums)
		.map(Some)
		.map_err(Error::from)
}

pub fn build_layer_exponent_claims<F, FGenerator>(
	claims: &mut [GeneratorExponentClaim<F>],
	sumcheck_output: BatchSumcheckOutput<F>,
	layer_no: usize,
) -> Result<Vec<LayerClaim<F>>, Error>
where
	F: TowerField + ExtensionField<FGenerator>,
	FGenerator: BinaryField,
{
	let mut layer_eval_claims_on_exponent_bit_columns = Vec::new();

	let mut multilinears_index = 0;

	let mut multilinear_evals_index = 0;

	let mut is_eq_ind_unverified = true;

	let mut current_eval_point = claims[0].eval_point.clone();

	for claim in claims {
		if claim.is_last_layer(layer_no) && !claim.with_dynamic_generator {
			let exponent_claim = LayerClaim::<F> {
				eval_point: claim.eval_point.clone(),
				eval: first_layer_inverse::<FGenerator, _>(claim.eval),
			};
			layer_eval_claims_on_exponent_bit_columns.push(exponent_claim);

			continue;
		}

		if claim.eval_point != current_eval_point {
			current_eval_point = claim.eval_point.clone();
			if multilinears_index != 0 {
				multilinears_index = 0;
				multilinear_evals_index += 1;
			}
			is_eq_ind_unverified = true;
		}

		let multilinear_evals = &sumcheck_output.multilinear_evals[multilinear_evals_index];

		let eval_point = &sumcheck_output.challenges
			[sumcheck_output.challenges.len() - current_eval_point.len()..];

		if !claim.is_last_layer(layer_no) {
			claim.eval = multilinear_evals[multilinears_index];
			claim.eval_point = eval_point.to_vec();
		}

		let exponent_claim = LayerClaim {
			eval: multilinear_evals[multilinears_index + 1],
			eval_point: eval_point.to_vec(),
		};

		layer_eval_claims_on_exponent_bit_columns.push(exponent_claim);

		multilinears_index += claim.layer_n_multilinears(layer_no);

		if is_eq_ind_unverified {
			let expected_eq_ind_eval =
				EqIndPartialEval::new(current_eval_point.len(), current_eval_point.clone())
					.and_then(|eq_ind| eq_ind.evaluate(eval_point))?;

			let eq_ind_eval = *sumcheck_output.multilinear_evals[multilinear_evals_index]
				.last()
				.expect("multilinear_evals contains the evaluation of the equality indicator");

			if expected_eq_ind_eval != eq_ind_eval {
				return Err(VerificationError::IncorrectEqIndEvaluation.into());
			}

			is_eq_ind_unverified = false;
		}
	}
	Ok(layer_eval_claims_on_exponent_bit_columns)
}
