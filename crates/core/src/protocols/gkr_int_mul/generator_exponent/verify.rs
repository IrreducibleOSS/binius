// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{ExtensionField, TowerField};
use binius_math::EvaluationOrder;

use super::{
	super::error::Error,
	common::{GeneratorExponentClaim, GeneratorExponentReductionOutput},
	utils::first_layer_inverse,
};
use crate::{
	fiat_shamir::Challenger,
	polynomial::MultivariatePoly,
	protocols::{
		gkr_gpa::LayerClaim,
		gkr_int_mul::{error::VerificationError, generator_exponent::compositions::MultiplyOrDont},
		sumcheck::{self, zerocheck::ExtraProduct, CompositeSumClaim, SumcheckClaim},
	},
	transcript::VerifierTranscript,
	transparent::eq_ind::EqIndPartialEval,
};

/// This is the verification side of the following interactive protocol
/// Consider the multilinears a_0, a_1, ..., a_63 (here EXPONENT_BIT_WIDTH = 64)
/// At each point on the hypercube, we construct the 64-bit integer a(X) as
/// a(X) = 2^0 * a_0(X) + 2^1 * a_1(X) + 2^2 * a_2(X) ... + 2^63 * a_63(X)
///
/// The multilinear n has values at each point on the hypercube such that
///
/// g^a(X) = n(X) for all X on the hypercube
///
/// This interactive protocol reduces a claimed evaluation of n to claimed evaluations of
/// the a_i's
///
/// Input: One evaluation claim on n
///
/// Output: EXPONENT_BITS_WIDTH separate claims (at different points) on each of the a_i's
pub fn verify<FGenerator, F, Challenger_, const EXPONENT_BIT_WIDTH: usize>(
	claim: &GeneratorExponentClaim<F>,
	transcript: &mut VerifierTranscript<Challenger_>,
	log_size: usize,
) -> Result<GeneratorExponentReductionOutput<F, EXPONENT_BIT_WIDTH>, Error>
where
	FGenerator: TowerField,
	F: TowerField + ExtensionField<FGenerator>,
	Challenger_: Challenger,
{
	let mut eval_claims_on_bit_columns: [_; EXPONENT_BIT_WIDTH] =
		array::from_fn(|_| LayerClaim::<F>::default());

	let mut eval_point = claim.eval_point.clone();
	let mut eval = claim.eval;
	for exponent_bit_number in (1..EXPONENT_BIT_WIDTH).rev() {
		let generator_power_constant =
			F::from(FGenerator::MULTIPLICATIVE_GENERATOR.pow([1 << exponent_bit_number]));

		let this_round_sumcheck_claim = SumcheckClaim::new(
			log_size,
			3,
			vec![CompositeSumClaim {
				composition: ExtraProduct {
					inner: MultiplyOrDont {
						generator_power_constant,
					},
				},
				sum: eval,
			}],
		)?;

		let sumcheck_verification_output = sumcheck::batch_verify(
			EvaluationOrder::LowToHigh,
			&[this_round_sumcheck_claim],
			transcript,
		)?;

		// Verify claims on transparent polynomials

		let sumcheck_query_point = sumcheck_verification_output.challenges;

		let eq_eval =
			EqIndPartialEval::new(log_size, sumcheck_query_point.clone())?.evaluate(&eval_point)?;

		if sumcheck_verification_output.multilinear_evals[0][2] != eq_eval {
			return Err(VerificationError::IncorrectEqIndEvaluation.into());
		}

		eval_claims_on_bit_columns[exponent_bit_number] = LayerClaim {
			eval_point: sumcheck_query_point.clone(),
			eval: sumcheck_verification_output.multilinear_evals[0][1],
		};

		eval_point = sumcheck_query_point;
		eval = sumcheck_verification_output.multilinear_evals[0][0];
	}

	eval_claims_on_bit_columns[0] = LayerClaim {
		eval_point,
		eval: first_layer_inverse::<FGenerator, _>(eval),
	};

	Ok(GeneratorExponentReductionOutput {
		eval_claims_on_exponent_bit_columns: eval_claims_on_bit_columns,
	})
}
