// Copyright 2024-2025 Irreducible Inc.

use std::array;

use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{EvaluationDomainFactory, MLEEmbeddingAdapter, MultilinearExtension};

use super::{
	common::GeneratorExponentReductionOutput, compositions::MultiplyOrDont,
	utils::first_layer_inverse, witness::GeneratorExponentWitness,
};
use crate::{
	fiat_shamir::CanSample,
	protocols::{
		gkr_gpa::{gpa_sumcheck::prove::GPAProver, Error, LayerClaim},
		sumcheck::{self, CompositeSumClaim},
	},
	transcript::CanWrite,
};

pub fn prove<
	FGenerator,
	F,
	PBits,
	PChallenge,
	PGenerator,
	FDomain,
	Transcript,
	Backend,
	const EXPONENT_BIT_WIDTH: usize,
>(
	witness: &GeneratorExponentWitness<'_, PBits, PGenerator, PChallenge, EXPONENT_BIT_WIDTH>,
	claim: &LayerClaim<F>, // this is a claim about the evaluation of the result layer at a random point
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	mut transcript: Transcript,
	backend: &Backend,
) -> Result<GeneratorExponentReductionOutput<F, EXPONENT_BIT_WIDTH>, Error>
where
	FDomain: Field,
	PBits: PackedField,
	PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	PGenerator::Scalar: ExtensionField<PBits::Scalar>,
	PGenerator: PackedFieldIndexable<Scalar = FGenerator>,
	PGenerator: PackedExtension<FDomain>,
	PGenerator::Scalar: ExtensionField<FDomain>,
	PChallenge: PackedField + PackedFieldIndexable<Scalar = F>,
	PChallenge:
		PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator> + PackedExtension<FDomain>,
	F: ExtensionField<PGenerator::Scalar> + ExtensionField<FDomain> + BinaryField + TowerField,
	FGenerator: Field + TowerField,
	Backend: ComputationBackend,
	Transcript: CanSample<F> + CanWrite,
{
	let mut eval_claims_on_bit_columns: [_; EXPONENT_BIT_WIDTH] =
		array::from_fn(|_| LayerClaim::<F>::default());

	let mut eval_point = claim.eval_point.clone();
	let mut eval = claim.eval;
	for exponent_bit_number in (1..EXPONENT_BIT_WIDTH).rev() {
		let this_round_exponent_bit = witness.exponent[exponent_bit_number].clone();
		let this_round_generator_power_constant =
			F::from(FGenerator::MULTIPLICATIVE_GENERATOR.pow([1 << exponent_bit_number]));

		let this_round_input_data =
			witness.single_bit_output_layers_data[exponent_bit_number - 1].clone();

		let this_round_input = MLEEmbeddingAdapter::<PGenerator, PChallenge>::from(
			MultilinearExtension::from_values(this_round_input_data)?,
		)
		.upcast_arc_dyn();

		let this_round_multilinears = [this_round_input, this_round_exponent_bit];

		let this_round_composite_claim = CompositeSumClaim {
			sum: eval,
			composition: MultiplyOrDont {
				generator_power_constant: this_round_generator_power_constant,
			},
		};

		let this_round_prover = GPAProver::<FDomain, PChallenge, _, _, Backend>::new(
			this_round_multilinears.to_vec(),
			None,
			[this_round_composite_claim],
			evaluation_domain_factory.clone(),
			&eval_point,
			backend,
		)?;

		let sumcheck_proof_output =
			sumcheck::batch_prove(vec![this_round_prover], &mut transcript)?;

		eval_point = sumcheck_proof_output.challenges.clone();
		eval = sumcheck_proof_output.multilinear_evals[0][0];

		eval_claims_on_bit_columns[exponent_bit_number] = LayerClaim::<F> {
			eval_point: sumcheck_proof_output.challenges,
			eval: sumcheck_proof_output.multilinear_evals[0][1],
		}
	}

	eval_claims_on_bit_columns[0] = LayerClaim::<F> {
		eval_point,
		eval: first_layer_inverse::<FGenerator, _>(eval),
	};

	Ok(GeneratorExponentReductionOutput {
		eval_claims_on_exponent_bit_columns: eval_claims_on_bit_columns,
	})
}
