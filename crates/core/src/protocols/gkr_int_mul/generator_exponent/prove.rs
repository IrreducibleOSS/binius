// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{
	EvaluationDomainFactory, MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_utils::sorting::{stable_sort, unsort};
use itertools::izip;
use tracing::instrument;

use super::{
	common::{GeneratorExponentClaim, GeneratorExponentReductionOutput},
	compositions::ExponentCompositions,
	utils::first_layer_inverse,
	witness::GeneratorExponentWitness,
};
use crate::{
	composition::{ComplexIndexComposition, IndexComposition},
	fiat_shamir::Challenger,
	protocols::{
		gkr_gpa::{gpa_sumcheck::prove::GPAProver, LayerClaim},
		gkr_int_mul::error::Error,
		sumcheck::{self, prove::SumcheckProver, BatchSumcheckOutput, CompositeSumClaim},
	},
	transcript::ProverTranscript,
	witness::MultilinearWitness,
};

/// REQUIRES:
/// * witnesses and claims are of the same length
/// * The ith witness corresponds to the ith claim
///
/// RECOMMENDATIONS:
/// * Witnesses and claims should be grouped by evaluation points from claims
pub fn batch_prove<'a, FGenerator, F, P, PGenerator, FDomain, Challenger_, Backend>(
	witnesses: impl IntoIterator<Item = GeneratorExponentWitness<'a, PGenerator, P>>,
	claims: &[GeneratorExponentClaim<F>],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<GeneratorExponentReductionOutput<F>, Error>
where
	F: ExtensionField<PGenerator::Scalar> + ExtensionField<FDomain> + TowerField,
	FDomain: Field,
	FGenerator: TowerField + ExtensionField<FDomain>,
	PGenerator: PackedField<Scalar = FGenerator> + PackedExtension<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>
		+ PackedExtension<FDomain>,
	Backend: ComputationBackend,
	Challenger_: Challenger,
{
	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	let mut provers = witnesses
		.into_iter()
		.zip(claims)
		.map(|(witness, claim)| GeneratorExponentProverState::new(witness, claim.clone()))
		.collect::<Result<Vec<_>, Error>>()?;

	let max_exponent_bit_number = provers.first().map(|p| p.exponent_bit_width()).unwrap_or(0);

	for layer_no in 0..max_exponent_bit_number {
		let gkr_sumcheck_provers = GeneratorExponentProverState::build_layer_gkr_sumcheck_provers(
			&mut provers,
			layer_no,
			evaluation_domain_factory.clone(),
			backend,
		)?;

		// Since the provers can have different n_vars with the same exponent_bit_number, we need to sort them before the sumcheck proof.
		let (original_indices, sorted_gkr_sumcheck_provers) =
			stable_sort(gkr_sumcheck_provers, |prover| prover.n_vars(), true);

		let mut sumcheck_proof_output =
			sumcheck::batch_prove(sorted_gkr_sumcheck_provers, transcript)?;

		// Unsort multilinear_evals to match the provers.
		sumcheck_proof_output.multilinear_evals =
			unsort(original_indices, sumcheck_proof_output.multilinear_evals);

		let layer_exponent_claims = GeneratorExponentProverState::build_layer_exponent_claims(
			&mut provers,
			sumcheck_proof_output,
			layer_no,
		)?;

		eval_claims_on_exponent_bit_columns.push(layer_exponent_claims);

		provers.retain(|prover| !prover.is_last_layer(layer_no));
	}

	Ok(GeneratorExponentReductionOutput {
		eval_claims_on_exponent_bit_columns,
	})
}

struct GeneratorExponentProverState<'a, PGenerator, P>
where
	P: PackedField,
	PGenerator: PackedField,
{
	witness: GeneratorExponentWitness<'a, PGenerator, P>,
	current_layer_claim: LayerClaim<P::Scalar>,
}

impl<'a, P, PGenerator> GeneratorExponentProverState<'a, PGenerator, P>
where
	P: PackedField + PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>,
	P::Scalar: ExtensionField<PGenerator::Scalar>,
	PGenerator: PackedField,
	PGenerator::Scalar: BinaryField,
{
	fn new(
		witness: GeneratorExponentWitness<'a, PGenerator, P>,
		claim: GeneratorExponentClaim<P::Scalar>,
	) -> Result<Self, Error> {
		Ok(Self {
			witness,
			current_layer_claim: claim.into(),
		})
	}

	pub fn exponent_bit_width(&self) -> usize {
		self.witness.exponent.len()
	}

	pub fn current_layer_exponent_bit(&self, layer_no: usize) -> MultilinearWitness<'a, P> {
		self.witness.exponent[self.current_layer_exponent_bit_no(layer_no)].clone()
	}

	fn current_layer_exponent_bit_no(&self, layer_no: usize) -> usize {
		if self.with_dynamic_generator() {
			layer_no
		} else {
			self.exponent_bit_width() - 1 - layer_no
		}
	}

	pub fn current_layer_single_bit_output_layers_data(
		&mut self,
		layer_no: usize,
	) -> Result<MultilinearWitness<'a, P>, Error> {
		let index = self.witness.single_bit_output_layers_data.len() - layer_no - 2;

		let single_bit_output_layers_data =
			std::mem::take(&mut self.witness.single_bit_output_layers_data[index]);

		Ok(MLEEmbeddingAdapter::<PGenerator, P>::from(MultilinearExtension::from_values(
			single_bit_output_layers_data,
		)?)
		.upcast_arc_dyn())
	}

	fn with_dynamic_generator(&self) -> bool {
		self.witness.generator.is_some()
	}

	fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}

	fn layer_n_multilinears(&self, layer_no: usize) -> usize {
		if self.with_dynamic_generator() && !self.is_last_layer(layer_no) {
			3
		} else if !self.with_dynamic_generator() && self.is_last_layer(layer_no) {
			0
		} else {
			2
		}
	}

	#[allow(clippy::type_complexity)]
	#[instrument(skip_all, level = "debug")]
	fn build_layer_gkr_sumcheck_provers<FDomain, Backend>(
		provers: &mut [Self],
		layer_no: usize,
		evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
		backend: &'a Backend,
	) -> Result<
		Vec<
			GPAProver<
				'a,
				FDomain,
				P,
				ComplexIndexComposition<ExponentCompositions<P::Scalar>>,
				impl MultilinearPoly<P> + Send + Sync + 'a,
				Backend,
			>,
		>,
		Error,
	>
	where
		FDomain: Field,
		P: PackedFieldIndexable + PackedExtension<FDomain>,
		P::Scalar: BinaryField + ExtensionField<FDomain>,
		Backend: ComputationBackend,
	{
		assert!(!provers.is_empty());

		let mut composite_claims = Vec::new();
		let mut multilinears = Vec::new();

		let first_eval_point = provers[0].current_layer_claim.eval_point.clone();
		let mut eval_points = vec![first_eval_point];

		let mut active_index = 0;

		for i in 0..provers.len() {
			if provers[i].current_layer_claim.eval_point != *eval_points[eval_points.len() - 1] {
				let (eval_point_composite_claims, eval_point_multilinears) =
					Self::build_eval_point_claims(&mut provers[active_index..i], layer_no)?;

				if eval_point_composite_claims.is_empty() {
					eval_points.pop();
				} else {
					composite_claims.push(eval_point_composite_claims);
					multilinears.push(eval_point_multilinears);
				}
				eval_points.push(provers[i].current_layer_claim.eval_point.clone());
				active_index = i;
			}

			if i == provers.len() - 1 {
				let (eval_point_composite_claims, eval_point_multilinears) =
					Self::build_eval_point_claims(&mut provers[active_index..=i], layer_no)?;

				if !eval_point_composite_claims.is_empty() {
					composite_claims.push(eval_point_composite_claims);
					multilinears.push(eval_point_multilinears);
				}
			}
		}
		izip!(composite_claims, multilinears, eval_points)
			.map(|(composite_claims, multilinears, eval_point)| {
				GPAProver::<'a, FDomain, P, _, _, Backend>::new(
					multilinears,
					None,
					composite_claims,
					evaluation_domain_factory.clone(),
					&eval_point,
					backend,
				)
			})
			.collect::<Result<Vec<_>, _>>()
			.map_err(Error::from)
	}

	#[allow(clippy::type_complexity)]
	fn build_eval_point_claims(
		provers: &mut [Self],
		layer_no: usize,
	) -> Result<
		(
			Vec<
				CompositeSumClaim<
					P::Scalar,
					ComplexIndexComposition<ExponentCompositions<P::Scalar>>,
				>,
			>,
			Vec<MultilinearWitness<'a, P>>,
		),
		Error,
	> {
		let (composite_claims_n_multilinears, n_claims) =
			provers
				.iter()
				.fold((0, 0), |(mut n_multilinears, mut n_claims), prover| {
					n_multilinears += prover.layer_n_multilinears(layer_no);
					if prover.with_dynamic_generator() || !prover.is_last_layer(layer_no) {
						n_claims += 1;
					}
					(n_multilinears, n_claims)
				});

		let mut multilinears = Vec::with_capacity(composite_claims_n_multilinears);

		let mut composite_claims = Vec::with_capacity(n_claims);

		for prover in provers {
			let ml_index = multilinears.len();

			let exponent_bit = prover.current_layer_exponent_bit(layer_no);

			let (composition, this_round_multilinears) =
				match (prover.witness.generator.clone(), prover.is_last_layer(layer_no)) {
					// Last internal layer with dynamic generator
					(Some(generator), true) => {
						let this_round_multilinears = [generator, exponent_bit].to_vec();

						let composition = IndexComposition::new(
							composite_claims_n_multilinears,
							[ml_index, ml_index + 1],
							ExponentCompositions::DynamicGeneratorLastLayer,
						)?;
						let composition = ComplexIndexComposition::Bivariate(composition);
						(composition, this_round_multilinears)
					}
					// Non-last internal layer with a dynamic generator.
					(Some(generator), false) => {
						let this_round_input =
							prover.current_layer_single_bit_output_layers_data(layer_no)?;

						let this_round_multilinears =
							[this_round_input.clone(), exponent_bit.clone(), generator].to_vec();
						let composition = IndexComposition::new(
							composite_claims_n_multilinears,
							[ml_index, ml_index + 1, ml_index + 2],
							ExponentCompositions::DynamicGenerator,
						)?;
						let composition = ComplexIndexComposition::Trivariate(composition);
						(composition, this_round_multilinears)
					}
					// Non-last internal layer with static generator
					(None, false) => {
						let this_round_input =
							prover.current_layer_single_bit_output_layers_data(layer_no)?;

						let this_round_multilinears =
							[this_round_input.clone(), exponent_bit.clone()].to_vec();

						let generator_power_constant = P::Scalar::from(
							PGenerator::Scalar::MULTIPLICATIVE_GENERATOR
								.pow(1 << prover.current_layer_exponent_bit_no(layer_no)),
						);
						let composition = IndexComposition::new(
							composite_claims_n_multilinears,
							[ml_index, ml_index + 1],
							ExponentCompositions::StaticGenerator {
								generator_power_constant,
							},
						)?;
						let composition = ComplexIndexComposition::Bivariate(composition);

						(composition, this_round_multilinears)
					}
					// Last internal layer with static generator
					(None, true) => continue,
				};

			let this_round_composite_claim = CompositeSumClaim {
				sum: prover.current_layer_claim.eval,
				composition,
			};

			composite_claims.push(this_round_composite_claim);

			multilinears.extend(this_round_multilinears);
		}
		Ok((composite_claims, multilinears))
	}

	pub fn build_layer_exponent_claims(
		provers: &mut [Self],
		sumcheck_output: BatchSumcheckOutput<P::Scalar>,
		layer_no: usize,
	) -> Result<Vec<LayerClaim<P::Scalar>>, Error>
	where
		P::Scalar: BinaryField,
	{
		let mut eval_claims_on_exponent_bit_columns = Vec::new();

		let mut multilinears_index = 0;
		let mut multilinear_evals_index = 0;

		let mut current_eval_point = provers[0].current_layer_claim.eval_point.clone();

		for prover in provers {
			if prover.is_last_layer(layer_no) && !prover.with_dynamic_generator() {
				let LayerClaim { eval_point, eval } = prover.current_layer_claim.clone();

				let exponent_claim = LayerClaim::<P::Scalar> {
					eval_point,
					eval: first_layer_inverse::<PGenerator::Scalar, _>(eval),
				};
				eval_claims_on_exponent_bit_columns.push(exponent_claim);

				continue;
			}

			if prover.current_layer_claim.eval_point != current_eval_point {
				current_eval_point = prover.current_layer_claim.eval_point.clone();
				if multilinears_index != 0 {
					multilinears_index = 0;
					multilinear_evals_index += 1;
				}
			}

			let multilinear_evals = &sumcheck_output.multilinear_evals[multilinear_evals_index];

			let eval_point = &sumcheck_output.challenges
				[sumcheck_output.challenges.len() - current_eval_point.len()..];

			if !prover.is_last_layer(layer_no) {
				prover.current_layer_claim = LayerClaim {
					eval: multilinear_evals[multilinears_index],
					eval_point: eval_point.to_vec(),
				};
			}

			let exponent_claim = LayerClaim {
				eval: multilinear_evals[multilinears_index + 1],
				eval_point: eval_point.to_vec(),
			};

			eval_claims_on_exponent_bit_columns.push(exponent_claim);

			multilinears_index += prover.layer_n_multilinears(layer_no);
		}
		Ok(eval_claims_on_exponent_bit_columns)
	}
}
