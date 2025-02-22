// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::EvaluationDomainFactory;
use binius_utils::{bail, sorting::is_sorted_ascending};
use itertools::izip;
use tracing::instrument;

use super::{
	common::{GeneratorExponentClaim, GeneratorExponentReductionOutput},
	compositions::ExponentCompositions,
	provers::{DynamicProver, GeneratorProver, ProverLayerClaimMeta, StaticProver},
	witness::GeneratorExponentWitness,
};
use crate::{
	composition::ComplexIndexComposition,
	fiat_shamir::Challenger,
	protocols::{
		gkr_gpa::{gpa_sumcheck::prove::GPAProver, LayerClaim},
		gkr_int_mul::error::Error,
		sumcheck::{self, BatchSumcheckOutput, CompositeSumClaim},
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
pub fn batch_prove<'a, FGenerator, F, P, FDomain, Challenger_, Backend>(
	witnesses: impl IntoIterator<Item = GeneratorExponentWitness<'a, P>>,
	claims: &[GeneratorExponentClaim<F>],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<GeneratorExponentReductionOutput<F>, Error>
where
	F: ExtensionField<FGenerator> + ExtensionField<FDomain> + TowerField,
	FDomain: Field,
	FGenerator: TowerField + ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>,
	Backend: ComputationBackend,
	Challenger_: Challenger,
{
	let witness_vec = witnesses.into_iter().collect::<Vec<_>>();

	if witness_vec.len() != claims.len() {
		bail!(Error::MismatchedWitnessClaimLength);
	}

	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	if witness_vec.is_empty() {
		return Ok(GeneratorExponentReductionOutput {
			eval_claims_on_exponent_bit_columns,
		});
	}

	// Check that the witnesses are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let mut provers: Vec<Box<dyn GeneratorProver<'a, P>>> =
		make_provers::<_, FGenerator>(witness_vec, claims)?;

	let max_exponent_bit_number = provers.first().map(|p| p.exponent_bit_width()).unwrap_or(0);

	for layer_no in 0..max_exponent_bit_number {
		let gkr_sumcheck_provers = build_layer_gkr_sumcheck_provers::<_, FGenerator, _, _>(
			&mut provers,
			layer_no,
			evaluation_domain_factory.clone(),
			backend,
		)?;

		let sumcheck_proof_output = sumcheck::batch_prove(gkr_sumcheck_provers, transcript)?;

		let layer_exponent_claims = build_layer_exponent_claims::<_, FGenerator>(
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

type GKRProvers<'a, 'b, F, P, FDomain, Backend> = Vec<
	GPAProver<
		'b,
		FDomain,
		P,
		ComplexIndexComposition<ExponentCompositions<F>>,
		MultilinearWitness<'a, P>,
		Backend,
	>,
>;

#[instrument(skip_all, level = "debug")]
fn build_layer_gkr_sumcheck_provers<'a, 'b, P, FGenerator, FDomain, Backend>(
	provers: &mut [Box<dyn GeneratorProver<'a, P>>],
	layer_no: usize,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	backend: &'b Backend,
) -> Result<GKRProvers<'a, 'b, P::Scalar, P, FDomain, Backend>, Error>
where
	FDomain: Field,
	FGenerator: BinaryField,
	P: PackedFieldIndexable + PackedExtension<FDomain>,
	P::Scalar: BinaryField + ExtensionField<FDomain> + ExtensionField<FGenerator>,
	Backend: ComputationBackend,
{
	assert!(!provers.is_empty());

	let mut composite_claims = Vec::new();
	let mut multilinears = Vec::new();

	let first_eval_point = provers[0].eval_point().to_vec();
	let mut eval_points = vec![first_eval_point];

	let mut active_index = 0;

	for i in 0..provers.len() {
		if provers[i].eval_point() != eval_points[eval_points.len() - 1] {
			let (eval_point_composite_claims, eval_point_multilinears) =
				build_eval_point_claims::<P, FGenerator>(&mut provers[active_index..i], layer_no)?;

			if eval_point_composite_claims.is_empty() {
				eval_points.pop();
			} else {
				composite_claims.push(eval_point_composite_claims);
				multilinears.push(eval_point_multilinears);
			}
			eval_points.push(provers[i].eval_point().to_vec());
			active_index = i;
		}

		if i == provers.len() - 1 {
			let (eval_point_composite_claims, eval_point_multilinears) =
				build_eval_point_claims::<P, FGenerator>(&mut provers[active_index..], layer_no)?;

			if !eval_point_composite_claims.is_empty() {
				composite_claims.push(eval_point_composite_claims);
				multilinears.push(eval_point_multilinears);
			}
		}
	}

	izip!(composite_claims, multilinears, eval_points)
		.map(|(composite_claims, multilinears, eval_point)| {
			GPAProver::<'b, FDomain, P, _, _, Backend>::new(
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
fn build_eval_point_claims<'a, P, FGenerator>(
	provers: &mut [Box<dyn GeneratorProver<'a, P>>],
	layer_no: usize,
) -> Result<
	(
		Vec<CompositeSumClaim<P::Scalar, ComplexIndexComposition<ExponentCompositions<P::Scalar>>>>,
		Vec<MultilinearWitness<'a, P>>,
	),
	Error,
>
where
	P: PackedField,
	FGenerator: BinaryField,
	P::Scalar: BinaryField + ExtensionField<FGenerator>,
{
	let (composite_claims_n_multilinears, n_claims) =
		provers
			.iter()
			.fold((0, 0), |(n_multilinears, n_claims), prover| {
				let (layer_n_multilinears, layer_n_claims) =
					prover.layer_n_multilinears_n_claims(layer_no);

				(n_multilinears + layer_n_multilinears, n_claims + layer_n_claims)
			});

	let mut multilinears = Vec::with_capacity(composite_claims_n_multilinears);

	let mut composite_claims = Vec::with_capacity(n_claims);

	for prover in provers {
		let multilinears_index = multilinears.len();

		let ProverLayerClaimMeta {
			claim: composite_claim,
			multilinears: this_round_multilinears,
		} = prover.get_layer_composite_sum_claim(
			layer_no,
			composite_claims_n_multilinears,
			multilinears_index,
		)?;

		if let Some(composite_claim) = composite_claim {
			composite_claims.push(composite_claim);

			multilinears.extend(this_round_multilinears);
		}
	}
	Ok((composite_claims, multilinears))
}

pub fn build_layer_exponent_claims<'a, P, FGenerator>(
	provers: &mut [Box<dyn GeneratorProver<'a, P>>],
	mut sumcheck_output: BatchSumcheckOutput<P::Scalar>,
	layer_no: usize,
) -> Result<Vec<LayerClaim<P::Scalar>>, Error>
where
	P: PackedField,
	P::Scalar: BinaryField + ExtensionField<FGenerator>,
	FGenerator: BinaryField,
{
	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	sumcheck_output
		.multilinear_evals
		.iter_mut()
		.for_each(|multilinear_evals| {
			multilinear_evals.pop();
		});

	let mut multilinear_evals = sumcheck_output.multilinear_evals.into_iter().flatten();

	for prover in provers {
		let this_porver_multilinear_evals = multilinear_evals
			.by_ref()
			.take(prover.layer_n_multilinears_n_claims(layer_no).0)
			.collect::<Vec<_>>();

		let layer_claim = prover.finish_layer(
			layer_no,
			&this_porver_multilinear_evals,
			&sumcheck_output.challenges,
		);

		eval_claims_on_exponent_bit_columns.push(layer_claim);
	}

	Ok(eval_claims_on_exponent_bit_columns)
}

fn make_provers<'a, P, FGenerator>(
	witnesses: Vec<GeneratorExponentWitness<'a, P>>,
	claims: &[GeneratorExponentClaim<P::Scalar>],
) -> Result<Vec<Box<dyn GeneratorProver<'a, P> + 'a>>, Error>
where
	P: PackedField,
	FGenerator: BinaryField,
	P::Scalar: BinaryField + ExtensionField<FGenerator>,
{
	witnesses
		.into_iter()
		.zip(claims)
		.map(|(witness, claim)| {
			let is_dynamic_prover = witness.generator.is_some();

			if is_dynamic_prover {
				DynamicProver::new(witness, claim)
					.map(move |prover| Box::new(prover) as Box<dyn GeneratorProver<'a, P> + 'a>)
			} else {
				StaticProver::<'a, P, FGenerator>::new(witness, claim)
					.map(move |prover| Box::new(prover) as Box<dyn GeneratorProver<'a, P> + 'a>)
			}
		})
		.collect::<Result<Vec<_>, Error>>()
}
