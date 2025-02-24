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
	common::{BaseExponentReductionOutput, ExponentiationClaim},
	compositions::ProverExponentiationComposition,
	error::Error,
	provers::{
		DynamicBaseExponentiationProver, ExponentiationProver, GeneratorExponentiationProver,
		ProverLayerClaimMeta,
	},
	witness::BaseExponentWitness,
};
use crate::{
	fiat_shamir::Challenger,
	protocols::{
		gkr_gpa::{gpa_sumcheck::prove::GPAProver, LayerClaim},
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
pub fn batch_prove<'a, FBase, F, P, FDomain, Challenger_, Backend>(
	witnesses: impl IntoIterator<Item = BaseExponentWitness<'a, P>>,
	claims: &[ExponentiationClaim<F>],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<BaseExponentReductionOutput<F>, Error>
where
	F: ExtensionField<FBase> + ExtensionField<FDomain> + TowerField,
	FDomain: Field,
	FBase: TowerField + ExtensionField<FDomain>,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>,
	Backend: ComputationBackend,
	Challenger_: Challenger,
{
	let witnesses = witnesses.into_iter().collect::<Vec<_>>();

	if witnesses.len() != claims.len() {
		bail!(Error::MismatchedWitnessClaimLength);
	}

	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	if witnesses.is_empty() {
		return Ok(BaseExponentReductionOutput {
			eval_claims_on_exponent_bit_columns,
		});
	}

	// Check that the witnesses are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let mut provers = make_provers::<_, FBase>(witnesses, claims)?;

	let max_exponent_bit_number = provers.first().map(|p| p.exponent_bit_width()).unwrap_or(0);

	for layer_no in 0..max_exponent_bit_number {
		let gkr_sumcheck_provers = build_layer_gkr_sumcheck_provers(
			&mut provers,
			layer_no,
			evaluation_domain_factory.clone(),
			backend,
		)?;

		let sumcheck_proof_output = sumcheck::batch_prove(gkr_sumcheck_provers, transcript)?;

		let layer_exponent_claims =
			build_layer_exponent_claims::<_>(&mut provers, sumcheck_proof_output, layer_no)?;

		eval_claims_on_exponent_bit_columns.push(layer_exponent_claims);

		provers.retain(|prover| !prover.is_last_layer(layer_no));
	}

	Ok(BaseExponentReductionOutput {
		eval_claims_on_exponent_bit_columns,
	})
}

type GKRProvers<'a, F, P, FDomain, Backend> = Vec<
	GPAProver<
		'a,
		FDomain,
		P,
		ProverExponentiationComposition<F>,
		MultilinearWitness<'a, P>,
		Backend,
	>,
>;

#[instrument(skip_all, level = "debug")]
fn build_layer_gkr_sumcheck_provers<'a, P, FDomain, Backend>(
	provers: &mut [Box<dyn ExponentiationProver<'a, P> + 'a>],
	layer_no: usize,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	backend: &'a Backend,
) -> Result<GKRProvers<'a, P::Scalar, P, FDomain, Backend>, Error>
where
	FDomain: Field,
	P: PackedFieldIndexable + PackedExtension<FDomain>,
	P::Scalar: ExtensionField<FDomain>,
	Backend: ComputationBackend,
{
	assert!(!provers.is_empty());

	let mut composite_claims = Vec::new();
	let mut multilinears = Vec::new();

	let first_eval_point = provers[0].layer_claim_eval_point().to_vec();
	let mut eval_points = vec![first_eval_point];

	let mut active_index = 0;

	for i in 0..provers.len() {
		if provers[i].layer_claim_eval_point() != eval_points[eval_points.len() - 1] {
			let CompositeSumClaimWithMultilinears {
				composite_claims: eval_point_composite_claims,
				multilinears: eval_point_multilinears,
			} = build_eval_point_claims::<P>(&mut provers[active_index..i], layer_no)?;

			if eval_point_composite_claims.is_empty() {
				// extract the last point because provers with this point will not participate in the sumcheck.
				eval_points.pop();
			} else {
				composite_claims.push(eval_point_composite_claims);
				multilinears.push(eval_point_multilinears);
			}

			eval_points.push(provers[i].layer_claim_eval_point().to_vec());
			active_index = i;
		}

		if i == provers.len() - 1 {
			let CompositeSumClaimWithMultilinears {
				composite_claims: eval_point_composite_claims,
				multilinears: eval_point_multilinears,
			} = build_eval_point_claims::<P>(&mut provers[active_index..], layer_no)?;

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

struct CompositeSumClaimWithMultilinears<'a, P: PackedField> {
	composite_claims: Vec<CompositeSumClaim<P::Scalar, ProverExponentiationComposition<P::Scalar>>>,
	multilinears: Vec<MultilinearWitness<'a, P>>,
}

fn build_eval_point_claims<'a, P>(
	provers: &mut [Box<dyn ExponentiationProver<'a, P> + 'a>],
	layer_no: usize,
) -> Result<CompositeSumClaimWithMultilinears<'a, P>, Error>
where
	P: PackedField,
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
			multilinears: this_layer_multilinears,
		} = prover.layer_composite_sum_claim(
			layer_no,
			composite_claims_n_multilinears,
			multilinears_index,
		)?;

		if let Some(composite_claim) = composite_claim {
			composite_claims.push(composite_claim);

			multilinears.extend(this_layer_multilinears);
		}
	}
	Ok(CompositeSumClaimWithMultilinears {
		composite_claims,
		multilinears,
	})
}

pub fn build_layer_exponent_claims<'a, P>(
	provers: &mut [Box<dyn ExponentiationProver<'a, P> + 'a>],
	mut sumcheck_output: BatchSumcheckOutput<P::Scalar>,
	layer_no: usize,
) -> Result<Vec<LayerClaim<P::Scalar>>, Error>
where
	P: PackedField,
{
	let mut eval_claims_on_exponent_bit_columns = Vec::new();

	// extract eq_ind_evals
	for multilinear_evals in &mut sumcheck_output.multilinear_evals {
		multilinear_evals.pop();
	}

	let mut multilinear_evals = sumcheck_output.multilinear_evals.into_iter().flatten();

	for prover in provers {
		let (this_prover_n_multilinears, _) = prover.layer_n_multilinears_n_claims(layer_no);

		let this_prover_multilinear_evals = multilinear_evals
			.by_ref()
			.take(this_prover_n_multilinears)
			.collect::<Vec<_>>();

		let exponent_bit_claim = prover.finish_layer(
			layer_no,
			&this_prover_multilinear_evals,
			&sumcheck_output.challenges,
		);

		eval_claims_on_exponent_bit_columns.push(exponent_bit_claim);
	}

	Ok(eval_claims_on_exponent_bit_columns)
}

fn make_provers<'a, P, FBase>(
	witnesses: Vec<BaseExponentWitness<'a, P>>,
	claims: &[ExponentiationClaim<P::Scalar>],
) -> Result<Vec<Box<dyn ExponentiationProver<'a, P> + 'a>>, Error>
where
	P: PackedField,
	FBase: BinaryField,
	P::Scalar: BinaryField + ExtensionField<FBase>,
{
	witnesses
		.into_iter()
		.zip(claims)
		.map(|(witness, claim)| {
			let is_dynamic_prover = witness.base.is_some();

			if is_dynamic_prover {
				DynamicBaseExponentiationProver::new(witness, claim).map(move |prover| {
					Box::new(prover) as Box<dyn ExponentiationProver<'a, P> + 'a>
				})
			} else {
				GeneratorExponentiationProver::<'a, P, FBase>::new(witness, claim).map(
					move |prover| Box::new(prover) as Box<dyn ExponentiationProver<'a, P> + 'a>,
				)
			}
		})
		.collect::<Result<Vec<_>, Error>>()
}
