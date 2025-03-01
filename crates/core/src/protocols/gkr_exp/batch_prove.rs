// Copyright 2025 Irreducible Inc.

use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{EvaluationDomainFactory, EvaluationOrder};
use binius_utils::{bail, sorting::is_sorted_ascending};
use itertools::izip;
use tracing::instrument;

use super::{
	common::{BaseExpReductionOutput, ExpClaim, GKRExpProver, LayerClaim},
	compositions::ProverExpComposition,
	error::Error,
	provers::{
		CompositeSumClaimWithMultilinears, DynamicBaseExpProver, ExpProver, GeneratorExpProver,
	},
	witness::BaseExpWitness,
};
use crate::{
	fiat_shamir::Challenger,
	protocols::sumcheck::{self, BatchSumcheckOutput, CompositeSumClaim},
	transcript::ProverTranscript,
	witness::MultilinearWitness,
};

/// Prove a batched GKR exponentiation protocol execution.
///
/// The protocol can be batched over multiple instances by grouping consecutive provers over
/// `eval_points` in internal `LayerClaims` into `GkrExpProvers`. To achieve this, we use
/// [`crate::composition::IndexComposition`]. Since exponents can have different bit sizes, resulting
/// in a varying number of layers, we group them starting from the first layer to maximize the
/// opportunity to share the same evaluation point.
///
/// # Requirements
/// - Witnesses and claims must be in the same order as in [`super::batch_verify`] during proof verification.
/// - Witnesses and claims must be sorted in descending order by n_vars.
/// - Witnesses and claims must be of the same length.
/// - The `i`th witness must correspond to the `i`th claim.
///
/// # Recommendations
/// - Witnesses and claims should be grouped by evaluation points from the claims.
pub fn batch_prove<'a, FBase, F, P, FDomain, Challenger_, Backend>(
	evaluation_order: EvaluationOrder,
	witnesses: impl IntoIterator<Item = BaseExpWitness<'a, P, FBase>>,
	claims: &[ExpClaim<F>],
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<BaseExpReductionOutput<F>, Error>
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

	let mut layers_claims = Vec::new();

	if witnesses.is_empty() {
		return Ok(BaseExpReductionOutput { layers_claims });
	}

	// Check that the witnesses are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let mut provers = make_provers::<_, FBase>(witnesses, claims)?;

	let max_exponent_bit_number = provers.first().map(|p| p.exponent_bit_width()).unwrap_or(0);

	for layer_no in 0..max_exponent_bit_number {
		let gkr_sumcheck_provers = build_layer_gkr_sumcheck_provers(
			evaluation_order,
			&mut provers,
			layer_no,
			evaluation_domain_factory.clone(),
			backend,
		)?;

		let sumcheck_proof_output = sumcheck::batch_prove(gkr_sumcheck_provers, transcript)?;

		let layer_exponent_claims = build_layer_exponent_bit_claims(
			evaluation_order,
			&mut provers,
			sumcheck_proof_output,
			layer_no,
		)?;

		layers_claims.push(layer_exponent_claims);

		provers.retain(|prover| !prover.is_last_layer(layer_no));
	}

	Ok(BaseExpReductionOutput { layers_claims })
}

type GKRExpProvers<'a, F, P, FDomain, Backend> =
	Vec<GKRExpProver<'a, FDomain, P, ProverExpComposition<F>, MultilinearWitness<'a, P>, Backend>>;

/// Groups consecutive provers by their `eval_point` and reduces them to sumcheck provers.
#[instrument(skip_all, level = "debug")]
fn build_layer_gkr_sumcheck_provers<'a, P, FDomain, Backend>(
	evaluation_order: EvaluationOrder,
	provers: &mut [Box<dyn ExpProver<'a, P> + 'a>],
	layer_no: usize,
	evaluation_domain_factory: impl EvaluationDomainFactory<FDomain>,
	backend: &'a Backend,
) -> Result<GKRExpProvers<'a, P::Scalar, P, FDomain, Backend>, Error>
where
	FDomain: Field,
	P: PackedFieldIndexable + PackedExtension<FDomain>,
	P::Scalar: TowerField + ExtensionField<FDomain>,
	Backend: ComputationBackend,
{
	assert!(!provers.is_empty());

	let mut composite_claims = Vec::new();
	let mut multilinears = Vec::new();

	let first_eval_point = provers[0].layer_claim_eval_point().to_vec();
	let mut eval_points = vec![first_eval_point];

	let mut active_index = 0;

	// group provers by evaluation points and build composite sum claims.
	for i in 0..provers.len() {
		if provers[i].layer_claim_eval_point() != eval_points[eval_points.len() - 1] {
			let CompositeSumClaimsWithMultilinears {
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
			let CompositeSumClaimsWithMultilinears {
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
			GKRExpProver::<'a, FDomain, P, _, _, Backend>::new(
				evaluation_order,
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

struct CompositeSumClaimsWithMultilinears<'a, P: PackedField> {
	composite_claims: Vec<CompositeSumClaim<P::Scalar, ProverExpComposition<P::Scalar>>>,
	multilinears: Vec<MultilinearWitness<'a, P>>,
}

/// Builds composite claims and multilinears for provers that share the same `eval_point` from their internal [LayerClaim]s.
fn build_eval_point_claims<'a, P>(
	provers: &mut [Box<dyn ExpProver<'a, P> + 'a>],
	layer_no: usize,
) -> Result<CompositeSumClaimsWithMultilinears<'a, P>, Error>
where
	P: PackedField,
{
	let (composite_claims_n_multilinears, n_claims) =
		provers
			.iter()
			.fold((0, 0), |(n_multilinears, n_claims), prover| {
				let layer_n_multilinears = prover.layer_n_multilinears(layer_no);
				let layer_n_claims = prover.layer_n_claims(layer_no);

				(n_multilinears + layer_n_multilinears, n_claims + layer_n_claims)
			});

	let mut multilinears = Vec::with_capacity(composite_claims_n_multilinears);

	let mut composite_claims = Vec::with_capacity(n_claims);

	for prover in provers {
		let multilinears_index = multilinears.len();

		let meta = prover.layer_composite_sum_claim(
			layer_no,
			composite_claims_n_multilinears,
			multilinears_index,
		)?;

		if let Some(meta) = meta {
			let CompositeSumClaimWithMultilinears {
				claim,
				multilinears: this_layer_multilinears,
			} = meta;

			composite_claims.push(claim);

			multilinears.extend(this_layer_multilinears);
		}
	}
	Ok(CompositeSumClaimsWithMultilinears {
		composite_claims,
		multilinears,
	})
}

/// Reduces the sumcheck output to [LayerClaim]s and updates the internal provers [LayerClaim]s for the next layer.
fn build_layer_exponent_bit_claims<'a, P>(
	evaluation_order: EvaluationOrder,
	provers: &mut [Box<dyn ExpProver<'a, P> + 'a>],
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
		let this_prover_n_multilinears = prover.layer_n_multilinears(layer_no);

		let this_prover_multilinear_evals = multilinear_evals
			.by_ref()
			.take(this_prover_n_multilinears)
			.collect::<Vec<_>>();

		let exponent_bit_claims = prover.finish_layer(
			evaluation_order,
			layer_no,
			&this_prover_multilinear_evals,
			&sumcheck_output.challenges,
		);

		eval_claims_on_exponent_bit_columns.extend(exponent_bit_claims);
	}

	Ok(eval_claims_on_exponent_bit_columns)
}

/// Creates a vector of boxed [ExpProver]s from the given witnesses and claims.
fn make_provers<'a, P, FBase>(
	witnesses: Vec<BaseExpWitness<'a, P, FBase>>,
	claims: &[ExpClaim<P::Scalar>],
) -> Result<Vec<Box<dyn ExpProver<'a, P> + 'a>>, Error>
where
	P: PackedField,
	FBase: BinaryField,
	P::Scalar: BinaryField + ExtensionField<FBase>,
{
	witnesses
		.into_iter()
		.zip(claims)
		.map(|(witness, claim)| {
			if witness.uses_dynamic_base() {
				DynamicBaseExpProver::new(witness, claim)
					.map(|prover| Box::new(prover) as Box<dyn ExpProver<'a, P> + 'a>)
			} else {
				GeneratorExpProver::<'a, P, FBase>::new(witness, claim)
					.map(|prover| Box::new(prover) as Box<dyn ExpProver<'a, P> + 'a>)
			}
		})
		.collect::<Result<Vec<_>, Error>>()
}
