// Copyright 2024-2025 Irreducible Inc.

use std::{iter, sync::Arc};

use binius_field::{
	PackedField, PackedFieldIndexable, TowerField,
	tower::{PackedTop, TowerFamily},
};
use binius_hal::ComputationBackend;
use binius_math::{MLEDirectAdapter, MultilinearPoly, MultilinearQuery};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use tracing::instrument;

use super::{
	common::{EvalClaimPrefixDesc, EvalClaimSystem, PIOPSumcheckClaimDesc},
	eq_ind::RowBatchCoeffs,
	error::Error,
	logging::MLEFoldHisgDimensionsData,
	tower_tensor_algebra::TowerTensorAlgebra,
};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	piop::PIOPSumcheckClaim,
	protocols::evalcheck::subclaims::MemoizedData,
	ring_switch::{
		common::EvalClaimSuffixDesc, eq_ind::RingSwitchEqInd, logging::CalculateRingSwitchEqIndData,
	},
	transcript::ProverTranscript,
	witness::MultilinearWitness,
};

type FExt<Tower> = <Tower as TowerFamily>::B128;

#[derive(Debug)]
pub struct ReducedWitness<P: PackedField> {
	pub transparents: Vec<MultilinearWitness<'static, P>>,
	pub sumcheck_claims: Vec<PIOPSumcheckClaim<P::Scalar>>,
}

pub fn prove<F, P, M, Tower, Challenger_, Backend>(
	system: &EvalClaimSystem<F>,
	witnesses: &[M],
	transcript: &mut ProverTranscript<Challenger_>,
	memoized_data: MemoizedData<P, Backend>,
	backend: &Backend,
) -> Result<ReducedWitness<P>, Error>
where
	F: TowerField + PackedTop<Tower>,
	P: PackedFieldIndexable<Scalar = F>,
	M: MultilinearPoly<P> + Sync,
	Tower: TowerFamily<B128 = F>,
	Challenger_: Challenger,
	Backend: ComputationBackend,
{
	if witnesses.len() != system.commit_meta.total_multilins() {
		return Err(Error::InvalidWitness(
			"witness length does not match the number of multilinears".into(),
		));
	}

	// Sample enough randomness to batch tensor elements corresponding to claims that share an
	// evaluation point prefix.
	let n_mixing_challenges = log2_ceil_usize(system.sumcheck_claim_descs.len());
	let mixing_challenges = transcript.sample_vec(n_mixing_challenges);
	let dimensions_data = MLEFoldHisgDimensionsData::new(witnesses);
	let mle_fold_high_span = tracing::debug_span!(
		"[task] (Ring Switch) MLE Fold High",
		phase = "ring_switch",
		perfetto_category = "task.main",
		?dimensions_data,
	)
	.entered();

	let mixing_coeffs = MultilinearQuery::expand(&mixing_challenges).into_expansion();

	// For each evaluation point prefix, send one batched partial evaluation.
	let tensor_elems =
		compute_partial_evals::<_, _, _, Tower, _>(system, witnesses, memoized_data, backend)?;
	let scaled_tensor_elems = scale_tensor_elems(tensor_elems, &mixing_coeffs);
	let mixed_tensor_elems = mix_tensor_elems_for_prefixes(
		&scaled_tensor_elems,
		&system.prefix_descs,
		&system.eval_claim_to_prefix_desc_index,
	)?;
	drop(mle_fold_high_span);
	let mut writer = transcript.message();
	for (mixed_tensor_elem, prefix_desc) in iter::zip(mixed_tensor_elems, &system.prefix_descs) {
		debug_assert_eq!(mixed_tensor_elem.vertical_elems().len(), 1 << prefix_desc.kappa());
		writer.write_scalar_slice(mixed_tensor_elem.vertical_elems());
	}

	// Sample the row-batching randomness.
	let row_batch_challenges = transcript.sample_vec(system.max_claim_kappa());
	let row_batch_coeffs = Arc::new(RowBatchCoeffs::new(
		MultilinearQuery::<F, _>::expand(&row_batch_challenges).into_expansion(),
		row_batch_challenges,
	));

	let row_batched_evals =
		compute_row_batched_sumcheck_evals(scaled_tensor_elems, row_batch_coeffs.coeffs());
	transcript.message().write_scalar_slice(&row_batched_evals);

	// Create the reduced PIOP sumcheck witnesses.
	let dimensions_data = CalculateRingSwitchEqIndData::new(system.suffix_descs.iter());
	let calculate_ring_switch_eq_ind_span = tracing::debug_span!(
		"[task] Calculate Ring Switch Eq Ind",
		phase = "ring_switch",
		perfetto_category = "task.main",
		?dimensions_data,
	)
	.entered();

	let ring_switch_eq_inds = make_ring_switch_eq_inds::<_, P, Tower>(
		&system.sumcheck_claim_descs,
		&system.suffix_descs,
		row_batch_coeffs,
		&mixing_coeffs,
	)?;
	drop(calculate_ring_switch_eq_ind_span);

	let sumcheck_claims = iter::zip(&system.sumcheck_claim_descs, row_batched_evals)
		.enumerate()
		.map(|(idx, (claim_desc, eval))| {
			let suffix_desc = &system.suffix_descs[claim_desc.suffix_desc_idx];
			PIOPSumcheckClaim {
				n_vars: suffix_desc.suffix.len(),
				committed: claim_desc.committed_idx,
				transparent: idx,
				sum: eval,
			}
		})
		.collect::<Vec<_>>();

	Ok(ReducedWitness {
		transparents: ring_switch_eq_inds,
		sumcheck_claims,
	})
}

#[instrument(skip_all)]
fn compute_partial_evals<F, P, M, Tower, Backend>(
	system: &EvalClaimSystem<F>,
	witnesses: &[M],
	mut memoized_data: MemoizedData<P, Backend>,
	backend: &Backend,
) -> Result<Vec<TowerTensorAlgebra<Tower>>, Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Sync,
	Tower: TowerFamily<B128 = F>,
	Backend: ComputationBackend,
{
	let suffixes = system
		.suffix_descs
		.iter()
		.map(|desc| Arc::as_ref(&desc.suffix))
		.collect::<Vec<_>>();

	memoized_data.memoize_query_par(suffixes, backend)?;

	let tensor_elems = system
		.sumcheck_claim_descs
		.par_iter()
		.map(
			|PIOPSumcheckClaimDesc {
			     committed_idx,
			     suffix_desc_idx,
			     eval_claim,
			 }| {
				let suffix_desc = &system.suffix_descs[*suffix_desc_idx];

				let mut elems = if let Some(partial_eval) =
					memoized_data.partial_eval(eval_claim.id, Arc::as_ref(&suffix_desc.suffix))
				{
					PackedField::iter_slice(
						partial_eval.packed_evals().expect("packed_evals exist"),
					)
					.take((1 << suffix_desc.kappa).min(1 << partial_eval.n_vars()))
					.collect::<Vec<_>>()
				} else {
					let suffix_query = memoized_data
						.full_query_readonly(&suffix_desc.suffix)
						.expect("memoized above");
					let partial_eval =
						witnesses[*committed_idx].evaluate_partial_high(suffix_query.into())?;
					PackedField::iter_slice(partial_eval.evals())
						.take((1 << suffix_desc.kappa).min(1 << partial_eval.n_vars()))
						.collect::<Vec<_>>()
				};

				if elems.len() < (1 << suffix_desc.kappa) {
					elems = elems
						.into_iter()
						.cycle()
						.take(1 << suffix_desc.kappa)
						.collect();
				}
				TowerTensorAlgebra::new(suffix_desc.kappa, elems)
			},
		)
		.collect::<Result<Vec<_>, _>>()?;

	Ok(tensor_elems)
}

fn scale_tensor_elems<F, Tower>(
	tensor_elems: Vec<TowerTensorAlgebra<Tower>>,
	mixing_coeffs: &[F],
) -> Vec<TowerTensorAlgebra<Tower>>
where
	F: TowerField,
	Tower: TowerFamily<B128 = F>,
{
	// Precondition
	assert!(tensor_elems.len() <= mixing_coeffs.len());

	tensor_elems
		.into_par_iter()
		.zip(mixing_coeffs)
		.map(|(tensor_elem, &mixing_coeff)| tensor_elem.scale_vertical(mixing_coeff))
		.collect()
}

fn mix_tensor_elems_for_prefixes<F, Tower>(
	scaled_tensor_elems: &[TowerTensorAlgebra<Tower>],
	prefix_descs: &[EvalClaimPrefixDesc<F>],
	eval_claim_to_prefix_desc_index: &[usize],
) -> Result<Vec<TowerTensorAlgebra<Tower>>, Error>
where
	F: TowerField,
	Tower: TowerFamily<B128 = F>,
{
	// Precondition
	assert_eq!(scaled_tensor_elems.len(), eval_claim_to_prefix_desc_index.len());

	let mut batched_tensor_elems = prefix_descs
		.iter()
		.map(|desc| TowerTensorAlgebra::zero(desc.kappa()))
		.collect::<Result<Vec<_>, _>>()?;
	for (tensor_elem, &desc_index) in
		iter::zip(scaled_tensor_elems, eval_claim_to_prefix_desc_index)
	{
		let mixed_val = &mut batched_tensor_elems[desc_index];
		debug_assert_eq!(mixed_val.kappa(), tensor_elem.kappa());
		mixed_val.add_assign(tensor_elem)?;
	}
	Ok(batched_tensor_elems)
}

#[instrument(skip_all)]
fn compute_row_batched_sumcheck_evals<F, Tower>(
	tensor_elems: Vec<TowerTensorAlgebra<Tower>>,
	row_batch_coeffs: &[F],
) -> Vec<F>
where
	F: TowerField,
	Tower: TowerFamily<B128 = F>,
	F: PackedTop<Tower>,
{
	tensor_elems
		.into_par_iter()
		.map(|tensor_elem| tensor_elem.fold_vertical(row_batch_coeffs))
		.collect()
}

fn make_ring_switch_eq_inds<F, P, Tower>(
	sumcheck_claim_descs: &[PIOPSumcheckClaimDesc<F>],
	suffix_descs: &[EvalClaimSuffixDesc<F>],
	row_batch_coeffs: Arc<RowBatchCoeffs<F>>,
	mixing_coeffs: &[F],
) -> Result<Vec<MultilinearWitness<'static, P>>, Error>
where
	F: TowerField + PackedTop<Tower>,
	P: PackedFieldIndexable<Scalar = F>,
	Tower: TowerFamily<B128 = F>,
{
	sumcheck_claim_descs
		.par_iter()
		.zip(mixing_coeffs)
		.map(|(claim_desc, &mixing_coeff)| {
			let suffix_desc = &suffix_descs[claim_desc.suffix_desc_idx];
			make_ring_switch_eq_ind::<P, Tower>(suffix_desc, row_batch_coeffs.clone(), mixing_coeff)
		})
		.collect()
}

fn make_ring_switch_eq_ind<P, Tower>(
	suffix_desc: &EvalClaimSuffixDesc<FExt<Tower>>,
	row_batch_coeffs: Arc<RowBatchCoeffs<FExt<Tower>>>,
	mixing_coeff: FExt<Tower>,
) -> Result<MultilinearWitness<'static, P>, Error>
where
	P: PackedFieldIndexable<Scalar = FExt<Tower>>,
	Tower: TowerFamily,
{
	let eq_ind = match suffix_desc.kappa {
		7 => RingSwitchEqInd::<Tower::B1, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension::<P>(),
		4 => RingSwitchEqInd::<Tower::B8, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(),
		3 => RingSwitchEqInd::<Tower::B16, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(),
		2 => RingSwitchEqInd::<Tower::B32, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(),
		1 => RingSwitchEqInd::<Tower::B64, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(),
		0 => RingSwitchEqInd::<Tower::B128, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(),
		_ => Err(Error::PackingDegreeNotSupported {
			kappa: suffix_desc.kappa,
		}),
	}?;
	Ok(MLEDirectAdapter::from(eq_ind).upcast_arc_dyn())
}
