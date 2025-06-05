// Copyright 2024-2025 Irreducible Inc.

use std::{iter, sync::Arc};

use binius_compute::{
	ComputeLayer, FSlice,
	alloc::{BumpAllocator, HostBumpAllocator},
	layer,
};
use binius_field::{Field, PackedField, PackedFieldIndexable};
use binius_math::{
	B1, B8, B16, B32, B64, B128, MultilinearPoly, MultilinearQuery, PackedTop, TowerTop,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use itertools::izip;
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
};

pub struct ReducedWitness<'a, F: Field, Hal: ComputeLayer<F>> {
	pub transparents: Vec<FSlice<'a, F, Hal>>,
	pub sumcheck_claims: Vec<PIOPSumcheckClaim<F>>,
}

pub fn prove<'a, 'alloc, F, P, M, Challenger_, Hal>(
	system: &EvalClaimSystem<F>,
	witnesses: &[M],
	transcript: &mut ProverTranscript<Challenger_>,
	memoized_data: MemoizedData<P>,
	hal: &'a Hal,
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'a HostBumpAllocator<'a, F>,
) -> Result<ReducedWitness<'a, F, Hal>, Error>
where
	F: TowerTop + PackedTop<Scalar = F>,
	P: PackedFieldIndexable<Scalar = F> + PackedTop,
	M: MultilinearPoly<P> + Sync,
	Challenger_: Challenger,
	Hal: ComputeLayer<F>,
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
	let tensor_elems = compute_partial_evals(system, witnesses, memoized_data)?;
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

	let ring_switch_eq_inds = make_ring_switch_eq_inds::<_, _>(
		&system.sumcheck_claim_descs,
		&system.suffix_descs,
		row_batch_coeffs,
		&mixing_coeffs,
		hal,
		dev_alloc,
		host_alloc,
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
fn compute_partial_evals<F, P, M>(
	system: &EvalClaimSystem<F>,
	witnesses: &[M],
	mut memoized_data: MemoizedData<P>,
) -> Result<Vec<TowerTensorAlgebra<F>>, Error>
where
	F: TowerTop,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Sync,
{
	let suffixes = system
		.suffix_descs
		.iter()
		.map(|desc| Arc::as_ref(&desc.suffix))
		.collect::<Vec<_>>();

	memoized_data.memoize_query_par(suffixes)?;

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

fn scale_tensor_elems<F>(
	tensor_elems: Vec<TowerTensorAlgebra<F>>,
	mixing_coeffs: &[F],
) -> Vec<TowerTensorAlgebra<F>>
where
	F: TowerTop,
{
	// Precondition
	assert!(tensor_elems.len() <= mixing_coeffs.len());

	tensor_elems
		.into_par_iter()
		.zip(mixing_coeffs)
		.map(|(tensor_elem, &mixing_coeff)| tensor_elem.scale_vertical(mixing_coeff))
		.collect()
}

fn mix_tensor_elems_for_prefixes<F>(
	scaled_tensor_elems: &[TowerTensorAlgebra<F>],
	prefix_descs: &[EvalClaimPrefixDesc<F>],
	eval_claim_to_prefix_desc_index: &[usize],
) -> Result<Vec<TowerTensorAlgebra<F>>, Error>
where
	F: TowerTop,
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
fn compute_row_batched_sumcheck_evals<F>(
	tensor_elems: Vec<TowerTensorAlgebra<F>>,
	row_batch_coeffs: &[F],
) -> Vec<F>
where
	F: TowerTop + PackedTop,
{
	tensor_elems
		.into_par_iter()
		.map(|tensor_elem| tensor_elem.fold_vertical(row_batch_coeffs))
		.collect()
}

fn make_ring_switch_eq_inds<'a, 'alloc, F, Hal>(
	sumcheck_claim_descs: &[PIOPSumcheckClaimDesc<F>],
	suffix_descs: &[EvalClaimSuffixDesc<F>],
	row_batch_coeffs: Arc<RowBatchCoeffs<F>>,
	mixing_coeffs: &[F],
	hal: &'a Hal,
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'a HostBumpAllocator<'a, F>,
) -> Result<Vec<FSlice<'a, F, Hal>>, Error>
where
	F: TowerTop,
	Hal: ComputeLayer<F>,
{
	let mut eq_inds = Vec::with_capacity(sumcheck_claim_descs.len());
	let _ = hal.execute(|exec| {
		let res = hal
			.map(
				exec,
				izip!(sumcheck_claim_descs, mixing_coeffs),
				|exec, (claim_desc, &mixing_coeff)| {
					let suffix_desc = &suffix_descs[claim_desc.suffix_desc_idx];

					make_ring_switch_eq_ind(
						suffix_desc,
						row_batch_coeffs.clone(),
						mixing_coeff,
						hal,
						exec,
						dev_alloc,
						host_alloc,
					)
					.map_err(|e| layer::Error::CoreLibError(Box::new(e)))
				},
			)
			.map_err(|e| layer::Error::CoreLibError(Box::new(e)))?;

		eq_inds.extend(res);
		Ok(vec![])
	})?;

	Ok(eq_inds)
}

fn make_ring_switch_eq_ind<'a, 'alloc, F, Hal>(
	suffix_desc: &EvalClaimSuffixDesc<F>,
	row_batch_coeffs: Arc<RowBatchCoeffs<F>>,
	mixing_coeff: F,
	hal: &'a Hal,
	exec: &mut Hal::Exec,
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'a HostBumpAllocator<'a, F>,
) -> Result<FSlice<'a, F, Hal>, Error>
where
	F: TowerTop,
	Hal: ComputeLayer<F>,
{
	let eq_ind = match F::TOWER_LEVEL - suffix_desc.kappa {
		0 => RingSwitchEqInd::<B1, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(hal, exec, dev_alloc, host_alloc, 0),
		3 => RingSwitchEqInd::<B8, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(hal, exec, dev_alloc, host_alloc, 3),
		4 => RingSwitchEqInd::<B16, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(hal, exec, dev_alloc, host_alloc, 4),
		5 => RingSwitchEqInd::<B32, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(hal, exec, dev_alloc, host_alloc, 5),
		6 => RingSwitchEqInd::<B64, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(hal, exec, dev_alloc, host_alloc, 6),
		7 => RingSwitchEqInd::<B128, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(hal, exec, dev_alloc, host_alloc, 7),
		_ => Err(Error::PackingDegreeNotSupported {
			kappa: suffix_desc.kappa,
		}),
	}?;
	Ok(eq_ind)
}
