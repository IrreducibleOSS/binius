// Copyright 2024 Irreducible, Inc

use std::{iter, sync::Arc};

use binius_field::{packed::iter_packed_slice, PackedField, PackedFieldIndexable, TowerField};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{MLEDirectAdapter, MultilinearPoly, MultilinearQuery};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use rayon::prelude::*;

use super::{
	common::{EvalClaimPrefixDesc, EvalClaimSystem, PIOPSumcheckClaimDesc},
	error::Error,
	tower_tensor_algebra::TowerTensorAlgebra,
};
use crate::{
	fiat_shamir::CanSample,
	piop::PIOPSumcheckClaim,
	ring_switch::{common::EvalClaimSuffixDesc, eq_ind::RingSwitchEqInd},
	tower::{PackedTop, TowerFamily},
	transcript::{CanWrite, Proof},
	witness::MultilinearWitness,
};

type FExt<Tower> = <Tower as TowerFamily>::B128;

#[derive(Debug)]
pub struct ReducedWitness<P: PackedField> {
	pub transparents: Vec<MultilinearWitness<'static, P>>,
	pub sumcheck_claims: Vec<PIOPSumcheckClaim<P::Scalar>>,
}

#[tracing::instrument("ring_switch::prove", skip_all)]
pub fn prove<F, P, M, Tower, Transcript, Advice, Backend>(
	system: &EvalClaimSystem<F>,
	witnesses: &[M],
	proof: &mut Proof<Transcript, Advice>,
	backend: &Backend,
) -> Result<ReducedWitness<P>, Error>
where
	F: TowerField,
	P: PackedFieldIndexable<Scalar = F>,
	M: MultilinearPoly<P> + Sync,
	Tower: TowerFamily<B128 = F>,
	F: PackedTop<Tower>,
	Transcript: CanWrite + CanSample<F>,
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
	let mixing_challenges = proof.transcript.sample_vec(n_mixing_challenges);
	let mixing_coeffs = MultilinearQuery::expand(&mixing_challenges).into_expansion();

	// For each evaluation point prefix, send one batched partial evaluation.
	let tensor_elems = compute_partial_evals::<_, _, _, Tower, _>(system, witnesses, backend)?;
	let scaled_tensor_elems = scale_tensor_elems(tensor_elems, &mixing_coeffs);
	let mixed_tensor_elems = mix_tensor_elems_for_prefixes(
		&scaled_tensor_elems,
		&system.prefix_descs,
		&system.eval_claim_to_prefix_desc_index,
	)?;
	for (mixed_tensor_elem, prefix_desc) in iter::zip(mixed_tensor_elems, &system.prefix_descs) {
		debug_assert_eq!(mixed_tensor_elem.vertical_elems().len(), 1 << prefix_desc.kappa());
		proof
			.transcript
			.write_scalar_slice(mixed_tensor_elem.vertical_elems());
	}

	// Sample the row-batching randomness.
	let row_batch_challenges = proof.transcript.sample_vec(system.max_claim_kappa());
	let row_batch_coeffs =
		Arc::from(MultilinearQuery::<F, _>::expand(&row_batch_challenges).into_expansion());

	let row_batched_evals =
		compute_row_batched_sumcheck_evals(scaled_tensor_elems, &row_batch_coeffs);
	proof.transcript.write_scalar_slice(&row_batched_evals);

	// Create the reduced PIOP sumcheck witnesses.
	let ring_switch_eq_inds = make_ring_switch_eq_inds::<_, P, Tower, _>(
		&system.sumcheck_claim_descs,
		&system.suffix_descs,
		row_batch_coeffs,
		&mixing_coeffs,
		backend,
	)?;
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

fn compute_partial_evals<F, P, M, Tower, Backend>(
	system: &EvalClaimSystem<F>,
	witnesses: &[M],
	backend: &Backend,
) -> Result<Vec<TowerTensorAlgebra<Tower>>, Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Sync,
	Tower: TowerFamily<B128 = F>,
	Backend: ComputationBackend,
{
	let suffix_queries = system
		.suffix_descs
		.par_iter()
		.map(|desc| backend.multilinear_query(&desc.suffix))
		.collect::<Result<Vec<_>, _>>()?;

	let tensor_elems = system
		.sumcheck_claim_descs
		.par_iter()
		.map(
			|PIOPSumcheckClaimDesc {
			     committed_idx,
			     suffix_desc_idx,
			     eval_claim: _,
			 }| {
				let suffix = &system.suffix_descs[*suffix_desc_idx];
				let suffix_query = &suffix_queries[*suffix_desc_idx];
				let partial_eval =
					witnesses[*committed_idx].evaluate_partial_high(suffix_query.to_ref())?;
				TowerTensorAlgebra::new(
					suffix.kappa,
					iter_packed_slice(partial_eval.evals())
						.take(1 << suffix.kappa)
						.collect(),
				)
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

fn make_ring_switch_eq_inds<F, P, Tower, Backend>(
	sumcheck_claim_descs: &[PIOPSumcheckClaimDesc<F>],
	suffix_descs: &[EvalClaimSuffixDesc<F>],
	row_batch_coeffs: Arc<[F]>,
	mixing_coeffs: &[F],
	backend: &Backend,
) -> Result<Vec<MultilinearWitness<'static, P>>, Error>
where
	F: TowerField,
	P: PackedFieldIndexable<Scalar = F>,
	Tower: TowerFamily<B128 = F>,
	F: PackedTop<Tower>,
	Backend: ComputationBackend,
{
	sumcheck_claim_descs
		.par_iter()
		.zip(mixing_coeffs)
		.map(|(claim_desc, &mixing_coeff)| {
			let suffix_desc = &suffix_descs[claim_desc.suffix_desc_idx];
			make_ring_switch_eq_ind::<P, Tower, _>(
				suffix_desc,
				row_batch_coeffs.clone(),
				mixing_coeff,
				backend,
			)
		})
		.collect()
}

fn make_ring_switch_eq_ind<P, Tower, Backend>(
	suffix_desc: &EvalClaimSuffixDesc<FExt<Tower>>,
	row_batch_coeffs: Arc<[FExt<Tower>]>,
	mixing_coeff: FExt<Tower>,
	backend: &Backend,
) -> Result<MultilinearWitness<'static, P>, Error>
where
	P: PackedFieldIndexable<Scalar = FExt<Tower>>,
	Tower: TowerFamily,
	Backend: ComputationBackend,
{
	let eq_ind = match suffix_desc.kappa {
		7 => RingSwitchEqInd::<Tower::B1, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension::<P, _>(backend),
		4 => RingSwitchEqInd::<Tower::B8, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(backend),
		3 => RingSwitchEqInd::<Tower::B16, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(backend),
		2 => RingSwitchEqInd::<Tower::B32, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(backend),
		1 => RingSwitchEqInd::<Tower::B64, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(backend),
		0 => RingSwitchEqInd::<Tower::B128, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?
		.multilinear_extension(backend),
		_ => Err(Error::PackingDegreeNotSupported {
			kappa: suffix_desc.kappa,
		}),
	}?;
	Ok(MLEDirectAdapter::from(eq_ind).upcast_arc_dyn())
}
