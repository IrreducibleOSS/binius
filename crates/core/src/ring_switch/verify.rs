// Copyright 2024-2025 Irreducible Inc.

use std::{iter, sync::Arc};

use binius_field::{Field, TowerField};
use binius_math::{MultilinearExtension, MultilinearQuery};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bytes::Buf;
use itertools::izip;

use super::eq_ind::RowBatchCoeffs;
use crate::{
	fiat_shamir::{CanSample, Challenger},
	piop::PIOPSumcheckClaim,
	polynomial::MultivariatePoly,
	ring_switch::{
		eq_ind::RingSwitchEqInd, tower_tensor_algebra::TowerTensorAlgebra, Error,
		EvalClaimSuffixDesc, EvalClaimSystem, PIOPSumcheckClaimDesc, VerificationError,
	},
	tower::{PackedTop, TowerFamily},
	transcript::{TranscriptReader, VerifierTranscript},
};

type FExt<Tower> = <Tower as TowerFamily>::B128;

#[derive(Debug)]
pub struct ReducedClaim<'a, F: Field> {
	pub transparents: Vec<Box<dyn MultivariatePoly<F> + 'a>>,
	pub sumcheck_claims: Vec<PIOPSumcheckClaim<F>>,
}

pub fn verify<'a, F, Tower, Challenger_>(
	system: &'a EvalClaimSystem<F>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<ReducedClaim<'a, F>, Error>
where
	F: TowerField + PackedTop<Tower>,
	Tower: TowerFamily<B128 = F>,
	Challenger_: Challenger,
{
	// Sample enough randomness to batch tensor elements corresponding to claims that share an
	// evaluation point prefix.
	let n_mixing_challenges = log2_ceil_usize(system.sumcheck_claim_descs.len());
	let mixing_challenges = transcript.sample_vec(n_mixing_challenges);
	let mixing_coeffs = MultilinearQuery::expand(&mixing_challenges).into_expansion();

	// For each evaluation point prefix, receive one batched tensor algebra element and verify
	// that it is consistent with the evaluation claims.
	let tensor_elems =
		verify_receive_tensor_elems(system, &mixing_coeffs, &mut transcript.message())?;

	// Sample the row-batching randomness.
	let row_batch_challenges = transcript.sample_vec(system.max_claim_kappa());
	let row_batch_coeffs = Arc::new(RowBatchCoeffs::new(
		MultilinearQuery::<F, _>::expand(&row_batch_challenges).into_expansion(),
	));

	// For each original evaluation claim, receive the row-batched evaluation claim.
	let row_batched_evals = transcript
		.message()
		.read_scalar_slice(system.sumcheck_claim_descs.len())?;

	// Check that the row-batched evaluation claims sent by the prover are consistent with the
	// tensor algebra sum elements previously sent.
	let mixed_row_batched_evals = accumulate_evaluations_by_prefixes(
		row_batched_evals.iter().copied(),
		system.prefix_descs.len(),
		&system.eval_claim_to_prefix_desc_index,
	);
	for (expected, tensor_elem) in iter::zip(mixed_row_batched_evals, tensor_elems) {
		if tensor_elem.fold_vertical(row_batch_coeffs.coeffs()) != expected {
			return Err(VerificationError::IncorrectRowBatchedSum.into());
		}
	}

	// Create the reduced PIOP sumcheck claims.
	let ring_switch_eq_inds = make_ring_switch_eq_inds::<_, Tower>(
		&system.sumcheck_claim_descs,
		&system.suffix_descs,
		&row_batch_coeffs,
		&mixing_coeffs,
	)?;

	let one_point = vec![F::ONE; ring_switch_eq_inds[0].n_vars()];

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

	Ok(ReducedClaim {
		transparents: ring_switch_eq_inds,
		sumcheck_claims,
	})
}

fn verify_receive_tensor_elems<F, Tower, B>(
	system: &EvalClaimSystem<F>,
	mixing_coeffs: &[F],
	transcript: &mut TranscriptReader<B>,
) -> Result<Vec<TowerTensorAlgebra<Tower>>, Error>
where
	F: TowerField + PackedTop<Tower>,
	Tower: TowerFamily<B128 = F>,
	B: Buf,
{
	let expected_tensor_elem_evals = compute_mixed_evaluations(
		system
			.sumcheck_claim_descs
			.iter()
			.map(|desc| desc.eval_claim.eval),
		system.prefix_descs.len(),
		&system.eval_claim_to_prefix_desc_index,
		mixing_coeffs,
	);

	let mut tensor_elems = Vec::with_capacity(system.prefix_descs.len());
	for (desc, expected_eval) in iter::zip(&system.prefix_descs, expected_tensor_elem_evals) {
		let kappa = desc.kappa();
		let tensor_elem =
			TowerTensorAlgebra::new(kappa, transcript.read_scalar_slice(1 << kappa)?)?;

		let query = MultilinearQuery::<F>::expand(&desc.prefix);
		let tensor_elem_eval =
			MultilinearExtension::<F, _>::new(kappa, tensor_elem.vertical_elems())
				.expect("tensor_elem has length 1 << kappa")
				.evaluate(&query)
				.expect("query has kappa variables");
		if tensor_elem_eval != expected_eval {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		tensor_elems.push(tensor_elem);
	}

	Ok(tensor_elems)
}

fn compute_mixed_evaluations<F: TowerField>(
	evals: impl ExactSizeIterator<Item = F>,
	n_prefixes: usize,
	eval_claim_to_prefix_desc_index: &[usize],
	mixing_coeffs: &[F],
) -> Vec<F> {
	// Pre-condition
	debug_assert!(evals.len() <= mixing_coeffs.len());

	accumulate_evaluations_by_prefixes(
		iter::zip(evals, mixing_coeffs).map(|(eval, &mixing_coeff)| eval * mixing_coeff),
		n_prefixes,
		eval_claim_to_prefix_desc_index,
	)
}

fn accumulate_evaluations_by_prefixes<F: TowerField>(
	evals: impl ExactSizeIterator<Item = F>,
	n_prefixes: usize,
	eval_claim_to_prefix_desc_index: &[usize],
) -> Vec<F> {
	// Pre-condition
	debug_assert_eq!(evals.len(), eval_claim_to_prefix_desc_index.len());

	let mut batched_evals = vec![F::ZERO; n_prefixes];
	for (eval, &desc_index) in izip!(evals, eval_claim_to_prefix_desc_index) {
		batched_evals[desc_index] += eval;
	}
	batched_evals
}

fn make_ring_switch_eq_inds<F, Tower>(
	sumcheck_claim_descs: &[PIOPSumcheckClaimDesc<F>],
	suffix_descs: &[EvalClaimSuffixDesc<F>],
	row_batch_coeffs: &Arc<RowBatchCoeffs<F>>,
	mixing_coeffs: &[F],
) -> Result<Vec<Box<dyn MultivariatePoly<F>>>, Error>
where
	F: TowerField + PackedTop<Tower>,
	Tower: TowerFamily<B128 = F>,
{
	iter::zip(sumcheck_claim_descs, mixing_coeffs)
		.map(|(claim_desc, &mixing_coeff)| {
			let suffix_desc = &suffix_descs[claim_desc.suffix_desc_idx];
			make_ring_switch_eq_ind::<Tower>(suffix_desc, row_batch_coeffs.clone(), mixing_coeff)
		})
		.collect()
}

fn make_ring_switch_eq_ind<Tower>(
	suffix_desc: &EvalClaimSuffixDesc<FExt<Tower>>,
	row_batch_coeffs: Arc<RowBatchCoeffs<FExt<Tower>>>,
	mixing_coeff: FExt<Tower>,
) -> Result<Box<dyn MultivariatePoly<FExt<Tower>>>, Error>
where
	Tower: TowerFamily,
	FExt<Tower>: PackedTop<Tower>,
{
	let eq_ind = match suffix_desc.kappa {
		7 => Box::new(RingSwitchEqInd::<Tower::B1, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?) as Box<dyn MultivariatePoly<_>>,
		4 => Box::new(RingSwitchEqInd::<Tower::B8, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?) as Box<dyn MultivariatePoly<_>>,
		3 => Box::new(RingSwitchEqInd::<Tower::B16, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?) as Box<dyn MultivariatePoly<_>>,
		2 => Box::new(RingSwitchEqInd::<Tower::B32, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?) as Box<dyn MultivariatePoly<_>>,
		1 => Box::new(RingSwitchEqInd::<Tower::B64, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?) as Box<dyn MultivariatePoly<_>>,
		0 => Box::new(RingSwitchEqInd::<Tower::B128, _>::new(
			suffix_desc.suffix.clone(),
			row_batch_coeffs,
			mixing_coeff,
		)?) as Box<dyn MultivariatePoly<_>>,
		_ => {
			return Err(Error::PackingDegreeNotSupported {
				kappa: suffix_desc.kappa,
			})
		}
	};
	Ok(eq_ind)
}
