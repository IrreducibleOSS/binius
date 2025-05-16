// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_fast_compute::arith_circuit::ArithCircuitPoly;
use binius_field::{Field, PackedField, TowerField};
use binius_math::EvaluationOrder;
use binius_utils::{bail, sorting::is_sorted_ascending};

use super::{
	BatchSumcheckOutput, BatchZerocheckOutput, CompositeSumClaim, EqIndSumcheckClaim, Error,
	SumcheckClaim, ZerocheckClaim,
};
use crate::{
	oracle::{Constraint, ConstraintPredicate, ConstraintSet, OracleId, TypeErasedComposition},
	protocols::evalcheck::EvalcheckMultilinearClaim,
};

#[derive(Debug)]
pub enum ConcreteClaim<P: PackedField> {
	Sumcheck(SumcheckClaim<P::Scalar, TypeErasedComposition<P>>),
	Zerocheck(ZerocheckClaim<P::Scalar, TypeErasedComposition<P>>),
}

pub struct OracleClaimMeta {
	pub n_vars: usize,
	pub oracle_ids: Vec<OracleId>,
}

/// Create a sumcheck claim out of constraint set. Fails when the constraint set contains
/// zerochecks. Returns claim and metadata used for evalcheck claim construction.
#[allow(clippy::type_complexity)]
pub fn constraint_set_sumcheck_claim<F: TowerField>(
	constraint_set: ConstraintSet<F>,
) -> Result<(SumcheckClaim<F, ArithCircuitPoly<F>>, OracleClaimMeta), Error> {
	let (constraints, meta) = split_constraint_set(constraint_set);
	let n_multilinears = meta.oracle_ids.len();

	let mut sums = Vec::new();
	for Constraint {
		composition,
		predicate,
		..
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: ArithCircuitPoly::with_n_vars(n_multilinears, composition)?,
				sum,
			}),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let claim = SumcheckClaim::new(meta.n_vars, n_multilinears, sums)?;
	Ok((claim, meta))
}

/// Create a zerocheck claim from the constraint set. Fails when the constraint set contains regular
/// sumchecks. Returns claim and metadata used for evalcheck claim construction.
#[allow(clippy::type_complexity)]
pub fn constraint_set_zerocheck_claim<F: TowerField>(
	constraint_set: ConstraintSet<F>,
) -> Result<(ZerocheckClaim<F, ArithCircuitPoly<F>>, OracleClaimMeta), Error> {
	let (constraints, meta) = split_constraint_set(constraint_set);
	let n_multilinears = meta.oracle_ids.len();

	let mut zeros = Vec::new();
	for Constraint {
		composition,
		predicate,
		..
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Zero => {
				zeros.push(ArithCircuitPoly::with_n_vars(n_multilinears, composition)?)
			}
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let claim = ZerocheckClaim::new(meta.n_vars, n_multilinears, zeros)?;
	Ok((claim, meta))
}

#[allow(clippy::type_complexity)]
pub fn constraint_set_mlecheck_claim<F: TowerField>(
	constraint_set: ConstraintSet<F>,
) -> Result<(EqIndSumcheckClaim<F, ArithCircuitPoly<F>>, OracleClaimMeta), Error> {
	let (constraints, meta) = split_constraint_set(constraint_set);
	let n_multilinears = meta.oracle_ids.len();

	let mut sums = Vec::new();
	for Constraint {
		composition,
		predicate,
		..
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: ArithCircuitPoly::with_n_vars(n_multilinears, composition)?,
				sum,
			}),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let claim = EqIndSumcheckClaim::new(meta.n_vars, n_multilinears, sums)?;
	Ok((claim, meta))
}

fn split_constraint_set<F: Field>(
	constraint_set: ConstraintSet<F>,
) -> (Vec<Constraint<F>>, OracleClaimMeta) {
	let ConstraintSet {
		oracle_ids,
		constraints,
		n_vars,
	} = constraint_set;
	let meta = OracleClaimMeta { n_vars, oracle_ids };
	(constraints, meta)
}

/// Constructs evalcheck claims from metadata returned by constraint set claim constructors.
pub fn make_eval_claims<F: TowerField>(
	evaluation_order: EvaluationOrder,
	metas: impl IntoIterator<Item = OracleClaimMeta>,
	batch_sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let metas = metas.into_iter().collect::<Vec<_>>();

	if !is_sorted_ascending(metas.iter().map(|meta| meta.n_vars)) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = metas.last().map_or(0, |meta| meta.n_vars);

	if metas.len() != batch_sumcheck_output.multilinear_evals.len() {
		bail!(Error::ClaimProofMismatch);
	}

	if max_n_vars != batch_sumcheck_output.challenges.len() {
		bail!(Error::ClaimProofMismatch);
	}

	let mut evalcheck_claims = Vec::new();
	for (meta, prover_evals) in iter::zip(metas, batch_sumcheck_output.multilinear_evals) {
		if meta.oracle_ids.len() != prover_evals.len() {
			bail!(Error::ClaimProofMismatch);
		}

		for (oracle_id, eval) in iter::zip(meta.oracle_ids, prover_evals) {
			let eval_points_range = match evaluation_order {
				EvaluationOrder::LowToHigh => 0..meta.n_vars,
				EvaluationOrder::HighToLow => max_n_vars - meta.n_vars..max_n_vars,
			};
			let eval_point = batch_sumcheck_output.challenges[eval_points_range].to_vec();

			let claim = EvalcheckMultilinearClaim {
				id: oracle_id,
				eval_point: eval_point.into(),
				eval,
			};

			evalcheck_claims.push(claim);
		}
	}

	Ok(evalcheck_claims)
}

/// Construct eval claims from the batched zerocheck output.
pub fn make_zerocheck_eval_claims<F: Field>(
	metas: impl IntoIterator<Item = OracleClaimMeta>,
	batch_zerocheck_output: BatchZerocheckOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let BatchZerocheckOutput {
		skipped_challenges,
		unskipped_challenges,
		concat_multilinear_evals,
	} = batch_zerocheck_output;

	let metas = metas.into_iter().collect::<Vec<_>>();

	if !is_sorted_ascending(metas.iter().map(|meta| meta.n_vars)) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = metas.last().map_or(0, |meta| meta.n_vars);
	let n_multilinears = metas
		.iter()
		.map(|meta| meta.oracle_ids.len())
		.sum::<usize>();

	if n_multilinears != concat_multilinear_evals.len() {
		bail!(Error::ClaimProofMismatch);
	}

	if max_n_vars != skipped_challenges.len() + unskipped_challenges.len() {
		bail!(Error::ClaimProofMismatch);
	}

	let ids_with_n_vars = metas.into_iter().flat_map(|meta| {
		meta.oracle_ids
			.into_iter()
			.map(move |oracle_id| (oracle_id, meta.n_vars))
	});

	let mut evalcheck_claims = Vec::new();
	for ((oracle_id, n_vars), eval) in iter::zip(ids_with_n_vars, concat_multilinear_evals) {
		// NB. Two stages of zerocheck reduction (univariate skip and front-loaded high-to-low
		// sumchecks)     may result in a "gap" between challenges prefix and suffix.
		let eval_point = [
			&skipped_challenges[..n_vars.min(skipped_challenges.len())],
			&unskipped_challenges[(max_n_vars - n_vars).min(unskipped_challenges.len())..],
		]
		.concat()
		.into();

		let claim = EvalcheckMultilinearClaim {
			id: oracle_id,
			eval_point,
			eval,
		};

		evalcheck_claims.push(claim);
	}

	Ok(evalcheck_claims)
}

pub struct SumcheckClaimsWithMeta<F: TowerField, C> {
	pub claims: Vec<SumcheckClaim<F, C>>,
	pub metas: Vec<OracleClaimMeta>,
}

/// Constructs sumcheck claims and metas from the vector of [`ConstraintSet`]
pub fn constraint_set_sumcheck_claims<F: TowerField>(
	constraint_sets: Vec<ConstraintSet<F>>,
) -> Result<SumcheckClaimsWithMeta<F, ArithCircuitPoly<F>>, Error> {
	let mut claims = Vec::with_capacity(constraint_sets.len());
	let mut metas = Vec::with_capacity(constraint_sets.len());

	for constraint_set in constraint_sets {
		let (claim, meta) = constraint_set_sumcheck_claim(constraint_set)?;
		metas.push(meta);
		claims.push(claim);
	}
	Ok(SumcheckClaimsWithMeta { claims, metas })
}

pub struct MLEcheckClaimsWithMeta<F: TowerField, C> {
	pub claims: Vec<EqIndSumcheckClaim<F, C>>,
	pub metas: Vec<OracleClaimMeta>,
}

/// Constructs sumcheck claims and metas from the vector of [`ConstraintSet`]
pub fn constraint_set_mlecheck_claims<F: TowerField>(
	constraint_sets: Vec<ConstraintSet<F>>,
) -> Result<MLEcheckClaimsWithMeta<F, ArithCircuitPoly<F>>, Error> {
	let mut claims = Vec::with_capacity(constraint_sets.len());
	let mut metas = Vec::with_capacity(constraint_sets.len());

	for constraint_set in constraint_sets {
		let (claim, meta) = constraint_set_mlecheck_claim(constraint_set)?;
		metas.push(meta);
		claims.push(claim);
	}
	Ok(MLEcheckClaimsWithMeta { claims, metas })
}
