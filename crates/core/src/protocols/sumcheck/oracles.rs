// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{Field, PackedField, TowerField};
use binius_math::EvaluationOrder;
use binius_utils::bail;

use super::{BatchSumcheckOutput, CompositeSumClaim, Error, SumcheckClaim, ZerocheckClaim};
use crate::{
	oracle::{Constraint, ConstraintPredicate, ConstraintSet, OracleId, TypeErasedComposition},
	polynomial::ArithCircuitPoly,
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

/// Create a sumcheck claim out of constraint set. Fails when the constraint set contains zerochecks.
/// Returns claim and metadata used for evalcheck claim construction.
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

/// Create a zerocheck claim from the constraint set. Fails when the constraint set contains regular sumchecks.
/// Returns claim and metadata used for evalcheck claim construction.
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
	let max_n_vars = metas.first().map_or(0, |meta| meta.n_vars);

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
				EvaluationOrder::LowToHigh => max_n_vars - meta.n_vars..max_n_vars,
				EvaluationOrder::HighToLow => 0..meta.n_vars,
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
