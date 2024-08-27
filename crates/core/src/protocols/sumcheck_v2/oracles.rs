// Copyright 2024 Ulvetanna Inc.

use super::{BatchSumcheckOutput, CompositeSumClaim, Error, SumcheckClaim, ZerocheckClaim};
use crate::{
	oracle::{
		Constraint, ConstraintPredicate, ConstraintSet, Error as OracleError, MultilinearOracleSet,
		OracleId, TypeErasedComposition,
	},
	protocols::evalcheck::EvalcheckMultilinearClaim,
};

use binius_field::{PackedField, TowerField};
use binius_math::polynomial::{CompositionPoly, CompositionScalarAdapter};
use binius_utils::bail;
use std::iter;

#[derive(Debug)]
pub enum ConcreteClaim<P: PackedField> {
	Sumcheck(SumcheckClaim<P::Scalar, TypeErasedComposition<P>>),
	Zerocheck(ZerocheckClaim<P::Scalar, TypeErasedComposition<P>>),
}

pub struct OracleClaimMeta {
	n_vars: usize,
	oracle_ids: Vec<OracleId>,
}

/// Create a sumcheck claim out of constraint set. Fails when the constraint set contains zerochecks.
/// Returns claim and metadata used for evalcheck claim construction.
#[allow(clippy::type_complexity)]
pub fn constraint_set_sumcheck_claim<F: TowerField, P: PackedField>(
	constraint_set: ConstraintSet<P>,
	oracles: &MultilinearOracleSet<F>,
) -> Result<(SumcheckClaim<P::Scalar, impl CompositionPoly<P::Scalar>>, OracleClaimMeta), Error> {
	let (constraints, meta) = split_constraint_set(constraint_set, oracles)?;

	let mut sums = Vec::new();
	for Constraint {
		composition,
		predicate,
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Sum(sum) => sums.push(CompositeSumClaim {
				composition: CompositionScalarAdapter::new(composition),
				sum,
			}),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let claim = SumcheckClaim::new(meta.n_vars, meta.oracle_ids.len(), sums)?;
	Ok((claim, meta))
}

/// Create a zerocheck claim from the constraint set. Fails when the constraint set contains regular sumchecks.
/// Returns claim and metadata used for evalcheck claim construction.
#[allow(clippy::type_complexity)]
pub fn constraint_set_zerocheck_claim<F: TowerField, P: PackedField>(
	constraint_set: ConstraintSet<P>,
	oracles: &MultilinearOracleSet<F>,
) -> Result<(ZerocheckClaim<P::Scalar, impl CompositionPoly<P::Scalar>>, OracleClaimMeta), Error> {
	let (constraints, meta) = split_constraint_set(constraint_set, oracles)?;

	let mut zeros = Vec::new();
	for Constraint {
		composition,
		predicate,
	} in constraints
	{
		match predicate {
			ConstraintPredicate::Zero => zeros.push(CompositionScalarAdapter::new(composition)),
			_ => bail!(Error::MixedBatchingNotSupported),
		}
	}

	let claim = ZerocheckClaim::new(meta.n_vars, meta.oracle_ids.len(), zeros)?;
	Ok((claim, meta))
}

fn split_constraint_set<F: TowerField, P: PackedField>(
	constraint_set: ConstraintSet<P>,
	oracles: &MultilinearOracleSet<F>,
) -> Result<(Vec<Constraint<P>>, OracleClaimMeta), Error> {
	let ConstraintSet {
		oracle_ids,
		constraints,
	} = constraint_set;

	let all_n_vars = oracle_ids
		.iter()
		.map(|&oracle_id| {
			if !oracles.is_valid_oracle_id(oracle_id) {
				bail!(OracleError::InvalidOracleId(oracle_id));
			}
			Ok(oracles.n_vars(oracle_id))
		})
		.collect::<Result<Vec<_>, Error>>()?;

	let n_vars = if let Some(&n_vars) = all_n_vars.first() {
		n_vars
	} else {
		bail!(Error::EmptyConstraintSet);
	};

	let meta = OracleClaimMeta { n_vars, oracle_ids };
	Ok((constraints, meta))
}

/// Constructs evalcheck claims from metadata returned by constraint set claim constructors.
pub fn make_eval_claims<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
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
			let poly = oracles.oracle(oracle_id);
			let eval_point = batch_sumcheck_output.challenges[max_n_vars - meta.n_vars..].to_vec();

			let claim = EvalcheckMultilinearClaim {
				poly,
				eval_point,
				eval,
				is_random_point: true,
			};

			evalcheck_claims.push(claim);
		}
	}

	Ok(evalcheck_claims)
}
