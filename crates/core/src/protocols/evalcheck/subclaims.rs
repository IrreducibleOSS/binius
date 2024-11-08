// Copyright 2024 Irreducible Inc.

//! This module contains helpers to create bivariate sumcheck instances originating from:
//!  * products with shift indicators (shifted virtual polynomials)
//!  * products with tower basis (packed virtual polynomials)
//!  * products with equality indicator ([`CommittedEvalClaim`])
//!
//! All of them have common traits:
//!  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
//!  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
//!  * other multilin is a projection of one of the evalcheck claim multilins to its first variables

use super::{
	error::Error,
	evalcheck::{BatchCommittedEvalClaims, CommittedEvalClaim},
	EvalcheckMultilinearClaim, EvalcheckProver, EvalcheckVerifier,
};
use crate::{
	composition::BivariateProduct,
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet, OracleId,
		Packed, ProjectionVariant, Shifted,
	},
	polynomial::MultivariatePoly,
	protocols::sumcheck::{
		self,
		prove::oracles::{constraint_sets_sumcheck_provers_metas, SumcheckProversWithMetas},
		Error as SumcheckError, Proof as SumcheckBatchProof,
	},
	transcript::CanWrite,
	transparent::{
		eq_ind::EqIndPartialEval, shift_ind::ShiftIndPartialEval, tower_basis::TowerBasis,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{EvaluationDomainFactory, MLEDirectAdapter, MultilinearQuery};
use binius_utils::bail;
use p3_challenger::{CanObserve, CanSample};
use tracing::instrument;

/// Create oracles for the bivariate product of an inner oracle with shift indicator.
///
/// Projects to first `block_size()` vars.
pub fn shifted_sumcheck_meta<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	shifted: &Shifted<F>,
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	projected_bivariate_meta(
		oracles,
		shifted.inner().id(),
		shifted.block_size(),
		eval_point,
		|projected_eval_point| {
			Ok(ShiftIndPartialEval::new(
				shifted.block_size(),
				shifted.shift_offset(),
				shifted.shift_variant(),
				projected_eval_point.to_vec(),
			)?)
		},
	)
}

/// Creates bivariate witness and adds them to the witness index, and add bivariate sumcheck constraint to the [`ConstraintSetBuilder`]
#[allow(clippy::too_many_arguments)]
pub fn process_shifted_sumcheck<U, F, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	shifted: &Shifted<F>,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	memoized_queries: &mut MemoizedQueries<PackedType<U, F>, Backend>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<PackedType<U, F>>>,
	backend: &Backend,
) -> Result<(), Error>
where
	PackedType<U, F>: PackedFieldIndexable,
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let meta = shifted_sumcheck_meta(oracles, shifted, eval_point)?;

	process_projected_bivariate_witness(
		witness_index,
		memoized_queries,
		meta,
		eval_point,
		|projected_eval_point| {
			let shift_ind = ShiftIndPartialEval::new(
				projected_eval_point.len(),
				shifted.shift_offset(),
				shifted.shift_variant(),
				projected_eval_point.to_vec(),
			)?;

			let shift_ind_mle = shift_ind.multilinear_extension::<PackedType<U, F>>()?;
			Ok(MLEDirectAdapter::from(shift_ind_mle).upcast_arc_dyn())
		},
		backend,
	)?;

	add_bivariate_sumcheck_to_constraints(meta, constraint_builders, shifted.block_size(), eval);

	Ok(())
}

/// Create oracles for the bivariate product of an inner oracle with the tower basis.
///
/// Projects to first `log_degree()` vars.
/// Returns metadata object with oracle identifiers.
pub fn packed_sumcheck_meta<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	packed: &Packed<F>,
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	let n_vars = packed.inner().n_vars();
	let log_degree = packed.log_degree();
	let binary_tower_level = packed.inner().binary_tower_level();

	if log_degree > n_vars {
		bail!(OracleError::NotEnoughVarsForPacking { n_vars, log_degree });
	}

	// NB. projected_n_vars = 0 because eval_point length is log_degree less than inner n_vars
	projected_bivariate_meta(oracles, packed.inner().id(), 0, eval_point, |_| {
		Ok(TowerBasis::new(log_degree, binary_tower_level)?)
	})
}

pub fn add_bivariate_sumcheck_to_constraints<P: PackedField>(
	meta: ProjectedBivariateMeta,
	constraint_builders: &mut Vec<ConstraintSetBuilder<P>>,
	n_vars: usize,
	eval: P::Scalar,
) {
	if n_vars > constraint_builders.len() {
		constraint_builders.resize_with(n_vars, || ConstraintSetBuilder::new());
	}

	add_bivariate_sumcheck_to_constraint_builder(meta, &mut constraint_builders[n_vars - 1], eval);
}

fn add_bivariate_sumcheck_to_constraint_builder<P: PackedField>(
	meta: ProjectedBivariateMeta,
	constraint_builder: &mut ConstraintSetBuilder<P>,
	eval: P::Scalar,
) {
	constraint_builder.add_sumcheck(meta.oracle_ids(), BivariateProduct {}, eval);
}

/// Creates bivariate witness and adds them to the witness index, and add bivariate sumcheck constraint to the [`ConstraintSetBuilder`]
#[allow(clippy::too_many_arguments)]
pub fn process_packed_sumcheck<U, F, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	packed: &Packed<F>,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	memoized_queries: &mut MemoizedQueries<PackedType<U, F>, Backend>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<PackedType<U, F>>>,
	backend: &Backend,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let log_degree = packed.log_degree();
	let binary_tower_level = packed.inner().binary_tower_level();

	let meta = packed_sumcheck_meta(oracles, packed, eval_point)?;

	process_projected_bivariate_witness(
		witness_index,
		memoized_queries,
		meta,
		eval_point,
		|_projected_eval_point| {
			let tower_basis = TowerBasis::new(log_degree, binary_tower_level)?;
			let tower_basis_mle = tower_basis.multilinear_extension::<PackedType<U, F>>()?;
			Ok(MLEDirectAdapter::from(tower_basis_mle).upcast_arc_dyn())
		},
		backend,
	)?;

	add_bivariate_sumcheck_to_constraints(meta, constraint_builders, packed.log_degree(), eval);

	Ok(())
}

#[derive(Clone)]
pub struct NonSameQueryPcsClaimMeta<F> {
	pub projected_bivariate_meta: ProjectedBivariateMeta,
	eval_point: Vec<F>,
	pub eval: F,
}

/// Create sumchecks for committed evalcheck claims on differing eval points.
///
/// Each sumcheck instance is bivariate product of a column projection and equality indicator.
/// Common suffix is optimized out, degenerate zero variable sumchecks are not emitted, and
/// PCS claims are inserted directly into [`BatchCommittedEvalClaims`] instead.
pub fn non_same_query_pcs_sumcheck_metas<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
	new_batch_committed_eval_claims: &mut BatchCommittedEvalClaims<F>,
) -> Result<Vec<NonSameQueryPcsClaimMeta<F>>, Error> {
	let common_suffix_len = compute_common_suffix_len(
		committed_eval_claims
			.iter()
			.map(|claim| claim.eval_point.as_slice()),
	);

	let mut metas = Vec::new();

	for claim in committed_eval_claims {
		let eval_point = &claim.eval_point;
		debug_assert!(eval_point.len() >= common_suffix_len);

		if eval_point.len() == common_suffix_len {
			new_batch_committed_eval_claims.insert(claim.clone());
			continue;
		}

		let projected_bivariate_meta = projected_bivariate_meta(
			oracles,
			oracles.committed_oracle_id(claim.id),
			eval_point.len() - common_suffix_len,
			eval_point,
			|projected_eval_point| {
				Ok(EqIndPartialEval::new(
					projected_eval_point.len(),
					projected_eval_point.to_vec(),
				)?)
			},
		)?;

		let meta = NonSameQueryPcsClaimMeta {
			projected_bivariate_meta,
			eval_point: eval_point.to_vec(),
			eval: claim.eval,
		};

		metas.push(meta);
	}

	Ok(metas)
}

fn compute_common_suffix_len<'a, F: PartialEq + 'a>(
	mut eval_points: impl Iterator<Item = &'a [F]>,
) -> usize {
	if let Some(first_eval_point) = eval_points.next() {
		let common_suffix = first_eval_point.iter().rev().collect::<Vec<_>>();
		let common_suffix = eval_points.fold(common_suffix, |common_suffix, eval_point| {
			eval_point
				.iter()
				.rev()
				.zip(common_suffix)
				.take_while(|(a, b)| a == b)
				.unzip::<_, _, Vec<_>, Vec<_>>()
				.0
		});
		common_suffix.len()
	} else {
		0
	}
}

pub fn process_non_same_query_pcs_sumcheck_witness<U, F, Backend>(
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	memoized_queries: &mut MemoizedQueries<PackedType<U, F>, Backend>,
	meta: NonSameQueryPcsClaimMeta<F>,
	backend: &Backend,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	process_projected_bivariate_witness(
		witness_index,
		memoized_queries,
		meta.projected_bivariate_meta,
		&meta.eval_point,
		|projected_eval_point| {
			let eq_ind =
				EqIndPartialEval::new(projected_eval_point.len(), projected_eval_point.to_vec())?;
			let eq_ind_mle = eq_ind.multilinear_extension::<PackedType<U, F>, _>(backend)?;
			Ok(MLEDirectAdapter::from(eq_ind_mle).upcast_arc_dyn())
		},
		backend,
	)
}

#[derive(Clone, Copy)]
pub struct ProjectedBivariateMeta {
	inner_id: OracleId,
	projected_id: Option<OracleId>,
	multiplier_id: OracleId,
	projected_n_vars: usize,
}

impl ProjectedBivariateMeta {
	pub fn oracle_ids(&self) -> [OracleId; 2] {
		[
			self.projected_id.unwrap_or(self.inner_id),
			self.multiplier_id,
		]
	}
}

fn projected_bivariate_meta<F: TowerField, T: MultivariatePoly<F> + 'static>(
	oracles: &mut MultilinearOracleSet<F>,
	inner_id: OracleId,
	projected_n_vars: usize,
	eval_point: &[F],
	multiplier_transparent_ctr: impl FnOnce(&[F]) -> Result<T, Error>,
) -> Result<ProjectedBivariateMeta, Error> {
	let inner = oracles.oracle(inner_id);

	let (projected_eval_point, projected_id) = if projected_n_vars < inner.n_vars() {
		let projected_id = oracles.add_projected(
			inner_id,
			eval_point[projected_n_vars..].to_vec(),
			ProjectionVariant::LastVars,
		)?;

		(&eval_point[..projected_n_vars], Some(projected_id))
	} else {
		(eval_point, None)
	};

	let projected_n_vars = projected_eval_point.len();

	let multiplier_id =
		oracles.add_transparent(multiplier_transparent_ctr(projected_eval_point)?)?;

	let meta = ProjectedBivariateMeta {
		inner_id,
		projected_id,
		multiplier_id,
		projected_n_vars,
	};

	Ok(meta)
}

fn process_projected_bivariate_witness<'a, U, F, Backend>(
	witness_index: &mut MultilinearExtensionIndex<'a, U, F>,
	memoized_queries: &mut MemoizedQueries<PackedType<U, F>, Backend>,
	meta: ProjectedBivariateMeta,
	eval_point: &[F],
	multiplier_witness_ctr: impl FnOnce(&[F]) -> Result<MultilinearWitness<'a, PackedType<U, F>>, Error>,
	backend: &Backend,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let ProjectedBivariateMeta {
		inner_id,
		projected_id,
		multiplier_id,
		projected_n_vars,
	} = meta;

	let inner_multilin = witness_index.get_multilin_poly(inner_id)?;

	let projected_eval_point = if let Some(projected_id) = projected_id {
		let query = memoized_queries.full_query(&eval_point[projected_n_vars..], backend)?;
		let projected = inner_multilin.evaluate_partial_high(query.to_ref())?;
		witness_index.update_multilin_poly(vec![(
			projected_id,
			MLEDirectAdapter::from(projected).upcast_arc_dyn(),
		)])?;

		&eval_point[..projected_n_vars]
	} else {
		eval_point
	};

	let m = multiplier_witness_ctr(projected_eval_point)?;

	if !witness_index.has(multiplier_id) {
		witness_index.update_multilin_poly(vec![(multiplier_id, m)])?;
	}
	Ok(())
}

#[allow(clippy::type_complexity)]
pub struct MemoizedQueries<P: PackedField, Backend: ComputationBackend> {
	memo: Vec<(Vec<P::Scalar>, MultilinearQuery<P, Backend::Vec<P>>)>,
}

impl<P: PackedField, Backend: ComputationBackend> MemoizedQueries<P, Backend> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self { memo: Vec::new() }
	}

	pub fn full_query(
		&mut self,
		eval_point: &[P::Scalar],
		backend: &Backend,
	) -> Result<&MultilinearQuery<P, Backend::Vec<P>>, Error> {
		if let Some(index) = self
			.memo
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
		{
			let (_, ref query) = &self.memo[index];
			return Ok(query);
		}

		let query = backend.multilinear_query(eval_point)?;
		self.memo.push((eval_point.to_vec(), query));

		let (_, ref query) = self.memo.last().expect("pushed query immediately above");
		Ok(query)
	}
}

type SumcheckProofEvalcheckClaims<F> = (SumcheckBatchProof<F>, Vec<EvalcheckMultilinearClaim<F>>);

pub fn prove_bivariate_sumchecks_with_switchover<U, F, DomainField, CH, Backend>(
	oracles: &MultilinearOracleSet<F>,
	witness: &MultilinearExtensionIndex<U, F>,
	constraint_sets: Vec<ConstraintSet<PackedType<U, F>>>,
	challenger: &mut CH,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<SumcheckProofEvalcheckClaims<F>, SumcheckError>
where
	U: UnderlierType + PackScalar<F> + PackScalar<DomainField>,
	F: TowerField + ExtensionField<DomainField>,
	DomainField: Field,
	CH: CanObserve<F> + CanSample<F> + CanWrite,
	Backend: ComputationBackend,
{
	let SumcheckProversWithMetas { provers, metas } = constraint_sets_sumcheck_provers_metas(
		constraint_sets,
		witness,
		domain_factory.clone(),
		&switchover_fn,
		backend,
	)?;

	let (sumcheck_output, proof) = sumcheck::batch_prove(provers, challenger)?;

	let evalcheck_claims = sumcheck::make_eval_claims(oracles, metas, sumcheck_output)?;

	Ok((proof, evalcheck_claims))
}

pub fn make_non_same_query_pcs_sumcheck_claims<F>(
	verifier: &mut EvalcheckVerifier<F>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
) -> Result<ConstraintSet<F>, Error>
where
	F: TowerField,
{
	let metas = non_same_query_pcs_sumcheck_metas(
		verifier.oracles,
		committed_eval_claims,
		&mut verifier.batch_committed_eval_claims,
	)?;

	let mut constraint_set_builder = ConstraintSetBuilder::new();
	for meta in metas {
		add_bivariate_sumcheck_to_constraint_builder(
			meta.projected_bivariate_meta,
			&mut constraint_set_builder,
			meta.eval,
		)
	}
	Ok(constraint_set_builder.build_one(verifier.oracles)?)
}

#[instrument(skip_all, level = "debug")]
pub fn make_non_same_query_pcs_sumchecks<U, F, Backend>(
	prover: &mut EvalcheckProver<U, F, Backend>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
	backend: &Backend,
) -> Result<ConstraintSet<PackedType<U, F>>, Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let metas = non_same_query_pcs_sumcheck_metas(
		prover.oracles,
		committed_eval_claims,
		&mut prover.batch_committed_eval_claims,
	)?;

	let mut memoized_queries = MemoizedQueries::new();

	let mut constraint_set_builder = ConstraintSetBuilder::new();

	for meta in metas {
		add_bivariate_sumcheck_to_constraint_builder(
			meta.projected_bivariate_meta,
			&mut constraint_set_builder,
			meta.eval,
		);
		process_non_same_query_pcs_sumcheck_witness(
			prover.witness_index,
			&mut memoized_queries,
			meta,
			backend,
		)?;
	}
	Ok(constraint_set_builder.build_one(prover.oracles)?)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_compute_common_suffix_len() {
		let tests = vec![
			(vec![], 0),
			(vec![vec![1, 2, 3]], 3),
			(vec![vec![1, 2, 3], vec![2, 3]], 2),
			(vec![vec![1, 2, 3], vec![2, 3], vec![3]], 1),
			(vec![vec![1, 2, 3], vec![4, 2, 3], vec![6, 5, 3]], 1),
			(vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]], 3),
			(vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]], 0),
		];
		for test in tests {
			let eval_points = test.0.iter().map(|x| x.as_slice());
			let expected = test.1;
			let got = compute_common_suffix_len(eval_points);
			assert_eq!(got, expected);
		}
	}
}
