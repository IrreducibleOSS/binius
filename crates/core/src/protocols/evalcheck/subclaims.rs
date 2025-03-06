// Copyright 2024-2025 Irreducible Inc.

//! This module contains helpers to create bivariate sumcheck instances originating from:
//!  * products with shift indicators (shifted virtual polynomials)
//!  * products with tower basis (packed virtual polynomials)
//!
//! All of them have common traits:
//!  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
//!  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
//!  * other multilin is a projection of one of the evalcheck claim multilins to its first variables

use std::collections::HashSet;

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{
	ArithExpr, CompositionPoly, EvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension,
	MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;

use super::{error::Error, evalcheck::EvalcheckMultilinearClaim};
use crate::{
	fiat_shamir::Challenger,
	oracle::{
		CompositeMLE, ConstraintSet, ConstraintSetBuilder, Error as OracleError,
		MultilinearOracleSet, OracleId, Packed, ProjectionVariant, Shifted,
	},
	polynomial::MultivariatePoly,
	protocols::sumcheck::{
		self,
		prove::oracles::{constraint_sets_sumcheck_provers_metas, SumcheckProversWithMetas},
		Error as SumcheckError,
	},
	transcript::ProverTranscript,
	transparent::{
		eq_ind::EqIndPartialEval, shift_ind::ShiftIndPartialEval, tower_basis::TowerBasis,
	},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};

/// Create oracles for the bivariate product of an inner oracle with shift indicator.
///
/// Projects to first `block_size()` vars.
pub fn shifted_sumcheck_meta<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	shifted: &Shifted,
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	projected_bivariate_meta(
		oracles,
		shifted.id(),
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
pub fn process_shifted_sumcheck<U, F>(
	shifted: &Shifted,
	meta: ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	projected: MultilinearExtension<PackedType<U, F>>,
) -> Result<(), Error>
where
	PackedType<U, F>: PackedFieldIndexable,
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	process_projected_bivariate_witness(
		witness_index,
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
		projected,
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
	packed: &Packed,
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	let n_vars = oracles.n_vars(packed.id());
	let log_degree = packed.log_degree();
	let binary_tower_level = oracles.oracle(packed.id()).binary_tower_level();

	if log_degree > n_vars {
		bail!(OracleError::NotEnoughVarsForPacking { n_vars, log_degree });
	}

	// NB. projected_n_vars = 0 because eval_point length is log_degree less than inner n_vars
	projected_bivariate_meta(oracles, packed.id(), 0, eval_point, |_| {
		Ok(TowerBasis::new(log_degree, binary_tower_level)?)
	})
}

pub fn composite_sumcheck_meta<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	id: OracleId,
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	Ok(ProjectedBivariateMeta {
		multiplier_id: oracles.add_transparent(EqIndPartialEval::new(eval_point.to_vec()))?,
		inner_id: id,
		projected_id: None,
		projected_n_vars: eval_point.len(),
	})
}

pub fn add_bivariate_sumcheck_to_constraints<F: TowerField>(
	meta: ProjectedBivariateMeta,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	n_vars: usize,
	eval: F,
) {
	if n_vars > constraint_builders.len() {
		constraint_builders.resize_with(n_vars, || ConstraintSetBuilder::new());
	}
	let bivariate_product = ArithExpr::Var(0) * ArithExpr::Var(1);
	constraint_builders[n_vars - 1].add_sumcheck(meta.oracle_ids(), bivariate_product, eval);
}

pub fn add_composite_sumcheck_to_constraints<F: TowerField>(
	meta: ProjectedBivariateMeta,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	comp: CompositeMLE<F>,
	eval: F,
) {
	let n_vars = comp.n_vars();
	let mut oracle_ids = comp.inner().clone();
	oracle_ids.push(meta.multiplier_id); // eq

	// Var(comp.n_polys()) corresponds to eq_oracle
	let expr = <_ as CompositionPoly<F>>::expression(comp.c()) * ArithExpr::Var(comp.n_polys());
	if n_vars > constraint_builders.len() {
		constraint_builders.resize_with(n_vars, || ConstraintSetBuilder::new());
	}
	constraint_builders[n_vars - 1].add_sumcheck(oracle_ids, expr, eval);
}

/// Creates bivariate witness and adds them to the witness index, and add bivariate sumcheck constraint to the [`ConstraintSetBuilder`]
#[allow(clippy::too_many_arguments)]
pub fn process_packed_sumcheck<U, F>(
	oracles: &MultilinearOracleSet<F>,
	packed: &Packed,
	meta: ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	projected: MultilinearExtension<PackedType<U, F>>,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	let log_degree = packed.log_degree();
	let binary_tower_level = oracles.oracle(packed.id()).binary_tower_level();

	process_projected_bivariate_witness(
		witness_index,
		meta,
		eval_point,
		|_projected_eval_point| {
			let tower_basis = TowerBasis::new(log_degree, binary_tower_level)?;
			let tower_basis_mle = tower_basis.multilinear_extension::<PackedType<U, F>>()?;
			Ok(MLEDirectAdapter::from(tower_basis_mle).upcast_arc_dyn())
		},
		projected,
	)?;

	add_bivariate_sumcheck_to_constraints(meta, constraint_builders, packed.log_degree(), eval);
	Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn process_composite_sumcheck<U, F>(
	comp: &CompositeMLE<F>,
	meta: ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	backend: &impl ComputationBackend,
	projected: MultilinearExtension<PackedType<U, F>>,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	process_projected_bivariate_witness(
		witness_index,
		meta,
		eval_point,
		|_projected_eval_point| {
			let eq = EqIndPartialEval::new(eval_point.to_vec());
			let eq_mle = eq.multilinear_extension::<PackedType<U, F>, _>(backend)?;
			Ok(MLEDirectAdapter::from(eq_mle).upcast_arc_dyn())
		},
		projected,
	)?;

	let mut oracle_ids = comp.inner().clone();
	oracle_ids.push(meta.multiplier_id);

	if comp.n_vars() > constraint_builders.len() {
		constraint_builders.resize_with(comp.n_vars(), || ConstraintSetBuilder::new());
	}
	// Var(comp.n_polys()) corresponds to eq_oracle
	let expr = <_ as CompositionPoly<F>>::expression(comp.c()) * ArithExpr::Var(comp.n_polys());
	constraint_builders[comp.n_vars() - 1].add_sumcheck(oracle_ids, expr, eval);
	Ok(())
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

fn process_projected_bivariate_witness<'a, U, F>(
	witness_index: &mut MultilinearExtensionIndex<'a, U, F>,
	meta: ProjectedBivariateMeta,
	eval_point: &[F],
	multiplier_witness_ctr: impl FnOnce(&[F]) -> Result<MultilinearWitness<'a, PackedType<U, F>>, Error>,
	projected: MultilinearExtension<PackedType<U, F>>,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	let ProjectedBivariateMeta {
		projected_id,
		multiplier_id,
		projected_n_vars,
		..
	} = meta;

	let projected_eval_point = if let Some(projected_id) = projected_id {
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

pub fn calculate_projected_mles<U, F, Backend>(
	metas: &[ProjectedBivariateMeta],
	memoized_queries: &mut MemoizedQueries<PackedType<U, F>, Backend>,
	projected_bivariate_claims: &[EvalcheckMultilinearClaim<F>],
	witness_index: &MultilinearExtensionIndex<U, F>,
	backend: &Backend,
) -> Result<Vec<MultilinearExtension<PackedType<U, F>>>, Error>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let inner_multilins = metas
		.iter()
		.map(|meta| {
			witness_index
				.get_multilin_poly(meta.inner_id)
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, Error>>()?;

	// Memoize queries for calculate_projected_mle
	for (meta, claim) in metas.iter().zip(projected_bivariate_claims) {
		let eval_point = &claim.eval_point[meta.projected_n_vars..];
		memoized_queries.full_query(eval_point, backend)?;
	}

	inner_multilins
		.par_iter()
		.zip(projected_bivariate_claims)
		.zip(metas)
		.map(|((inner_multilin, claim), meta)| {
			let eval_point = &claim.eval_point[meta.projected_n_vars..];
			let query = memoized_queries
				.full_query_readonly(eval_point)
				.ok_or(Error::MissingQuery)?;

			backend
				.evaluate_partial_high(&inner_multilin, query.to_ref())
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, Error>>()
}

#[allow(clippy::type_complexity)]
pub struct MemoizedQueries<P: PackedField, Backend: ComputationBackend> {
	memo: Vec<(Vec<P::Scalar>, MultilinearQuery<P, Backend::Vec<P>>)>,
}

impl<P: PackedField, Backend: ComputationBackend> MemoizedQueries<P, Backend> {
	#[allow(clippy::new_without_default)]
	pub const fn new() -> Self {
		Self { memo: Vec::new() }
	}

	/// Constructs `MemoizedQueries` from a list of eval_points and corresponding MultilinearQueries.
	/// Assumes that each `eval_point` is given at most once.
	/// Does not check that the input is valid.
	#[allow(clippy::type_complexity)]
	pub const fn new_from_known_queries(
		data: Vec<(Vec<P::Scalar>, MultilinearQuery<P, Backend::Vec<P>>)>,
	) -> Self {
		Self { memo: data }
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

	/// Finds a `MultilinearQuery` corresponding to the given `eval_point`.
	pub fn full_query_readonly(
		&self,
		eval_point: &[P::Scalar],
	) -> Option<&MultilinearQuery<P, Backend::Vec<P>>> {
		self.memo
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
			.map(|index| {
				let (_, ref query) = &self.memo[index];
				query
			})
	}

	pub fn memoize_query_par(
		&mut self,
		eval_points: Vec<&[P::Scalar]>,
		backend: &Backend,
	) -> Result<(), Error> {
		let deduplicated_eval_points = eval_points.into_iter().collect::<HashSet<_>>();

		let new_queries = deduplicated_eval_points
			.into_par_iter()
			.filter(|ep| self.full_query_readonly(ep).is_none())
			.map(|ep| {
				backend
					.multilinear_query::<P>(ep)
					.map(|res| (ep.to_vec(), res))
					.map_err(Error::from)
			})
			.collect::<Result<Vec<_>, Error>>()?;

		self.memo.extend(new_queries);

		Ok(())
	}
}

type SumcheckProofEvalcheckClaims<F> = Vec<EvalcheckMultilinearClaim<F>>;

pub fn prove_bivariate_sumchecks_with_switchover<U, F, DomainField, Transcript, Backend>(
	witness: &MultilinearExtensionIndex<U, F>,
	constraint_sets: Vec<ConstraintSet<F>>,
	transcript: &mut ProverTranscript<Transcript>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<SumcheckProofEvalcheckClaims<F>, SumcheckError>
where
	U: UnderlierType + PackScalar<F> + PackScalar<DomainField>,
	F: TowerField + ExtensionField<DomainField>,
	DomainField: Field,
	Transcript: Challenger,
	Backend: ComputationBackend,
{
	let SumcheckProversWithMetas { provers, metas } = constraint_sets_sumcheck_provers_metas(
		constraint_sets,
		witness,
		domain_factory,
		&switchover_fn,
		backend,
	)?;

	let sumcheck_output = sumcheck::batch_prove(provers, transcript)?;

	let evalcheck_claims = sumcheck::make_eval_claims(metas, sumcheck_output)?;

	Ok(evalcheck_claims)
}
