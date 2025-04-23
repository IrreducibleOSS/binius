// Copyright 2024-2025 Irreducible Inc.

//! This module contains helpers to create bivariate sumcheck instances originating from:
//!  * products with shift indicators (shifted virtual polynomials)
//!  * products with tower basis (packed virtual polynomials)
//!
//! All of them have common traits:
//!  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
//!  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
//!  * other multilin is a projection of one of the evalcheck claim multilins to its first variables

use std::collections::{HashMap, HashSet};

use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_math::{
	ArithExpr, CompositionPoly, EvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter,
	MultilinearExtension, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use tracing::instrument;

use super::{error::Error, evalcheck::EvalcheckMultilinearClaim, EvalPointOracleIdMap};
use crate::{
	fiat_shamir::Challenger,
	oracle::{
		CompositeMLE, ConstraintSet, ConstraintSetBuilder, Error as OracleError,
		MultilinearOracleSet, MultilinearPolyVariant, OracleId, Packed, Shifted,
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
pub fn process_shifted_sumcheck<F, P>(
	shifted: &Shifted,
	meta: &ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<P>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	projected: Option<MultilinearExtension<P>>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
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

			let shift_ind_mle = shift_ind.multilinear_extension::<P>()?;
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
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	Ok(ProjectedBivariateMeta {
		multiplier_id: oracles.add_transparent(EqIndPartialEval::new(eval_point.to_vec()))?,
		inner_id: None,
		projected_id: None,
		// not used in case of composite
		projected_n_vars: 0,
	})
}

pub fn add_bivariate_sumcheck_to_constraints<F: TowerField>(
	meta: &ProjectedBivariateMeta,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	n_vars: usize,
	eval: F,
) {
	if n_vars > constraint_builders.len() {
		constraint_builders.resize_with(n_vars, || ConstraintSetBuilder::new());
	}
	let bivariate_product = ArithExpr::var(0) * ArithExpr::var(1);
	constraint_builders[n_vars - 1].add_sumcheck(meta.oracle_ids(), bivariate_product, eval);
}

pub fn add_composite_sumcheck_to_constraints<F: TowerField>(
	meta: &ProjectedBivariateMeta,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	comp: &CompositeMLE<F>,
	eval: F,
) {
	let n_vars = comp.n_vars();
	let mut oracle_ids = comp.inner().clone();
	oracle_ids.push(meta.multiplier_id); // eq

	// Var(comp.n_polys()) corresponds to the eq MLE (meta.multiplier_id)
	let expr = <_ as CompositionPoly<F>>::expression(comp.c()) * ArithExpr::var(comp.n_polys());
	if n_vars > constraint_builders.len() {
		constraint_builders.resize_with(n_vars, || ConstraintSetBuilder::new());
	}
	constraint_builders[n_vars - 1].add_sumcheck(oracle_ids, expr, eval);
}

/// Creates bivariate witness and adds them to the witness index, and add bivariate sumcheck constraint to the [`ConstraintSetBuilder`]
#[allow(clippy::too_many_arguments)]
pub fn process_packed_sumcheck<F, P>(
	oracles: &MultilinearOracleSet<F>,
	packed: &Packed,
	meta: &ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<P>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	projected: Option<MultilinearExtension<P>>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
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
			let tower_basis_mle = tower_basis.multilinear_extension::<P>()?;
			Ok(MLEDirectAdapter::from(tower_basis_mle).upcast_arc_dyn())
		},
		projected,
	)?;

	add_bivariate_sumcheck_to_constraints(meta, constraint_builders, packed.log_degree(), eval);
	Ok(())
}

#[derive(Clone, Copy)]
pub struct ProjectedBivariateMeta {
	/// `Some` if shifted / packed, `None` if composite
	inner_id: Option<OracleId>,
	projected_id: Option<OracleId>,
	multiplier_id: OracleId,
	projected_n_vars: usize,
}

impl ProjectedBivariateMeta {
	pub fn oracle_ids(&self) -> [OracleId; 2] {
		[
			self.projected_id.unwrap_or_else(|| {
				self.inner_id
					.expect("oracle_ids() is only defined for shifted / packed")
			}),
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
		let projected_id =
			oracles.add_projected_last_vars(inner_id, eval_point[projected_n_vars..].to_vec())?;

		(&eval_point[..projected_n_vars], Some(projected_id))
	} else {
		(eval_point, None)
	};

	let projected_n_vars = projected_eval_point.len();

	let multiplier_id =
		oracles.add_transparent(multiplier_transparent_ctr(projected_eval_point)?)?;

	let meta = ProjectedBivariateMeta {
		inner_id: Some(inner_id),
		projected_id,
		multiplier_id,
		projected_n_vars,
	};

	Ok(meta)
}

fn process_projected_bivariate_witness<'a, F, P>(
	witness_index: &mut MultilinearExtensionIndex<'a, P>,
	meta: &ProjectedBivariateMeta,
	eval_point: &[F],
	multiplier_witness_ctr: impl FnOnce(&[F]) -> Result<MultilinearWitness<'a, P>, Error>,
	projected: Option<MultilinearExtension<P>>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
{
	let &ProjectedBivariateMeta {
		projected_id,
		multiplier_id,
		projected_n_vars,
		..
	} = meta;

	let projected_eval_point = if let Some(projected_id) = projected_id {
		witness_index.update_multilin_poly(vec![(
			projected_id,
			MLEDirectAdapter::from(
				projected.expect("projected should exist if projected_id exist"),
			)
			.upcast_arc_dyn(),
		)])?;

		&eval_point[..projected_n_vars]
	} else {
		eval_point
	};

	let m = multiplier_witness_ctr(projected_eval_point)?;

	if !witness_index.has(multiplier_id) {
		witness_index.update_multilin_poly([(multiplier_id, m)])?;
	}
	Ok(())
}

/// shifted / packed oracle -> compute the projected MLE (i.e. the inner oracle evaluated on the projected eval_point)
/// composite oracle -> None
#[allow(clippy::type_complexity)]
#[instrument(
	skip_all,
	name = "Evalcheck::calculate_projected_mles",
	level = "debug"
)]
pub fn calculate_projected_mles<F, P, Backend>(
	metas: &[ProjectedBivariateMeta],
	memoized_queries: &mut MemoizedData<P, Backend>,
	projected_bivariate_claims: &[EvalcheckMultilinearClaim<F>],
	witness_index: &MultilinearExtensionIndex<P>,
	backend: &Backend,
) -> Result<Vec<Option<MultilinearExtension<P>>>, Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let mut queries_to_memoize = Vec::new();
	for (meta, claim) in metas.iter().zip(projected_bivariate_claims) {
		if meta.inner_id.is_some() {
			// packed / shifted
			queries_to_memoize.push(&claim.eval_point[meta.projected_n_vars..]);
		}
	}
	memoized_queries.memoize_query_par(&queries_to_memoize, backend)?;

	projected_bivariate_claims
		.par_iter()
		.zip(metas)
		.map(|(claim, meta)| match (meta.inner_id, meta.projected_id) {
			(Some(inner_id), Some(_)) => {
				let inner_multilin = witness_index.get_multilin_poly(inner_id)?;
				let eval_point = &claim.eval_point[meta.projected_n_vars..];
				let query = memoized_queries
					.full_query_readonly(eval_point)
					.ok_or(Error::MissingQuery)?;
				Ok(Some(
					backend
						.evaluate_partial_high(&inner_multilin, query.to_ref())
						.map_err(Error::from)?,
				))
			}
			_ => Ok(None),
		})
		.collect::<Result<Vec<Option<_>>, Error>>()
}

/// Each composite oracle induces a new eq oracle, for which we need to fill the witness
pub fn fill_eq_witness_for_composites<F, P, Backend>(
	metas: &[ProjectedBivariateMeta],
	memoized_queries: &mut MemoizedData<P, Backend>,
	projected_bivariate_claims: &[EvalcheckMultilinearClaim<F>],
	witness_index: &mut MultilinearExtensionIndex<P>,
	backend: &Backend,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	let dedup_eval_points = metas
		.iter()
		.zip(projected_bivariate_claims)
		.filter(|(meta, _)| meta.inner_id.is_none())
		.map(|(_, claim)| claim.eval_point.as_ref())
		.collect::<HashSet<_>>();

	memoized_queries
		.memoize_query_par(&dedup_eval_points.iter().copied().collect::<Vec<_>>(), backend)?;

	let eq_indicators = dedup_eval_points
		.into_iter()
		.map(|eval_point| {
			let mle = MLEDirectAdapter::from(MultilinearExtension::new(
				eval_point.len(),
				memoized_queries
					.full_query_readonly(eval_point)
					.expect("computed above")
					.expansion()
					.to_vec(),
			)?)
			.upcast_arc_dyn();
			Ok((eval_point, mle))
		})
		.collect::<Result<HashMap<_, _>, Error>>()?;

	for (meta, claim) in metas
		.iter()
		.zip(projected_bivariate_claims)
		.filter(|(meta, _)| meta.inner_id.is_none())
	{
		let eq_ind = eq_indicators
			.get(claim.eval_point.as_ref())
			.expect("was added above");

		witness_index.update_multilin_poly(vec![(meta.multiplier_id, eq_ind.clone())])?;
	}

	Ok(())
}

/// Struct for memoizing tensor expansions of evaluation points and partial evaluations of multilinears
#[allow(clippy::type_complexity)]
pub struct MemoizedData<'a, P: PackedField, Backend: ComputationBackend> {
	query: Vec<(Vec<P::Scalar>, MultilinearQuery<P, Backend::Vec<P>>)>,
	partial_evals: EvalPointOracleIdMap<MultilinearWitness<'a, P>, P::Scalar>,
}

impl<'a, P: PackedField, Backend: ComputationBackend> MemoizedData<'a, P, Backend> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			query: Vec::new(),
			partial_evals: EvalPointOracleIdMap::new(),
		}
	}

	pub fn full_query(
		&mut self,
		eval_point: &[P::Scalar],
		backend: &Backend,
	) -> Result<&MultilinearQuery<P, Backend::Vec<P>>, Error> {
		if let Some(index) = self
			.query
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
		{
			let (_, ref query) = &self.query[index];
			return Ok(query);
		}

		let query = backend.multilinear_query(eval_point)?;
		self.query.push((eval_point.to_vec(), query));

		let (_, ref query) = self.query.last().expect("pushed query immediately above");
		Ok(query)
	}

	/// Finds a `MultilinearQuery` corresponding to the given `eval_point`.
	pub fn full_query_readonly(
		&self,
		eval_point: &[P::Scalar],
	) -> Option<&MultilinearQuery<P, Backend::Vec<P>>> {
		self.query
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
			.map(|index| {
				let (_, ref query) = &self.query[index];
				query
			})
	}

	#[instrument(skip_all, name = "Evalcheck::memoize_query_par", level = "debug")]
	pub fn memoize_query_par(
		&mut self,
		eval_points: &[&[P::Scalar]],
		backend: &Backend,
	) -> Result<(), binius_hal::Error> {
		let deduplicated_eval_points = eval_points.iter().collect::<HashSet<_>>();

		let new_queries = deduplicated_eval_points
			.into_par_iter()
			.filter(|ep| self.full_query_readonly(ep).is_none())
			.map(|ep| {
				backend
					.multilinear_query::<P>(ep)
					.map(|res| (ep.to_vec(), res))
			})
			.collect::<Result<Vec<_>, binius_hal::Error>>()?;

		self.query.extend(new_queries);

		Ok(())
	}

	pub fn memoize_partial_evals(
		&mut self,
		metas: &[ProjectedBivariateMeta],
		projected_bivariate_claims: &[EvalcheckMultilinearClaim<P::Scalar>],
		oracles: &mut MultilinearOracleSet<P::Scalar>,
		witness_index: &MultilinearExtensionIndex<'a, P>,
	) where
		P::Scalar: TowerField,
	{
		projected_bivariate_claims
			.iter()
			.zip(metas)
			.filter(|(_, meta)| meta.inner_id.is_some())
			.for_each(|(claim, meta)| {
				let inner_id = meta.inner_id.expect("filtered by Some");
				if matches!(oracles.oracle(inner_id).variant, MultilinearPolyVariant::Committed)
					&& meta.projected_id.is_some()
				{
					let eval_point = claim.eval_point[meta.projected_n_vars..].to_vec().into();

					let projected_id = meta.projected_id.expect("checked above");

					let projected = witness_index
						.get_multilin_poly(projected_id)
						.expect("witness_index contains projected if projected_id exist");

					self.partial_evals.insert(inner_id, eval_point, projected);
				}
			});
	}

	pub fn partial_eval(
		&self,
		id: OracleId,
		eval_point: &[P::Scalar],
	) -> Option<&MultilinearWitness<'a, P>> {
		self.partial_evals.get(id, eval_point)
	}
}

type SumcheckProofEvalcheckClaims<F> = Vec<EvalcheckMultilinearClaim<F>>;

pub fn prove_bivariate_sumchecks_with_switchover<F, P, DomainField, Transcript, Backend>(
	witness: &MultilinearExtensionIndex<P>,
	constraint_sets: Vec<ConstraintSet<F>>,
	transcript: &mut ProverTranscript<Transcript>,
	switchover_fn: impl Fn(usize) -> usize + 'static,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<SumcheckProofEvalcheckClaims<F>, SumcheckError>
where
	P: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<DomainField>,
	F: TowerField + ExtensionField<DomainField>,
	DomainField: Field,
	Transcript: Challenger,
	Backend: ComputationBackend,
{
	let SumcheckProversWithMetas { provers, metas } = constraint_sets_sumcheck_provers_metas(
		EvaluationOrder::HighToLow,
		constraint_sets,
		witness,
		domain_factory,
		&switchover_fn,
		backend,
	)?;

	let sumcheck_output = sumcheck::batch_prove(provers, transcript)?;

	let evalcheck_claims =
		sumcheck::make_eval_claims(EvaluationOrder::HighToLow, metas, sumcheck_output)?;

	Ok(evalcheck_claims)
}
