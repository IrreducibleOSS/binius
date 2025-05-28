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

use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::{
	ArithExpr, CompositionPoly, EvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter,
	MultilinearExtension, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use tracing::instrument;

use super::{EvalPoint, EvalPointOracleIdMap, error::Error, evalcheck::EvalcheckMultilinearClaim};
use crate::{
	fiat_shamir::Challenger,
	oracle::{
		CompositeMLE, ConstraintSet, ConstraintSetBuilder, Error as OracleError,
		MultilinearOracleSet, OracleId, Packed, Shifted,
	},
	polynomial::MultivariatePoly,
	protocols::sumcheck::{
		self, Error as SumcheckError,
		prove::{
			front_loaded,
			oracles::{
				MLECheckProverWithMeta, SumcheckProversWithMetas,
				constraint_sets_mlecheck_prover_meta, constraint_sets_sumcheck_provers_metas,
			},
		},
	},
	transcript::ProverTranscript,
	transparent::{shift_ind::ShiftIndPartialEval, tower_basis::TowerBasis},
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

/// Creates bivariate witness and adds them to the witness index, and add bivariate sumcheck
/// constraint to the [`ConstraintSetBuilder`]
#[allow(clippy::too_many_arguments)]
pub fn process_shifted_sumcheck<F, P>(
	shifted: &Shifted,
	meta: &ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<P>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	partial_evals: &EvalPointOracleIdMap<MultilinearExtension<P>, F>,
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
		partial_evals,
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

pub fn add_bivariate_sumcheck_to_constraints<F: TowerField>(
	meta: &ProjectedBivariateMeta,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	n_vars: usize,
	eval: F,
) {
	if n_vars >= constraint_builders.len() {
		constraint_builders.resize_with(n_vars + 1, || ConstraintSetBuilder::new());
	}
	let bivariate_product = ArithExpr::Var(0) * ArithExpr::Var(1);
	constraint_builders[n_vars].add_sumcheck(meta.oracle_ids(), bivariate_product.into(), eval);
}

pub fn add_composite_sumcheck_to_constraints<F: TowerField>(
	position: usize,
	eval_point: &EvalPoint<F>,
	constraint_builders: &mut Vec<(EvalPoint<F>, ConstraintSetBuilder<F>)>,
	comp: &CompositeMLE<F>,
	eval: F,
) {
	let oracle_ids = comp.inner().clone();

	if let Some((_, constraint_builder)) = constraint_builders.get_mut(position) {
		constraint_builder.add_sumcheck(
			oracle_ids,
			<_ as CompositionPoly<F>>::expression(comp.c()),
			eval,
		);
	} else {
		let mut new_builder = ConstraintSetBuilder::new();
		new_builder.add_sumcheck(oracle_ids, <_ as CompositionPoly<F>>::expression(comp.c()), eval);
		constraint_builders.push((eval_point.clone(), new_builder));
	}
}

/// Creates bivariate witness and adds them to the witness index, and add bivariate sumcheck
/// constraint to the [`ConstraintSetBuilder`]
#[allow(clippy::too_many_arguments)]
pub fn process_packed_sumcheck<F, P>(
	oracles: &MultilinearOracleSet<F>,
	packed: &Packed,
	meta: &ProjectedBivariateMeta,
	eval_point: &[F],
	eval: F,
	witness_index: &mut MultilinearExtensionIndex<P>,
	constraint_builders: &mut Vec<ConstraintSetBuilder<F>>,
	partial_evals: &EvalPointOracleIdMap<MultilinearExtension<P>, F>,
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
		partial_evals,
	)?;

	add_bivariate_sumcheck_to_constraints(meta, constraint_builders, packed.log_degree(), eval);
	Ok(())
}

/// Metadata about a sumcheck over a bivariate product of two multilinears.
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
		inner_id,
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
	partial_evals: &EvalPointOracleIdMap<MultilinearExtension<P>, F>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
{
	let &ProjectedBivariateMeta {
		projected_id,
		multiplier_id,
		projected_n_vars,
		inner_id,
	} = meta;

	let projected_eval_point = if let Some(projected_id) = projected_id {
		let (prefix, suffix) = eval_point.split_at(projected_n_vars);

		let projected = partial_evals
			.get(inner_id, suffix)
			.expect("projected should exist if projected_id exist")
			.clone();

		witness_index.update_multilin_poly(vec![(
			projected_id,
			MLEDirectAdapter::from(projected).upcast_arc_dyn(),
		)])?;
		prefix
	} else {
		eval_point
	};

	let m = multiplier_witness_ctr(projected_eval_point)?;

	if !witness_index.has(multiplier_id) {
		witness_index.update_multilin_poly([(multiplier_id, m)])?;
	}
	Ok(())
}

pub struct OracleIdPartialEval<P: PackedField> {
	pub id: OracleId,
	pub suffix: EvalPoint<P::Scalar>,
	pub partial_eval: MultilinearExtension<P>,
}

/// shifted / packed oracle compute the projected MLE (i.e. the inner oracle evaluated on the
/// projected eval_point)
#[allow(clippy::type_complexity)]
#[instrument(
	skip_all,
	name = "Evalcheck::calculate_projected_mles",
	level = "debug"
)]
pub fn collect_projected_mles<F, P>(
	metas: &[ProjectedBivariateMeta],
	memoized_queries: &mut MemoizedData<P>,
	projected_bivariate_claims: &[EvalcheckMultilinearClaim<F>],
	witness_index: &MultilinearExtensionIndex<P>,
	partial_evals: &mut EvalPointOracleIdMap<MultilinearExtension<P>, F>,
) -> Result<(), Error>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
{
	let mut queries_to_memoize = Vec::new();
	for (meta, claim) in metas.iter().zip(projected_bivariate_claims) {
		queries_to_memoize.push(&claim.eval_point[meta.projected_n_vars..]);
	}
	memoized_queries.memoize_query_par(queries_to_memoize)?;

	let new_partial_evals = projected_bivariate_claims
		.par_iter()
		.zip(metas)
		.map(|(claim, meta)| match meta.projected_id {
			Some(_) => {
				let inner_multilin = witness_index.get_multilin_poly(meta.inner_id)?;
				let eval_point = &claim.eval_point[meta.projected_n_vars..];
				let query = memoized_queries
					.full_query_readonly(eval_point)
					.ok_or(Error::MissingQuery)?;

				if partial_evals.get(meta.inner_id, eval_point).is_some() {
					return Ok(None);
				}

				let partial_eval = inner_multilin
					.evaluate_partial_high(query.to_ref())
					.map_err(Error::from)?;

				Ok(Some(OracleIdPartialEval {
					id: meta.inner_id,
					suffix: eval_point.into(),
					partial_eval,
				}))
			}
			_ => Ok(None),
		})
		.collect::<Result<Vec<Option<_>>, Error>>();

	for OracleIdPartialEval {
		id,
		suffix,
		partial_eval,
	} in new_partial_evals?.into_iter().flatten()
	{
		partial_evals.insert(id, suffix, partial_eval)
	}

	Ok(())
}

/// Struct for memoizing tensor expansions of evaluation points and partial evaluations of
/// multilinears
#[allow(clippy::type_complexity)]
pub struct MemoizedData<'a, P: PackedField> {
	query: Vec<(Vec<P::Scalar>, MultilinearQuery<P>)>,
	partial_evals: EvalPointOracleIdMap<MultilinearWitness<'a, P>, P::Scalar>,
}

impl<'a, P: PackedField> MemoizedData<'a, P> {
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
	) -> Result<&MultilinearQuery<P>, binius_hal::Error> {
		if let Some(index) = self
			.query
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
		{
			let (_, query) = &self.query[index];
			return Ok(query);
		}

		let query = MultilinearQuery::expand(eval_point);
		self.query.push((eval_point.to_vec(), query));

		let (_, query) = self.query.last().expect("pushed query immediately above");
		Ok(query)
	}

	/// Finds a `MultilinearQuery` corresponding to the given `eval_point`.
	pub fn full_query_readonly(&self, eval_point: &[P::Scalar]) -> Option<&MultilinearQuery<P>> {
		self.query
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
			.map(|index| {
				let (_, query) = &self.query[index];
				query
			})
	}

	#[instrument(skip_all, name = "Evalcheck::memoize_query_par", level = "debug")]
	pub fn memoize_query_par<'b>(
		&mut self,
		eval_points: impl IntoIterator<Item = &'b [P::Scalar]>,
	) -> Result<(), binius_hal::Error> {
		let deduplicated_eval_points = eval_points.into_iter().collect::<HashSet<_>>();

		let new_queries = deduplicated_eval_points
			.into_par_iter()
			.filter(|ep| self.full_query_readonly(ep).is_none())
			.map(|ep| {
				let query = MultilinearQuery::<P>::expand(ep);
				(ep.to_vec(), query)
			})
			.collect::<Vec<_>>();

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
			.for_each(|(claim, meta)| {
				let inner_id = meta.inner_id;
				if oracles.oracle(inner_id).variant.is_committed() && meta.projected_id.is_some() {
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

	let batch_prover = front_loaded::BatchProver::new(provers, transcript)?;

	let mut sumcheck_output = batch_prover.run(transcript)?;

	// Reverse challenges since folding high-to-low
	sumcheck_output.challenges.reverse();

	let evalcheck_claims =
		sumcheck::make_eval_claims(EvaluationOrder::HighToLow, metas, sumcheck_output)?;

	Ok(evalcheck_claims)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_mlecheck_with_switchover<'a, F, P, DomainField, Transcript, Backend>(
	witness: &MultilinearExtensionIndex<P>,
	constraint_set: ConstraintSet<F>,
	eq_ind_challenges: EvalPoint<F>,
	memoized_data: &mut MemoizedData<'a, P>,
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
	let MLECheckProverWithMeta { prover, meta } = constraint_sets_mlecheck_prover_meta(
		EvaluationOrder::HighToLow,
		constraint_set,
		eq_ind_challenges,
		memoized_data,
		witness,
		domain_factory,
		&switchover_fn,
		backend,
	)?;

	let batch_prover = front_loaded::BatchProver::new(vec![prover], transcript)?;

	let mut sumcheck_output = batch_prover.run(transcript)?;

	// Reverse challenges since folding high-to-low
	sumcheck_output.challenges.reverse();

	// extract eq_ind_eval
	sumcheck_output.multilinear_evals[0].pop();

	let evalcheck_claims =
		sumcheck::make_eval_claims(EvaluationOrder::HighToLow, vec![meta], sumcheck_output)?;

	Ok(evalcheck_claims)
}
