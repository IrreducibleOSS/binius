// Copyright 2024 Ulvetanna Inc.

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
};
use crate::{
	oracle::{
		CompositePolyOracle, Error as OracleError, MultilinearOracleSet, OracleId, Packed,
		ProjectionVariant, Shifted,
	},
	polynomial::{
		composition::BivariateProduct,
		transparent::{
			eq_ind::EqIndPartialEval, shift_ind::ShiftIndPartialEval, tower_basis::TowerBasis,
		},
		MultilinearQuery, MultivariatePoly,
	},
	protocols::sumcheck::{SumcheckClaim, SumcheckWitness},
	witness::{MultilinearWitness, MultilinearWitnessIndex},
};
use binius_field::{Field, PackedField, PackedFieldIndexable, TowerField};
use std::sync::Arc;

// type aliases for bivariate claims/witnesses and their pairs to shorten type signatures
pub type BivariateSumcheck<'a, F, PW> = (SumcheckClaim<F>, BivariateSumcheckWitness<'a, PW>);
pub type BivariateSumcheckWitness<'a, PW> =
	SumcheckWitness<PW, BivariateProduct, MultilinearWitness<'a, PW>>;

/// Create oracles for the bivariate product of an inner oracle with shift indicator.
///
/// Projects to first `block_size()` vars.
/// Returns metadata object with oracle identifiers. Pass this object to:
///  - [`projected_bivariate_claim`] to obtain sumcheck claim
///  - [`shifted_sumcheck_witness`] to obtain sumcheck witness
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

/// Takes in metadata object and creates a witness for a bivariate claim on shift indicator.
///
/// `wf_eval_point` should be isomorphic to `eval_point` in `shifted_sumcheck_meta`.
pub fn shifted_sumcheck_witness<'a, F: Field, PW: PackedFieldIndexable>(
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: ProjectedBivariateMeta,
	shifted: &Shifted<F>,
	wf_eval_point: &[PW::Scalar],
) -> Result<BivariateSumcheckWitness<'a, PW>, Error> {
	projected_bivariate_witness(
		witness_index,
		memoized_queries,
		meta,
		wf_eval_point,
		|projected_eval_point| {
			let shift_ind = ShiftIndPartialEval::new(
				projected_eval_point.len(),
				shifted.shift_offset(),
				shifted.shift_variant(),
				projected_eval_point.to_vec(),
			)?;

			Ok(shift_ind
				.multilinear_extension::<PW>()?
				.specialize_arc_dyn())
		},
	)
}

/// Create oracles for the bivariate product of an inner oracle with the tower basis.
///
/// Projects to first `log_degree()` vars.
/// Returns metadata object with oracle identifiers. Pass this object to:
///  - [`projected_bivariate_claim`] to obtain sumcheck claim
///  - [`packed_sumcheck_witness`] to obtain sumcheck witness
pub fn packed_sumcheck_meta<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	packed: &Packed<F>,
	eval_point: &[F],
) -> Result<ProjectedBivariateMeta, Error> {
	let n_vars = packed.inner().n_vars();
	let log_degree = packed.log_degree();
	let binary_tower_level = packed.inner().binary_tower_level();

	if log_degree > n_vars {
		return Err(OracleError::NotEnoughVarsForPacking { n_vars, log_degree }.into());
	}

	projected_bivariate_meta(oracles, packed.inner().id(), log_degree, eval_point, |_| {
		Ok(TowerBasis::new(log_degree, binary_tower_level)?)
	})
}

/// Takes in metadata object and creates a witness for a bivariate claim on tower basis.
///
/// `wf_eval_point` should be isomorphic to `eval_point` in `shifted_sumcheck_meta`.
pub fn packed_sumcheck_witness<'a, F: Field, PW: PackedField>(
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: ProjectedBivariateMeta,
	packed: &Packed<F>,
	wf_eval_point: &[PW::Scalar],
) -> Result<BivariateSumcheckWitness<'a, PW>, Error>
where
	PW::Scalar: TowerField,
{
	let log_degree = packed.log_degree();
	let binary_tower_level = packed.inner().binary_tower_level();

	projected_bivariate_witness(
		witness_index,
		memoized_queries,
		meta,
		wf_eval_point,
		|_projected_eval_point| {
			let tower_basis = TowerBasis::new(log_degree, binary_tower_level)?;
			Ok(tower_basis
				.multilinear_extension::<PW>()?
				.specialize_arc_dyn())
		},
	)
}

#[derive(Clone)]
pub struct NonSameQueryPcsClaimMeta<F> {
	projected_bivariate_meta: ProjectedBivariateMeta,
	eval_point: Vec<F>,
	eval: F,
}

/// Create sumchecks for committed evalcheck claims on differing eval points.
///
/// Each sumcheck instance is bivariate product of a column projection and equality indicator.
/// Common suffix is optimized out, degenerate zero variable sumchecks are not emitted, and
/// PCS claims are inserted directly into [`BatchCommittedEvalClaims`] instead.
#[allow(clippy::type_complexity)]
pub fn non_same_query_pcs_sumcheck_metas<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	committed_eval_claims: &[CommittedEvalClaim<F>],
	new_batch_committed_eval_claims: &mut BatchCommittedEvalClaims<F>,
) -> Result<Vec<NonSameQueryPcsClaimMeta<F>>, Error> {
	let common_suffix_len = if let Some((first, rest)) = committed_eval_claims.split_first() {
		let mut common_suffix = first.eval_point.as_slice();

		for claim in rest {
			let diff = claim
				.eval_point
				.iter()
				.rev()
				.zip(common_suffix.iter().rev())
				.position(|(a, b)| a != b);
			common_suffix = &common_suffix[diff.map_or(0, |diff| diff + 1)..];
		}

		common_suffix.len()
	} else {
		0
	};

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

pub fn non_same_query_pcs_sumcheck_claim<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	meta: NonSameQueryPcsClaimMeta<F>,
) -> Result<SumcheckClaim<F>, Error> {
	projected_bivariate_claim(oracles, meta.projected_bivariate_meta, meta.eval)
}

pub fn non_same_query_pcs_sumcheck_witness<'a, F, PW>(
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: NonSameQueryPcsClaimMeta<F>,
) -> Result<BivariateSumcheckWitness<'a, PW>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	let wf_eval_point = meta
		.eval_point
		.into_iter()
		.map(Into::into)
		.collect::<Vec<_>>();

	projected_bivariate_witness(
		witness_index,
		memoized_queries,
		meta.projected_bivariate_meta,
		&wf_eval_point,
		|projected_eval_point| {
			let eq_ind =
				EqIndPartialEval::new(projected_eval_point.len(), projected_eval_point.to_vec())?;

			Ok(eq_ind.multilinear_extension::<PW>()?.specialize_arc_dyn())
		},
	)
}

#[derive(Clone, Copy)]
pub struct ProjectedBivariateMeta {
	inner_id: OracleId,
	projected_id: Option<OracleId>,
	multiplier_id: OracleId,
	projected_n_vars: usize,
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
		oracles.add_transparent(Arc::new(multiplier_transparent_ctr(projected_eval_point)?))?;

	let meta = ProjectedBivariateMeta {
		inner_id,
		projected_id,
		multiplier_id,
		projected_n_vars,
	};

	Ok(meta)
}

/// Take in projected bivariate metadata and produce a sumcheck claim.
pub fn projected_bivariate_claim<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	meta: ProjectedBivariateMeta,
	eval: F,
) -> Result<SumcheckClaim<F>, Error> {
	let ProjectedBivariateMeta {
		projected_n_vars,
		multiplier_id,
		inner_id,
		projected_id,
	} = meta;

	let inner = oracles.oracle(projected_id.unwrap_or(inner_id));
	let multiplier = oracles.oracle(multiplier_id);

	let product =
		CompositePolyOracle::new(projected_n_vars, vec![inner, multiplier], BivariateProduct)?;

	let sumcheck_claim = SumcheckClaim {
		poly: product,
		sum: eval,
	};

	Ok(sumcheck_claim)
}

fn projected_bivariate_witness<'a, PW: PackedField>(
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: ProjectedBivariateMeta,
	wf_eval_point: &[PW::Scalar],
	multiplier_witness_ctr: impl FnOnce(&[PW::Scalar]) -> Result<MultilinearWitness<'a, PW>, Error>,
) -> Result<BivariateSumcheckWitness<'a, PW>, Error> {
	let ProjectedBivariateMeta {
		inner_id,
		projected_id,
		multiplier_id,
		projected_n_vars,
	} = meta;

	let inner_multilin = witness_index
		.get(inner_id)
		.ok_or(Error::InvalidWitness(inner_id))?;

	let (projected_inner_multilin, projected_eval_point) = if let Some(projected_id) = projected_id
	{
		let query = memoized_queries.full_query(&wf_eval_point[projected_n_vars..])?;

		let projected = inner_multilin
			.evaluate_partial_high(query)?
			.upcast_arc_dyn();
		witness_index.set(projected_id, projected.clone());

		(projected, &wf_eval_point[..projected_n_vars])
	} else {
		(inner_multilin.clone(), wf_eval_point)
	};

	let projected_n_vars = projected_eval_point.len();
	let multiplier_multilin = multiplier_witness_ctr(projected_eval_point)?;
	witness_index.set(multiplier_id, multiplier_multilin.clone());

	let witness = SumcheckWitness::new(
		projected_n_vars,
		BivariateProduct,
		vec![projected_inner_multilin, multiplier_multilin],
	)?;

	Ok(witness)
}

pub struct MemoizedQueries<P: PackedField> {
	memo: Vec<(Vec<P::Scalar>, MultilinearQuery<P>)>,
}

impl<P: PackedField> MemoizedQueries<P> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self { memo: Vec::new() }
	}

	pub fn full_query(&mut self, eval_point: &[P::Scalar]) -> Result<&MultilinearQuery<P>, Error> {
		if let Some(index) = self
			.memo
			.iter()
			.position(|(memo_eval_point, _)| memo_eval_point.as_slice() == eval_point)
		{
			let (_, ref query) = &self.memo[index];
			return Ok(query);
		}

		let wf_query = MultilinearQuery::with_full_query(eval_point)?;
		self.memo.push((eval_point.to_vec(), wf_query));

		let (_, ref query) = self.memo.last().expect("pushed query immediately above");
		Ok(query)
	}
}
