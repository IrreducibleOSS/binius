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
		ProjectionVariant, ShiftVariant, Shifted,
	},
	polynomial::{
		composition::BivariateProduct,
		transparent::{
			eq_ind::EqIndPartialEval, shift_ind::ShiftIndPartialEval, tower_basis::TowerBasis,
		},
		MultilinearComposite, MultilinearPoly, MultilinearQuery, MultivariatePoly,
	},
	protocols::sumcheck::SumcheckClaim,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::PackScalar, underlier::WithUnderlier, Field, PackedField,
	PackedFieldIndexable, TowerField,
};
use std::sync::Arc;

// type aliases for bivariate claims/witnesses and their pairs to shorten type signatures
pub type BivariateSumcheck<'a, F, PW> = (SumcheckClaim<F>, BivariateSumcheckWitness<'a, PW>);
pub type BivariateSumcheckWitness<'a, PW> =
	MultilinearComposite<PW, BivariateProduct, MultilinearWitness<'a, PW>>;

// type alias for the simple linear map used to deduplicate transparent polynomials
pub type MemoizedTransparentPolynomials<K> = Vec<(K, OracleId)>;

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
	shift_ind_memo: Option<&mut MemoizedTransparentPolynomials<(usize, ShiftVariant, Vec<F>)>>,
) -> Result<ProjectedBivariateMeta, Error> {
	projected_bivariate_meta(
		oracles,
		shifted.inner().id(),
		shifted.block_size(),
		eval_point,
		shift_ind_memo,
		|projected_eval_point| {
			(shifted.shift_offset(), shifted.shift_variant(), projected_eval_point.to_vec())
		},
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
pub fn shifted_sumcheck_witness<'a, F, PW>(
	witness_index: &mut MultilinearExtensionIndex<'a, PW::Underlier, PW::Scalar>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: ProjectedBivariateMeta,
	shifted: &Shifted<F>,
	wf_eval_point: &[PW::Scalar],
) -> Result<BivariateSumcheckWitness<'a, PW>, Error>
where
	F: Field,
	PW: PackedFieldIndexable + WithUnderlier,
	PW::Scalar: TowerField,
	PW::Underlier: PackScalar<PW::Scalar, Packed = PW>,
{
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

	// NB. projected_n_vars = 0 because eval_point length is log_degree less than inner n_vars
	projected_bivariate_meta(
		oracles,
		packed.inner().id(),
		0,
		eval_point,
		None,
		|_| (),
		|_| Ok(TowerBasis::new(log_degree, binary_tower_level)?),
	)
}

/// Takes in metadata object and creates a witness for a bivariate claim on tower basis.
///
/// `wf_eval_point` should be isomorphic to `eval_point` in `shifted_sumcheck_meta`.
pub fn packed_sumcheck_witness<'a, F, PW>(
	witness_index: &mut MultilinearExtensionIndex<'a, PW::Underlier, PW::Scalar>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: ProjectedBivariateMeta,
	packed: &Packed<F>,
	wf_eval_point: &[PW::Scalar],
) -> Result<BivariateSumcheckWitness<'a, PW>, Error>
where
	F: Field,
	PW: PackedFieldIndexable + WithUnderlier,
	PW::Scalar: TowerField,
	PW::Underlier: PackScalar<PW::Scalar, Packed = PW>,
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
	mut eq_ind_memo: Option<&mut MemoizedTransparentPolynomials<Vec<F>>>,
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
			eq_ind_memo.as_deref_mut(),
			|projected_eval_point| projected_eval_point.to_vec(),
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

pub fn non_same_query_pcs_sumcheck_claim<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	meta: NonSameQueryPcsClaimMeta<F>,
) -> Result<SumcheckClaim<F>, Error> {
	projected_bivariate_claim(oracles, meta.projected_bivariate_meta, meta.eval)
}

pub fn non_same_query_pcs_sumcheck_witness<'a, F, PW>(
	witness_index: &mut MultilinearExtensionIndex<'a, PW::Underlier, PW::Scalar>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: NonSameQueryPcsClaimMeta<F>,
) -> Result<BivariateSumcheckWitness<'a, PW>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField + WithUnderlier,
	PW::Scalar: From<F> + TowerField,
	PW::Underlier: PackScalar<PW::Scalar, Packed = PW>,
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

fn projected_bivariate_meta<F: TowerField, T: MultivariatePoly<F> + 'static, K: PartialEq>(
	oracles: &mut MultilinearOracleSet<F>,
	inner_id: OracleId,
	projected_n_vars: usize,
	eval_point: &[F],
	mut multiplier_memo: Option<&mut MemoizedTransparentPolynomials<K>>,
	multiplier_memo_key: impl FnOnce(&[F]) -> K,
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
	let memo_key = multiplier_memo_key(projected_eval_point);

	let opt_multiplier_id = multiplier_memo
		.as_ref()
		.and_then(|memo| {
			memo.iter()
				.find(|(other_memo_key, _)| other_memo_key == &memo_key)
		})
		.map(|(_, oracle_id)| *oracle_id);

	let multiplier_id = if let Some(multiplier_id) = opt_multiplier_id {
		multiplier_id
	} else {
		let multiplier_id =
			oracles.add_transparent(multiplier_transparent_ctr(projected_eval_point)?)?;

		if let Some(memo) = multiplier_memo.as_mut() {
			memo.push((memo_key, multiplier_id));
		}
		multiplier_id
	};

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
		multiplier_id,
		inner_id,
		projected_id,
		..
	} = meta;

	let inner = oracles.oracle(projected_id.unwrap_or(inner_id));
	let multiplier = oracles.oracle(multiplier_id);

	let product =
		CompositePolyOracle::new(multiplier.n_vars(), vec![inner, multiplier], BivariateProduct)?;

	let sumcheck_claim = SumcheckClaim {
		poly: product,
		sum: eval,
	};

	Ok(sumcheck_claim)
}

fn projected_bivariate_witness<'a, PW>(
	witness_index: &mut MultilinearExtensionIndex<'a, PW::Underlier, PW::Scalar>,
	memoized_queries: &mut MemoizedQueries<PW>,
	meta: ProjectedBivariateMeta,
	wf_eval_point: &[PW::Scalar],
	multiplier_witness_ctr: impl FnOnce(&[PW::Scalar]) -> Result<MultilinearWitness<'a, PW>, Error>,
) -> Result<BivariateSumcheckWitness<'a, PW>, Error>
where
	PW: PackedField + WithUnderlier,
	PW::Scalar: TowerField,
	PW::Underlier: PackScalar<PW::Scalar, Packed = PW>,
{
	let ProjectedBivariateMeta {
		inner_id,
		projected_id,
		multiplier_id,
		projected_n_vars,
	} = meta;

	let inner_multilin = witness_index.get_multilin_poly(inner_id)?;

	let (projected_inner_multilin, projected_eval_point) = if let Some(projected_id) = projected_id
	{
		let query = memoized_queries.full_query(&wf_eval_point[projected_n_vars..])?;
		// upcast_arc_dyn() doesn't compile, but an explicit Arc::new() does compile. Beats me.
		let projected: Arc<dyn MultilinearPoly<PW> + Send + Sync> =
			Arc::new(inner_multilin.evaluate_partial_high(query)?);
		witness_index.update_multilin_poly(vec![(projected_id, projected.clone())])?;

		(projected, &wf_eval_point[..projected_n_vars])
	} else {
		(inner_multilin, wf_eval_point)
	};

	if !witness_index.has(multiplier_id) {
		witness_index.update_multilin_poly(vec![(
			multiplier_id,
			multiplier_witness_ctr(projected_eval_point)?,
		)])?;
	}

	let multiplier_multilin = witness_index
		.get_multilin_poly(multiplier_id)
		.expect("multilinear forcibly created if absent")
		.clone();

	let witness = MultilinearComposite::new(
		multiplier_multilin.n_vars(),
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
