// Copyright 2024 Ulvetanna Inc.

use tracing::instrument;

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckMultilinearClaim,
		EvalcheckProof,
	},
	subclaims::{
		packed_sumcheck_meta, packed_sumcheck_witness, projected_bivariate_claim,
		shifted_sumcheck_meta, shifted_sumcheck_witness, BivariateSumcheck, MemoizedQueries,
	},
};
use crate::{
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, ProjectionVariant},
	witness::MultilinearWitnessIndex,
};
use binius_field::{PackedField, TowerField};

/// Prove an evalcheck claim.
///
/// Given a [`MultilinearOracleSet`] indexing into given[`MultilinearWitnessIndex`], we
/// prove an [`EvalcheckClaim`] (stating that given composite `poly` equals `eval` at `eval_point`)
/// by recursively processing each of the multilinears in the composition. This way the evalcheck claim
/// gets transformed into an [`EvalcheckProof`] and a new set of claims on
///   * PCS openings (which get inserted into [`BatchCommittedEvalClaims`] accumulator)
///   * New sumcheck instances that need to be proven in subsequent rounds (those get appended to `new_sumchecks`)
///
/// All of the `new_sumchecks` instances follow the same pattern:
///  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
///  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
///  * other multilin is a projection of one of the evalcheck claim multilins to its first variables
#[instrument(skip_all, name = "evalcheck::prove")]
pub fn prove<'a, F, PW, C>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	evalcheck_claim: EvalcheckClaim<F, C>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	new_sumchecks: &mut Vec<BivariateSumcheck<'a, F, PW>>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: TowerField + From<F>,
{
	let mut memoized_queries = MemoizedQueries::new();

	let EvalcheckClaim {
		poly: composite,
		eval_point,
		is_random_point,
		..
	} = evalcheck_claim;

	prove_composite(
		oracles,
		witness_index,
		composite.inner_polys().into_iter(),
		eval_point,
		is_random_point,
		batch_commited_eval_claims,
		new_sumchecks,
		&mut memoized_queries,
	)
}

#[allow(clippy::too_many_arguments)]
fn prove_composite<'a, F, PW>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	multilin_oracles: impl Iterator<Item = MultilinearPolyOracle<F>>,
	eval_point: Vec<F>,
	is_random_point: bool,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	new_sumchecks: &mut Vec<BivariateSumcheck<'a, F, PW>>,
	memoized_queries: &mut MemoizedQueries<PW>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: TowerField + From<F>,
{
	let wf_eval_point = eval_point
		.iter()
		.copied()
		.map(Into::into)
		.collect::<Vec<_>>();

	let subproofs = multilin_oracles
		.map(|suboracle| {
			let eval_query = memoized_queries.full_query(&wf_eval_point)?;

			let witness_poly = witness_index
				.get(suboracle.id())
				.ok_or(Error::InvalidWitness(suboracle.id()))?;

			let eval = witness_poly.evaluate(eval_query)?.into();

			let subclaim = EvalcheckMultilinearClaim {
				poly: suboracle,
				eval_point: eval_point.clone(),
				eval,
				is_random_point,
			};

			let proof = prove_multilinear(
				oracles,
				witness_index,
				subclaim,
				batch_commited_eval_claims,
				new_sumchecks,
				memoized_queries,
			)?;

			Ok((eval, proof))
		})
		.collect::<Result<_, Error>>()?;

	Ok(EvalcheckProof::Composite { subproofs })
}

fn prove_multilinear<'a, F, PW>(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearWitnessIndex<'a, PW>,
	evalcheck_claim: EvalcheckMultilinearClaim<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	new_sumchecks: &mut Vec<BivariateSumcheck<'a, F, PW>>,
	memoized_queries: &mut MemoizedQueries<PW>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: TowerField + From<F>,
{
	let EvalcheckMultilinearClaim {
		poly: multilinear,
		eval_point,
		eval,
		is_random_point,
	} = evalcheck_claim;

	let wf_eval_point = eval_point
		.iter()
		.copied()
		.map(Into::into)
		.collect::<Vec<_>>();

	use MultilinearPolyOracle::*;

	let proof = match multilinear {
		Transparent { .. } => EvalcheckProof::Transparent,

		Committed { id, .. } => {
			let subclaim = CommittedEvalClaim {
				id,
				eval_point,
				eval,
				is_random_point,
			};

			batch_commited_eval_claims.insert(subclaim);
			EvalcheckProof::Committed
		}

		Repeating { inner, .. } => {
			let n_vars = inner.n_vars();
			let inner_eval_point = eval_point[..n_vars].to_vec();
			let subclaim = EvalcheckMultilinearClaim {
				poly: *inner,
				eval_point: inner_eval_point,
				eval,
				is_random_point,
			};

			let subproof = prove_multilinear(
				oracles,
				witness_index,
				subclaim,
				batch_commited_eval_claims,
				new_sumchecks,
				memoized_queries,
			)?;

			EvalcheckProof::Repeating(Box::new(subproof))
		}

		Merged(..) => todo!(),
		Interleaved(..) => todo!(),

		Shifted(_id, shifted) => {
			let meta = shifted_sumcheck_meta(oracles, &shifted, eval_point.as_slice())?;
			let sumcheck_claim = projected_bivariate_claim(oracles, meta, eval)?;
			let sumcheck_witness = shifted_sumcheck_witness(
				witness_index,
				memoized_queries,
				meta,
				&shifted,
				&wf_eval_point,
			)?;

			new_sumchecks.push((sumcheck_claim, sumcheck_witness));
			EvalcheckProof::Shifted
		}

		Packed(_id, packed) => {
			let meta = packed_sumcheck_meta(oracles, &packed, eval_point.as_slice())?;
			let sumcheck_claim = projected_bivariate_claim(oracles, meta, eval)?;
			let sumcheck_witness = packed_sumcheck_witness(
				witness_index,
				memoized_queries,
				meta,
				&packed,
				&wf_eval_point,
			)?;
			new_sumchecks.push((sumcheck_claim, sumcheck_witness));
			EvalcheckProof::Packed
		}

		Projected(_id, projected) => {
			let (inner, values) = (projected.inner(), projected.values());
			let new_eval_point = match projected.projection_variant() {
				ProjectionVariant::LastVars => {
					let mut new_eval_point = eval_point.clone();
					new_eval_point.extend(values);
					new_eval_point
				}
				ProjectionVariant::FirstVars => values.iter().cloned().chain(eval_point).collect(),
			};

			let new_poly = *inner.clone();

			let subclaim = EvalcheckMultilinearClaim {
				poly: new_poly,
				eval_point: new_eval_point,
				eval,
				is_random_point,
			};

			prove_multilinear(
				oracles,
				witness_index,
				subclaim,
				batch_commited_eval_claims,
				new_sumchecks,
				memoized_queries,
			)?
		}

		LinearCombination(_id, lin_com) => prove_composite(
			oracles,
			witness_index,
			lin_com.polys().cloned(),
			eval_point,
			is_random_point,
			batch_commited_eval_claims,
			new_sumchecks,
			memoized_queries,
		)?,
	};

	Ok(proof)
}
