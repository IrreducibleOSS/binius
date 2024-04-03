// Copyright 2024 Ulvetanna Inc.

use tracing::instrument;

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckMultilinearClaim,
		EvalcheckProof, PackedEvalClaim, ShiftedEvalClaim,
	},
};
use crate::{
	oracle::{MultilinearPolyOracle, ProjectionVariant},
	polynomial::multilinear_query::MultilinearQuery,
	witness::MultilinearWitnessIndex,
};
use binius_field::{Field, PackedField};

#[instrument(skip_all, name = "evalcheck::prove")]
pub fn prove<F, PW, C>(
	witness_index: &MultilinearWitnessIndex<'_, PW>,
	evalcheck_claim: EvalcheckClaim<F, C>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	let mut memoized_queries = Vec::new();

	let EvalcheckClaim {
		poly: composite,
		eval_point,
		is_random_point,
		..
	} = evalcheck_claim;

	prove_composite(
		witness_index,
		composite.inner_polys().into_iter(),
		eval_point,
		is_random_point,
		batch_commited_eval_claims,
		shifted_eval_claims,
		packed_eval_claims,
		&mut memoized_queries,
	)
}

#[allow(clippy::too_many_arguments)]
fn prove_composite<F, PW>(
	witness_index: &MultilinearWitnessIndex<'_, PW>,
	oracles: impl Iterator<Item = MultilinearPolyOracle<F>>,
	eval_point: Vec<F>,
	is_random_point: bool,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
	memoized_queries: &mut Vec<(Vec<F>, MultilinearQuery<PW>)>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	let subproofs = oracles
		.map(|suboracle| {
			let eval_query = if let Some((_, ref query)) = memoized_queries
				.iter()
				.find(|(memoized_eval_point, _)| memoized_eval_point == &eval_point)
			{
				query
			} else {
				let wf_eval_point = eval_point
					.iter()
					.copied()
					.map(Into::into)
					.collect::<Vec<_>>();

				memoized_queries.push((
					eval_point.clone(),
					MultilinearQuery::<PW>::with_full_query(&wf_eval_point)?,
				));

				let (_, ref query) = memoized_queries
					.last()
					.expect("pushed query immediately above");
				query
			};

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
				witness_index,
				subclaim,
				batch_commited_eval_claims,
				shifted_eval_claims,
				packed_eval_claims,
				memoized_queries,
			)?;

			Ok((eval, proof))
		})
		.collect::<Result<_, Error>>()?;

	Ok(EvalcheckProof::Composite { subproofs })
}

pub fn prove_multilinear<F, PW>(
	witness_index: &MultilinearWitnessIndex<'_, PW>,
	evalcheck_claim: EvalcheckMultilinearClaim<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
	memoized_queries: &mut Vec<(Vec<F>, MultilinearQuery<PW>)>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field + From<PW::Scalar>,
	PW: PackedField,
	PW::Scalar: From<F>,
{
	let EvalcheckMultilinearClaim {
		poly: multilinear,
		eval_point,
		eval,
		is_random_point,
	} = evalcheck_claim;

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

			batch_commited_eval_claims.insert(subclaim)?;
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
				witness_index,
				subclaim,
				batch_commited_eval_claims,
				shifted_eval_claims,
				packed_eval_claims,
				memoized_queries,
			)?;

			EvalcheckProof::Repeating(Box::new(subproof))
		}

		Merged(..) => todo!(),
		Interleaved(..) => todo!(),

		Shifted(_id, shifted) => {
			let subclaim = ShiftedEvalClaim {
				eval_point,
				eval,
				is_random_point,
				shifted,
			};

			shifted_eval_claims.push(subclaim);
			EvalcheckProof::Shifted
		}

		Packed(_id, packed) => {
			let subclaim = PackedEvalClaim {
				eval_point,
				eval,
				is_random_point,
				packed,
			};

			packed_eval_claims.push(subclaim);
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
				witness_index,
				subclaim,
				batch_commited_eval_claims,
				shifted_eval_claims,
				packed_eval_claims,
				memoized_queries,
			)?
		}

		LinearCombination(_id, lin_com) => prove_composite(
			witness_index,
			lin_com.polys().cloned(),
			eval_point,
			is_random_point,
			batch_commited_eval_claims,
			shifted_eval_claims,
			packed_eval_claims,
			memoized_queries,
		)?,
	};

	Ok(proof)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
		polynomial::{
			transparent::select_row::SelectRow, CompositionPoly, Error as PolynomialError,
			MultilinearExtension, MultilinearPoly, MultivariatePoly,
		},
		protocols::evalcheck::verify::verify,
	};
	use assert_matches::assert_matches;
	use binius_field::{BinaryField128b, PackedBinaryField128x1b, PackedBinaryField4x32b};
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	type EF = BinaryField128b;
	type PF = PackedBinaryField4x32b;

	#[derive(Clone, Debug)]
	struct QuadProduct;

	impl CompositionPoly<EF> for QuadProduct {
		fn n_vars(&self) -> usize {
			4
		}

		fn degree(&self) -> usize {
			4
		}

		fn evaluate(&self, query: &[EF]) -> Result<EF, PolynomialError> {
			self.evaluate_packed(query)
		}

		fn evaluate_packed(&self, query: &[EF]) -> Result<EF, PolynomialError> {
			if query.len() != 4 {
				return Err(PolynomialError::IncorrectQuerySize { expected: 4 });
			}
			let (a, b, c, d) = (query[0], query[1], query[2], query[3]);
			Ok(a * b * c * d)
		}

		fn binary_tower_level(&self) -> usize {
			0
		}
	}

	#[test]
	fn test_prove_verify_interaction() {
		crate::util::tracing::init_tracing();

		let mut rng = StdRng::seed_from_u64(0);

		let log_size = 8;
		let tower_level = 5;

		// random multilinear polys in BF32
		let multilins = repeat_with(|| {
			let evals = repeat_with(|| PF::random(&mut rng))
				.take(1 << (log_size - 2))
				.collect();
			MultilinearExtension::from_values(evals)
				.unwrap()
				.specialize::<EF>()
		})
		.take(4)
		.collect::<Vec<_>>();

		// eval point & eval in BF128
		let eval_point = repeat_with(|| <EF as PackedField>::random(&mut rng))
			.take(log_size)
			.collect::<Vec<_>>();

		let query = MultilinearQuery::with_full_query(&eval_point).unwrap();
		let batch_evals = multilins
			.iter()
			.map(|multilin| multilin.evaluate(&query).unwrap())
			.collect::<Vec<_>>();

		let eval = batch_evals.iter().fold(EF::ONE, |acc, cur| acc * cur);

		let mut oracles = MultilinearOracleSet::new();
		let batch_0_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 1,
			n_vars: log_size,
			n_polys: 2,
			tower_level,
		});
		let batch_1_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 1,
			n_vars: log_size,
			n_polys: 2,
			tower_level,
		});

		let suboracles = vec![
			oracles.committed_oracle(CommittedId {
				batch_id: batch_0_id,
				index: 0,
			}),
			oracles.committed_oracle(CommittedId {
				batch_id: batch_1_id,
				index: 0,
			}),
			oracles.committed_oracle(CommittedId {
				batch_id: batch_0_id,
				index: 1,
			}),
			oracles.committed_oracle(CommittedId {
				batch_id: batch_1_id,
				index: 1,
			}),
		];

		let mut witness_index = MultilinearWitnessIndex::new();
		for (oracle, multilin) in suboracles.iter().zip(multilins.into_iter()) {
			witness_index.set(oracle.id(), multilin.upcast_arc_dyn());
		}

		let oracle = CompositePolyOracle::new(log_size, suboracles.clone(), QuadProduct).unwrap();

		let claim = EvalcheckClaim {
			poly: oracle,
			eval_point: eval_point.clone(),
			eval,
			is_random_point: true,
		};

		let mut bcec_prove = BatchCommittedEvalClaims::new(&oracles.committed_batches());
		let mut shifted_claims_prove = Vec::new();
		let mut packed_claims_prove = Vec::new();
		let proof = prove(
			&witness_index,
			claim.clone(),
			&mut bcec_prove,
			&mut shifted_claims_prove,
			&mut packed_claims_prove,
		)
		.unwrap();

		let mut bcec_verify = BatchCommittedEvalClaims::new(&oracles.committed_batches());
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();
		verify(
			claim,
			proof,
			&mut bcec_verify,
			&mut shifted_claims_verify,
			&mut packed_claims_verify,
		)
		.unwrap();

		let prove_batch0 = bcec_prove
			.try_extract_same_query_pcs_claim(0)
			.unwrap()
			.unwrap();
		let verify_batch0 = bcec_verify
			.try_extract_same_query_pcs_claim(0)
			.unwrap()
			.unwrap();

		assert_eq!(prove_batch0.eval_point, eval_point);
		assert_eq!(verify_batch0.eval_point, eval_point);

		assert_eq!(prove_batch0.evals, [batch_evals[0], batch_evals[2]]);
		assert_eq!(verify_batch0.evals, [batch_evals[0], batch_evals[2]]);

		let prove_batch1 = bcec_prove
			.try_extract_same_query_pcs_claim(1)
			.unwrap()
			.unwrap();
		let verify_batch1 = bcec_verify
			.try_extract_same_query_pcs_claim(1)
			.unwrap()
			.unwrap();

		assert_eq!(prove_batch1.eval_point, eval_point);
		assert_eq!(verify_batch1.eval_point, eval_point);

		assert_eq!(prove_batch1.evals, [batch_evals[1], batch_evals[3]]);
		assert_eq!(verify_batch1.evals, [batch_evals[1], batch_evals[3]]);
	}

	#[test]
	fn test_evalcheck_linear_combination() {
		let n_vars = 8;

		let select_row1 = SelectRow::new(n_vars, 0).unwrap();
		let select_row2 = SelectRow::new(n_vars, 5).unwrap();
		let select_row3 = SelectRow::new(n_vars, 10).unwrap();

		let mut oracles = MultilinearOracleSet::new();

		let select_row1_oracle_id = oracles
			.add_transparent(Arc::new(select_row1.clone()), 0)
			.unwrap();
		let select_row2_oracle_id = oracles
			.add_transparent(Arc::new(select_row2.clone()), 0)
			.unwrap();
		let select_row3_oracle_id = oracles
			.add_transparent(Arc::new(select_row3.clone()), 0)
			.unwrap();

		let lin_com_id = oracles
			.add_linear_combination(
				n_vars,
				[
					(select_row1_oracle_id, EF::new(2)),
					(select_row2_oracle_id, EF::new(3)),
					(select_row3_oracle_id, EF::new(4)),
				],
			)
			.unwrap();
		let lin_com = oracles.oracle(lin_com_id);

		let mut rng = StdRng::seed_from_u64(0);
		let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let eval = select_row1.evaluate(&eval_point).unwrap() * EF::new(2)
			+ select_row2.evaluate(&eval_point).unwrap() * EF::new(3)
			+ select_row3.evaluate(&eval_point).unwrap() * EF::new(4);

		let select_row1_witness = select_row1
			.multilinear_extension::<PackedBinaryField128x1b>()
			.unwrap();
		let select_row2_witness = select_row2
			.multilinear_extension::<PackedBinaryField128x1b>()
			.unwrap();
		let select_row3_witness = select_row3
			.multilinear_extension::<PackedBinaryField128x1b>()
			.unwrap();

		let lin_com_values = (0..1 << n_vars)
			.map(|i| {
				select_row1_witness.evaluate_on_hypercube(i).unwrap() * EF::new(2)
					+ select_row2_witness.evaluate_on_hypercube(i).unwrap() * EF::new(3)
					+ select_row3_witness.evaluate_on_hypercube(i).unwrap() * EF::new(4)
			})
			.collect();
		let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

		// Make the claim a composite oracle over a linear combination, in order to test the case
		// of requiring nested composite evalcheck proofs.
		let claim_oracle = lin_com.clone();
		let claim_oracle = claim_oracle.into_composite();

		let claim = EvalcheckClaim {
			poly: claim_oracle,
			eval_point,
			eval,
			is_random_point: true,
		};

		let mut witness = MultilinearWitnessIndex::<EF>::new();
		witness.set(select_row1_oracle_id, select_row1_witness.specialize_arc_dyn());
		witness.set(select_row2_oracle_id, select_row2_witness.specialize_arc_dyn());
		witness.set(select_row3_oracle_id, select_row3_witness.specialize_arc_dyn());
		witness.set(lin_com_id, lin_com_witness.specialize_arc_dyn());

		let mut batch_claims = BatchCommittedEvalClaims::new(&[]);
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();
		let proof = prove(
			&witness,
			claim.clone(),
			&mut batch_claims,
			&mut shifted_claims_verify,
			&mut packed_claims_verify,
		)
		.unwrap();

		let mut batch_claims = BatchCommittedEvalClaims::new(&[]);
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();
		verify(
			claim,
			proof,
			&mut batch_claims,
			&mut shifted_claims_verify,
			&mut packed_claims_verify,
		)
		.unwrap();
	}

	#[test]
	fn test_evalcheck_repeating() {
		let n_vars = 7;
		let row_id = 11;

		let mut oracles = MultilinearOracleSet::new();

		let select_row = SelectRow::new(n_vars, row_id).unwrap();
		let select_row_oracle_id = oracles
			.add_transparent(Arc::new(select_row.clone()), 0)
			.unwrap();

		let select_row_subwitness = select_row
			.multilinear_extension::<PackedBinaryField128x1b>()
			.unwrap();
		let repeated_values = (0..4)
			.flat_map(|_| select_row_subwitness.evals().iter().copied())
			.collect::<Vec<_>>();

		let select_row_witness = MultilinearExtension::from_values(repeated_values)
			.unwrap()
			.specialize_arc_dyn();

		let repeating_id = oracles.add_repeating(select_row_oracle_id, 2).unwrap();
		let repeating = oracles.oracle(repeating_id);

		let mut witness_index = MultilinearWitnessIndex::<EF>::new();

		let mut rng = StdRng::seed_from_u64(0);
		let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
			.take(n_vars + 2)
			.collect::<Vec<_>>();

		let eval = select_row.evaluate(&eval_point[..n_vars]).unwrap();

		let claim = EvalcheckClaim {
			poly: repeating.clone().into_composite(),
			eval_point,
			eval,
			is_random_point: true,
		};

		let mut batch_claims = BatchCommittedEvalClaims::new(&[]);
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();

		witness_index.set(repeating_id, select_row_witness);

		let proof = prove(
			&witness_index,
			claim.clone(),
			&mut batch_claims,
			&mut shifted_claims_verify,
			&mut packed_claims_verify,
		)
		.unwrap();

		if let EvalcheckProof::Composite { ref subproofs } = proof {
			assert_eq!(subproofs.len(), 1);
			assert_matches!(subproofs[0].1, EvalcheckProof::Repeating(..));
		} else {
			panic!("Proof should be Composite.");
		}

		let mut batch_claims = BatchCommittedEvalClaims::new(&[]);
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();
		verify(
			claim,
			proof,
			&mut batch_claims,
			&mut shifted_claims_verify,
			&mut packed_claims_verify,
		)
		.unwrap();
	}
}
