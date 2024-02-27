// Copyright 2024 Ulvetanna Inc.

use tracing::instrument;

use crate::{
	field::{Field, PackedField},
	oracle::{MultilinearPolyOracle, MultivariatePolyOracle, ProjectionVariant},
	polynomial::{multilinear_query::MultilinearQuery, MultilinearPoly},
};
use std::borrow::Borrow;

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckProof,
		EvalcheckWitness, PackedEvalClaim, ShiftedEvalClaim,
	},
};

#[instrument(skip_all, name = "evalcheck::prove")]
pub fn prove<F, P, M, BM>(
	evalcheck_witness: &EvalcheckWitness<P, M, BM>,
	evalcheck_claim: EvalcheckClaim<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + ?Sized,
	BM: Borrow<M>,
{
	let mut memoized_queries = Vec::new();
	prove_inner(
		evalcheck_witness,
		evalcheck_claim,
		batch_commited_eval_claims,
		shifted_eval_claims,
		packed_eval_claims,
		&mut memoized_queries,
	)
}

pub fn prove_inner<F, P, M, BM>(
	evalcheck_witness: &EvalcheckWitness<P, M, BM>,
	evalcheck_claim: EvalcheckClaim<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
	memoized_queries: &mut Vec<(Vec<F>, MultilinearQuery<P>)>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + ?Sized,
	BM: Borrow<M>,
{
	let EvalcheckClaim {
		poly,
		eval_point,
		eval,
		is_random_point,
	} = evalcheck_claim;

	let proof = match poly {
		MultivariatePolyOracle::Multilinear(multilinear) => {
			use MultilinearPolyOracle::*;
			match multilinear {
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
				Repeating {
					inner,
					log_count: _,
				} => {
					let n_vars = inner.n_vars();
					let inner_eval_point = eval_point[..n_vars].to_vec();
					let subclaim = EvalcheckClaim {
						poly: MultivariatePolyOracle::Multilinear(*inner),
						eval_point: inner_eval_point,
						eval,
						is_random_point,
					};

					let subproof = prove_inner(
						evalcheck_witness,
						subclaim,
						batch_commited_eval_claims,
						shifted_eval_claims,
						packed_eval_claims,
						memoized_queries,
					)?;

					EvalcheckProof::Repeating(Box::new(subproof))
				}
				Merged(_, _) => todo!(),
				Interleaved(_, _) => todo!(),
				Shifted(shifted) => {
					let subclaim = ShiftedEvalClaim {
						eval_point,
						eval,
						is_random_point,
						shifted,
					};

					shifted_eval_claims.push(subclaim);
					EvalcheckProof::Shifted
				}
				Packed(packed) => {
					let subclaim = PackedEvalClaim {
						eval_point,
						eval,
						is_random_point,
						packed,
					};

					packed_eval_claims.push(subclaim);
					EvalcheckProof::Packed
				}
				Projected(projected) => {
					let (inner, values) = (projected.inner(), projected.values());
					let new_eval_point = match projected.projection_variant() {
						ProjectionVariant::LastVars => {
							let mut new_eval_point = eval_point.clone();
							new_eval_point.extend(values);
							new_eval_point
						}
						ProjectionVariant::FirstVars => {
							values.iter().cloned().chain(eval_point).collect()
						}
					};

					let new_poly = MultivariatePolyOracle::Multilinear(*inner.clone());

					let subclaim = EvalcheckClaim {
						poly: new_poly,
						eval_point: new_eval_point,
						eval,
						is_random_point,
					};

					prove_inner(
						evalcheck_witness,
						subclaim,
						batch_commited_eval_claims,
						shifted_eval_claims,
						packed_eval_claims,
						memoized_queries,
					)?
				}
				LinearCombination(lin_com) => prove_composite(
					lin_com.polys().cloned(),
					eval_point,
					is_random_point,
					evalcheck_witness,
					batch_commited_eval_claims,
					shifted_eval_claims,
					packed_eval_claims,
					memoized_queries,
				)?,
			}
		}
		MultivariatePolyOracle::Composite(composite) => prove_composite(
			composite.inner_polys().into_iter(),
			eval_point,
			is_random_point,
			evalcheck_witness,
			batch_commited_eval_claims,
			shifted_eval_claims,
			packed_eval_claims,
			memoized_queries,
		)?,
	};

	Ok(proof)
}

#[allow(clippy::too_many_arguments)]
fn prove_composite<F, P, M, BM>(
	oracles: impl Iterator<Item = MultilinearPolyOracle<F>>,
	eval_point: Vec<F>,
	is_random_point: bool,
	witness: &EvalcheckWitness<P, M, BM>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
	memoized_queries: &mut Vec<(Vec<F>, MultilinearQuery<P>)>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + ?Sized,
	BM: Borrow<M>,
{
	let subproofs = oracles
		.map(|suboracle| {
			let eval_query = if let Some((_, ref query)) = memoized_queries
				.iter()
				.find(|(memoized_eval_point, _)| memoized_eval_point == &eval_point)
			{
				query
			} else {
				memoized_queries.push((
					eval_point.clone(),
					MultilinearQuery::<P>::with_full_query(&eval_point)?,
				));
				let (_, ref query) = memoized_queries
					.last()
					.expect("pushed query immediately above");
				query
			};

			let eval = witness.evaluate(&suboracle, eval_query)?;
			let subclaim = EvalcheckClaim {
				poly: MultivariatePolyOracle::Multilinear(suboracle),
				eval_point: eval_point.clone(),
				eval,
				is_random_point,
			};

			let proof = prove_inner(
				witness,
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

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	use crate::{
		field::{BinaryField128b, BinaryField1b, PackedBinaryField128x1b, PackedBinaryField4x32b},
		oracle::{CommittedBatch, CompositePolyOracle, LinearCombination},
		polynomial::{
			transparent::select_row::SelectRow, CompositionPoly, Error as PolynomialError,
			MultilinearExtension, MultivariatePoly,
		},
		protocols::evalcheck::verify::verify,
	};

	type EF = BinaryField128b;
	type PF = PackedBinaryField4x32b;

	#[derive(Debug)]
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
		crate::util::init_tracing();

		let mut rng = StdRng::seed_from_u64(0);

		let log_size = 8;
		let tower_level = 5;

		// random multilinear polys in BF32
		let multilins: Vec<MultilinearExtension<PF>> = repeat_with(|| {
			let evals = repeat_with(|| PF::random(&mut rng))
				.take(1 << (log_size - 2))
				.collect();
			MultilinearExtension::from_values(evals).unwrap()
		})
		.take(4)
		.collect();

		// eval point & eval in BF128
		let eval_point = repeat_with(|| <EF as PackedField>::random(&mut rng))
			.take(log_size)
			.collect::<Vec<_>>();

		let query = MultilinearQuery::with_full_query(&eval_point).unwrap();
		let batch_evals = multilins
			.iter()
			.map(|multilin| {
				<MultilinearExtension<PF> as MultilinearPoly<EF>>::evaluate(multilin, &query)
					.unwrap()
			})
			.collect::<Vec<_>>();

		let eval = batch_evals.iter().fold(EF::ONE, |acc, cur| acc * cur);

		let batches = [
			CommittedBatch {
				id: 0,
				round_id: 1,
				n_vars: log_size,
				n_polys: 2,
				tower_level,
			},
			CommittedBatch {
				id: 1,
				round_id: 1,
				n_vars: log_size,
				n_polys: 2,
				tower_level,
			},
		];

		let suboracles = vec![
			batches[0].oracle(0).unwrap(),
			batches[1].oracle(0).unwrap(),
			batches[0].oracle(1).unwrap(),
			batches[1].oracle(1).unwrap(),
		];

		let oracle = MultivariatePolyOracle::Composite(
			CompositePolyOracle::new(log_size, suboracles.clone(), Arc::new(QuadProduct)).unwrap(),
		);

		let claim = EvalcheckClaim {
			poly: oracle,
			eval_point: eval_point.clone(),
			eval,
			is_random_point: true,
		};

		let mut bcec_prove = BatchCommittedEvalClaims::new(&batches);
		let mut shifted_claims_prove = Vec::new();
		let mut packed_claims_prove = Vec::new();
		let proof = prove(
			&EvalcheckWitness::<EF, _, _>::new(suboracles.into_iter().zip(multilins).collect()),
			claim.clone(),
			&mut bcec_prove,
			&mut shifted_claims_prove,
			&mut packed_claims_prove,
		)
		.unwrap();

		let mut bcec_verify = BatchCommittedEvalClaims::new(&batches);
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

		let select_row1_oracle = select_row1.multilinear_poly_oracle();
		let select_row2_oracle = select_row2.multilinear_poly_oracle();
		let select_row3_oracle = select_row3.multilinear_poly_oracle();

		let lin_com = MultilinearPolyOracle::LinearCombination(
			LinearCombination::new(
				n_vars,
				[
					(select_row1_oracle.clone(), EF::new(2)),
					(select_row2_oracle.clone(), EF::new(3)),
					(select_row3_oracle.clone(), EF::new(4)),
				],
			)
			.unwrap(),
		);

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
				MultilinearPoly::<BinaryField1b>::evaluate_on_hypercube(&select_row1_witness, i)
					.unwrap() * EF::new(2)
					+ MultilinearPoly::<BinaryField1b>::evaluate_on_hypercube(
						&select_row2_witness,
						i,
					)
					.unwrap() * EF::new(3)
					+ MultilinearPoly::<BinaryField1b>::evaluate_on_hypercube(
						&select_row3_witness,
						i,
					)
					.unwrap() * EF::new(4)
			})
			.collect();
		let lin_com_witness = MultilinearExtension::from_values(lin_com_values).unwrap();

		// Make the claim a composite oracle over a linear combination, in order to test the case
		// of requiring nested composite evalcheck proofs.
		let claim_oracle = MultivariatePolyOracle::Multilinear(lin_com.clone());
		let claim_oracle = MultivariatePolyOracle::Composite(claim_oracle.into_composite());

		let claim = EvalcheckClaim {
			poly: claim_oracle,
			eval_point,
			eval,
			is_random_point: true,
		};

		let witness = EvalcheckWitness::<_, dyn MultilinearPoly<EF>, _>::new(vec![
			(select_row1_oracle, Arc::new(select_row1_witness) as Arc<dyn MultilinearPoly<EF>>),
			(select_row2_oracle, Arc::new(select_row2_witness) as Arc<dyn MultilinearPoly<EF>>),
			(select_row3_oracle, Arc::new(select_row3_witness) as Arc<dyn MultilinearPoly<EF>>),
			(lin_com, Arc::new(lin_com_witness) as Arc<dyn MultilinearPoly<EF>>),
		]);

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

		let select_row = SelectRow::new(n_vars, 11).unwrap();
		let repeating = MultilinearPolyOracle::repeating(select_row.multilinear_poly_oracle(), 2);

		let mut rng = StdRng::seed_from_u64(0);
		let eval_point = repeat_with(|| <EF as Field>::random(&mut rng))
			.take(n_vars + 2)
			.collect::<Vec<_>>();

		let eval = select_row.evaluate(&eval_point[..n_vars]).unwrap();

		let claim = EvalcheckClaim {
			poly: repeating.into(),
			eval_point,
			eval,
			is_random_point: true,
		};

		let mut batch_claims = BatchCommittedEvalClaims::new(&[]);
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();
		let proof = prove(
			&EvalcheckWitness::<_, dyn MultilinearPoly<EF>, Arc<_>>::new(vec![]),
			claim.clone(),
			&mut batch_claims,
			&mut shifted_claims_verify,
			&mut packed_claims_verify,
		)
		.unwrap();
		assert_matches!(proof, EvalcheckProof::Repeating(_));

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
