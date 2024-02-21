// Copyright 2024 Ulvetanna Inc.

use tracing::instrument;

use crate::{
	field::Field,
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
pub fn prove<F, M, BM>(
	evalcheck_witness: EvalcheckWitness<F, M, BM>,
	evalcheck_claim: EvalcheckClaim<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
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
				Transparent { .. } => {
					match evalcheck_witness {
						EvalcheckWitness::Multilinear => {}
						_ => return Err(Error::InvalidWitness),
					};
					EvalcheckProof::Transparent
				}
				Committed { id, .. } => {
					match evalcheck_witness {
						EvalcheckWitness::Multilinear => {}
						_ => return Err(Error::InvalidWitness),
					};

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
					let subclaim = EvalcheckClaim {
						poly: MultivariatePolyOracle::Multilinear(*inner),
						eval_point: eval_point[..n_vars].to_vec(),
						eval,
						is_random_point,
					};

					let subproof = prove(
						evalcheck_witness,
						subclaim,
						batch_commited_eval_claims,
						shifted_eval_claims,
						packed_eval_claims,
					)?;

					EvalcheckProof::Repeating(Box::new(subproof))
				}
				Merged(_, _) => todo!(),
				Interleaved(_, _) => todo!(),
				Shifted(shifted) => {
					match evalcheck_witness {
						EvalcheckWitness::Multilinear => {}
						_ => return Err(Error::InvalidWitness),
					};

					let subclaim = ShiftedEvalClaim {
						poly: *shifted.inner().clone(),
						eval_point,
						eval,
						is_random_point,
						shifted,
					};

					shifted_eval_claims.push(subclaim);
					EvalcheckProof::Shifted
				}
				Packed(packed) => {
					match evalcheck_witness {
						EvalcheckWitness::Multilinear => {}
						_ => return Err(Error::InvalidWitness),
					};

					let subclaim = PackedEvalClaim {
						poly: *packed.inner().clone(),
						eval_point,
						eval,
						is_random_point,
						packed,
					};

					packed_eval_claims.push(subclaim);
					EvalcheckProof::Packed
				}
				Projected(projected) => {
					match evalcheck_witness {
						EvalcheckWitness::Multilinear => {}
						_ => return Err(Error::InvalidWitness),
					};

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

					prove(
						evalcheck_witness,
						subclaim,
						batch_commited_eval_claims,
						shifted_eval_claims,
						packed_eval_claims,
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
		)?,
	};

	Ok(proof)
}

fn prove_composite<F, M, BM>(
	oracles: impl Iterator<Item = MultilinearPolyOracle<F>>,
	eval_point: Vec<F>,
	is_random_point: bool,
	witness: EvalcheckWitness<F, M, BM>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
	packed_eval_claims: &mut Vec<PackedEvalClaim<F>>,
) -> Result<EvalcheckProof<F>, Error>
where
	F: Field,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	let witness = match witness {
		EvalcheckWitness::Composite(witness) => witness,
		_ => return Err(Error::InvalidWitness),
	};

	let query = MultilinearQuery::with_full_query(&eval_point)?;
	let evals = witness
		.multilinears()
		.iter()
		.map(|multilin| multilin.borrow().evaluate(&query))
		.collect::<Result<Vec<_>, _>>()?;

	let subproofs = evals
		.iter()
		.zip(oracles)
		.map(|(&eval, suboracle)| {
			let subclaim = EvalcheckClaim {
				poly: MultivariatePolyOracle::Multilinear(suboracle),
				eval_point: eval_point.clone(),
				eval,
				is_random_point,
			};

			prove::<F, M, BM>(
				EvalcheckWitness::Multilinear,
				subclaim,
				batch_commited_eval_claims,
				shifted_eval_claims,
				packed_eval_claims,
			)
		})
		.collect::<Result<_, _>>()?;

	Ok(EvalcheckProof::Composite { evals, subproofs })
}

#[cfg(test)]
mod tests {
	use super::*;
	use assert_matches::assert_matches;
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	use crate::{
		field::{BinaryField128b, PackedBinaryField128x1b, PackedBinaryField4x32b, PackedField},
		oracle::{CommittedBatch, CompositePolyOracle, LinearCombination, MultivariatePolyOracle},
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
			CompositePolyOracle::new(log_size, suboracles, Arc::new(QuadProduct)).unwrap(),
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
			EvalcheckWitness::composite(multilins),
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

		let lin_com = MultilinearPolyOracle::LinearCombination(
			LinearCombination::new(
				n_vars,
				vec![
					(Box::new(select_row1.multilinear_poly_oracle()), EF::new(2)),
					(Box::new(select_row2.multilinear_poly_oracle()), EF::new(3)),
					(Box::new(select_row3.multilinear_poly_oracle()), EF::new(4)),
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

		let claim = EvalcheckClaim {
			poly: MultivariatePolyOracle::Multilinear(lin_com),
			eval_point,
			eval,
			is_random_point: true,
		};

		let witness = EvalcheckWitness::composite(vec![
			select_row1
				.multilinear_extension::<PackedBinaryField128x1b>()
				.unwrap(),
			select_row2
				.multilinear_extension::<PackedBinaryField128x1b>()
				.unwrap(),
			select_row3
				.multilinear_extension::<PackedBinaryField128x1b>()
				.unwrap(),
		]);

		let mut batch_claims = BatchCommittedEvalClaims::new(&[]);
		let mut shifted_claims_verify = Vec::new();
		let mut packed_claims_verify = Vec::new();
		let proof = prove(
			witness,
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
			EvalcheckWitness::<_, MultilinearExtension<EF>, MultilinearExtension<EF>>::Multilinear,
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
