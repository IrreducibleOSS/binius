// Copyright 2024 Ulvetanna Inc.

use tracing::instrument;

use crate::{
	field::Field,
	iopoly::{MultilinearPolyOracle, MultivariatePolyOracle, ProjectionVariant},
	polynomial::{multilinear_query::MultilinearQuery, MultilinearPoly},
};
use std::borrow::Borrow;

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckProof,
		EvalcheckWitness, ShiftedEvalClaim,
	},
};

#[instrument(skip_all, name = "evalcheck::prove")]
pub fn prove<F: Field, M: MultilinearPoly<F> + ?Sized, BM: Borrow<M>>(
	evalcheck_witness: EvalcheckWitness<F, M, BM>,
	evalcheck_claim: EvalcheckClaim<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	shifted_eval_claims: &mut Vec<ShiftedEvalClaim<F>>,
) -> Result<EvalcheckProof<F>, Error> {
	let EvalcheckClaim {
		poly,
		eval_point,
		eval,
		is_random_point,
	} = evalcheck_claim;

	let proof = match poly {
		MultivariatePolyOracle::Multilinear(multilinear) => {
			match evalcheck_witness {
				EvalcheckWitness::Multilinear => (),
				EvalcheckWitness::Composite(_) => return Err(Error::InvalidWitness),
			};
			match multilinear {
				MultilinearPolyOracle::Transparent { .. } => EvalcheckProof::Transparent,
				MultilinearPolyOracle::Committed { id, .. } => {
					let subclaim = CommittedEvalClaim {
						id,
						eval_point,
						eval,
						is_random_point,
					};

					batch_commited_eval_claims.insert(subclaim)?;
					EvalcheckProof::Committed
				}
				MultilinearPolyOracle::Shifted(shifted) => {
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
				MultilinearPolyOracle::Projected(projected) => {
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
					)?
				}

				_ => todo!(),
			}
		}

		MultivariatePolyOracle::Composite(composite) => {
			let query = MultilinearQuery::with_full_query(&eval_point)?;

			let composite_witness = match evalcheck_witness {
				EvalcheckWitness::Multilinear => return Err(Error::InvalidWitness),
				EvalcheckWitness::Composite(composite) => composite,
			};

			let evals = composite_witness
				.iter_multilinear_polys()
				.map(|multilin| multilin.evaluate(&query))
				.collect::<Result<Vec<_>, _>>()?;

			let mut subproofs = vec![];

			for (&eval, suboracle) in evals.iter().zip(composite.inner_polys()) {
				let subclaim = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(suboracle),
					eval_point: eval_point.clone(),
					eval,
					is_random_point,
				};

				let subproof = prove::<F, M, BM>(
					EvalcheckWitness::Multilinear,
					subclaim,
					batch_commited_eval_claims,
					shifted_eval_claims,
				)?;

				subproofs.push(subproof);
			}

			EvalcheckProof::Composite { evals, subproofs }
		}
	};

	Ok(proof)
}

#[cfg(test)]
mod tests {
	use super::*;
	use rand::{rngs::StdRng, SeedableRng};
	use std::{iter::repeat_with, sync::Arc};

	use crate::{
		field::{BinaryField128b, PackedBinaryField4x32b, PackedField},
		iopoly::{CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle},
		polynomial::{
			CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearExtension,
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

		type DynMP = dyn MultilinearPoly<EF> + Send + Sync;

		// sample BF32 polys as BF128
		let witness = MultilinearComposite::<EF, DynMP, _>::new(
			log_size,
			Arc::new(QuadProduct),
			multilins
				.into_iter()
				.map(|multilin| Arc::new(multilin) as Arc<DynMP>)
				.collect(),
		)
		.unwrap();

		let suboracles = (0..4)
			.map(|id| MultilinearPolyOracle::<EF>::Committed {
				id,
				n_vars: log_size,
				tower_level,
			})
			.collect();

		let oracle = MultivariatePolyOracle::Composite(
			CompositePolyOracle::new(log_size, suboracles, Arc::new(QuadProduct)).unwrap(),
		);

		let claim = EvalcheckClaim {
			poly: oracle,
			eval_point: eval_point.clone(),
			eval,
			is_random_point: true,
		};

		let batches = vec![vec![0, 2], vec![1, 3]];

		let mut bcec_prove = BatchCommittedEvalClaims::new(&batches);
		let mut shifted_claims_prove = Vec::new();
		let proof = prove(
			EvalcheckWitness::Composite(witness),
			claim.clone(),
			&mut bcec_prove,
			&mut shifted_claims_prove,
		)
		.unwrap();

		let mut bcec_verify = BatchCommittedEvalClaims::new(&batches);
		let mut shifted_claims_verify = Vec::new();
		verify(claim, proof, &mut bcec_verify, &mut shifted_claims_verify).unwrap();

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
}
