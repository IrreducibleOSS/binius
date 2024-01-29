// Copyright 2024 Ulvetanna Inc.

use crate::{
	field::Field,
	iopoly::{MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::{multilinear_query::MultilinearQuery, MultilinearComposite, MultilinearPoly},
};
use std::borrow::Borrow;

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckProof,
		EvalcheckWitness, ShiftedEvalClaim,
	},
};

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
		MultivariatePolyOracle::Multilinear(multilinear) => match multilinear {
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

			_ => todo!(),
		},

		MultivariatePolyOracle::Composite(composite) => {
			let query = MultilinearQuery::with_full_query(&eval_point)?;

			let evals = evalcheck_witness
				.iter_multilinear_polys()
				.map(|multilin| multilin.evaluate(&query))
				.collect::<Result<Vec<_>, _>>()?;

			let mut subproofs = vec![];

			for ((multilin, &eval), suboracle) in evalcheck_witness
				.iter_multilinear_polys()
				.zip(&evals)
				.zip(composite.inner_polys())
			{
				let subwitness = MultilinearComposite::from_multilinear(multilin);

				let subclaim = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(suboracle),
					eval_point: eval_point.clone(),
					eval,
					is_random_point,
				};

				let subproof = prove::<F, M, &M>(
					subwitness,
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
