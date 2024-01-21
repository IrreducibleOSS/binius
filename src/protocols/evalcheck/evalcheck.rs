// Copyright 2023 Ulvetanna Inc.

use super::error::{Error, VerificationError};
use crate::{
	field::Field,
	iopoly::{CommittedId, MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::{extrapolate_line, MultilinearComposite},
};
use std::any::Any;

#[derive(Debug)]
pub struct EvalcheckClaim<F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: MultivariatePolyOracle<F>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
	/// Whether the evaluation point is random
	pub is_random_point: bool,
}

/// Polynomial must be representable as a composition of multilinear polynomials
pub type EvalcheckWitness<F, M, BM> = MultilinearComposite<F, M, BM>;

#[derive(Debug)]
pub enum EvalcheckProof<F: Field> {
	Transparent,
	Committed(Box<dyn Any>),
	Repeating(Box<EvalcheckProof<F>>),
	Merged {
		eval1: F,
		eval2: F,
		subproof1: Box<EvalcheckProof<F>>,
		subproof2: Box<EvalcheckProof<F>>,
	},
	Composite {
		evals: Vec<F>,
		subproofs: Vec<EvalcheckProof<F>>,
	},
}

#[derive(Debug)]
pub struct CommittedEvalClaim<F> {
	pub id: CommittedId,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
	/// Whether the evaluation point is random
	pub is_random_point: bool,
}

pub fn verify<F: Field>(
	claim: EvalcheckClaim<F>,
	proof: EvalcheckProof<F>,
	committed_claims: &mut Vec<(CommittedEvalClaim<F>, Box<dyn Any>)>,
) -> Result<(), Error> {
	let EvalcheckClaim {
		poly,
		mut eval_point,
		eval,
		is_random_point,
	} = claim;
	match poly {
		MultivariatePolyOracle::Multilinear(multilinear) => match multilinear {
			MultilinearPolyOracle::Transparent {
				poly,
				tower_level: _,
			} => {
				match proof {
					EvalcheckProof::Transparent => {}
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let actual_eval = poly.evaluate(&eval_point)?;
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation.into());
				}
			}
			MultilinearPolyOracle::Committed { id, .. } => {
				let subproof = match proof {
					EvalcheckProof::Committed(subproof) => subproof,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let subclaim = CommittedEvalClaim {
					id,
					eval_point,
					eval,
					is_random_point,
				};
				committed_claims.push((subclaim, subproof));
			}
			MultilinearPolyOracle::Repeating { inner, log_count } => {
				let subproof = match proof {
					EvalcheckProof::Repeating(subproof) => subproof,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};
				let n_vars = inner.n_vars();
				let subclaim = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(*inner),
					eval_point: eval_point[..n_vars - log_count].to_vec(),
					eval,
					is_random_point,
				};
				verify(subclaim, *subproof, committed_claims)?;
			}
			MultilinearPolyOracle::Interleaved(_poly1, _poly2) => {
				// TODO: Implement interleaved reduction, similar to merged
				todo!()
			}
			MultilinearPolyOracle::Merged(poly1, poly2) => {
				let (eval1, eval2, subproof1, subproof2) = match proof {
					EvalcheckProof::Merged {
						eval1,
						eval2,
						subproof1,
						subproof2,
					} => (eval1, eval2, subproof1, subproof2),
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				// Verify the evaluation of the merged function over the claimed evaluations
				let n_vars = poly1.n_vars();
				let subclaim_eval_point = &eval_point[..n_vars];
				let actual_eval = extrapolate_line(eval1, eval2, eval_point[n_vars]);
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation.into());
				}

				let claim1 = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(*poly1),
					eval_point: subclaim_eval_point.to_vec(),
					eval: eval1,
					is_random_point,
				};
				verify(claim1, *subproof1, committed_claims)?;

				let claim2 = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(*poly2),
					eval_point: subclaim_eval_point.to_vec(),
					eval: eval2,
					is_random_point,
				};
				verify(claim2, *subproof2, committed_claims)?;
			}
			MultilinearPolyOracle::ProjectFirstVar { inner, value } => {
				eval_point.insert(0, value);
				let new_claim = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(*inner),
					eval_point,
					eval,
					is_random_point,
				};
				verify(new_claim, proof, committed_claims)?;
			}
			MultilinearPolyOracle::ProjectLastVar { inner, value } => {
				eval_point.push(value);
				let new_claim = EvalcheckClaim {
					poly: MultivariatePolyOracle::Multilinear(*inner),
					eval_point,
					eval,
					is_random_point,
				};
				verify(new_claim, proof, committed_claims)?;
			}
			MultilinearPolyOracle::Shifted(_) => {
				// TODO
				todo!()
			}
			MultilinearPolyOracle::Packed(_) => {
				// TODO
				todo!()
			}
		},
		MultivariatePolyOracle::Composite(composite) => {
			let (evals, subproofs) = match proof {
				EvalcheckProof::Composite { evals, subproofs } => (evals, subproofs),
				_ => return Err(VerificationError::SubproofMismatch.into()),
			};

			if evals.len() != composite.n_multilinears() {
				return Err(VerificationError::SubproofMismatch.into());
			}
			if subproofs.len() != composite.n_multilinears() {
				return Err(VerificationError::SubproofMismatch.into());
			}

			// Verify the evaluation of the composition function over the claimed evaluations
			let actual_eval = composite.composition().evaluate(&evals)?;
			if actual_eval != eval {
				return Err(VerificationError::IncorrectEvaluation.into());
			}

			evals
				.into_iter()
				.zip(subproofs.into_iter())
				.zip(composite.inner_polys().into_iter())
				.try_for_each(|((eval, subproof), suboracle)| {
					let subclaim = EvalcheckClaim {
						poly: MultivariatePolyOracle::Multilinear(suboracle),
						eval_point: eval_point.clone(),
						eval,
						is_random_point,
					};
					verify(subclaim, subproof, committed_claims)
				})?;
		}
	}

	Ok(())
}
