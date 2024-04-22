// Copyright 2024 Ulvetanna Inc.

use super::{
	error::{Error, VerificationError},
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckMultilinearClaim,
		EvalcheckProof,
	},
	subclaims::{
		packed_sumcheck_meta, projected_bivariate_claim, shifted_sumcheck_meta,
		BivariateSumcheckClaim,
	},
};
use crate::{
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, ProjectionVariant},
	polynomial::{extrapolate_line, CompositionPoly},
};
use binius_field::{util::inner_product_unchecked, TowerField};
use tracing::instrument;

/// Verify an evalcheck claim.
///
/// See [`evalcheck::prove`](`super::prove::prove`) docs for comments on [`BatchCommittedEvalClaims`] and `new_sumchecks`.
#[instrument(skip_all, name = "evalcheck::verify")]
pub fn verify<F: TowerField, C>(
	oracles: &mut MultilinearOracleSet<F>,
	evalcheck_claim: EvalcheckClaim<F, C>,
	evalcheck_proof: EvalcheckProof<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	new_sumchecks: &mut Vec<BivariateSumcheckClaim<F>>,
) -> Result<(), Error>
where
	C: CompositionPoly<F>,
{
	let EvalcheckClaim {
		poly: composite,
		eval_point,
		eval,
		is_random_point,
	} = evalcheck_claim;

	let subproofs = match evalcheck_proof {
		EvalcheckProof::Composite { subproofs } => subproofs,
		_ => return Err(VerificationError::SubproofMismatch.into()),
	};

	if subproofs.len() != composite.n_multilinears() {
		return Err(VerificationError::SubproofMismatch.into());
	}

	// Verify the evaluation of the composition function over the claimed evaluations
	let evals = subproofs.iter().map(|(eval, _)| *eval).collect::<Vec<_>>();
	let actual_eval = composite.composition().evaluate(&evals)?;
	if actual_eval != eval {
		return Err(VerificationError::IncorrectEvaluation.into());
	}

	subproofs
		.into_iter()
		.zip(composite.inner_polys().into_iter())
		.try_for_each(|((eval, subproof), suboracle)| {
			let subclaim = EvalcheckMultilinearClaim {
				poly: suboracle,
				eval_point: eval_point.clone(),
				eval,
				is_random_point,
			};

			verify_multilinear(
				oracles,
				subclaim,
				subproof,
				batch_commited_eval_claims,
				new_sumchecks,
			)
		})?;

	Ok(())
}

fn verify_multilinear<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	evalcheck_claim: EvalcheckMultilinearClaim<F>,
	evalcheck_proof: EvalcheckProof<F>,
	batch_commited_eval_claims: &mut BatchCommittedEvalClaims<F>,
	new_sumchecks: &mut Vec<BivariateSumcheckClaim<F>>,
) -> Result<(), Error> {
	let EvalcheckMultilinearClaim {
		poly: multilinear,
		mut eval_point,
		eval,
		is_random_point,
	} = evalcheck_claim;

	match multilinear {
		MultilinearPolyOracle::Transparent(_id, transparent) => {
			match evalcheck_proof {
				EvalcheckProof::Transparent => {}
				_ => return Err(VerificationError::SubproofMismatch.into()),
			};

			let actual_eval = transparent.poly().evaluate(&eval_point)?;
			if actual_eval != eval {
				return Err(VerificationError::IncorrectEvaluation.into());
			}
		}

		MultilinearPolyOracle::Committed { id, .. } => {
			match evalcheck_proof {
				EvalcheckProof::Committed => {}
				_ => return Err(VerificationError::SubproofMismatch.into()),
			}

			let subclaim = CommittedEvalClaim {
				id,
				eval_point,
				eval,
				is_random_point,
			};

			batch_commited_eval_claims.insert(subclaim);
		}

		MultilinearPolyOracle::Repeating { inner, .. } => {
			let subproof = match evalcheck_proof {
				EvalcheckProof::Repeating(subproof) => subproof,
				_ => return Err(VerificationError::SubproofMismatch.into()),
			};

			let n_vars = inner.n_vars();
			let subclaim = EvalcheckMultilinearClaim {
				poly: *inner,
				eval_point: eval_point[..n_vars].to_vec(),
				eval,
				is_random_point,
			};

			verify_multilinear(
				oracles,
				subclaim,
				*subproof,
				batch_commited_eval_claims,
				new_sumchecks,
			)?;
		}

		MultilinearPolyOracle::Interleaved(..) => {
			// TODO: Implement interleaved reduction, similar to merged
			todo!()
		}

		MultilinearPolyOracle::Merged(_id, poly1, poly2) => {
			let (eval1, eval2, subproof1, subproof2) = match evalcheck_proof {
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

			let claim1 = EvalcheckMultilinearClaim {
				poly: *poly1,
				eval_point: subclaim_eval_point.to_vec(),
				eval: eval1,
				is_random_point,
			};

			verify_multilinear(
				oracles,
				claim1,
				*subproof1,
				batch_commited_eval_claims,
				new_sumchecks,
			)?;

			let claim2 = EvalcheckMultilinearClaim {
				poly: *poly2,
				eval_point: subclaim_eval_point.to_vec(),
				eval: eval2,
				is_random_point,
			};

			verify_multilinear(
				oracles,
				claim2,
				*subproof2,
				batch_commited_eval_claims,
				new_sumchecks,
			)?;
		}

		MultilinearPolyOracle::Projected(_id, projected) => {
			let (inner, values) = (projected.inner(), projected.values());
			let eval_point = match projected.projection_variant() {
				ProjectionVariant::LastVars => {
					eval_point.extend(values);
					eval_point
				}
				ProjectionVariant::FirstVars => values.iter().cloned().chain(eval_point).collect(),
			};

			let new_claim = EvalcheckMultilinearClaim {
				poly: *inner.clone(),
				eval_point,
				eval,
				is_random_point,
			};

			verify_multilinear(
				oracles,
				new_claim,
				evalcheck_proof,
				batch_commited_eval_claims,
				new_sumchecks,
			)?;
		}

		MultilinearPolyOracle::Shifted(_id, shifted) => {
			match evalcheck_proof {
				EvalcheckProof::Shifted => {}
				_ => return Err(VerificationError::SubproofMismatch.into()),
			};

			let meta = shifted_sumcheck_meta(oracles, &shifted, eval_point.as_slice())?;
			let sumcheck_claim = projected_bivariate_claim(oracles, meta, eval)?;
			new_sumchecks.push(sumcheck_claim);
		}

		MultilinearPolyOracle::Packed(_id, packed) => {
			match evalcheck_proof {
				EvalcheckProof::Packed => {}
				_ => return Err(VerificationError::SubproofMismatch.into()),
			};

			let meta = packed_sumcheck_meta(oracles, &packed, eval_point.as_slice())?;
			let sumcheck_claim = projected_bivariate_claim(oracles, meta, eval)?;
			new_sumchecks.push(sumcheck_claim);
		}

		MultilinearPolyOracle::LinearCombination(_id, lin_com) => {
			let subproofs = match evalcheck_proof {
				EvalcheckProof::Composite { subproofs } => subproofs,
				_ => return Err(VerificationError::SubproofMismatch.into()),
			};

			if subproofs.len() != lin_com.n_polys() {
				return Err(VerificationError::SubproofMismatch.into());
			}

			// Verify the evaluation of the linear combination over the claimed evaluations
			let actual_eval = inner_product_unchecked::<F, F>(
				subproofs.iter().map(|(eval, _)| *eval),
				lin_com.coefficients(),
			);

			if actual_eval != eval {
				return Err(VerificationError::IncorrectEvaluation.into());
			}

			subproofs.into_iter().zip(lin_com.polys()).try_for_each(
				|((eval, subproof), suboracle)| {
					let subclaim = EvalcheckMultilinearClaim {
						poly: suboracle.clone(),
						eval_point: eval_point.clone(),
						eval,
						is_random_point,
					};

					verify_multilinear(
						oracles,
						subclaim,
						subproof,
						batch_commited_eval_claims,
						new_sumchecks,
					)
				},
			)?;
		}
	}

	Ok(())
}
