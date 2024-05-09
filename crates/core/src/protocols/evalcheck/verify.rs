// Copyright 2024 Ulvetanna Inc.

use super::{
	error::{Error, VerificationError},
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckMultilinearClaim,
		EvalcheckProof,
	},
	subclaims::{packed_sumcheck_meta, projected_bivariate_claim, shifted_sumcheck_meta},
};
use crate::{
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, ProjectionVariant},
	polynomial::extrapolate_line,
	protocols::sumcheck::SumcheckClaim,
};
use binius_field::{util::inner_product_unchecked, TowerField};
use getset::{Getters, MutGetters};
use tracing::instrument;

/// A mutable verifier state.
///
/// Can be persisted across [`EvalcheckVerifier::verify`] invocations. Accumulates
/// `new_sumchecks` bivariate sumcheck instances, as well as holds mutable references to
/// the trace (to which new oracles & multilinears may be added during verification)
#[derive(Getters, MutGetters)]
pub struct EvalcheckVerifier<'a, F: TowerField> {
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,

	#[getset(get = "pub", get_mut = "pub")]
	pub(crate) batch_committed_eval_claims: BatchCommittedEvalClaims<F>,

	#[get = "pub"]
	new_sumcheck_claims: Vec<SumcheckClaim<F>>,
}

impl<'a, F: TowerField> EvalcheckVerifier<'a, F> {
	/// Create a new verifier state from a mutable reference to the oracle set
	/// (it needs to be mutable because `new_sumcheck` reduction may add new
	/// oracles & multilinears)
	pub fn new(oracles: &'a mut MultilinearOracleSet<F>) -> Self {
		let new_sumcheck_claims = Vec::new();
		let batch_committed_eval_claims =
			BatchCommittedEvalClaims::new(&oracles.committed_batches());

		Self {
			oracles,
			batch_committed_eval_claims,
			new_sumcheck_claims,
		}
	}

	/// A helper method to move out the set of reduced claims
	pub fn take_new_sumchecks(&mut self) -> Vec<SumcheckClaim<F>> {
		std::mem::take(&mut self.new_sumcheck_claims)
	}

	/// Verify an evalcheck claim.
	///
	/// See [`EvalcheckProver::prove`](`super::prove::EvalcheckProver::prove`) docs for comments.
	#[instrument(skip_all, name = "EvalcheckVerifierState::verify")]
	pub fn verify(
		&mut self,
		evalcheck_claim: EvalcheckClaim<F>,
		evalcheck_proof: EvalcheckProof<F>,
	) -> Result<(), Error> {
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
				self.verify_multilinear_subclaim(
					eval,
					subproof,
					suboracle,
					&eval_point,
					is_random_point,
				)
			})?;

		Ok(())
	}

	fn verify_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		evalcheck_proof: EvalcheckProof<F>,
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

				self.batch_committed_eval_claims.insert(subclaim);
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

				self.verify_multilinear(subclaim, *subproof)?;
			}

			MultilinearPolyOracle::Interleaved(_id, poly1, poly2) => {
				let (eval1, eval2, subproof1, subproof2) = match evalcheck_proof {
					EvalcheckProof::Interleaved {
						eval1,
						eval2,
						subproof1,
						subproof2,
					} => (eval1, eval2, subproof1, subproof2),
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				// Verify the evaluation of the interleaved function over the claimed evaluations
				let subclaim_eval_point = &eval_point[1..];
				let actual_eval = extrapolate_line(eval1, eval2, eval_point[0]);
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation.into());
				}
				self.verify_multilinear_subclaim(
					eval1,
					*subproof1,
					*poly1,
					subclaim_eval_point,
					is_random_point,
				)?;
				self.verify_multilinear_subclaim(
					eval2,
					*subproof2,
					*poly2,
					subclaim_eval_point,
					is_random_point,
				)?;
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

				self.verify_multilinear_subclaim(
					eval1,
					*subproof1,
					*poly1,
					subclaim_eval_point,
					is_random_point,
				)?;
				self.verify_multilinear_subclaim(
					eval2,
					*subproof2,
					*poly2,
					subclaim_eval_point,
					is_random_point,
				)?;
			}

			MultilinearPolyOracle::Projected(_id, projected) => {
				let (inner, values) = (projected.inner(), projected.values());
				let eval_point = match projected.projection_variant() {
					ProjectionVariant::LastVars => {
						eval_point.extend(values);
						eval_point
					}
					ProjectionVariant::FirstVars => {
						values.iter().cloned().chain(eval_point).collect()
					}
				};

				let new_claim = EvalcheckMultilinearClaim {
					poly: *inner.clone(),
					eval_point,
					eval,
					is_random_point,
				};

				self.verify_multilinear(new_claim, evalcheck_proof)?;
			}

			MultilinearPolyOracle::Shifted(_id, shifted) => {
				match evalcheck_proof {
					EvalcheckProof::Shifted => {}
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let meta = shifted_sumcheck_meta(self.oracles, &shifted, eval_point.as_slice())?;
				let sumcheck_claim = projected_bivariate_claim(self.oracles, meta, eval)?;
				self.new_sumcheck_claims.push(sumcheck_claim);
			}

			MultilinearPolyOracle::Packed(_id, packed) => {
				match evalcheck_proof {
					EvalcheckProof::Packed => {}
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let meta = packed_sumcheck_meta(self.oracles, &packed, eval_point.as_slice())?;
				let sumcheck_claim = projected_bivariate_claim(self.oracles, meta, eval)?;
				self.new_sumcheck_claims.push(sumcheck_claim);
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
						self.verify_multilinear_subclaim(
							eval,
							subproof,
							suboracle.clone(),
							&eval_point,
							is_random_point,
						)
					},
				)?;
			}
		}

		Ok(())
	}

	fn verify_multilinear_subclaim(
		&mut self,
		eval: F,
		subproof: EvalcheckProof<F>,
		poly: MultilinearPolyOracle<F>,
		eval_point: &[F],
		is_random_point: bool,
	) -> Result<(), Error> {
		let subclaim = EvalcheckMultilinearClaim {
			poly,
			eval_point: eval_point.to_vec(),
			eval,
			is_random_point,
		};
		self.verify_multilinear(subclaim, subproof)
	}
}
