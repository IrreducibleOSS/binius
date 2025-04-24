// Copyright 2024-2025 Irreducible Inc.

use std::mem;

use binius_field::{util::inner_product_unchecked, TowerField};
use getset::{Getters, MutGetters};
use tracing::instrument;

use super::{
	error::{Error, VerificationError},
	evalcheck::{EvalcheckMultilinearClaim, EvalcheckProof},
	subclaims::{
		add_bivariate_sumcheck_to_constraints, add_composite_sumcheck_to_constraints,
		composite_sumcheck_meta, packed_sumcheck_meta, shifted_sumcheck_meta,
	},
};
use crate::{
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyVariant, OracleId,
	},
	polynomial::MultivariatePoly,
	transparent::select_row::SelectRow,
};

/// A mutable verifier state.
///
/// Can be persisted across [`EvalcheckVerifier::verify`] invocations. Accumulates
/// `new_sumchecks` bivariate sumcheck constraints, as well as holds mutable references to
/// the trace (to which new oracles & multilinears may be added during verification)
#[derive(Getters, MutGetters)]
pub struct EvalcheckVerifier<'a, F>
where
	F: TowerField,
{
	/// Mutable reference to the oracle set which is modified to create new claims arising from sumchecks
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,

	/// The committed evaluation claims in this round
	#[getset(get = "pub", get_mut = "pub")]
	committed_eval_claims: Vec<EvalcheckMultilinearClaim<F>>,

	/// The new sumcheck constraints in this round
	new_sumcheck_constraints: Vec<ConstraintSetBuilder<F>>,

	/// The list of claims that have been verified in this round
	round_claims: Vec<EvalcheckMultilinearClaim<F>>,
}

impl<'a, F: TowerField> EvalcheckVerifier<'a, F> {
	/// Create a new verifier state from a mutable reference to the oracle set
	/// (it needs to be mutable because `new_sumcheck` reduction may add new
	/// oracles & multilinears)
	pub const fn new(oracles: &'a mut MultilinearOracleSet<F>) -> Self {
		Self {
			oracles,
			committed_eval_claims: Vec::new(),
			new_sumcheck_constraints: Vec::new(),
			round_claims: Vec::new(),
		}
	}

	/// A helper method to move out sumcheck constraints
	pub fn take_new_sumcheck_constraints(&mut self) -> Result<Vec<ConstraintSet<F>>, OracleError> {
		self.new_sumcheck_constraints
			.iter_mut()
			.map(|builder| mem::take(builder).build_one(self.oracles))
			.filter(|constraint| !matches!(constraint, Err(OracleError::EmptyConstraintSet)))
			.rev()
			.collect()
	}

	/// Verify an evalcheck claim.
	///
	/// For each claim, we verify the proof by recursively verifying the subclaims in a DFS manner deduplicating previously verified claims
	/// See [`EvalcheckProver::prove`](`super::prove::EvalcheckProver::prove`) docs for more details.
	#[instrument(skip_all, name = "EvalcheckVerifierState::verify", level = "debug")]
	pub fn verify(
		&mut self,
		evalcheck_claims: Vec<EvalcheckMultilinearClaim<F>>,
		evalcheck_proofs: Vec<EvalcheckProof<F>>,
	) -> Result<(), Error> {
		self.round_claims.clear();

		for (claim, proof) in evalcheck_claims
			.into_iter()
			.zip(evalcheck_proofs.into_iter())
		{
			self.verify_multilinear(claim, proof)?;
		}

		Ok(())
	}

	fn verify_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		evalcheck_proof: EvalcheckProof<F>,
	) -> Result<(), Error> {
		// If the proof is a duplicate claim, we need to check if the claim is already in the round claims which has been verified
		if let EvalcheckProof::DuplicateClaim(index) = evalcheck_proof {
			if let Some(expected_claim) = self.round_claims.get(index) {
				if *expected_claim == evalcheck_claim {
					return Ok(());
				}
				return Err(VerificationError::DuplicateClaimMismatch.into());
			}
		}

		// If the proof is not a duplicate claim, we need to add the claim to the round claims
		self.round_claims.push(evalcheck_claim.clone());

		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		let multilinear = self.oracles.oracle(id);

		match multilinear.variant.clone() {
			MultilinearPolyVariant::Transparent(inner) => {
				if evalcheck_proof != EvalcheckProof::Transparent {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let actual_eval = inner.poly().evaluate(&eval_point)?;
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(
						self.oracles.oracle(id).label(),
					)
					.into());
				}
			}

			MultilinearPolyVariant::Committed => {
				if evalcheck_proof != EvalcheckProof::Committed {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let claim = EvalcheckMultilinearClaim {
					id: multilinear.id(),
					eval_point,
					eval,
				};

				self.committed_eval_claims.push(claim);
			}

			MultilinearPolyVariant::Repeating { id, .. } => {
				let subproof = match evalcheck_proof {
					EvalcheckProof::Repeating(subproof) => subproof,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};
				let n_vars = self.oracles.n_vars(id);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: eval_point[..n_vars].into(),
					eval,
				};

				self.verify_multilinear(subclaim, *subproof)?;
			}

			MultilinearPolyVariant::Projected(projected) => {
				let subproof = match evalcheck_proof {
					EvalcheckProof::Projected(subproof) => subproof,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let (id, values) = (projected.id(), projected.values());

				let new_eval_point = {
					let idx = projected.start_index();
					let mut new_eval_point = eval_point[0..idx].to_vec();
					new_eval_point.extend(values.clone());
					new_eval_point.extend(eval_point[idx..].to_vec());
					new_eval_point
				};

				let new_claim = EvalcheckMultilinearClaim {
					id,
					eval_point: new_eval_point.into(),
					eval,
				};

				self.verify_multilinear(new_claim, *subproof)?;
			}

			MultilinearPolyVariant::Shifted(shifted) => {
				if evalcheck_proof != EvalcheckProof::Shifted {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = shifted_sumcheck_meta(self.oracles, &shifted, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					shifted.block_size(),
					eval,
				)
			}

			MultilinearPolyVariant::Packed(packed) => {
				if evalcheck_proof != EvalcheckProof::Packed {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = packed_sumcheck_meta(self.oracles, &packed, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					packed.log_degree(),
					eval,
				)
			}

			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let subproofs = match evalcheck_proof {
					EvalcheckProof::LinearCombination { subproofs } => subproofs,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				if subproofs.len() != linear_combination.n_polys() {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let mut evals = Vec::new();

				for (subproof, sub_oracle_id) in subproofs.iter().zip(linear_combination.polys()) {
					// If the subproof is a duplicate claim, we need to check if the claim is already in the round claims which has been verified
					// otherwise, we verify the subclaim by DFS
					match subproof {
						(None, EvalcheckProof::DuplicateClaim(index)) => {
							if self.round_claims[*index].id != sub_oracle_id
								|| self.round_claims[*index].eval_point != eval_point
							{
								return Err(VerificationError::DuplicateClaimMismatch.into());
							}

							evals.push(self.round_claims[*index].eval);
						}
						(Some(eval), _) => {
							evals.push(*eval);
						}
						_ => return Err(VerificationError::MissingLinearCombinationEval.into()),
					}
				}

				// Verify the evaluation of the linear combination over the claimed evaluations
				let actual_eval = linear_combination.offset()
					+ inner_product_unchecked::<F, F>(evals, linear_combination.coefficients());

				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear.label()).into());
				}

				subproofs
					.into_iter()
					.zip(linear_combination.polys())
					.try_for_each(|(subclaim, suboracle_id)| match subclaim {
						(None, EvalcheckProof::DuplicateClaim(_)) => Ok(()),
						(Some(eval), proof) => {
							self.verify_multilinear_subclaim(eval, proof, suboracle_id, &eval_point)
						}
						_ => Err(VerificationError::MissingLinearCombinationEval.into()),
					})?;
			}
			MultilinearPolyVariant::ZeroPadded(padded) => {
				let inner = padded.id();
				let inner_n_vars = self.oracles.n_vars(inner);

				let start_index = padded.start_index();
				let extra_n_vars = padded.new_n_vars() - inner_n_vars;

				let (inner_eval, subproof) = match evalcheck_proof {
					EvalcheckProof::ZeroPadded(eval, subproof) => (eval, subproof),
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let (first_subclaim_eval_point, zs_second_subclaim) =
					eval_point.split_at(start_index);

				let (zs, second_subclaim) = zs_second_subclaim.split_at(extra_n_vars);
				let subclaim_eval_point = {
					let mut subclaim_eval_point = first_subclaim_eval_point.to_vec();
					subclaim_eval_point.extend_from_slice(second_subclaim);
					subclaim_eval_point
				};

				let select_row = SelectRow::new(zs.len(), padded.nonzero_index()).unwrap();
				let expected_eval = inner_eval * select_row.evaluate(zs).unwrap();

				if expected_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear.label()).into());
				}

				self.verify_multilinear_subclaim(
					inner_eval,
					*subproof,
					inner,
					&subclaim_eval_point,
				)?;
			}
			MultilinearPolyVariant::Composite(composition) => {
				if evalcheck_proof != EvalcheckProof::CompositeMLE {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = composite_sumcheck_meta(self.oracles, &eval_point)?;
				add_composite_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					&composition,
					eval,
				)
			}
		}

		Ok(())
	}

	fn verify_multilinear_subclaim(
		&mut self,
		eval: F,
		subproof: EvalcheckProof<F>,
		oracle_id: OracleId,
		eval_point: &[F],
	) -> Result<(), Error> {
		let subclaim = EvalcheckMultilinearClaim {
			id: oracle_id,
			eval_point: eval_point.into(),
			eval,
		};
		self.verify_multilinear(subclaim, subproof)
	}
}
