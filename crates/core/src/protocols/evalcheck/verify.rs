// Copyright 2024-2025 Irreducible Inc.

use std::mem;

use binius_field::{util::inner_product_unchecked, TowerField};
use binius_math::extrapolate_line_scalar;
use getset::{Getters, MutGetters};
use tracing::instrument;

use super::{
	error::{Error, VerificationError},
	evalcheck::{EvalcheckMultilinearClaim, EvalcheckProofEnum},
	subclaims::{
		add_bivariate_sumcheck_to_constraints, add_composite_sumcheck_to_constraints,
		composite_sumcheck_meta, packed_sumcheck_meta, shifted_sumcheck_meta,
	},
	ProofIndex,
};
use crate::oracle::{
	ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
	MultilinearPolyVariant, OracleId, ProjectionVariant,
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
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,

	#[getset(get = "pub", get_mut = "pub")]
	committed_eval_claims: Vec<EvalcheckMultilinearClaim<F>>,

	new_sumcheck_constraints: Vec<ConstraintSetBuilder<F>>,
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
	/// See [`EvalcheckProver::prove`](`super::prove::EvalcheckProver::prove`) docs for comments.
	#[instrument(skip_all, name = "EvalcheckVerifierState::verify", level = "debug")]
	pub fn verify(
		&mut self,
		evalcheck_claims: Vec<EvalcheckMultilinearClaim<F>>,
		all_proofs: Vec<EvalcheckProofEnum<F>>,
	) -> Result<(), Error> {
		if evalcheck_claims.len() != all_proofs.len() {
			return Err(Error::Verification(VerificationError::NumberOfClaimsMismatch));
		}

		let mut verified_idxs = vec![false; all_proofs.len()];
		for (current_idx, claim) in evalcheck_claims.into_iter().enumerate() {
			self.verify_multilinear(claim, current_idx, &all_proofs, &mut verified_idxs)?;
		}

		if !verified_idxs.into_iter().all(|x| x) {
			return Err(Error::Verification(VerificationError::NotAllProofsVerified));
		}

		Ok(())
	}

	fn verify_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		current_proof_idx: ProofIndex,
		all_proofs: &[EvalcheckProofEnum<F>],
		verified_indexs: &mut [bool],
	) -> Result<(), Error> {
		if verified_indexs[current_proof_idx] {
			return Ok(());
		}
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		let evalcheck_proof = all_proofs
			.get(current_proof_idx)
			.ok_or(VerificationError::ProofIndexOutOfRange {
				index: current_proof_idx,
				length: all_proofs.len(),
			})?
			.clone();

		let multilinear = self.oracles.oracle(id);
		match multilinear.variant.clone() {
			MultilinearPolyVariant::Transparent(inner) => {
				if evalcheck_proof != EvalcheckProofEnum::Transparent {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let actual_eval = inner.poly().evaluate(&eval_point)?;
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(
						self.oracles.oracle(id).label(),
					)
					.into());
				}
				verified_indexs[current_proof_idx] = true;
			}

			MultilinearPolyVariant::Committed => {
				if evalcheck_proof != EvalcheckProofEnum::Committed {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let claim = EvalcheckMultilinearClaim {
					id: multilinear.id(),
					eval_point,
					eval,
				};

				self.committed_eval_claims.push(claim);
				verified_indexs[current_proof_idx] = true;
			}

			MultilinearPolyVariant::Repeating { id, .. } => {
				let subproof_index = match evalcheck_proof {
					EvalcheckProofEnum::Repeating(subproof) => subproof,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};
				let n_vars = self.oracles.n_vars(id);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: eval_point[..n_vars].into(),
					eval,
				};

				self.verify_multilinear(subclaim, subproof_index, all_proofs, verified_indexs)?;
				verified_indexs[current_proof_idx] = true;
			}

			MultilinearPolyVariant::Projected(projected) => {
				let (id, values) = (projected.id(), projected.values());
				let eval_point = match projected.projection_variant() {
					ProjectionVariant::LastVars => {
						let mut eval_point = eval_point.to_vec();
						eval_point.extend(values);
						eval_point
					}
					ProjectionVariant::FirstVars => {
						values.iter().copied().chain(eval_point.to_vec()).collect()
					}
				};

				let new_claim = EvalcheckMultilinearClaim {
					id,
					eval_point: eval_point.into(),
					eval,
				};

				self.verify_multilinear(new_claim, current_proof_idx, all_proofs, verified_indexs)?;
				verified_indexs[current_proof_idx] = true;
			}

			MultilinearPolyVariant::Shifted(shifted) => {
				if evalcheck_proof != EvalcheckProofEnum::Shifted {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = shifted_sumcheck_meta(self.oracles, &shifted, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					meta,
					&mut self.new_sumcheck_constraints,
					shifted.block_size(),
					eval,
				);
				verified_indexs[current_proof_idx] = true;
			}

			MultilinearPolyVariant::Packed(packed) => {
				if evalcheck_proof != EvalcheckProofEnum::Packed {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = packed_sumcheck_meta(self.oracles, &packed, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					meta,
					&mut self.new_sumcheck_constraints,
					packed.log_degree(),
					eval,
				);
				verified_indexs[current_proof_idx] = true;
			}

			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let subproofs = match evalcheck_proof {
					EvalcheckProofEnum::LinearCombination { subproofs } => subproofs,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				if subproofs.len() != linear_combination.n_polys() {
					return Err(VerificationError::SubproofMismatch.into());
				}

				// Verify the evaluation of the linear combination over the claimed evaluations
				let actual_eval = linear_combination.offset()
					+ inner_product_unchecked::<F, F>(
						subproofs.iter().map(|(eval, _)| *eval),
						linear_combination.coefficients(),
					);

				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear.label()).into());
				}

				subproofs
					.into_iter()
					.zip(linear_combination.polys())
					.try_for_each(|((eval, subproof), suboracle_id)| {
						self.verify_multilinear_subclaim(
							eval,
							subproof,
							suboracle_id,
							&eval_point,
							all_proofs,
							verified_indexs,
						)
					})?;
				verified_indexs[current_proof_idx] = true;
			}
			MultilinearPolyVariant::ZeroPadded(inner) => {
				let (inner_eval, subproof) = match evalcheck_proof {
					EvalcheckProofEnum::ZeroPadded(eval, subproof) => (eval, subproof),
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				let inner_n_vars = self.oracles.n_vars(inner);

				let (subclaim_eval_point, zs) = eval_point.split_at(inner_n_vars);

				let mut extrapolate_eval = inner_eval;

				for z in zs {
					extrapolate_eval =
						extrapolate_line_scalar::<F, F>(F::ZERO, extrapolate_eval, *z);
				}

				if extrapolate_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear.label()).into());
				}

				self.verify_multilinear_subclaim(
					inner_eval,
					subproof,
					inner,
					subclaim_eval_point,
					all_proofs,
					verified_indexs,
				)?;
				verified_indexs[current_proof_idx] = true;
			}
			MultilinearPolyVariant::Composite(composition) => {
				if evalcheck_proof != EvalcheckProofEnum::CompositeMLE {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = composite_sumcheck_meta(self.oracles, &eval_point)?;
				add_composite_sumcheck_to_constraints(
					meta,
					&mut self.new_sumcheck_constraints,
					&composition,
					eval,
				);
				verified_indexs[current_proof_idx] = true;
			}
		}

		Ok(())
	}

	fn verify_multilinear_subclaim(
		&mut self,
		eval: F,
		current_index: ProofIndex,
		oracle_id: OracleId,
		eval_point: &[F],
		all_proofs: &[EvalcheckProofEnum<F>],
		verified_indicies: &mut [bool],
	) -> Result<(), Error> {
		let subclaim = EvalcheckMultilinearClaim {
			id: oracle_id,
			eval_point: eval_point.into(),
			eval,
		};
		self.verify_multilinear(subclaim, current_index, all_proofs, verified_indicies)
	}
}
