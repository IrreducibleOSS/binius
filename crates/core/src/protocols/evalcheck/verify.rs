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
	EvalcheckProofAdvice, ProofIndex, Subclaim,
};
use crate::oracle::{
	ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
	MultilinearPolyVariant,
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
	/// See [`EvalcheckProver::prove`](`super::prove::EvalcheckProver::prove`) docs for comments.
	#[instrument(skip_all, name = "EvalcheckVerifierState::verify", level = "debug")]
	pub fn verify(
		&mut self,
		evalcheck_claims: Vec<EvalcheckMultilinearClaim<F>>,
		proofs: Vec<EvalcheckProofEnum<F>>,
		advices: Vec<EvalcheckProofAdvice>,
	) -> Result<(), Error> {
		self.round_claims.clear();
		self.round_claims.extend(evalcheck_claims.clone());

		let mut proof_idx = 0;
		// proof_len <= total_claim_len == advice_len
		for (current_claim_idx, advice) in advices.into_iter().enumerate() {
			if self.round_claims.len() <= current_claim_idx {
				return Err(VerificationError::ClaimIndexOutOfRange {
					index: current_claim_idx,
					length: self.round_claims.len(),
				}
				.into());
			}
			let claim = &self.round_claims[current_claim_idx];
			match advice {
				EvalcheckProofAdvice::DuplicateClaim(claim_idx) => {
					if claim_idx >= self.round_claims.len() {
						return Err(VerificationError::ClaimIndexOutOfRange {
							index: claim_idx,
							length: self.round_claims.len(),
						}
						.into());
					}
					if claim_idx <= current_claim_idx {
						return Err(VerificationError::DuplicateClaimIndexTooSmall.into());
					}
					if claim != &self.round_claims[claim_idx] {
						return Err(VerificationError::DuplicateClaimMismatch.into());
					}
				}
				EvalcheckProofAdvice::HandleClaim => {
					self.verify_multilinear(claim.clone(), current_claim_idx, &proofs[proof_idx])?;
					proof_idx += 1;
				}
			};
		}

		if proof_idx != proofs.len() {
			return Err(Error::Verification(VerificationError::NotAllProofsVerified));
		}

		Ok(())
	}

	fn verify_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		current_idx: ProofIndex,
		proof: &EvalcheckProofEnum<F>,
	) -> Result<(), Error> {
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		let multilinear = self.oracles.oracle(id);
		match multilinear.variant.clone() {
			MultilinearPolyVariant::Transparent(inner) => {
				if !matches!(proof, EvalcheckProofEnum::Transparent) {
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
				if !matches!(proof, EvalcheckProofEnum::Committed) {
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
				if !matches!(proof, EvalcheckProofEnum::Repeating) {
					return Err(VerificationError::SubproofMismatch.into());
				};
				let n_vars = self.oracles.n_vars(id);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: eval_point[..n_vars].into(),
					eval,
				};

				self.round_claims.push(subclaim);
			}

			MultilinearPolyVariant::Projected(projected) => {
				if !matches!(proof, EvalcheckProofEnum::Projected) {
					return Err(VerificationError::SubproofMismatch.into());
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

				self.round_claims.push(new_claim);
			}

			MultilinearPolyVariant::Shifted(shifted) => {
				if !matches!(proof, EvalcheckProofEnum::Shifted) {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = shifted_sumcheck_meta(self.oracles, &shifted, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					shifted.block_size(),
					eval,
				);
			}

			MultilinearPolyVariant::Packed(packed) => {
				if !matches!(proof, EvalcheckProofEnum::Packed) {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = packed_sumcheck_meta(self.oracles, &packed, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					packed.log_degree(),
					eval,
				);
			}

			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let subproofs = match proof {
					EvalcheckProofEnum::LinearCombination { subproofs } => subproofs,
					_ => return Err(VerificationError::SubproofMismatch.into()),
				};

				if subproofs.len() != linear_combination.n_polys() {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let inner_evals = subproofs
					.iter()
					.zip(linear_combination.polys())
					.map(|(subclaim, sub_oracle_id)| match subclaim {
						Subclaim::ExistingClaim(index) => {
							if *index == current_idx {
								return Err(VerificationError::ExistingClaimEqCurrentClaim);
							}
							if *index >= self.round_claims.len() {
								return Err(VerificationError::ClaimIndexOutOfRange {
									index: *index,
									length: self.round_claims.len(),
								});
							}
							Ok(self.round_claims[*index].eval)
						}
						Subclaim::NewClaim(claim) => {
							self.round_claims.push(EvalcheckMultilinearClaim {
								id: sub_oracle_id,
								eval_point: eval_point.clone(),
								eval: *claim,
							});
							Ok(*claim)
						}
					})
					.collect::<Result<Vec<_>, _>>()?;

				// Verify the evaluation of the linear combination over the claimed evaluations
				let actual_eval = linear_combination.offset()
					+ inner_product_unchecked::<F, F>(
						inner_evals.into_iter(),
						linear_combination.coefficients(),
					);

				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear.label()).into());
				}
			}
			MultilinearPolyVariant::ZeroPadded(inner) => {
				let inner_eval = match proof {
					EvalcheckProofEnum::ZeroPadded(eval) => *eval,
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

				self.round_claims.push(EvalcheckMultilinearClaim {
					id: inner,
					eval_point: subclaim_eval_point.into(),
					eval: inner_eval,
				})
			}
			MultilinearPolyVariant::Composite(composition) => {
				if !matches!(proof, EvalcheckProofEnum::CompositeMLE) {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = composite_sumcheck_meta(self.oracles, &eval_point)?;
				add_composite_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					&composition,
					eval,
				);
			}
		}

		Ok(())
	}
}
