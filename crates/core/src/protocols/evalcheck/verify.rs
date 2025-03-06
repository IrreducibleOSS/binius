// Copyright 2024-2025 Irreducible Inc.

use std::mem;

use binius_field::{util::inner_product_unchecked, TowerField};
use binius_math::extrapolate_line_scalar;
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
		evalcheck_proofs: Vec<EvalcheckProof<F>>,
	) -> Result<(), Error> {
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

				self.verify_multilinear(new_claim, evalcheck_proof)?;
			}

			MultilinearPolyVariant::Shifted(shifted) => {
				if evalcheck_proof != EvalcheckProof::Shifted {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = shifted_sumcheck_meta(self.oracles, &shifted, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					meta,
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
					meta,
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
						self.verify_multilinear_subclaim(eval, subproof, suboracle_id, &eval_point)
					})?;
			}
			MultilinearPolyVariant::ZeroPadded(inner) => {
				let (inner_eval, subproof) = match evalcheck_proof {
					EvalcheckProof::ZeroPadded(eval, subproof) => (eval, subproof),
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
					*subproof,
					inner,
					subclaim_eval_point,
				)?;
			}
			MultilinearPolyVariant::Composite(composition) => {
				if evalcheck_proof != EvalcheckProof::CompositeMLE {
					return Err(VerificationError::SubproofMismatch.into());
				}

				let meta = composite_sumcheck_meta(&mut self.oracles, id, &eval_point)?;
				add_composite_sumcheck_to_constraints(
					meta,
					&mut self.new_sumcheck_constraints,
					composition,
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
