// Copyright 2024-2025 Irreducible Inc.

use std::{iter, mem};

use binius_field::{Field, TowerField, util::inner_product_unchecked};
use getset::{Getters, MutGetters};
use itertools::chain;
use tracing::instrument;

use super::{
	EvalPoint, deserialize_evalcheck_proof,
	error::{Error, VerificationError},
	evalcheck::{EvalcheckHint, EvalcheckMultilinearClaim},
	subclaims::{
		add_bivariate_sumcheck_to_constraints, add_composite_sumcheck_to_constraints,
		packed_sumcheck_meta, shifted_sumcheck_meta,
	},
};
use crate::{
	fiat_shamir::Challenger,
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyVariant, OracleId,
	},
	polynomial::MultivariatePoly,
	transcript::VerifierTranscript,
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
	/// Mutable reference to the oracle set which is modified to create new claims arising from
	/// sumchecks
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,

	/// The committed evaluation claims in this round
	#[getset(get = "pub", get_mut = "pub")]
	committed_eval_claims: Vec<EvalcheckMultilinearClaim<F>>,

	/// The new sumcheck constraints in this round
	new_sumcheck_constraints: Vec<ConstraintSetBuilder<F>>,

	// The new mle sumcheck constraints arising in this round
	new_mlechecks_constraints: Vec<(EvalPoint<F>, ConstraintSetBuilder<F>)>,

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
			new_mlechecks_constraints: Vec::new(),
			round_claims: Vec::new(),
		}
	}

	/// A helper method to move out sumcheck constraints
	pub fn take_new_sumcheck_constraints(&mut self) -> Result<Vec<ConstraintSet<F>>, OracleError> {
		self.new_sumcheck_constraints
			.iter_mut()
			.map(|builder| mem::take(builder).build_one(self.oracles))
			.filter(|constraint| !matches!(constraint, Err(OracleError::EmptyConstraintSet)))
			.collect()
	}

	/// A helper method to move out mlechecks constraints
	pub fn take_new_mlechecks_constraints(
		&mut self,
	) -> Result<ConstraintSetsEqIndPoints<F>, OracleError> {
		let new_mlechecks_constraints = std::mem::take(&mut self.new_mlechecks_constraints);

		let mut eq_ind_challenges = Vec::with_capacity(new_mlechecks_constraints.len());
		let mut constraint_sets = Vec::with_capacity(new_mlechecks_constraints.len());

		for (ep, builder) in new_mlechecks_constraints {
			eq_ind_challenges.push(ep.to_vec());
			constraint_sets.push(builder.build_one(self.oracles)?)
		}
		Ok(ConstraintSetsEqIndPoints {
			eq_ind_challenges,
			constraint_sets,
		})
	}

	/// Verify an evalcheck claim.
	///
	/// For each claim, we verify the proof by recursively verifying the subclaims in a DFS manner
	/// deduplicating previously verified claims
	/// See [`EvalcheckProver::prove`](`super::prove::EvalcheckProver::prove`) docs for more
	/// details.
	#[instrument(skip_all, name = "EvalcheckVerifierState::verify", level = "debug")]
	pub fn verify<Challenger_: Challenger>(
		&mut self,
		evalcheck_claims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		self.round_claims.clear();
		for claim in evalcheck_claims {
			self.verify_multilinear(claim, transcript)?;
		}
		Ok(())
	}

	fn verify_multilinear<Challenger_: Challenger>(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		let evalcheck_proof = deserialize_evalcheck_proof(&mut transcript.message())?;

		// If the proof is a duplicate claim, we need to check if the claim is already in the round
		// claims, which have been verified.
		if let EvalcheckHint::DuplicateClaim(index) = evalcheck_proof {
			if let Some(expected_claim) = self.round_claims.get(index as usize) {
				if *expected_claim == evalcheck_claim {
					return Ok(());
				}
			}
			return Err(VerificationError::DuplicateClaimMismatch.into());
		}

		self.verify_multilinear_skip_duplicate_check(evalcheck_claim, transcript)
	}

	fn verify_multilinear_skip_duplicate_check<Challenger_: Challenger>(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(), Error> {
		self.round_claims.push(evalcheck_claim.clone());

		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		let multilinear = &self.oracles[id];
		let multilinear_label = multilinear.label();
		match multilinear.variant {
			MultilinearPolyVariant::Transparent(ref inner) => {
				let actual_eval = inner.poly().evaluate(&eval_point)?;
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear_label).into());
				}
			}
			MultilinearPolyVariant::Structured(ref inner) => {
				// Here we need to extend the eval_point to the input domain of the arith circuit
				// by assigning zeroes to the variables.
				let eval_point: &[F] = &eval_point;
				let n_pad_zeros = inner.n_vars() - eval_point.len();
				let query = eval_point
					.iter()
					.copied()
					.chain(iter::repeat_n(F::ZERO, n_pad_zeros))
					.collect::<Vec<_>>();
				let actual_eval = inner.evaluate(&query)?;
				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear_label).into());
				}
			}

			MultilinearPolyVariant::Committed => {
				self.committed_eval_claims.push(EvalcheckMultilinearClaim {
					id,
					eval_point,
					eval,
				});
			}

			MultilinearPolyVariant::Repeating { id, log_count } => {
				let n_vars = eval_point.len() - log_count;
				self.verify_multilinear(
					EvalcheckMultilinearClaim {
						id,
						eval_point: eval_point.slice(0..n_vars),
						eval,
					},
					transcript,
				)?;
			}

			MultilinearPolyVariant::Projected(ref projected) => {
				let new_eval_point = {
					let (lo, hi) = eval_point.split_at(projected.start_index());
					chain!(lo, projected.values(), hi)
						.copied()
						.collect::<Vec<_>>()
				};

				self.verify_multilinear(
					EvalcheckMultilinearClaim {
						id: projected.id(),
						eval_point: new_eval_point.into(),
						eval,
					},
					transcript,
				)?;
			}

			MultilinearPolyVariant::Shifted(ref shifted) => {
				let shifted = shifted.clone();
				let meta = shifted_sumcheck_meta(self.oracles, &shifted, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					shifted.block_size(),
					eval,
				)
			}

			MultilinearPolyVariant::Packed(ref packed) => {
				let packed = packed.clone();
				let meta = packed_sumcheck_meta(self.oracles, &packed, &eval_point)?;
				add_bivariate_sumcheck_to_constraints(
					&meta,
					&mut self.new_sumcheck_constraints,
					packed.log_degree(),
					eval,
				)
			}

			MultilinearPolyVariant::LinearCombination(ref linear_combination) => {
				let linear_combination = linear_combination.clone();
				let evals = linear_combination
					.polys()
					.map(|sub_oracle_id| {
						self.verify_multilinear_subclaim(
							sub_oracle_id,
							eval_point.clone(),
							transcript,
						)
					})
					.collect::<Result<Vec<_>, _>>()?;

				// Verify the evaluation of the linear combination over the claimed evaluations
				let actual_eval = linear_combination.offset()
					+ inner_product_unchecked::<F, F>(evals, linear_combination.coefficients());

				if actual_eval != eval {
					return Err(VerificationError::IncorrectEvaluation(multilinear_label).into());
				}
			}
			MultilinearPolyVariant::ZeroPadded(ref padded) => {
				let subclaim_eval_point = chain!(
					&eval_point[..padded.start_index()],
					&eval_point[padded.start_index() + padded.n_pad_vars()..],
				)
				.copied()
				.collect::<Vec<_>>();

				let zs =
					&eval_point[padded.start_index()..padded.start_index() + padded.n_pad_vars()];
				let select_row = SelectRow::new(zs.len(), padded.nonzero_index())?;
				let select_row_term = select_row
					.evaluate(zs)
					.expect("select_row is constructor with zs.len() variables");

				let inner_eval = match select_row_term.invert() {
					Some(invert) => eval * invert,
					None if eval.is_zero() => return Ok(()),
					None => {
						return Err(
							VerificationError::IncorrectEvaluation(multilinear_label).into()
						);
					}
				};

				self.verify_multilinear(
					EvalcheckMultilinearClaim {
						id: padded.id(),
						eval_point: subclaim_eval_point.into(),
						eval: inner_eval,
					},
					transcript,
				)?;
			}
			MultilinearPolyVariant::Composite(ref composite) => {
				let position = transcript.message().read::<u32>()? as usize;

				if let Some((constraints_eval_point, _)) =
					self.new_mlechecks_constraints.get(position)
				{
					if *constraints_eval_point != eval_point {
						return Err(VerificationError::MLECheckConstraintSetPositionMismatch.into());
					}
				}

				add_composite_sumcheck_to_constraints(
					position,
					&eval_point,
					&mut self.new_mlechecks_constraints,
					composite,
					eval,
				);
			}
		}

		Ok(())
	}

	fn verify_multilinear_subclaim<Challenger_: Challenger>(
		&mut self,
		oracle_id: OracleId,
		eval_point: EvalPoint<F>,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<F, Error> {
		// If the subproof is a duplicate claim, we need to check that the claim is already in the
		// round claims and return the evaluation. Otherwise, we verify the subclaim recursively.
		let subproof = deserialize_evalcheck_proof(&mut transcript.message())?;
		match subproof {
			EvalcheckHint::DuplicateClaim(index) => {
				let index = index as usize;
				if self.round_claims[index].id != oracle_id
					|| self.round_claims[index].eval_point != eval_point
				{
					return Err(VerificationError::DuplicateClaimMismatch.into());
				}

				Ok(self.round_claims[index].eval)
			}
			EvalcheckHint::NewClaim => {
				let eval = transcript.message().read_scalar()?;
				let subclaim = EvalcheckMultilinearClaim {
					id: oracle_id,
					eval_point,
					eval,
				};
				self.verify_multilinear_skip_duplicate_check(subclaim, transcript)?;
				Ok(eval)
			}
		}
	}
}

pub struct ConstraintSetsEqIndPoints<F: Field> {
	pub eq_ind_challenges: Vec<Vec<F>>,
	pub constraint_sets: Vec<ConstraintSet<F>>,
}
