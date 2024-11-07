// Copyright 2024 Irreducible Inc.

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckMultilinearClaim, EvalcheckProof,
	},
	subclaims::MemoizedQueries,
};
use crate::{
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyOracle, ProjectionVariant,
	},
	protocols::evalcheck::subclaims::{process_packed_sumcheck, process_shifted_sumcheck},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use getset::{Getters, MutGetters};
use tracing::instrument;

/// A mutable prover state.
///
/// Can be persisted across [`EvalcheckProver::prove`] invocations. Accumulates
/// `new_sumchecks` bivariate sumcheck instances, as well as holds mutable references to
/// the trace (to which new oracles & multilinears may be added during proving)
#[derive(Getters, MutGetters)]
pub struct EvalcheckProver<'a, 'b, U, F, Backend>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,
	pub(crate) witness_index: &'a mut MultilinearExtensionIndex<'b, U, F>,

	#[getset(get = "pub", get_mut = "pub")]
	pub(crate) batch_committed_eval_claims: BatchCommittedEvalClaims<F>,

	new_sumchecks_constraints: Vec<ConstraintSetBuilder<PackedType<U, F>>>,
	memoized_queries: MemoizedQueries<PackedType<U, F>, Backend>,

	backend: &'a Backend,
}

impl<'a, 'b, U, F, Backend> EvalcheckProver<'a, 'b, U, F, Backend>
where
	U: UnderlierType + PackScalar<F>,
	PackedType<U, F>: PackedFieldIndexable,
	F: TowerField,
	Backend: ComputationBackend,
{
	/// Create a new prover state by tying together the mutable references to the oracle set and
	/// witness index (they need to be mutable because `new_sumcheck` reduction may add new oracles & multilinears)
	/// as well as committed eval claims accumulator.
	pub fn new(
		oracles: &'a mut MultilinearOracleSet<F>,
		witness_index: &'a mut MultilinearExtensionIndex<'b, U, F>,
		backend: &'a Backend,
	) -> Self {
		let memoized_queries = MemoizedQueries::new();
		let new_sumchecks_constraints = Vec::new();
		let batch_committed_eval_claims =
			BatchCommittedEvalClaims::new(&oracles.committed_batches());

		Self {
			oracles,
			witness_index,
			batch_committed_eval_claims,
			new_sumchecks_constraints,
			memoized_queries,
			backend,
		}
	}

	/// A helper method to move out sumcheck constraints
	pub fn take_new_sumchecks_constraints(
		&mut self,
	) -> Result<Vec<ConstraintSet<PackedType<U, F>>>, OracleError> {
		self.new_sumchecks_constraints
			.iter_mut()
			.map(|builder| std::mem::take(builder).build_one(self.oracles))
			.filter(|constraint| !matches!(constraint, Err(OracleError::EmptyConstraintSet)))
			.rev()
			.collect()
	}

	/// Prove an evalcheck claim.
	///
	/// Given a prover state containing [`MultilinearOracleSet`] indexing into given
	/// [`MultilinearExtensionIndex`], we prove an [`EvalcheckMultilinearClaim`] (stating that given composite
	/// `poly` equals `eval` at `eval_point`) by recursively processing each of the multilinears.
	/// This way the evalcheck claim gets transformed into an [`EvalcheckProof`]
	/// and a new set of claims on:
	///  * PCS openings (which get inserted into [`BatchCommittedEvalClaims`] accumulator)
	///  * New sumcheck constraints that need to be proven in subsequent rounds (those get appended to `new_sumchecks`)
	///
	/// All of the `new_sumchecks` constraints follow the same pattern:
	///  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
	///  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
	///  * other multilin is a projection of one of the evalcheck claim multilins to its first variables
	#[instrument(skip_all, name = "EvalcheckProverState::prove", level = "debug")]
	pub fn prove(
		&mut self,
		evalcheck_claims: Vec<EvalcheckMultilinearClaim<F>>,
	) -> Result<Vec<EvalcheckProof<F>>, Error> {
		evalcheck_claims
			.into_iter()
			.map(|claim| self.prove_multilinear(claim))
			.collect::<Result<Vec<_>, _>>()
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::prove_multilinear",
		level = "debug"
	)]
	fn prove_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
	) -> Result<EvalcheckProof<F>, Error> {
		let EvalcheckMultilinearClaim {
			poly: multilinear,
			eval_point,
			eval,
		} = evalcheck_claim;

		use MultilinearPolyOracle::*;

		let proof = match multilinear {
			Transparent { .. } => EvalcheckProof::Transparent,

			Committed { id, .. } => {
				let subclaim = CommittedEvalClaim {
					id,
					eval_point,
					eval,
				};

				self.batch_committed_eval_claims.insert(subclaim);
				EvalcheckProof::Committed
			}

			Repeating { inner, .. } => {
				let n_vars = inner.n_vars();
				let inner_eval_point = eval_point[..n_vars].to_vec();
				let subclaim = EvalcheckMultilinearClaim {
					poly: *inner,
					eval_point: inner_eval_point,
					eval,
				};

				let subproof = self.prove_multilinear(subclaim)?;
				EvalcheckProof::Repeating(Box::new(subproof))
			}

			Merged { poly0, poly1, .. } => {
				let n_vars = poly0.n_vars();
				assert_eq!(poly0.n_vars(), poly1.n_vars());
				let inner_eval_point = &eval_point[..n_vars];

				let (eval1, subproof1) = self.eval_and_proof(*poly0, inner_eval_point)?;
				let (eval2, subproof2) = self.eval_and_proof(*poly1, inner_eval_point)?;

				EvalcheckProof::Merged {
					eval1,
					eval2,
					subproof1: Box::new(subproof1),
					subproof2: Box::new(subproof2),
				}
			}
			Interleaved { poly0, poly1, .. } => {
				assert_eq!(poly0.n_vars(), poly1.n_vars());
				let inner_eval_point = &eval_point[1..];

				let (eval1, subproof1) = self.eval_and_proof(*poly0, inner_eval_point)?;
				let (eval2, subproof2) = self.eval_and_proof(*poly1, inner_eval_point)?;

				EvalcheckProof::Interleaved {
					eval1,
					eval2,
					subproof1: Box::new(subproof1),
					subproof2: Box::new(subproof2),
				}
			}

			Shifted { shifted, .. } => {
				process_shifted_sumcheck(
					self.oracles,
					&shifted,
					eval_point.as_slice(),
					eval,
					self.witness_index,
					&mut self.memoized_queries,
					&mut self.new_sumchecks_constraints,
					self.backend,
				)?;
				EvalcheckProof::Shifted
			}

			Packed { packed, .. } => {
				process_packed_sumcheck(
					self.oracles,
					&packed,
					eval_point.as_slice(),
					eval,
					self.witness_index,
					&mut self.memoized_queries,
					&mut self.new_sumchecks_constraints,
					self.backend,
				)?;

				EvalcheckProof::Packed
			}

			Projected { projected, .. } => {
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

				let new_poly = *inner.clone();

				let subclaim = EvalcheckMultilinearClaim {
					poly: new_poly,
					eval_point: new_eval_point,
					eval,
				};

				self.prove_multilinear(subclaim)?
			}

			LinearCombination {
				linear_combination, ..
			} => {
				let subproofs = linear_combination
					.polys()
					.cloned()
					.map(|suboracle| self.eval_and_proof(suboracle, &eval_point))
					.collect::<Result<_, Error>>()?;

				EvalcheckProof::Composite { subproofs }
			}

			ZeroPadded { inner, .. } => {
				let inner_n_vars = inner.n_vars();

				let inner_eval_point = &eval_point[..inner_n_vars];
				let (eval, subproof) = self.eval_and_proof(*inner, inner_eval_point)?;

				EvalcheckProof::ZeroPadded(eval, Box::new(subproof))
			}
		};

		Ok(proof)
	}

	fn eval_and_proof(
		&mut self,
		poly: MultilinearPolyOracle<F>,
		eval_point: &[F],
	) -> Result<(F, EvalcheckProof<F>), Error> {
		let eval_query = self.memoized_queries.full_query(eval_point, self.backend)?;
		let witness_poly = self
			.witness_index
			.get_multilin_poly(poly.id())
			.map_err(Error::Witness)?;
		let eval = witness_poly.evaluate(eval_query.to_ref())?;
		let subclaim = EvalcheckMultilinearClaim {
			poly,
			eval_point: eval_point.to_vec(),
			eval,
		};
		let subproof = self.prove_multilinear(subclaim)?;
		Ok((eval, subproof))
	}
}
