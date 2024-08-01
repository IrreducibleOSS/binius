// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	evalcheck::{
		BatchCommittedEvalClaims, CommittedEvalClaim, EvalcheckClaim, EvalcheckMultilinearClaim,
		EvalcheckProof,
	},
	subclaims::{
		packed_sumcheck_meta, packed_sumcheck_witness, projected_bivariate_claim,
		shifted_sumcheck_meta, shifted_sumcheck_witness, BivariateSumcheck, MemoizedQueries,
		MemoizedTransparentPolynomials,
	},
};
use crate::{
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, ProjectionVariant, ShiftVariant},
	witness::MultilinearWitnessIndex,
};
use binius_field::{PackedField, PackedFieldIndexable, TowerField};
use getset::{Getters, MutGetters};
use std::mem;
use tracing::instrument;

/// A mutable prover state.
///
/// Can be persisted across [`EvalcheckProver::prove`] invocations. Accumulates
/// `new_sumchecks` bivariate sumcheck instances, as well as holds mutable references to
/// the trace (to which new oracles & multilinears may be added during proving)
#[derive(Getters, MutGetters)]
pub struct EvalcheckProver<'a, 'b, F: TowerField, PW: PackedField> {
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,
	pub(crate) witness_index: &'a mut MultilinearWitnessIndex<'b, PW>,

	#[getset(get = "pub", get_mut = "pub")]
	pub(crate) batch_committed_eval_claims: BatchCommittedEvalClaims<F>,

	pub(crate) memoized_eq_ind: MemoizedTransparentPolynomials<Vec<F>>,
	pub(crate) memoized_shift_ind: MemoizedTransparentPolynomials<(usize, ShiftVariant, Vec<F>)>,

	#[get = "pub"]
	new_sumchecks: Vec<BivariateSumcheck<'b, F, PW>>,
	memoized_queries: MemoizedQueries<PW>,
}

impl<'a, 'b, F, PW> EvalcheckProver<'a, 'b, F, PW>
where
	F: TowerField + From<PW::Scalar>,
	PW: PackedFieldIndexable,
	PW::Scalar: TowerField + From<F>,
{
	/// Create a new prover state by tying together the mutable references to the oracle set and
	/// witness index (they need to be mutable because `new_sumcheck` reduction may add new oracles & multilinears)
	/// as well as committed eval claims accumulator.
	pub fn new(
		oracles: &'a mut MultilinearOracleSet<F>,
		witness_index: &'a mut MultilinearWitnessIndex<'b, PW>,
	) -> Self {
		let memoized_queries = MemoizedQueries::new();
		let memoized_eq_ind = MemoizedTransparentPolynomials::new();
		let memoized_shift_ind = MemoizedTransparentPolynomials::new();
		let new_sumchecks = Vec::new();
		let batch_committed_eval_claims =
			BatchCommittedEvalClaims::new(&oracles.committed_batches());

		Self {
			oracles,
			witness_index,
			batch_committed_eval_claims,
			new_sumchecks,
			memoized_queries,
			memoized_eq_ind,
			memoized_shift_ind,
		}
	}

	/// A helper method to move out the set of reduced sumcheck instances
	pub fn take_new_sumchecks(&mut self) -> Vec<BivariateSumcheck<'b, F, PW>> {
		mem::take(&mut self.new_sumchecks)
	}

	/// Prove an evalcheck claim.
	///
	/// Given a prover state containing [`MultilinearOracleSet`] indexing into given
	/// [`MultilinearWitnessIndex`], we prove an [`EvalcheckClaim`] (stating that given composite
	/// `poly` equals `eval` at `eval_point`) by recursively processing each of the multilinears in
	/// the composition. This way the evalcheck claim gets transformed into an [`EvalcheckProof`]
	/// and a new set of claims on:
	///  * PCS openings (which get inserted into [`BatchCommittedEvalClaims`] accumulator)
	///  * New sumcheck instances that need to be proven in subsequent rounds (those get appended to `new_sumchecks`)
	///
	/// All of the `new_sumchecks` instances follow the same pattern:
	///  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
	///  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
	///  * other multilin is a projection of one of the evalcheck claim multilins to its first variables
	#[instrument(skip_all, name = "EvalcheckProverState::prove")]
	pub fn prove(
		&mut self,
		evalcheck_claim: EvalcheckClaim<F>,
	) -> Result<EvalcheckProof<F>, Error> {
		let EvalcheckClaim {
			poly: composite,
			eval_point,
			is_random_point,
			..
		} = evalcheck_claim;

		self.prove_composite(composite.inner_polys().into_iter(), eval_point, is_random_point)
	}

	fn prove_composite(
		&mut self,
		multilin_oracles: impl Iterator<Item = MultilinearPolyOracle<F>>,
		eval_point: Vec<F>,
		is_random_point: bool,
	) -> Result<EvalcheckProof<F>, Error> {
		let wf_eval_point = eval_point
			.iter()
			.copied()
			.map(Into::into)
			.collect::<Vec<_>>();

		let subproofs = multilin_oracles
			.map(|suboracle| {
				self.eval_and_proof(suboracle, &eval_point, &wf_eval_point, is_random_point)
			})
			.collect::<Result<_, Error>>()?;

		Ok(EvalcheckProof::Composite { subproofs })
	}

	fn prove_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
	) -> Result<EvalcheckProof<F>, Error> {
		let EvalcheckMultilinearClaim {
			poly: multilinear,
			eval_point,
			eval,
			is_random_point,
		} = evalcheck_claim;

		let wf_eval_point = eval_point
			.iter()
			.copied()
			.map(Into::into)
			.collect::<Vec<_>>();

		use MultilinearPolyOracle::*;

		let proof = match multilinear {
			Transparent { .. } => EvalcheckProof::Transparent,

			Committed { id, .. } => {
				let subclaim = CommittedEvalClaim {
					id,
					eval_point,
					eval,
					is_random_point,
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
					is_random_point,
				};

				let subproof = self.prove_multilinear(subclaim)?;
				EvalcheckProof::Repeating(Box::new(subproof))
			}

			Merged(_id, poly1, poly2) => {
				let n_vars = poly1.n_vars();
				assert_eq!(poly1.n_vars(), poly2.n_vars());
				let inner_eval_point = &eval_point[..n_vars];
				let wf_inner_eval_point = &wf_eval_point[0..n_vars];

				let (eval1, subproof1) = self.eval_and_proof(
					*poly1,
					inner_eval_point,
					wf_inner_eval_point,
					is_random_point,
				)?;
				let (eval2, subproof2) = self.eval_and_proof(
					*poly2,
					inner_eval_point,
					wf_inner_eval_point,
					is_random_point,
				)?;

				EvalcheckProof::Merged {
					eval1,
					eval2,
					subproof1: Box::new(subproof1),
					subproof2: Box::new(subproof2),
				}
			}
			Interleaved(_id, poly1, poly2) => {
				assert_eq!(poly1.n_vars(), poly2.n_vars());
				let inner_eval_point = &eval_point[1..];
				let wf_inner_eval_point = &wf_eval_point[1..];

				let (eval1, subproof1) = self.eval_and_proof(
					*poly1,
					inner_eval_point,
					wf_inner_eval_point,
					is_random_point,
				)?;
				let (eval2, subproof2) = self.eval_and_proof(
					*poly2,
					inner_eval_point,
					wf_inner_eval_point,
					is_random_point,
				)?;

				EvalcheckProof::Interleaved {
					eval1,
					eval2,
					subproof1: Box::new(subproof1),
					subproof2: Box::new(subproof2),
				}
			}

			Shifted(_id, shifted) => {
				let meta = shifted_sumcheck_meta(
					self.oracles,
					&shifted,
					eval_point.as_slice(),
					Some(&mut self.memoized_shift_ind),
				)?;
				let sumcheck_claim = projected_bivariate_claim(self.oracles, meta, eval)?;
				let sumcheck_witness = shifted_sumcheck_witness(
					self.witness_index,
					&mut self.memoized_queries,
					meta,
					&shifted,
					&wf_eval_point,
				)?;

				self.new_sumchecks.push((sumcheck_claim, sumcheck_witness));
				EvalcheckProof::Shifted
			}

			Packed(_id, packed) => {
				let meta = packed_sumcheck_meta(self.oracles, &packed, eval_point.as_slice())?;
				let sumcheck_claim = projected_bivariate_claim(self.oracles, meta, eval)?;
				let sumcheck_witness = packed_sumcheck_witness(
					self.witness_index,
					&mut self.memoized_queries,
					meta,
					&packed,
					&wf_eval_point,
				)?;
				self.new_sumchecks.push((sumcheck_claim, sumcheck_witness));
				EvalcheckProof::Packed
			}

			Projected(_id, projected) => {
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
					is_random_point,
				};

				self.prove_multilinear(subclaim)?
			}

			LinearCombination(_id, lin_com) => {
				self.prove_composite(lin_com.polys().cloned(), eval_point, is_random_point)?
			}
		};

		Ok(proof)
	}

	fn eval_and_proof(
		&mut self,
		poly: MultilinearPolyOracle<F>,
		eval_point: &[F],
		wf_eval_point: &[PW::Scalar],
		is_random_point: bool,
	) -> Result<(F, EvalcheckProof<F>), Error> {
		let eval_query = self.memoized_queries.full_query(wf_eval_point)?;
		let witness_poly = self
			.witness_index
			.get(poly.id())
			.ok_or(Error::InvalidWitness(poly.id()))?;
		let eval = witness_poly.evaluate(eval_query)?.into();
		let subclaim = EvalcheckMultilinearClaim {
			poly,
			eval_point: eval_point.to_vec(),
			eval,
			is_random_point,
		};
		let subproof = self.prove_multilinear(subclaim)?;
		Ok((eval, subproof))
	}
}
