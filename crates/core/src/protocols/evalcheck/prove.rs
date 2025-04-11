// Copyright 2024-2025 Irreducible Inc.

use std::collections::HashSet;

use binius_field::{PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use binius_maybe_rayon::prelude::*;
use getset::{Getters, MutGetters};
use itertools::izip;
use tracing::instrument;

use super::{
	error::Error,
	evalcheck::{EvalcheckMultilinearClaim, EvalcheckProof},
	subclaims::{
		add_composite_sumcheck_to_constraints, calculate_projected_mles, composite_sumcheck_meta,
		fill_eq_witness_for_composites, MemoizedData, ProjectedBivariateMeta,
	},
	EvalPoint, EvalPointOracleIdMap, EvalcheckProofAdvice, Subproof,
};
use crate::{
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyOracle, MultilinearPolyVariant, OracleId,
	},
	protocols::evalcheck::subclaims::{
		packed_sumcheck_meta, process_packed_sumcheck, process_shifted_sumcheck,
		shifted_sumcheck_meta,
	},
	witness::MultilinearExtensionIndex,
};

/// A mutable prover state.
///
/// Can be persisted across [`EvalcheckProver::prove`] invocations. Accumulates
/// `new_sumchecks` bivariate sumcheck instances, as well as holds mutable references to
/// the trace (to which new oracles & multilinears may be added during proving)
#[derive(Getters, MutGetters)]
pub struct EvalcheckProver<'a, 'b, F, P, Backend>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,
	pub(crate) witness_index: &'a mut MultilinearExtensionIndex<'b, P>,

	#[getset(get = "pub", get_mut = "pub")]
	committed_eval_claims: Vec<EvalcheckMultilinearClaim<F>>,

	claims_queue: Vec<EvalcheckMultilinearClaim<F>>,
	claims_without_evals: Vec<(MultilinearPolyOracle<F>, EvalPoint<F>)>,
	projected_bivariate_claims: Vec<EvalcheckMultilinearClaim<F>>,

	new_sumchecks_constraints: Vec<ConstraintSetBuilder<F>>,
	pub memoized_data: MemoizedData<'b, P, Backend>,
	backend: &'a Backend,

	claim_to_index: EvalPointOracleIdMap<usize, F>,
	visited_claims: EvalPointOracleIdMap<(), F>,
	new_evals_memoization: EvalPointOracleIdMap<F, F>,
	round_claim_index: usize,
	advices: Vec<EvalcheckProofAdvice>,
}

impl<'a, 'b, F, P, Backend> EvalcheckProver<'a, 'b, F, P, Backend>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	/// Create a new prover state by tying together the mutable references to the oracle set and
	/// witness index (they need to be mutable because `new_sumcheck` reduction may add new oracles & multilinears)
	/// as well as committed eval claims accumulator.
	pub fn new(
		oracles: &'a mut MultilinearOracleSet<F>,
		witness_index: &'a mut MultilinearExtensionIndex<'b, P>,
		backend: &'a Backend,
	) -> Self {
		Self {
			oracles,
			witness_index,
			committed_eval_claims: Vec::new(),
			new_sumchecks_constraints: Vec::new(),
			claims_queue: Vec::new(),
			claims_without_evals: Vec::new(),
			projected_bivariate_claims: Vec::new(),
			memoized_data: MemoizedData::new(),
			backend,

			claim_to_index: EvalPointOracleIdMap::new(),
			visited_claims: EvalPointOracleIdMap::new(),
			new_evals_memoization: EvalPointOracleIdMap::new(),
			round_claim_index: 0,
			advices: Vec::new(),
		}
	}

	/// A helper method to move out sumcheck constraints
	pub fn take_new_sumchecks_constraints(&mut self) -> Result<Vec<ConstraintSet<F>>, OracleError> {
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
	///  * Committed polynomial evaluations
	///  * New sumcheck constraints that need to be proven in subsequent rounds (those get appended to `new_sumchecks`)
	///
	/// All of the `new_sumchecks` constraints follow the same pattern:
	///  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
	///  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
	///  * other multilin is a projection of one of the evalcheck claim multilins to its first variables
	#[instrument(skip_all, name = "EvalcheckProver::prove", level = "debug")]
	pub fn prove(
		&mut self,
		evalcheck_claims: Vec<EvalcheckMultilinearClaim<F>>,
	) -> Result<ProofsWithAdvices<F>, Error> {
		self.round_claim_index = 0;
		self.visited_claims.clear();
		self.claim_to_index.clear();
		self.new_evals_memoization.clear();

		for claim in &evalcheck_claims {
			if self
				.new_evals_memoization
				.get(claim.id, &claim.eval_point)
				.is_some()
			{
				continue;
			}

			self.new_evals_memoization
				.insert(claim.id, claim.eval_point.clone(), claim.eval);
		}

		self.claims_queue.extend(evalcheck_claims.clone());

		// Step 1: Use modified BFS to memoize new evaluation claims.
		while !self.claims_without_evals.is_empty() || !self.claims_queue.is_empty() {
			while !self.claims_queue.is_empty() {
				std::mem::take(&mut self.claims_queue)
					.into_iter()
					.for_each(|claim| self.collect_subclaims_for_precompute(claim));
			}

			let mut deduplicated_claims_without_evals = HashSet::new();

			for (poly, eval_point) in std::mem::take(&mut self.claims_without_evals) {
				if self
					.new_evals_memoization
					.get(poly.id(), &eval_point)
					.is_some()
				{
					continue;
				}

				deduplicated_claims_without_evals.insert((poly.id(), eval_point.clone()));
			}

			let deduplicated_eval_points = deduplicated_claims_without_evals
				.iter()
				.map(|(_, eval_point)| eval_point.as_ref())
				.collect::<Vec<_>>();

			self.memoized_data
				.memoize_query_par(&deduplicated_eval_points, self.backend)?;

			// Make new evaluation claims in parallel.
			let subclaims = deduplicated_claims_without_evals
				.into_par_iter()
				.map(|(id, eval_point)| {
					Self::make_new_eval_claim(
						id,
						eval_point,
						self.witness_index,
						&self.memoized_data,
					)
				})
				.collect::<Result<Vec<_>, Error>>()?;

			for subclaim in &subclaims {
				self.new_evals_memoization.insert(
					subclaim.id,
					subclaim.eval_point.clone(),
					subclaim.eval,
				);
			}

			subclaims
				.into_iter()
				.for_each(|claim| self.collect_subclaims_for_precompute(claim));
		}

		// Step 2: Prove multilinears

		let proofs = evalcheck_claims
			.iter()
			.cloned()
			.map(|claim| self.prove_multilinear(claim))
			.collect::<Result<Vec<_>, Error>>();

		// Step 3: Process projected_bivariate_claims
		let projected_bivariate_metas = self
			.projected_bivariate_claims
			.iter()
			.map(|claim| Self::projected_bivariate_meta(self.oracles, claim))
			.collect::<Result<Vec<_>, Error>>()?;

		let projected_mles = calculate_projected_mles(
			&projected_bivariate_metas,
			&mut self.memoized_data,
			&self.projected_bivariate_claims,
			self.witness_index,
			self.backend,
		)?;

		fill_eq_witness_for_composites(
			&projected_bivariate_metas,
			&mut self.memoized_data,
			&self.projected_bivariate_claims,
			self.witness_index,
			self.backend,
		)?;

		for (claim, meta, projected) in izip!(
			std::mem::take(&mut self.projected_bivariate_claims),
			&projected_bivariate_metas,
			projected_mles
		) {
			self.process_sumcheck(claim, meta, projected)?;
		}

		self.memoized_data.memoize_partial_evals(
			&projected_bivariate_metas,
			&self.projected_bivariate_claims,
			self.oracles,
			self.witness_index,
		);

		proofs.map(|proofs| ProofsWithAdvices {
			proofs,
			advices: std::mem::take(&mut self.advices),
		})
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::collect_subclaims_for_precompute",
		level = "debug"
	)]
	fn collect_subclaims_for_precompute(&mut self, evalcheck_claim: EvalcheckMultilinearClaim<F>) {
		let multilinear_id = evalcheck_claim.id;

		let eval_point = evalcheck_claim.eval_point;

		let eval = evalcheck_claim.eval;

		if self
			.visited_claims
			.get(multilinear_id, &eval_point)
			.is_some()
		{
			return;
		}

		self.visited_claims
			.insert(multilinear_id, eval_point.clone(), ());

		let multilinear = self.oracles.oracle(multilinear_id);

		match multilinear.variant {
			MultilinearPolyVariant::Repeating { id, .. } => {
				let n_vars = self.oracles.n_vars(id);
				let inner_eval_point = eval_point.slice(0..n_vars);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: inner_eval_point,
					eval,
				};
				self.claims_queue.push(subclaim);
			}

			MultilinearPolyVariant::Projected(projected) => {
				let (id, values) = (projected.id(), projected.values());
				let new_eval_point = {
					let idx = projected.start_index();
					let mut new_eval_point = eval_point[0..idx].to_vec();
					new_eval_point.extend(values.clone());
					new_eval_point.extend(eval_point[idx..].to_vec());
					new_eval_point
				};

				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: new_eval_point.into(),
					eval,
				};
				self.claims_queue.push(subclaim);
			}

			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let n_polys = linear_combination.n_polys();

				match linear_combination
					.polys()
					.zip(linear_combination.coefficients())
					.next()
				{
					Some((suboracle_id, coeff)) if n_polys == 1 && !coeff.is_zero() => {
						let eval = (eval - linear_combination.offset())
							* coeff.invert().expect("not zero");
						let subclaim = EvalcheckMultilinearClaim {
							id: suboracle_id,
							eval_point,
							eval,
						};
						self.claims_queue.push(subclaim);
					}
					_ => {
						for suboracle_id in linear_combination.polys() {
							self.claims_without_evals
								.push((self.oracles.oracle(suboracle_id), eval_point.clone()));
						}
					}
				};
			}

			MultilinearPolyVariant::ZeroPadded(id) => {
				let inner = self.oracles.oracle(id);
				let inner_n_vars = inner.n_vars();
				let inner_eval_point = eval_point.slice(0..inner_n_vars);
				self.claims_without_evals.push((inner, inner_eval_point));
			}
			_ => return,
		};
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::prove_multilinear",
		level = "debug"
	)]
	fn prove_multilinear(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
	) -> Result<Option<EvalcheckProof<F>>, Error> {
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim.clone();

		let claim_id = self.claim_to_index.get(id, &eval_point);

		if let Some(claim_id) = claim_id {
			self.advices
				.push(EvalcheckProofAdvice::DuplicateClaim(*claim_id));
			return Ok(None);
		}

		self.advices.push(EvalcheckProofAdvice::HandleClaim);

		self.claim_to_index
			.insert(id, eval_point.clone(), self.round_claim_index);

		self.round_claim_index += 1;

		let multilinear = self.oracles.oracle(id);
		let proof = match multilinear.variant {
			MultilinearPolyVariant::Transparent { .. } => Some(EvalcheckProof::Transparent),

			MultilinearPolyVariant::Committed => {
				let subclaim = EvalcheckMultilinearClaim {
					id: multilinear.id,
					eval_point,
					eval,
				};

				self.committed_eval_claims.push(subclaim);
				Some(EvalcheckProof::Committed)
			}
			MultilinearPolyVariant::Repeating { id, .. } => {
				let n_vars = self.oracles.n_vars(id);
				let inner_eval_point = eval_point.slice(0..n_vars);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: inner_eval_point,
					eval,
				};

				let subproof = self.prove_multilinear(subclaim)?;
				Some(EvalcheckProof::Repeating(Box::new(subproof)))
			}
			MultilinearPolyVariant::Projected(projected) => {
				let (id, values) = (projected.id(), projected.values());
				let new_eval_point = {
					let idx = projected.start_index();
					let mut new_eval_point = eval_point[0..idx].to_vec();
					new_eval_point.extend(values.clone());
					new_eval_point.extend(eval_point[idx..].to_vec());
					new_eval_point
				};

				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: new_eval_point.into(),
					eval,
				};
				Some(EvalcheckProof::Projected(Box::new(self.prove_multilinear(subclaim)?)))
			}
			MultilinearPolyVariant::Shifted { .. } => {
				self.projected_bivariate_claims.push(evalcheck_claim);
				Some(EvalcheckProof::Shifted)
			}
			MultilinearPolyVariant::Packed { .. } => {
				self.projected_bivariate_claims.push(evalcheck_claim);
				Some(EvalcheckProof::Packed)
			}
			MultilinearPolyVariant::Composite { .. } => {
				self.projected_bivariate_claims.push(evalcheck_claim);
				Some(EvalcheckProof::CompositeMLE)
			}
			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let n_polys = linear_combination.n_polys();

				let subproofs = match linear_combination
					.polys()
					.zip(linear_combination.coefficients())
					.next()
				{
					Some((suboracle_id, coeff)) if n_polys == 1 && !coeff.is_zero() => {
						let subclaim = if let Some(claim_index) =
							self.claim_to_index.get(suboracle_id, &eval_point)
						{
							Subproof::ExistingClaim(*claim_index)
						} else {
							let eval = (eval - linear_combination.offset())
								* coeff.invert().expect("not zero");
							let subclaim = EvalcheckMultilinearClaim {
								id: suboracle_id,
								eval_point,
								eval,
							};

							let proof = self.prove_multilinear(subclaim).unwrap();

							Subproof::NewProof { proof, eval }
						};

						vec![subclaim]
					}
					_ => linear_combination
						.polys()
						.map(|suboracle_id| {
							let eval = *self
								.new_evals_memoization
								.get(suboracle_id, &eval_point)
								.expect("precomputed above");

							let subclaim = EvalcheckMultilinearClaim {
								id: suboracle_id,
								eval_point: eval_point.clone(),
								eval,
							};

							let subclaim = if let Some(claim_index) =
								self.claim_to_index.get(suboracle_id, &eval_point)
							{
								Subproof::ExistingClaim(*claim_index)
							} else {
								let proof = self.prove_multilinear(subclaim).unwrap();

								Subproof::NewProof { proof, eval }
							};
							Ok(subclaim)
						})
						.collect::<Result<Vec<_>, Error>>()?,
				};

				Some(EvalcheckProof::LinearCombination { subproofs })
			}
			MultilinearPolyVariant::ZeroPadded(id) => {
				let inner_n_vars = self.oracles.n_vars(id);

				let inner_eval_point = &eval_point[..inner_n_vars];

				let eval = *self
					.new_evals_memoization
					.get(id, inner_eval_point)
					.expect("precomputed above");

				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: eval_point.clone(),
					eval,
				};

				let subproof = self.prove_multilinear(subclaim)?;

				Some(EvalcheckProof::ZeroPadded(eval, Box::new(subproof)))
			}
		};
		Ok(proof)
	}

	fn projected_bivariate_meta(
		oracles: &mut MultilinearOracleSet<F>,
		evalcheck_claim: &EvalcheckMultilinearClaim<F>,
	) -> Result<ProjectedBivariateMeta, Error> {
		let EvalcheckMultilinearClaim { id, eval_point, .. } = evalcheck_claim;

		match &oracles.oracle(*id).variant {
			MultilinearPolyVariant::Shifted(shifted) => {
				shifted_sumcheck_meta(oracles, shifted, eval_point)
			}
			MultilinearPolyVariant::Packed(packed) => {
				packed_sumcheck_meta(oracles, packed, eval_point)
			}
			MultilinearPolyVariant::Composite(_) => composite_sumcheck_meta(oracles, eval_point),
			_ => unreachable!(),
		}
	}

	fn process_sumcheck(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		meta: &ProjectedBivariateMeta,
		projected: Option<MultilinearExtension<P>>,
	) -> Result<(), Error> {
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		match self.oracles.oracle(id).variant {
			MultilinearPolyVariant::Shifted(shifted) => process_shifted_sumcheck(
				&shifted,
				meta,
				&eval_point,
				eval,
				self.witness_index,
				&mut self.new_sumchecks_constraints,
				projected,
			),

			MultilinearPolyVariant::Packed(packed) => process_packed_sumcheck(
				self.oracles,
				&packed,
				meta,
				&eval_point,
				eval,
				self.witness_index,
				&mut self.new_sumchecks_constraints,
				projected,
			),

			MultilinearPolyVariant::Composite(composite) => {
				// witness for eq MLE has been previously filled in `fill_eq_witness_for_composites`
				add_composite_sumcheck_to_constraints(
					meta,
					&mut self.new_sumchecks_constraints,
					&composite,
					eval,
				);
				Ok(())
			}
			_ => unreachable!(),
		}
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::make_new_eval_claim",
		level = "debug"
	)]
	fn make_new_eval_claim(
		oracle_id: OracleId,
		eval_point: EvalPoint<F>,
		witness_index: &MultilinearExtensionIndex<P>,
		memoized_queries: &MemoizedData<P, Backend>,
	) -> Result<EvalcheckMultilinearClaim<F>, Error> {
		let eval_query = memoized_queries
			.full_query_readonly(&eval_point)
			.ok_or(Error::MissingQuery)?;

		let witness_poly = witness_index
			.get_multilin_poly(oracle_id)
			.map_err(Error::Witness)?;

		let eval = witness_poly
			.evaluate(eval_query.to_ref())
			.map_err(Error::from)?;

		Ok(EvalcheckMultilinearClaim {
			id: oracle_id,
			eval_point,
			eval,
		})
	}
}

pub struct ProofsWithAdvices<F: Field> {
	pub proofs: Vec<Option<EvalcheckProof<F>>>,
	pub advices: Vec<EvalcheckProofAdvice>,
}
