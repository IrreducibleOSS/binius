// Copyright 2024-2025 Irreducible Inc.

use std::collections::HashSet;

use binius_field::{PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use binius_maybe_rayon::prelude::*;
use getset::{Getters, MutGetters};
use itertools::chain;
use tracing::instrument;

use super::{
	error::Error,
	evalcheck::{EvalcheckHint, EvalcheckMultilinearClaim},
	serialize_evalcheck_proof,
	subclaims::{
		add_composite_sumcheck_to_constraints, calculate_projected_mles, composite_mlecheck_meta,
		fill_eq_witness_for_composites, MemoizedData, ProjectedBivariateMeta, SumcheckClaims,
	},
	EvalPoint, EvalPointOracleIdMap,
};
use crate::{
	fiat_shamir::Challenger,
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyOracle, MultilinearPolyVariant, OracleId,
	},
	protocols::evalcheck::{
		logging::MLEFoldHighDimensionsData,
		subclaims::{
			packed_sumcheck_meta, process_packed_sumcheck, process_shifted_sumcheck,
			shifted_sumcheck_meta, CompositeMLECheckMeta,
		},
	},
	transcript::ProverTranscript,
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
	/// Mutable reference to the oracle set which is modified to create new claims arising from
	/// sumchecks
	pub(crate) oracles: &'a mut MultilinearOracleSet<F>,
	/// Mutable reference to the witness index which is is populated by the prover for new claims
	/// arising from sumchecks
	pub(crate) witness_index: &'a mut MultilinearExtensionIndex<'b, P>,

	/// The committed evaluation claims arising in this round
	#[getset(get = "pub", get_mut = "pub")]
	committed_eval_claims: Vec<EvalcheckMultilinearClaim<F>>,

	// Internally used to collect subclaims with evaluations to consume and further reduce.
	claims_queue: Vec<EvalcheckMultilinearClaim<F>>,
	// Internally used to collect subclaims without evaluations for future query and memoization
	claims_without_evals: Vec<(MultilinearPolyOracle<F>, EvalPoint<F>)>,
	// The list of claims that reduces to a bivariate sumcheck in a round.
	sumcheck_claims: Vec<SumcheckClaims<P::Scalar>>,

	// The new sumcheck constraints arising in this round
	new_sumchecks_constraints: Vec<ConstraintSetBuilder<F>>,
	// Tensor expansion of evaluation points and partial evaluations of multilinears
	pub memoized_data: MemoizedData<'b, P, Backend>,
	backend: &'a Backend,

	// The unique index of a claim in this round.
	claim_to_index: EvalPointOracleIdMap<usize, F>,
	// Claims that have been visited in this round, used to deduplicate claims when collecting
	// subclaims in a BFS manner.
	visited_claims: EvalPointOracleIdMap<(), F>,
	// Memoization of evaluations of claims the prover sees in this round
	evals_memoization: EvalPointOracleIdMap<F, F>,
	// The index of the next claim to be verified
	round_claim_index: usize,
}

impl<'a, 'b, F, P, Backend> EvalcheckProver<'a, 'b, F, P, Backend>
where
	P: PackedField<Scalar = F>,
	F: TowerField,
	Backend: ComputationBackend,
{
	/// Create a new prover state by tying together the mutable references to the oracle set and
	/// witness index (they need to be mutable because `new_sumcheck` reduction may add new oracles
	/// & multilinears) as well as committed eval claims accumulator.
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
			sumcheck_claims: Vec::new(),
			memoized_data: MemoizedData::new(),
			backend,

			claim_to_index: EvalPointOracleIdMap::new(),
			visited_claims: EvalPointOracleIdMap::new(),
			evals_memoization: EvalPointOracleIdMap::new(),
			round_claim_index: 0,
		}
	}

	/// A helper method to move out sumcheck constraints
	pub fn take_new_sumchecks_constraints(&mut self) -> Result<Vec<ConstraintSet<F>>, OracleError> {
		self.new_sumchecks_constraints
			.iter_mut()
			.map(|builder| std::mem::take(builder).build_one(self.oracles))
			.filter(|constraint| !matches!(constraint, Err(OracleError::EmptyConstraintSet)))
			.collect()
	}

	/// Prove an evalcheck claim.
	///
	/// Given a prover state containing [`MultilinearOracleSet`] indexing into given
	/// [`MultilinearExtensionIndex`], we prove an [`EvalcheckMultilinearClaim`] (stating that given
	/// composite `poly` equals `eval` at `eval_point`) by recursively processing each of the
	/// multilinears. This way the evalcheck claim gets transformed into an [`EvalcheckHint`]
	/// and a new set of claims on:
	///  * Committed polynomial evaluations
	///  * New sumcheck constraints that need to be proven in subsequent rounds (those get appended
	///    to `new_sumchecks`)
	///
	/// All of the `new_sumchecks` constraints follow the same pattern:
	///  * they are always a product of two multilins (composition polynomial is `BivariateProduct`)
	///  * one multilin (the multiplier) is transparent (`shift_ind`, `eq_ind`, or tower basis)
	///  * other multilin is a projection of one of the evalcheck claim multilins to its first
	///    variables
	pub fn prove<Challenger_: Challenger>(
		&mut self,
		evalcheck_claims: Vec<EvalcheckMultilinearClaim<F>>,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		// Reset the prover state for a new round.
		self.round_claim_index = 0;
		self.visited_claims.clear();
		self.claim_to_index.clear();
		self.evals_memoization.clear();

		for claim in &evalcheck_claims {
			if self
				.evals_memoization
				.get(claim.id, &claim.eval_point)
				.is_some()
			{
				continue;
			}

			self.evals_memoization
				.insert(claim.id, claim.eval_point.clone(), claim.eval);
		}

		self.claims_queue.extend(evalcheck_claims.clone());

		// Step 1: Use modified BFS to memoize evaluations. For each claim, if there is a subclaim
		// and we know the evaluation of the subclaim, we add the subclaim to the claims_queue
		// Otherwise, we find the evaluation of the claim by querying the witness data from the
		// oracle id and evaluation point
		let mle_fold_full_span = tracing::debug_span!(
			"[task] MLE Fold Full",
			phase = "evalcheck",
			perfetto_category = "task.main"
		)
		.entered();
		while !self.claims_without_evals.is_empty() || !self.claims_queue.is_empty() {
			while !self.claims_queue.is_empty() {
				std::mem::take(&mut self.claims_queue)
					.into_iter()
					.for_each(|claim| self.collect_subclaims_for_memoization(claim));
			}

			let mut deduplicated_claims_without_evals = HashSet::new();

			for (poly, eval_point) in std::mem::take(&mut self.claims_without_evals) {
				if self.evals_memoization.get(poly.id(), &eval_point).is_some() {
					continue;
				}

				deduplicated_claims_without_evals.insert((poly.id(), eval_point.clone()));
			}

			let deduplicated_eval_points = deduplicated_claims_without_evals
				.iter()
				.map(|(_, eval_point)| eval_point.as_ref())
				.collect::<Vec<_>>();

			// Tensor expansion of unique eval points.
			self.memoized_data
				.memoize_query_par(deduplicated_eval_points.iter().copied(), self.backend)?;

			// Query and fill missing evaluations.
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
				self.evals_memoization.insert(
					subclaim.id,
					subclaim.eval_point.clone(),
					subclaim.eval,
				);
			}

			subclaims
				.into_iter()
				.for_each(|claim| self.collect_subclaims_for_memoization(claim));
		}
		drop(mle_fold_full_span);

		// Step 2: Prove multilinears: For each claim, we prove the claim by recursively proving the
		// subclaims by stepping through subclaims in a DFS manner and deduplicating claims.
		for claim in evalcheck_claims {
			self.prove_multilinear(claim, transcript)?;
		}

		// Step 3: Process projected_bivariate_claims
		let mut projected_bivariate_metas = Vec::new();
		let mut composite_mle_metas = Vec::new();
		let mut projected_bivariate_claims = Vec::new();
		let mut composite_mle_claims = Vec::new();

		for claim in &self.sumcheck_claims {
			match claim {
				SumcheckClaims::Projected(claim) => {
					let meta = Self::projected_bivariate_meta(self.oracles, claim)?;
					projected_bivariate_metas.push(meta);
					projected_bivariate_claims.push(claim.clone())
				}
				SumcheckClaims::Composite(claim) => {
					let meta = composite_mlecheck_meta(self.oracles, &claim.eval_point)?;
					composite_mle_metas.push(meta);
					composite_mle_claims.push(claim.clone())
				}
			}
		}
		let dimensions_data = MLEFoldHighDimensionsData::new(projected_bivariate_claims.len());
		let evalcheck_mle_fold_high_span = tracing::debug_span!(
			"[task] (Evalcheck) MLE Fold High",
			phase = "evalcheck",
			perfetto_category = "task.main",
			?dimensions_data,
		)
		.entered();

		let projected_mles = calculate_projected_mles(
			&projected_bivariate_metas,
			&mut self.memoized_data,
			&projected_bivariate_claims,
			self.witness_index,
			self.backend,
		)?;
		drop(evalcheck_mle_fold_high_span);

		fill_eq_witness_for_composites(
			&composite_mle_metas,
			&mut self.memoized_data,
			&composite_mle_claims,
			self.witness_index,
			self.backend,
		)?;

		let mut projected_index = 0;
		let mut composite_index = 0;

		for claim in std::mem::take(&mut self.sumcheck_claims) {
			match claim {
				SumcheckClaims::Projected(claim) => {
					let meta = &projected_bivariate_metas[projected_index];
					let projected = projected_mles[projected_index].clone();
					self.process_bivariate_sumcheck(&claim, meta, projected)?;
					projected_index += 1;
				}
				SumcheckClaims::Composite(claim) => {
					let meta = composite_mle_metas[composite_index];
					self.process_composite_mlecheck(&claim, meta)?;
					composite_index += 1;
				}
			}
		}

		self.memoized_data.memoize_partial_evals(
			&projected_bivariate_metas,
			&projected_bivariate_claims,
			self.oracles,
			self.witness_index,
		);

		Ok(())
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::collect_subclaims_for_precompute",
		level = "debug"
	)]
	fn collect_subclaims_for_memoization(&mut self, evalcheck_claim: EvalcheckMultilinearClaim<F>) {
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
						let eval = if let Some(eval) =
							self.evals_memoization.get(suboracle_id, &eval_point)
						{
							*eval
						} else {
							let eval = (eval - linear_combination.offset())
								* coeff.invert().expect("not zero");
							self.evals_memoization
								.insert(suboracle_id, eval_point.clone(), eval);
							eval
						};

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

			MultilinearPolyVariant::ZeroPadded(padded) => {
				let id = padded.id();
				let inner = self.oracles.oracle(id);
				let inner_eval_point = chain!(
					&eval_point[..padded.start_index()],
					&eval_point[padded.start_index() + padded.n_pad_vars()..],
				)
				.copied()
				.collect::<Vec<_>>();
				self.claims_without_evals
					.push((inner, inner_eval_point.into()));
			}
			_ => return,
		};
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::prove_multilinear",
		level = "debug"
	)]
	fn prove_multilinear<Challenger_: Challenger>(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		let EvalcheckMultilinearClaim { id, eval_point, .. } = &evalcheck_claim;

		let claim_id = self.claim_to_index.get(*id, eval_point);

		if let Some(claim_id) = claim_id {
			serialize_evalcheck_proof(
				&mut transcript.message(),
				&EvalcheckHint::DuplicateClaim(*claim_id as u32),
			);
			return Ok(());
		}
		serialize_evalcheck_proof(&mut transcript.message(), &EvalcheckHint::NewClaim);

		self.prove_multilinear_skip_duplicate_check(evalcheck_claim, transcript)
	}

	fn prove_multilinear_skip_duplicate_check<Challenger_: Challenger>(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error> {
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		self.claim_to_index
			.insert(id, eval_point.clone(), self.round_claim_index);

		self.round_claim_index += 1;

		let multilinear = self.oracles.oracle(id);

		match multilinear.variant {
			MultilinearPolyVariant::Transparent { .. } => {}
			MultilinearPolyVariant::Committed => {
				self.committed_eval_claims.push(EvalcheckMultilinearClaim {
					id: multilinear.id,
					eval_point,
					eval,
				});
			}
			MultilinearPolyVariant::Repeating {
				id: inner_id,
				log_count,
			} => {
				let n_vars = eval_point.len() - log_count;
				self.prove_multilinear(
					EvalcheckMultilinearClaim {
						id: inner_id,
						eval_point: eval_point.slice(0..n_vars),
						eval,
					},
					transcript,
				)?;
			}
			MultilinearPolyVariant::Projected(projected) => {
				let new_eval_point = {
					let (lo, hi) = eval_point.split_at(projected.start_index());
					chain!(lo, projected.values(), hi)
						.copied()
						.collect::<Vec<_>>()
				};

				self.prove_multilinear(
					EvalcheckMultilinearClaim {
						id: projected.id(),
						eval_point: new_eval_point.into(),
						eval,
					},
					transcript,
				)?;
			}
			MultilinearPolyVariant::Shifted { .. } | MultilinearPolyVariant::Packed { .. } => {
				let claim = EvalcheckMultilinearClaim {
					id,
					eval_point,
					eval,
				};

				self.sumcheck_claims.push(SumcheckClaims::Projected(claim));
			}
			MultilinearPolyVariant::Composite { .. } => {
				let claim = EvalcheckMultilinearClaim {
					id,
					eval_point,
					eval,
				};

				self.sumcheck_claims.push(SumcheckClaims::Composite(claim));
			}
			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				for suboracle_id in linear_combination.polys() {
					if let Some(claim_index) = self.claim_to_index.get(suboracle_id, &eval_point) {
						serialize_evalcheck_proof(
							&mut transcript.message(),
							&EvalcheckHint::DuplicateClaim(*claim_index as u32),
						);
					} else {
						serialize_evalcheck_proof(
							&mut transcript.message(),
							&EvalcheckHint::NewClaim,
						);

						let eval = *self
							.evals_memoization
							.get(suboracle_id, &eval_point)
							.expect("precomputed above");

						transcript.message().write_scalar(eval);

						self.prove_multilinear_skip_duplicate_check(
							EvalcheckMultilinearClaim {
								id: suboracle_id,
								eval_point: eval_point.clone(),
								eval,
							},
							transcript,
						)?;
					}
				}
			}
			MultilinearPolyVariant::ZeroPadded(padded) => {
				let inner_eval_point = chain!(
					&eval_point[..padded.start_index()],
					&eval_point[padded.start_index() + padded.n_pad_vars()..],
				)
				.copied()
				.collect::<Vec<_>>();

				let inner_eval = *self
					.evals_memoization
					.get(padded.id(), &inner_eval_point)
					.expect("precomputed above");

				self.prove_multilinear(
					EvalcheckMultilinearClaim {
						id: padded.id(),
						eval_point: inner_eval_point.into(),
						eval: inner_eval,
					},
					transcript,
				)?;
			}
		}
		Ok(())
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
			_ => unreachable!(),
		}
	}

	fn process_bivariate_sumcheck(
		&mut self,
		evalcheck_claim: &EvalcheckMultilinearClaim<F>,
		meta: &ProjectedBivariateMeta,
		projected: Option<MultilinearExtension<P>>,
	) -> Result<(), Error> {
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim;

		match self.oracles.oracle(*id).variant {
			MultilinearPolyVariant::Shifted(shifted) => process_shifted_sumcheck(
				&shifted,
				meta,
				eval_point,
				*eval,
				self.witness_index,
				&mut self.new_sumchecks_constraints,
				projected,
			),

			MultilinearPolyVariant::Packed(packed) => process_packed_sumcheck(
				self.oracles,
				&packed,
				meta,
				eval_point,
				*eval,
				self.witness_index,
				&mut self.new_sumchecks_constraints,
				projected,
			),

			_ => unreachable!(),
		}
	}

	fn process_composite_mlecheck(
		&mut self,
		evalcheck_claim: &EvalcheckMultilinearClaim<F>,
		meta: CompositeMLECheckMeta,
	) -> Result<(), Error> {
		let EvalcheckMultilinearClaim {
			id,
			eval_point: _,
			eval,
		} = evalcheck_claim;

		match self.oracles.oracle(*id).variant {
			MultilinearPolyVariant::Composite(composite) => {
				// witness for eq MLE has been previously filled in `fill_eq_witness_for_composites`
				add_composite_sumcheck_to_constraints(
					meta,
					&mut self.new_sumchecks_constraints,
					&composite,
					*eval,
				);
				Ok(())
			}
			_ => unreachable!(),
		}
	}

	/// Function that queries the witness data from the oracle id and evaluation point to find the
	/// evaluation of the multilinear
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
