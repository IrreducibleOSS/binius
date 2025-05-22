// Copyright 2024-2025 Irreducible Inc.

use std::collections::HashSet;

use binius_field::{Field, PackedField, TowerField};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use binius_maybe_rayon::prelude::*;
use getset::{Getters, MutGetters};
use itertools::{chain, izip};
use tracing::instrument;

use super::{
	EvalPoint, EvalPointOracleIdMap,
	error::Error,
	evalcheck::{EvalcheckHint, EvalcheckMultilinearClaim},
	serialize_evalcheck_proof,
	subclaims::{
		MemoizedData, ProjectedBivariateMeta, add_composite_sumcheck_to_constraints,
		calculate_projected_mles,
	},
};
use crate::{
	fiat_shamir::Challenger,
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyVariant, OracleId,
	},
	polynomial::MultivariatePoly,
	protocols::evalcheck::{
		logging::MLEFoldHighDimensionsData,
		subclaims::{
			packed_sumcheck_meta, process_packed_sumcheck, process_shifted_sumcheck,
			shifted_sumcheck_meta,
		},
	},
	transcript::ProverTranscript,
	transparent::select_row::SelectRow,
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

	// Claims that need to be evaluated.
	claims_to_be_evaluated: HashSet<(OracleId, EvalPoint<F>)>,

	// Claims that can be evaluated using internal_evals.
	claims_without_evals: HashSet<(OracleId, EvalPoint<F>)>,

	// The list of claims that reduces to a bivariate sumcheck in a round.
	projected_bivariate_claims: Vec<EvalcheckMultilinearClaim<F>>,

	// The new bivariate sumcheck constraints arising in this round
	new_bivariate_sumchecks_constraints: Vec<ConstraintSetBuilder<F>>,
	// The new mle sumcheck constraints arising in this round
	new_mlechecks_constraints: Vec<(EvalPoint<F>, ConstraintSetBuilder<F>)>,
	// Tensor expansion of evaluation points and partial evaluations of multilinears
	pub memoized_data: MemoizedData<'b, P, Backend>,
	backend: &'a Backend,

	// The unique index of a claim in this round.
	claim_to_index: EvalPointOracleIdMap<usize, F>,
	// Claims that have been visited in this round, used to deduplicate claims when collecting
	// subclaims.
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
			new_bivariate_sumchecks_constraints: Vec::new(),
			new_mlechecks_constraints: Vec::new(),
			claims_without_evals: HashSet::new(),
			claims_to_be_evaluated: HashSet::new(),
			projected_bivariate_claims: Vec::new(),
			memoized_data: MemoizedData::new(),
			backend,

			claim_to_index: EvalPointOracleIdMap::new(),
			visited_claims: EvalPointOracleIdMap::new(),
			evals_memoization: EvalPointOracleIdMap::new(),
			round_claim_index: 0,
		}
	}

	/// A helper method to move out bivariate sumcheck constraints
	pub fn take_new_bivariate_sumchecks_constraints(
		&mut self,
	) -> Result<Vec<ConstraintSet<F>>, OracleError> {
		self.new_bivariate_sumchecks_constraints
			.iter_mut()
			.map(|builder| std::mem::take(builder).build_one(self.oracles))
			.filter(|constraint| !matches!(constraint, Err(OracleError::EmptyConstraintSet)))
			.collect()
	}

	/// A helper method to move out mlechecks constraints
	pub fn take_new_mlechecks_constraints(
		&mut self,
	) -> Result<Vec<ConstraintSetEqIndPoint<F>>, OracleError> {
		std::mem::take(&mut self.new_mlechecks_constraints)
			.into_iter()
			.map(|(ep, builder)| {
				builder
					.build_one(self.oracles)
					.map(|constraint| ConstraintSetEqIndPoint {
						eq_ind_challenges: ep.clone(),
						constraint_set: constraint,
					})
			})
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

		let mle_fold_full_span = tracing::debug_span!(
			"[task] MLE Fold Full",
			phase = "evalcheck",
			perfetto_category = "task.main"
		)
		.entered();

		// Step 1: Precompute claims that require additional evaluations.
		for claim in &evalcheck_claims {
			self.collect_subclaims_for_memoization(
				claim.id,
				claim.eval_point.clone(),
				Some(claim.eval),
			);
		}

		let eval_points = self
			.claims_to_be_evaluated
			.iter()
			.map(|(_, eval_point)| eval_point.as_ref())
			.collect::<Vec<_>>();

		self.memoized_data
			.memoize_query_par(eval_points, self.backend)?;

		let subclaims = std::mem::take(&mut self.claims_to_be_evaluated)
			.into_par_iter()
			.map(|(id, eval_point)| {
				Self::make_new_eval_claim(id, eval_point, self.witness_index, &self.memoized_data)
			})
			.collect::<Result<Vec<_>, Error>>()?;

		for subclaim in &subclaims {
			self.evals_memoization
				.insert(subclaim.id, subclaim.eval_point.clone(), subclaim.eval);
		}

		let mut claims_without_evals = std::mem::take(&mut self.claims_without_evals)
			.into_iter()
			.collect::<Vec<_>>();

		claims_without_evals.sort_unstable_by_key(|(id, _)| *id);

		for (id, eval_point) in claims_without_evals {
			self.collect_evals(id, &eval_point);
		}

		drop(mle_fold_full_span);

		// Step 2: Prove multilinears: For each claim, we prove the claim by recursively proving the
		// subclaims by stepping through subclaims in a DFS manner and deduplicating claims.
		for claim in evalcheck_claims {
			self.prove_multilinear(claim, transcript)?;
		}

		// Step 3: Process projected_bivariate_claims
		let dimensions_data = MLEFoldHighDimensionsData::new(self.projected_bivariate_claims.len());
		let evalcheck_mle_fold_high_span = tracing::debug_span!(
			"[task] (Evalcheck) MLE Fold High",
			phase = "evalcheck",
			perfetto_category = "task.main",
			?dimensions_data,
		)
		.entered();

		let projected_bivariate_metas = self
			.projected_bivariate_claims
			.iter()
			.map(|claim| Self::projected_bivariate_meta(self.oracles, claim))
			.collect::<Result<Vec<_>, Error>>()?;

		let projected_bivariate_claims = std::mem::take(&mut self.projected_bivariate_claims);

		let projected_mles = calculate_projected_mles(
			&projected_bivariate_metas,
			&mut self.memoized_data,
			&projected_bivariate_claims,
			self.witness_index,
			self.backend,
		)?;
		drop(evalcheck_mle_fold_high_span);

		// memoize eq_ind_partial_evals for HighToLow case
		self.memoized_data.memoize_query_par(
			self.new_mlechecks_constraints.iter().map(|(ep, _)| {
				let ep = ep.as_ref();
				&ep[0..ep.len().saturating_sub(1)]
			}),
			self.backend,
		)?;

		for (claim, meta, projected) in
			izip!(&projected_bivariate_claims, &projected_bivariate_metas, projected_mles)
		{
			self.process_bivariate_sumcheck(claim, meta, projected)?;
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
		name = "EvalcheckProverState::collect_subclaims_for_memoization",
		level = "debug"
	)]
	fn collect_subclaims_for_memoization(
		&mut self,
		multilinear_id: OracleId,
		eval_point: EvalPoint<F>,
		eval: Option<F>,
	) {
		if self
			.visited_claims
			.get(multilinear_id, &eval_point)
			.is_some()
		{
			return;
		}

		self.visited_claims
			.insert(multilinear_id, eval_point.clone(), ());

		if let Some(eval) = eval {
			if self
				.evals_memoization
				.get(multilinear_id, &eval_point)
				.is_none()
			{
				self.evals_memoization
					.insert(multilinear_id, eval_point.clone(), eval);
			}
		}

		let multilinear = self.oracles.oracle(multilinear_id);

		match multilinear.variant {
			MultilinearPolyVariant::Repeating { id, .. } => {
				let n_vars = self.oracles.n_vars(id);
				let inner_eval_point = eval_point.slice(0..n_vars);
				self.collect_subclaims_for_memoization(id, inner_eval_point, eval);
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

				self.collect_subclaims_for_memoization(id, new_eval_point.into(), eval);
			}

			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let n_polys = linear_combination.n_polys();

				match (
					izip!(linear_combination.polys(), linear_combination.coefficients()).next(),
					eval,
				) {
					(Some((suboracle_id, coeff)), Some(eval))
						if n_polys == 1 && !coeff.is_zero() =>
					{
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

						self.collect_subclaims_for_memoization(
							suboracle_id,
							eval_point,
							Some(eval),
						);
					}
					_ => {
						for suboracle_id in linear_combination.polys() {
							self.claims_without_evals
								.insert((suboracle_id, eval_point.clone()));

							self.collect_subclaims_for_memoization(
								suboracle_id,
								eval_point.clone(),
								None,
							);
						}
					}
				};
			}

			MultilinearPolyVariant::ZeroPadded(padded) => {
				let id = padded.id();
				let inner_eval_point = chain!(
					&eval_point[..padded.start_index()],
					&eval_point[padded.start_index() + padded.n_pad_vars()..],
				)
				.copied()
				.collect::<Vec<_>>();
				let inner_eval_point = EvalPoint::from(inner_eval_point);

				self.claims_without_evals
					.insert((id, inner_eval_point.clone()));

				self.collect_subclaims_for_memoization(id, inner_eval_point, None);
			}
			_ => {
				if self
					.evals_memoization
					.get(multilinear_id, &eval_point)
					.is_none()
				{
					self.claims_to_be_evaluated
						.insert((multilinear_id, eval_point));
				}
			}
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

				self.projected_bivariate_claims.push(claim);
			}
			MultilinearPolyVariant::Composite(composite) => {
				let position = self
					.new_mlechecks_constraints
					.iter()
					.position(|(ep, _)| *ep == eval_point)
					.unwrap_or(self.new_mlechecks_constraints.len());

				transcript.decommitment().write(&(position as u32));

				add_composite_sumcheck_to_constraints(
					position,
					&eval_point,
					&mut self.new_mlechecks_constraints,
					&composite,
					eval,
				);
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

				let zs =
					&eval_point[padded.start_index()..padded.start_index() + padded.n_pad_vars()];
				let select_row = SelectRow::new(zs.len(), padded.nonzero_index())?;
				let select_row_term = select_row
					.evaluate(zs)
					.expect("select_row is constructor with zs.len() variables");

				if eval.is_zero() && select_row_term.is_zero() {
					return Ok(());
				}

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

	pub fn collect_evals(&mut self, oracle_id: OracleId, eval_point: &EvalPoint<F>) -> F {
		if let Some(eval) = self.evals_memoization.get(oracle_id, eval_point) {
			return *eval;
		}

		let eval = match &self.oracles[oracle_id].variant {
			MultilinearPolyVariant::Repeating { id, log_count } => {
				let n_vars = eval_point.len() - log_count;
				self.collect_evals(*id, &eval_point.slice(0..n_vars))
			}
			MultilinearPolyVariant::Projected(projected) => {
				let new_eval_point = {
					let (lo, hi) = eval_point.split_at(projected.start_index());
					chain!(lo, projected.values(), hi)
						.copied()
						.collect::<Vec<_>>()
				};
				self.collect_evals(projected.id(), &new_eval_point.into())
			}
			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				let ids = linear_combination.polys().collect::<Vec<_>>();

				let coeffs = linear_combination.coefficients().collect::<Vec<_>>();
				let offset = linear_combination.offset();

				let mut evals = Vec::with_capacity(ids.len());

				for id in &ids {
					evals.push(self.collect_evals(*id, eval_point));
				}

				izip!(evals, coeffs).fold(offset, |acc, (eval, coeff)| {
					if coeff.is_zero() {
						return acc;
					}

					acc + eval * coeff
				})
			}
			MultilinearPolyVariant::ZeroPadded(padded) => {
				let subclaim_eval_point = chain!(
					&eval_point[..padded.start_index()],
					&eval_point[padded.start_index() + padded.n_pad_vars()..],
				)
				.copied()
				.collect::<Vec<_>>();

				let zs =
					&eval_point[padded.start_index()..padded.start_index() + padded.n_pad_vars()];
				let select_row = SelectRow::new(zs.len(), padded.nonzero_index())
					.expect("SelectRow receives the correct parameters");
				let select_row_term = select_row
					.evaluate(zs)
					.expect("select_row is constructor with zs.len() variables");

				let eval = self.collect_evals(padded.id(), &subclaim_eval_point.into());

				eval * select_row_term
			}
			_ => unreachable!(),
		};

		self.evals_memoization
			.insert(oracle_id, eval_point.clone(), eval);
		eval
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
				&mut self.new_bivariate_sumchecks_constraints,
				projected,
			),

			MultilinearPolyVariant::Packed(packed) => process_packed_sumcheck(
				self.oracles,
				&packed,
				meta,
				eval_point,
				*eval,
				self.witness_index,
				&mut self.new_bivariate_sumchecks_constraints,
				projected,
			),

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

pub struct ConstraintSetEqIndPoint<F: Field> {
	pub eq_ind_challenges: EvalPoint<F>,
	pub constraint_set: ConstraintSet<F>,
}
