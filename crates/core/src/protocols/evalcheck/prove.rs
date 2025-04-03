// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use binius_maybe_rayon::prelude::*;
use getset::{Getters, MutGetters};
use itertools::izip;
use tracing::instrument;

use super::{
	error::Error,
	evalcheck::{EvalcheckMultilinearClaim, EvalcheckProofEnum},
	subclaims::{
		add_composite_sumcheck_to_constraints, calculate_projected_mles, composite_sumcheck_meta,
		fill_eq_witness_for_composites, MemoizedQueries, ProjectedBivariateMeta,
	},
	EvalPoint, EvalPointOracleIdMap, ProofIndex,
};
use crate::{
	oracle::{
		ConstraintSet, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet,
		MultilinearPolyOracle, MultilinearPolyVariant, OracleId, ProjectionVariant,
	},
	protocols::evalcheck::subclaims::{
		packed_sumcheck_meta, process_packed_sumcheck, process_shifted_sumcheck,
		shifted_sumcheck_meta,
	},
	witness::MultilinearExtensionIndex,
};

#[derive(Debug, Clone)]
struct EvalcheckMultilinearClaimPartial<F: Field> {
	id: OracleId,
	eval_point: EvalPoint<F>,
	eval: Option<F>,
}

#[derive(Debug, Clone)]
enum EvalcheckProofWithStatus<F: Field> {
	Completed {
		proof_id: ProofIndex,
		proof: EvalcheckProofEnum<F>,
		eval: F,
	},
	Incomplete {
		proof_id: ProofIndex,
		subclaims: Vec<EvalcheckMultilinearClaimPartial<F>>,
	},
	SumcheckInducing {
		proof_id: ProofIndex,
		proof: EvalcheckProofEnum<F>,
		eval: F,
	},
}

impl<F: Field> EvalcheckProofWithStatus<F> {
	fn get_proof_id(&self) -> ProofIndex {
		match self {
			EvalcheckProofWithStatus::Completed { proof_id, .. } => *proof_id,
			EvalcheckProofWithStatus::Incomplete { proof_id, .. } => *proof_id,
			EvalcheckProofWithStatus::SumcheckInducing { proof_id, .. } => *proof_id,
		}
	}
}

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
	committed_eval_claims: Vec<EvalcheckMultilinearClaim<F>>,

	proofs: EvalPointOracleIdMap<EvalcheckProofWithStatus<F>, F>,
	// finalized_proofs: EvalPointOracleIdMap<(usize, F, EvalcheckProofEnum<F>), F>,
	claims_queue: Vec<EvalcheckMultilinearClaim<F>>,
	incomplete_proof_claims: EvalPointOracleIdMap<EvalcheckMultilinearClaim<F>, F>,
	claims_without_evals: Vec<(MultilinearPolyOracle<F>, EvalPoint<F>)>,
	claims_without_evals_dedup: EvalPointOracleIdMap<(), F>,
	projected_bivariate_claims: Vec<EvalcheckMultilinearClaim<F>>,

	new_sumchecks_constraints: Vec<ConstraintSetBuilder<F>>,
	memoized_queries: MemoizedQueries<PackedType<U, F>, Backend>,
	backend: &'a Backend,

	proof_context_index: usize,
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
		Self {
			oracles,
			witness_index,
			committed_eval_claims: Vec::new(),
			new_sumchecks_constraints: Vec::new(),
			// finalized_proofs: EvalPointOracleIdMap::new(),
			claims_queue: Vec::new(),
			claims_without_evals: Vec::new(),
			claims_without_evals_dedup: EvalPointOracleIdMap::new(),
			projected_bivariate_claims: Vec::new(),
			memoized_queries: MemoizedQueries::new(),
			backend,
			incomplete_proof_claims: EvalPointOracleIdMap::new(),
			proofs: EvalPointOracleIdMap::new(),
			proof_context_index: 0,
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

	fn reset_context(&mut self) {
		self.proof_context_index = 0;
	}

	fn next_proof(&mut self) -> usize {
		let idx = self.proof_context_index;
		self.proof_context_index += 1;
		idx
	}

	fn add_completed_proof(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		proof: EvalcheckProofEnum<F>,
		eval: F,
	) -> ProofIndex {
		let idx = self.next_proof();
		self.proofs.insert(
			evalcheck_claim.id,
			evalcheck_claim.eval_point,
			EvalcheckProofWithStatus::Completed {
				proof_id: idx,
				proof,
				eval,
			},
		);
		idx
	}

	fn add_sumcheck_inducing(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		proof: EvalcheckProofEnum<F>,
		eval: F,
	) -> ProofIndex {
		let idx = self.next_proof();
		self.proofs.insert(
			evalcheck_claim.id,
			evalcheck_claim.eval_point,
			EvalcheckProofWithStatus::SumcheckInducing {
				proof_id: idx,
				proof,
				eval,
			},
		);
		idx
	}

	fn add_incomplete_proof(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		subclaims: impl IntoIterator<Item = EvalcheckMultilinearClaim<F>>,
	) -> ProofIndex {
		let idx = self.next_proof();
		self.proofs.insert(
			evalcheck_claim.id,
			evalcheck_claim.eval_point.clone(),
			EvalcheckProofWithStatus::Incomplete {
				proof_id: idx,
				subclaims: subclaims
					.into_iter()
					.map(|subclaim| EvalcheckMultilinearClaimPartial {
						id: subclaim.id,
						eval_point: subclaim.eval_point,
						eval: Some(subclaim.eval),
					})
					.collect(),
			},
		);
		idx
	}

	fn add_incomplete_proof_partial(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		subclaims: impl IntoIterator<Item = (OracleId, EvalPoint<F>)>,
	) -> ProofIndex {
		let idx = self.next_proof();
		self.proofs.insert(
			evalcheck_claim.id,
			evalcheck_claim.eval_point.clone(),
			EvalcheckProofWithStatus::Incomplete {
				proof_id: idx,
				subclaims: subclaims
					.into_iter()
					.map(|(id, eval_point)| EvalcheckMultilinearClaimPartial {
						id,
						eval_point,
						eval: None,
					})
					.collect(),
			},
		);
		idx
	}

	// TODO: Maybe better to return Result  at this point
	fn replace_incomplete_with_complete_proof(
		&mut self,
		evalcheck_claim: EvalcheckMultilinearClaim<F>,
		proof: EvalcheckProofEnum<F>,
		eval: F,
	) -> Option<()> {
		self.proofs
			.get_mut(evalcheck_claim.id, evalcheck_claim.eval_point.as_ref())
			.map(|old| {
				let proof_id = match old {
					EvalcheckProofWithStatus::Completed { .. } => None,
					// TODO: You can probably add more checks to the subclaim here if need be?? or maybe the caller side??
					EvalcheckProofWithStatus::Incomplete { proof_id, .. } => Some(*proof_id),
					EvalcheckProofWithStatus::SumcheckInducing { .. } => {
						panic!("You can only call this function to replace incomplete proof")
					}
				}?;
				let mut new = EvalcheckProofWithStatus::Completed {
					proof_id,
					proof,
					eval,
				};
				std::mem::swap(&mut new, old);
				Some(())
			})
			.flatten()
	}

	// TODO: Maybe better to return Result  at this point
	fn replace_sumcheck_inducing_with_complete(
		&mut self,
		evalcheck_claim: &EvalcheckMultilinearClaim<F>,
	) -> Option<()> {
		self.proofs
			.get_mut(evalcheck_claim.id, evalcheck_claim.eval_point.as_ref())
			.map(|old| {
				let mut new = match old {
					EvalcheckProofWithStatus::Completed { .. } => None,
					// TODO: You can probably add more checks to the subclaim here if need be?? or maybe the caller side??
					EvalcheckProofWithStatus::Incomplete { .. } => {
						panic!("You can only call this function to replace sumcheck proofs")
					}
					EvalcheckProofWithStatus::SumcheckInducing {
						proof_id,
						proof,
						eval,
					} => Some(EvalcheckProofWithStatus::Completed {
						proof_id: *proof_id,
						proof: proof.clone(),
						eval: *eval,
					}),
				}?;
				std::mem::swap(&mut new, old);
				Some(())
			})
			.flatten()
	}

	/// Prove an evalcheck claim.
	///
	/// Given a prover state containing [`MultilinearOracleSet`] indexing into given
	/// [`MultilinearExtensionIndex`], we prove an [`EvalcheckMultilinearClaim`] (stating that given composite
	/// `poly` equals `eval` at `eval_point`) by recursively processing each of the multilinears.
	/// This way the evalcheck claim gets transformed into an [`EvalcheckProofEnum`]
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
	) -> Result<Vec<EvalcheckProofEnum<F>>, Error> {
		// DEBUGGGING
		// let with_ids = evalcheck_claims
		// 	.clone()
		// 	.into_iter()
		// 	.map(|claim| self.oracles.oracle(claim.id).variant)
		// 	.collect::<Vec<_>>();
		// println!("Proving claims: ");
		// for claim in evalcheck_claims.iter().enumerate() {
		// 	println!("\t{}: {:?}, ", claim.0, claim.1);
		// }
		// println!("with ids: ");
		// for ids in with_ids.into_iter().enumerate() {
		// 	println!("\t{}: id({}) {:?}, ", ids.0, evalcheck_claims[ids.0].id, ids.1);
		// }

		self.reset_context();
		for claim in &evalcheck_claims {
			self.claims_without_evals_dedup
				.insert(claim.id, claim.eval_point.clone(), ());
		}

		// Step 1: Collect proofs
		self.claims_queue.extend(evalcheck_claims.clone());

		// Use modified BFS approach with memoization to collect proofs.
		// The `prove_multilinear` function saves a proof if it can be generated immediately; otherwise, the claim is added to `incomplete_proof_claims` and resolved after BFS.
		// Claims requiring additional evaluation are stored in `claims_without_evals` and processed in parallel.
		while !self.claims_without_evals.is_empty() || !self.claims_queue.is_empty() {
			// Prove all available claims
			while !self.claims_queue.is_empty() {
				std::mem::take(&mut self.claims_queue)
					.into_iter()
					.for_each(|claim| self.prove_multilinear(claim));
			}

			let mut deduplicated_claims_without_evals = Vec::new();

			for (poly, eval_point) in std::mem::take(&mut self.claims_without_evals) {
				// if self.finalized_proofs.get(poly.id(), &eval_point).is_some() {
				// 	continue;
				// }
				if let Some(proof_with_status) = self.proofs.get(poly.id(), &eval_point) {
					if matches!(proof_with_status, EvalcheckProofWithStatus::Completed { .. }) {
						continue;
					}
				}
				if self
					.claims_without_evals_dedup
					.get(poly.id(), &eval_point)
					.is_some()
				{
					continue;
				}

				self.claims_without_evals_dedup
					.insert(poly.id(), eval_point.clone(), ());

				deduplicated_claims_without_evals.push((poly, eval_point.clone()))
			}

			let deduplicated_eval_points = deduplicated_claims_without_evals
				.iter()
				.map(|(_, eval_point)| eval_point.as_ref())
				.collect::<Vec<_>>();

			self.memoized_queries
				.memoize_query_par(&deduplicated_eval_points, self.backend)?;

			// Make new evaluation claims in parallel.
			let subclaims = deduplicated_claims_without_evals
				.into_par_iter()
				.map(|(poly, eval_point)| {
					Self::make_new_eval_claim(
						poly.id(),
						eval_point,
						self.witness_index,
						&self.memoized_queries,
					)
				})
				.collect::<Result<Vec<_>, Error>>()?;

			subclaims
				.into_iter()
				.for_each(|claim| self.prove_multilinear(claim));
		}

		let mut incomplete_proof_claims =
			std::mem::take(&mut self.incomplete_proof_claims).into_flatten();

		while !incomplete_proof_claims.is_empty() {
			for claim in std::mem::take(&mut incomplete_proof_claims) {
				if self.complete_proof(&claim) {
					continue;
				}
				incomplete_proof_claims.push(claim);
			}
		}

		// Step 2: Collect batch_committed_eval_claims and projected_bivariate_claims in right order

		// Since we use BFS for collecting proofs and DFS for verifying them,
		// it imposes restrictions on the correct order of collecting `batch_committed_eval_claims` and `projected_bivariate_claims`.
		// Therefore, we run a DFS to handle this.
		evalcheck_claims
			.iter()
			.cloned()
			.for_each(|claim| self.collect_projected_committed(claim));

		// Step 3: Process projected_bivariate_claims

		// println!("length of projected_bivariate_metas: {}", self.projected_bivariate_claims.len());

		let projected_bivariate_metas = self
			.projected_bivariate_claims
			.iter()
			.map(|claim| Self::projected_bivariate_meta(self.oracles, claim))
			.collect::<Result<Vec<_>, Error>>()?;

		let projected_mles = calculate_projected_mles(
			&projected_bivariate_metas,
			&mut self.memoized_queries,
			&self.projected_bivariate_claims,
			self.witness_index,
			self.backend,
		)?;

		fill_eq_witness_for_composites(
			&projected_bivariate_metas,
			&mut self.memoized_queries,
			&self.projected_bivariate_claims,
			self.witness_index,
			self.backend,
		)?;

		for (claim, meta, projected) in izip!(
			std::mem::take(&mut self.projected_bivariate_claims),
			projected_bivariate_metas,
			projected_mles
		) {
			self.process_sumcheck(claim, meta, projected)?;
		}

		// Step 4: Find and return the proofs of the original claims.

		let mut output = vec![None; self.proof_context_index];

		for proof_with_status in self.proofs.flatten().into_iter() {
			match proof_with_status {
				EvalcheckProofWithStatus::Completed {
					proof_id,
					proof,
					eval,
				} => {
					if output[proof_id].is_some() {
						panic!("Proof index :{} is already occupied", proof_id,);
					}
					output[proof_id] = Some(proof);
				}
				other => {
					panic!(
						"All claimed proofs must be complete by now but this claim is not {:?}",
						other
					)
				}
			}
		}

		let out = output
			.into_iter()
			.collect::<Option<Vec<_>>>()
			.expect("Every proof must be non empty");

		Ok(out)
	}

	#[instrument(
		skip_all,
		name = "EvalcheckProverState::prove_multilinear",
		level = "debug"
	)]
	fn prove_multilinear(&mut self, evalcheck_claim: EvalcheckMultilinearClaim<F>) {
		let multilinear_id = evalcheck_claim.id;

		let eval_point = evalcheck_claim.eval_point.clone();

		let eval = evalcheck_claim.eval;

		// TODO: Do we need these two??

		if self.proofs.get(multilinear_id, &eval_point).is_some() {
			return;
		}

		let multilinear = self.oracles.oracle(multilinear_id);

		match multilinear.variant {
			MultilinearPolyVariant::Transparent { .. } => {
				self.add_completed_proof(evalcheck_claim, EvalcheckProofEnum::Transparent, eval);
			}

			MultilinearPolyVariant::Committed => {
				self.add_sumcheck_inducing(evalcheck_claim, EvalcheckProofEnum::Committed, eval);
			}

			MultilinearPolyVariant::Repeating { id, .. } => {
				let n_vars = self.oracles.n_vars(id);
				let inner_eval_point = eval_point.slice(0..n_vars);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: inner_eval_point,
					eval,
				};
				self.incomplete_proof_claims.insert(
					multilinear_id,
					eval_point,
					evalcheck_claim.clone(),
				);
				self.claims_queue.push(subclaim.clone());
				self.add_incomplete_proof(evalcheck_claim, iter::once(subclaim));
			}

			MultilinearPolyVariant::Shifted { .. } => {
				self.add_sumcheck_inducing(evalcheck_claim, EvalcheckProofEnum::Shifted, eval);
			}

			MultilinearPolyVariant::Packed { .. } => {
				self.add_sumcheck_inducing(evalcheck_claim, EvalcheckProofEnum::Packed, eval);
			}

			MultilinearPolyVariant::Composite(_) => {
				self.add_sumcheck_inducing(evalcheck_claim, EvalcheckProofEnum::CompositeMLE, eval);
			}

			MultilinearPolyVariant::Projected(projected) => {
				let (id, values) = (projected.id(), projected.values());
				let new_eval_point = match projected.projection_variant() {
					ProjectionVariant::LastVars => {
						let mut new_eval_point = eval_point.to_vec();
						new_eval_point.extend(values);
						new_eval_point
					}
					ProjectionVariant::FirstVars => {
						values.iter().copied().chain(eval_point.to_vec()).collect()
					}
				};

				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: new_eval_point.into(),
					eval,
				};
				self.incomplete_proof_claims.insert(
					multilinear_id,
					eval_point,
					evalcheck_claim.clone(),
				);
				self.claims_queue.push(subclaim.clone());
				self.add_incomplete_proof(evalcheck_claim, iter::once(subclaim));
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
							eval_point: eval_point.clone(),
							eval,
						};
						self.claims_queue.push(subclaim.clone());
						self.add_incomplete_proof(evalcheck_claim.clone(), iter::once(subclaim));
					}
					_ => {
						let incomplete_subclaims = linear_combination
							.polys()
							.map(|suboracle_id| {
								self.claims_without_evals
									.push((self.oracles.oracle(suboracle_id), eval_point.clone()));
								(suboracle_id, eval_point.clone())
							})
							.collect::<Vec<_>>();
						self.add_incomplete_proof_partial(
							evalcheck_claim.clone(),
							incomplete_subclaims,
						);
					}
				};

				self.incomplete_proof_claims
					.insert(multilinear_id, eval_point, evalcheck_claim);
			}

			MultilinearPolyVariant::ZeroPadded(id) => {
				let inner = self.oracles.oracle(id);
				let inner_n_vars = inner.n_vars();
				let inner_eval_point = eval_point.slice(0..inner_n_vars);
				self.claims_without_evals
					.push((inner, inner_eval_point.clone()));
				self.incomplete_proof_claims.insert(
					multilinear_id,
					eval_point,
					evalcheck_claim.clone(),
				);
				self.add_incomplete_proof_partial(
					evalcheck_claim,
					iter::once((id, inner_eval_point)),
				);
			}
		};
	}

	fn complete_proof(&mut self, evalcheck_claim: &EvalcheckMultilinearClaim<F>) -> bool {
		let id = &evalcheck_claim.id;
		let eval_point = evalcheck_claim.eval_point.clone();
		let eval = evalcheck_claim.eval;

		let res = match self.oracles.oracle(*id).variant {
			MultilinearPolyVariant::Repeating { id: inner_id, .. } => {
				let n_vars = self.oracles.n_vars(inner_id);
				let inner_eval_point = &evalcheck_claim.eval_point[..n_vars];
				self.proofs
					.get(inner_id, inner_eval_point)
					.cloned()
					.map(|proof_with_status| match proof_with_status {
						EvalcheckProofWithStatus::Completed { proof_id, .. }
						| EvalcheckProofWithStatus::SumcheckInducing { proof_id, .. } => self
							.replace_incomplete_with_complete_proof(
								evalcheck_claim.clone(),
								EvalcheckProofEnum::Repeating(proof_id),
								eval,
							),
						// TODO: Should we panic here???
						EvalcheckProofWithStatus::Incomplete { .. } => None,
					})
					.flatten()
			}
			MultilinearPolyVariant::Projected(projected) => {
				let (inner_id, values) = (projected.id(), projected.values());
				let new_eval_point = match projected.projection_variant() {
					ProjectionVariant::LastVars => {
						let mut new_eval_point = eval_point.to_vec();
						new_eval_point.extend(values);
						new_eval_point
					}
					ProjectionVariant::FirstVars => values
						.iter()
						.copied()
						.chain((*eval_point).to_vec())
						.collect(),
				};

				self.proofs
					.get(inner_id, &new_eval_point)
					.cloned()
					.map(|proof_with_status| match proof_with_status {
						EvalcheckProofWithStatus::Completed { proof_id, .. }
						| EvalcheckProofWithStatus::SumcheckInducing { proof_id, .. } => self
							.replace_incomplete_with_complete_proof(
								evalcheck_claim.clone(),
								EvalcheckProofEnum::Projected(proof_id),
								eval,
							),
						EvalcheckProofWithStatus::Incomplete { .. } => None,
					})
					.flatten()
			}

			MultilinearPolyVariant::LinearCombination(linear_combination) => linear_combination
				.polys()
				.map(|suboracle_id| {
					self.proofs
						.get(suboracle_id, &evalcheck_claim.eval_point)
						.cloned()
				})
				.collect::<Option<Vec<_>>>()
				.map(|proofs_with_statuses| {
					let subproofs = proofs_with_statuses
						.into_iter()
						.map(|proof_with_status| match proof_with_status {
							EvalcheckProofWithStatus::Completed {
								proof_id,
								eval: inner_eval,
								..
							}
							| EvalcheckProofWithStatus::SumcheckInducing {
								proof_id,
								eval: inner_eval,
								..
							} => Some((inner_eval, proof_id)),
							EvalcheckProofWithStatus::Incomplete { .. } => None,
						})
						.collect::<Option<Vec<_>>>()?;
					let proof = EvalcheckProofEnum::LinearCombination { subproofs };
					self.replace_incomplete_with_complete_proof(
						evalcheck_claim.clone(),
						proof,
						eval,
					)
				})
				.flatten(),

			MultilinearPolyVariant::ZeroPadded(inner_id) => {
				let inner_n_vars = self.oracles.n_vars(inner_id);
				let inner_eval_point = &evalcheck_claim.eval_point[..inner_n_vars];

				self.proofs
					.get(inner_id, inner_eval_point)
					.cloned()
					.map(|proof_with_status| match proof_with_status {
						EvalcheckProofWithStatus::Completed {
							proof_id,
							eval: internal_eval,
							..
						}
						| EvalcheckProofWithStatus::SumcheckInducing {
							proof_id,
							eval: internal_eval,
							..
						} => self.replace_incomplete_with_complete_proof(
							evalcheck_claim.clone(),
							EvalcheckProofEnum::ZeroPadded(internal_eval, proof_id),
							eval,
						),
						EvalcheckProofWithStatus::Incomplete { .. } => None,
					})
					.flatten()
			}

			_ => unreachable!(),
		};
		res.is_some()
	}

	fn collect_projected_committed(&mut self, evalcheck_claim: EvalcheckMultilinearClaim<F>) {
		let EvalcheckMultilinearClaim {
			id,
			eval_point,
			eval,
		} = evalcheck_claim.clone();

		let multilinear = self.oracles.oracle(id);
		match multilinear.variant {
			MultilinearPolyVariant::Committed => {
				let subclaim = EvalcheckMultilinearClaim {
					id: multilinear.id,
					eval_point: eval_point.clone(),
					eval,
				};

				let proof_with_status = self
					.proofs
					.get(id, &eval_point)
					.expect("Sumcheck inducing claims were added in prove_multilinear");
				if matches!(proof_with_status, EvalcheckProofWithStatus::Completed { .. }) {
					return;
				}

				match proof_with_status {
					EvalcheckProofWithStatus::Completed { .. } => (),
					EvalcheckProofWithStatus::Incomplete { .. } => {
						unreachable!("Committed claims cannot be incomplete")
					}
					EvalcheckProofWithStatus::SumcheckInducing { .. } => {
						self.replace_sumcheck_inducing_with_complete(&subclaim);
						self.committed_eval_claims.push(subclaim)
					}
				}
			}
			MultilinearPolyVariant::Repeating { id, .. } => {
				let n_vars = self.oracles.n_vars(id);
				let inner_eval_point = eval_point.slice(0..n_vars);
				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: inner_eval_point,
					eval,
				};

				self.collect_projected_committed(subclaim);
			}
			MultilinearPolyVariant::Projected(projected) => {
				let (id, values) = (projected.id(), projected.values());
				let new_eval_point = match projected.projection_variant() {
					ProjectionVariant::LastVars => {
						let mut new_eval_point = eval_point.to_vec();
						new_eval_point.extend(values);
						new_eval_point
					}
					ProjectionVariant::FirstVars => {
						values.iter().copied().chain(eval_point.to_vec()).collect()
					}
				};

				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point: new_eval_point.into(),
					eval,
				};
				self.collect_projected_committed(subclaim);
			}
			MultilinearPolyVariant::Shifted { .. }
			| MultilinearPolyVariant::Packed { .. }
			| MultilinearPolyVariant::Composite { .. } => {
				let proof_with_status = self
					.proofs
					.get(id, &eval_point)
					.expect("Sumcheck inducing claims were added in prove_multilinear");

				match proof_with_status {
					EvalcheckProofWithStatus::Completed { .. } => (),
					EvalcheckProofWithStatus::Incomplete { .. } => {
						unreachable!("Sumcheck inducing claims cannot be incomplete")
					}
					EvalcheckProofWithStatus::SumcheckInducing { .. } => {
						self.replace_sumcheck_inducing_with_complete(&evalcheck_claim);
						self.projected_bivariate_claims.push(evalcheck_claim)
					}
				}
			}
			MultilinearPolyVariant::LinearCombination(linear_combination) => {
				for id in linear_combination.polys() {
					let proof_with_status = self
						.proofs
						.get(id, &eval_point)
						.expect("finalized_proofs contains all the proofs");
					let inner_eval = match proof_with_status {
						EvalcheckProofWithStatus::Incomplete { .. } => {
							panic!("SubProofs of Linear combinations must be completed by now")
						}
						EvalcheckProofWithStatus::Completed { eval, .. }
						| EvalcheckProofWithStatus::SumcheckInducing { eval, .. } => *eval,
					};
					let subclaim = EvalcheckMultilinearClaim {
						id,
						eval_point: eval_point.clone(),
						eval: inner_eval,
					};
					self.collect_projected_committed(subclaim);
				}
			}
			MultilinearPolyVariant::ZeroPadded(id) => {
				let inner_n_vars = self.oracles.n_vars(id);
				let inner_eval_point = eval_point.slice(0..inner_n_vars);

				let proof_with_status = self
					.proofs
					.get(id, &inner_eval_point)
					.expect("finalized_proofs contains all the proofs");
				let inner_eval = match proof_with_status {
					EvalcheckProofWithStatus::Incomplete { .. } => {
						panic!("SubProof of ZeroPadded must be completed by now")
					}
					EvalcheckProofWithStatus::Completed { eval, .. }
					| EvalcheckProofWithStatus::SumcheckInducing { eval, .. } => *eval,
				};

				let subclaim = EvalcheckMultilinearClaim {
					id,
					eval_point,
					eval: inner_eval,
				};
				self.collect_projected_committed(subclaim);
			}
			_ => {}
		}
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
		meta: ProjectedBivariateMeta,
		projected: Option<MultilinearExtension<PackedType<U, F>>>,
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
				projected.expect("projected is required by shifted oracle"),
			),

			MultilinearPolyVariant::Packed(packed) => process_packed_sumcheck(
				self.oracles,
				&packed,
				meta,
				&eval_point,
				eval,
				self.witness_index,
				&mut self.new_sumchecks_constraints,
				projected.expect("projected is required by packed oracle"),
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

	fn make_new_eval_claim(
		oracle_id: OracleId,
		eval_point: EvalPoint<F>,
		witness_index: &MultilinearExtensionIndex<U, F>,
		memoized_queries: &MemoizedQueries<PackedType<U, F>, Backend>,
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
