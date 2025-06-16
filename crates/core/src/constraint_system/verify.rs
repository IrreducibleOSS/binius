// Copyright 2024-2025 Irreducible Inc.

use std::collections::hash_map::Entry;

use binius_field::{
	BinaryField, PackedField, TowerField,
	tower::{PackedTop, TowerFamily, TowerUnderlier},
};
use binius_hash::PseudoCompressionFunction;
use binius_math::{ArithExpr, CompositionPoly, EvaluationOrder};
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize};
use digest::{Digest, Output, OutputSizeUser, core_api::BlockSizeUser};
use itertools::{Itertools, chain, izip};
use tracing::instrument;

use super::{
	ConstraintSystem, Proof,
	channel::{Boundary, OracleOrConst},
	error::{Error, VerificationError},
	exp::{self, reorder_exponents},
};
use crate::{
	constraint_system::{
		TableSizeSpec,
		channel::{Flush, FlushDirection},
		common::{FDomain, FEncode, FExt},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeScheme,
	oracle::{
		ConstraintSetBuilder, MultilinearOracleSet, MultilinearPolyVariant, OracleId,
		SizedConstraintSet,
	},
	piop,
	protocols::{
		evalcheck::{EvalPoint, EvalcheckMultilinearClaim},
		gkr_exp,
		gkr_gpa::{self},
		greedy_evalcheck,
		sumcheck::{
			self, MLEcheckClaimsWithMeta, ZerocheckClaim, constraint_set_mlecheck_claims,
			constraint_set_zerocheck_claim,
			eq_ind::{self, ClaimsSortingOrder, reduce_to_regular_sumchecks},
			front_loaded,
		},
	},
	ring_switch,
	transcript::VerifierTranscript,
	transparent::step_down::StepDown,
};

/// Verifies a proof against a constraint system.
#[instrument("constraint_system::verify", skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
pub fn verify<U, Tower, Hash, Compress, Challenger_>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	constraint_system_digest: &Output<Hash>,
	boundaries: &[Boundary<FExt<Tower>>],
	proof: Proof,
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: binius_math::TowerTop + binius_math::PackedTop + PackedTop<Tower>,
	Hash: Digest + BlockSizeUser + OutputSizeUser,
	Compress: PseudoCompressionFunction<Output<Hash>, 2> + Default + Sync,
	Challenger_: Challenger + Default,
{
	let _ = constraint_system_digest;
	let ConstraintSystem {
		mut oracles,
		table_constraints,
		mut flushes,
		non_zero_oracle_ids,
		channel_count,
		mut exponents,
		table_size_specs,
	} = constraint_system.clone();

	let Proof { transcript } = proof;

	let mut transcript = VerifierTranscript::<Challenger_>::new(transcript);
	transcript
		.observe()
		.write_slice(constraint_system_digest.as_ref());
	transcript.observe().write_slice(boundaries);

	let table_count = table_size_specs.len();
	let mut reader = transcript.message();
	let table_sizes: Vec<usize> = reader.read_vec(table_count)?;
	assert_eq!(table_sizes.len(), table_count);

	for (table_id, (&table_size, table_size_spec)) in
		table_sizes.iter().zip(table_size_specs.iter()).enumerate()
	{
		match table_size_spec {
			TableSizeSpec::PowerOfTwo => {
				if !table_size.is_power_of_two() {
					return Err(Error::TableSizePowerOfTwoRequired {
						table_id,
						size: table_size,
					});
				}
			}
			TableSizeSpec::Fixed { log_size } => {
				if table_size != 1 << log_size {
					return Err(Error::TableSizeFixedRequired {
						table_id,
						size: table_size,
					});
				}
			}
			TableSizeSpec::Arbitrary => (),
		}
	}

	let mut table_constraints = table_constraints
		.into_iter()
		.filter_map(|u| {
			if table_sizes[u.table_id] == 0 {
				None
			} else {
				let n_vars = u.log_values_per_row + log2_ceil_usize(table_sizes[u.table_id]);
				Some(SizedConstraintSet::new(n_vars, u))
			}
		})
		.collect::<Vec<_>>();
	// Stable sort constraint sets in ascending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| constraint_set.n_vars);

	let merkle_scheme = BinaryMerkleTreeScheme::<_, Hash, _>::new(Compress::default());
	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(&oracles)?;
	let fri_params = piop::make_commit_params_with_optimal_arity::<_, FEncode<Tower>, _>(
		&commit_meta,
		&merkle_scheme,
		security_bits,
		log_inv_rate,
	)?;

	// Read polynomial commitment polynomials
	let mut reader = transcript.message();
	let commitment = reader.read::<Output<Hash>>()?;

	// GKR exp multiplication
	reorder_exponents(&mut exponents, &oracles);

	let exp_challenge = transcript.sample_vec(exp::max_n_vars(&exponents, &oracles));

	let mut reader = transcript.message();
	let exp_evals = reader.read_scalar_slice(exponents.len())?;

	let exp_claims = exp::make_claims(&exponents, &oracles, &exp_challenge, &exp_evals)?
		.into_iter()
		.collect::<Vec<_>>();

	let base_exp_output =
		gkr_exp::batch_verify(EvaluationOrder::HighToLow, &exp_claims, &mut transcript)?;

	let exp_eval_claims = exp::make_eval_claims(&exponents, base_exp_output)?;

	// Grand product arguments
	// Grand products for non-zero checks
	let mut reader = transcript.message();
	let non_zero_products = reader.read_scalar_slice(non_zero_oracle_ids.len())?;
	if non_zero_products
		.iter()
		.any(|count| *count == Tower::B128::zero())
	{
		bail!(Error::Zeros);
	}

	let non_zero_prodcheck_claims = gkr_gpa::construct_grand_product_claims(
		&non_zero_oracle_ids,
		&oracles,
		&non_zero_products,
	)?;

	// Grand products for flushing
	let mixing_challenge = transcript.sample();
	// TODO(cryptographers): Find a way to sample less randomness
	let permutation_challenges = transcript.sample_vec(channel_count);

	flushes.retain(|flush| table_sizes[flush.table_id] > 0);
	flushes.sort_by_key(|flush| flush.channel_id);
	let _ =
		augument_flush_po2_step_down(&mut oracles, &mut flushes, &table_size_specs, &table_sizes)?;
	let flush_oracle_ids =
		make_flush_oracles(&mut oracles, &flushes, mixing_challenge, &permutation_challenges)?;

	let flush_products = transcript
		.message()
		.read_scalar_slice(flush_oracle_ids.len())?;
	verify_channels_balance(
		&flushes,
		&flush_products,
		boundaries,
		mixing_challenge,
		&permutation_challenges,
	)?;

	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracle_ids, &oracles, &flush_products)?;

	// Verify grand products
	let final_layer_claims = gkr_gpa::batch_verify(
		EvaluationOrder::HighToLow,
		[flush_prodcheck_claims, non_zero_prodcheck_claims].concat(),
		&mut transcript,
	)?;

	// Reduce non_zero_final_layer_claims to evalcheck claims
	let prodcheck_eval_claims = gkr_gpa::make_eval_claims(
		chain!(flush_oracle_ids.clone(), non_zero_oracle_ids),
		final_layer_claims,
	)?;

	let mut flush_prodcheck_eval_claims = prodcheck_eval_claims;

	let prodcheck_eval_claims = flush_prodcheck_eval_claims.split_off(flush_oracle_ids.len());

	let flush_eval_claims = reduce_flush_evalcheck_claims::<Tower, Challenger_>(
		flush_prodcheck_eval_claims,
		&oracles,
		&mut transcript,
	)?;

	// Zerocheck
	let (zerocheck_claims, zerocheck_oracle_metas) = table_constraints
		.iter()
		.cloned()
		.map(constraint_set_zerocheck_claim)
		.collect::<Result<Vec<_>, _>>()?
		.into_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	let (_max_n_vars, skip_rounds) =
		max_n_vars_and_skip_rounds(&zerocheck_claims, <FDomain<Tower>>::N_BITS);

	let zerocheck_output =
		sumcheck::batch_verify_zerocheck(&zerocheck_claims, skip_rounds, &mut transcript)?;

	let zerocheck_eval_claims =
		sumcheck::make_zerocheck_eval_claims(zerocheck_oracle_metas, zerocheck_output)?;

	// Evalcheck
	let eval_claims = greedy_evalcheck::verify(
		&mut oracles,
		chain!(flush_eval_claims, prodcheck_eval_claims, zerocheck_eval_claims, exp_eval_claims,),
		&mut transcript,
	)?;

	// Reduce committed evaluation claims to PIOP sumcheck claims
	let system = ring_switch::EvalClaimSystem::new(
		&oracles,
		&commit_meta,
		&oracle_to_commit_index,
		&eval_claims,
	)?;

	let ring_switch::ReducedClaim {
		transparents,
		sumcheck_claims: piop_sumcheck_claims,
	} = ring_switch::verify(&system, &mut transcript)?;

	// Prove evaluation claims using PIOP compiler
	piop::verify(
		&commit_meta,
		&merkle_scheme,
		&fri_params,
		&commitment,
		&transparents,
		&piop_sumcheck_claims,
		&mut transcript,
	)?;

	transcript.finalize()?;

	Ok(())
}

pub fn max_n_vars_and_skip_rounds<F, Composition>(
	zerocheck_claims: &[ZerocheckClaim<F, Composition>],
	domain_bits: usize,
) -> (usize, usize)
where
	F: TowerField,
	Composition: CompositionPoly<F>,
{
	let max_n_vars = max_n_vars(zerocheck_claims);

	// Univariate skip zerocheck domain size is degree * 2^skip_rounds, which
	// limits skip_rounds to ceil(log2(degree)) less than domain field bits.
	let domain_max_skip_rounds = zerocheck_claims
		.iter()
		.map(|claim| {
			let log_degree = log2_ceil_usize(claim.max_individual_degree());
			domain_bits.saturating_sub(log_degree)
		})
		.min()
		.unwrap_or(0);

	let max_skip_rounds = domain_max_skip_rounds.min(max_n_vars);
	(max_n_vars, max_skip_rounds)
}

fn max_n_vars<F, Composition>(zerocheck_claims: &[ZerocheckClaim<F, Composition>]) -> usize
where
	F: TowerField,
	Composition: CompositionPoly<F>,
{
	zerocheck_claims
		.iter()
		.map(|claim| claim.n_vars())
		.max()
		.unwrap_or(0)
}

fn verify_channels_balance<F: TowerField>(
	flushes: &[Flush<F>],
	flush_products: &[F],
	boundaries: &[Boundary<F>],
	mixing_challenge: F,
	permutation_challenges: &[F],
) -> Result<(), Error> {
	if flush_products.len() != flushes.len() {
		return Err(VerificationError::IncorrectNumberOfFlushProducts.into());
	}

	let mut flush_iter = flushes
		.iter()
		.zip(flush_products.iter().copied())
		.peekable();
	while let Some((flush, _)) = flush_iter.peek() {
		let channel_id = flush.channel_id;

		let boundary_products =
			boundaries
				.iter()
				.fold((F::ONE, F::ONE), |(pull_product, push_product), boundary| {
					let Boundary {
						channel_id: boundary_channel_id,
						direction,
						multiplicity,
						values,
						..
					} = boundary;

					if *boundary_channel_id == channel_id {
						let (mixed_values, _) = values.iter().fold(
							(permutation_challenges[channel_id], F::ONE),
							|(sum, mixing), values| {
								(sum + mixing * values, mixing * mixing_challenge)
							},
						);

						let mixed_values_with_multiplicity =
							mixed_values.pow_vartime([*multiplicity]);

						return match direction {
							FlushDirection::Pull => {
								(pull_product * mixed_values_with_multiplicity, push_product)
							}
							FlushDirection::Push => {
								(pull_product, push_product * mixed_values_with_multiplicity)
							}
						};
					}

					(pull_product, push_product)
				});

		let (pull_product, push_product) = flush_iter
			.peeking_take_while(|(flush, _)| flush.channel_id == channel_id)
			.fold(boundary_products, |(pull_product, push_product), (flush, flush_product)| {
				let flush_product_with_multiplicity =
					flush_product.pow_vartime([flush.multiplicity]);
				match flush.direction {
					FlushDirection::Pull => {
						(pull_product * flush_product_with_multiplicity, push_product)
					}
					FlushDirection::Push => {
						(pull_product, push_product * flush_product_with_multiplicity)
					}
				}
			});
		if pull_product != push_product {
			return Err(VerificationError::ChannelUnbalanced { id: channel_id }.into());
		}
	}

	Ok(())
}

/// This function will create a special selectors for the flushes, that are defined on tables that
/// are not of power-of-two size. Those artifical selectors are needed to bridge the gap between
/// the arbitrary sized tables and the oracles (oracles are always power-of-two sized).
///
/// Takes the vector of `flushes` and creates the corresponding list of oracles for them. If the
/// witness provided, then it will also fill the witness for those oracles.
pub fn augument_flush_po2_step_down<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	flushes: &mut [Flush<F>],
	table_size_specs: &[TableSizeSpec],
	table_sizes: &[usize],
) -> Result<Vec<(OracleId, StepDown)>, Error> {
	use std::collections::HashMap;

	use crate::transparent::step_down::StepDown;

	// Track created step-down oracles by (table_id, log_values_per_row)
	let mut step_down_oracles = HashMap::<(usize, usize), OracleId>::new();
	let mut step_down_polys = Vec::new();

	// First pass: create step-down oracles for arbitrary-sized tables
	for flush in flushes.iter() {
		let table_id = flush.table_id;
		let table_size = table_sizes[table_id];
		let table_spec = &table_size_specs[table_id];

		// Only process tables with arbitrary size that are not power-of-two
		if matches!(table_spec, TableSizeSpec::Arbitrary) {
			let log_values_per_row = flush.log_values_per_row;
			let key = (table_id, log_values_per_row);

			// Only create the step-down oracle once per (table_id, log_values_per_row) pair.
			if let Entry::Vacant(e) = step_down_oracles.entry(key) {
				let log_capacity = log2_ceil_usize(table_size);
				let n_vars = log_capacity + log_values_per_row;
				let size = table_size << log_values_per_row;

				let step_down_poly = StepDown::new(n_vars, size)?;
				let oracle_id = oracles
					.add_named(format!("stepdown_table_{table_id}_log_values_{log_values_per_row}"))
					.transparent(step_down_poly.clone())?;

				step_down_polys.push((oracle_id, step_down_poly));
				e.insert(oracle_id);
			}
		}
	}

	// Second pass: add step-down oracles as selectors to the appropriate flushes
	for flush in flushes.iter_mut() {
		let table_id = flush.table_id;
		let table_spec = &table_size_specs[table_id];

		if matches!(table_spec, TableSizeSpec::Arbitrary) {
			let key = (table_id, flush.log_values_per_row);
			if let Some(&oracle_id) = step_down_oracles.get(&key) {
				flush.selectors.push(oracle_id);
			}
		}
	}

	Ok(step_down_polys)
}

/// For each flush,
/// - if there is a selector $S$, we are taking the Grand product of the composite $1 + S * (-1 + r
///   + F_0 + F_1 s + F_2 s^1 + …)$
/// - otherwise the product is over the linear combination $r + F_0 + F_1 s + F_2 s^1 + …$
pub fn make_flush_oracles<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	flushes: &[Flush<F>],
	mixing_challenge: F,
	permutation_challenges: &[F],
) -> Result<Vec<OracleId>, Error> {
	let mut mixing_powers = vec![F::ONE];
	let mut flush_iter = flushes.iter();

	permutation_challenges
		.iter()
		.enumerate()
		.flat_map(|(channel_id, permutation_challenge)| {
			flush_iter
				.peeking_take_while(|flush| flush.channel_id == channel_id)
				.map(|flush| {
					// Check that all flushed oracles have the same number of variables
					let mut non_const_oracles =
						flush.oracles.iter().copied().filter_map(|id| match id {
							OracleOrConst::Oracle(oracle_id) => Some(oracle_id),
							_ => None,
						});

					let first_oracle = non_const_oracles.next().ok_or(Error::EmptyFlushOracles)?;
					let n_vars = oracles.n_vars(first_oracle);

					for selector_id in &flush.selectors {
						let got_tower_level = oracles[*selector_id].tower_level;
						if got_tower_level != 0 {
							return Err(Error::FlushSelectorTowerLevel {
								oracle: *selector_id,
								got_tower_level,
							});
						}
					}

					for oracle_id in non_const_oracles {
						let oracle_n_vars = oracles.n_vars(oracle_id);
						if oracle_n_vars != n_vars {
							return Err(Error::ChannelFlushNvarsMismatch {
								expected: n_vars,
								got: oracle_n_vars,
							});
						}
					}

					// Compute powers of the mixing challenge
					while mixing_powers.len() < flush.oracles.len() {
						let last_power = *mixing_powers.last().expect(
							"mixing_powers is initialized with one element; \
								mixing_powers never shrinks; \
								thus, it must not be empty",
						);
						mixing_powers.push(last_power * mixing_challenge);
					}

					let const_linear_combination = flush
						.oracles
						.iter()
						.copied()
						.zip(mixing_powers.iter())
						.filter_map(|(id, coeff)| match id {
							OracleOrConst::Const { base, .. } => Some(base * coeff),
							_ => None,
						})
						.sum::<F>();

					let poly = if flush.selectors.is_empty() {
						oracles
							.add_named(format!("flush channel_id={channel_id} linear combination"))
							.linear_combination_with_offset(
								n_vars,
								*permutation_challenge + const_linear_combination,
								flush
									.oracles
									.iter()
									.zip(mixing_powers.iter().copied())
									.filter_map(|(id, coeff)| match id {
										OracleOrConst::Oracle(oracle_id) => {
											Some((*oracle_id, coeff))
										}
										_ => None,
									}),
							)?
					} else {
						let offset = *permutation_challenge + const_linear_combination + F::ONE;
						let arith_expr_linear = ArithExpr::Const(offset);
						let var_offset = flush.selectors.len(); // Var's represents the selector columns.
						let (non_const_oracles, coeffs): (Vec<_>, Vec<_>) = flush
							.oracles
							.iter()
							.zip(mixing_powers.iter().copied())
							.filter_map(|(id, coeff)| match id {
								OracleOrConst::Oracle(id) => Some((*id, coeff)),
								_ => None,
							})
							.unzip();

						// Build the linear combination of the non-constant oracles.
						let arith_expr_linear = coeffs.into_iter().enumerate().fold(
							arith_expr_linear,
							|linear, (offset, coeff)| {
								linear
									+ ArithExpr::Var(offset + var_offset) * ArithExpr::Const(coeff)
							},
						);

						let selector = (0..var_offset)
							.map(ArithExpr::Var)
							.product::<ArithExpr<F>>();

						// The ArithExpr is of the form 1 + S * linear_factors
						oracles
							.add_named(format!("flush channel_id={channel_id} composite"))
							.composite_mle(
								n_vars,
								flush.selectors.iter().copied().chain(non_const_oracles),
								(ArithExpr::Const(F::ONE) + selector * arith_expr_linear).into(),
							)?
					};
					Ok(poly)
				})
				.collect::<Vec<_>>()
		})
		.collect()
}

fn reduce_flush_evalcheck_claims<Tower: TowerFamily, Challenger_>(
	claims: Vec<EvalcheckMultilinearClaim<FExt<Tower>>>,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<Vec<EvalcheckMultilinearClaim<FExt<Tower>>>, Error>
where
	Challenger_: Challenger + Default,
{
	let mut linear_claims = Vec::new();

	#[allow(clippy::type_complexity)]
	let mut new_mlechecks_constraints: Vec<(
		EvalPoint<FExt<Tower>>,
		ConstraintSetBuilder<FExt<Tower>>,
	)> = Vec::new();

	for claim in &claims {
		match &oracles[claim.id].variant {
			MultilinearPolyVariant::LinearCombination(_) => linear_claims.push(claim.clone()),
			MultilinearPolyVariant::Composite(composite) => {
				let eval_point = claim.eval_point.clone();
				let eval = claim.eval;

				let position = new_mlechecks_constraints
					.iter()
					.position(|(ep, _)| *ep == eval_point)
					.unwrap_or(new_mlechecks_constraints.len());

				let oracle_ids = composite.inner().clone();

				let exp = <_ as CompositionPoly<FExt<Tower>>>::expression(composite.c());
				if let Some((_, constraint_builder)) = new_mlechecks_constraints.get_mut(position) {
					constraint_builder.add_sumcheck(oracle_ids, exp, eval);
				} else {
					let mut new_builder = ConstraintSetBuilder::new();
					new_builder.add_sumcheck(oracle_ids, exp, eval);
					new_mlechecks_constraints.push((eval_point.clone(), new_builder));
				}
			}
			_ => unreachable!(),
		}
	}

	let new_mlechecks_constraints = new_mlechecks_constraints;

	let mut eq_ind_challenges = Vec::with_capacity(new_mlechecks_constraints.len());
	let mut constraint_sets = Vec::with_capacity(new_mlechecks_constraints.len());

	for (ep, builder) in new_mlechecks_constraints {
		eq_ind_challenges.push(ep.to_vec());
		constraint_sets.push(builder.build_one(oracles)?)
	}

	let MLEcheckClaimsWithMeta {
		claims: mlecheck_claims,
		metas,
	} = constraint_set_mlecheck_claims(constraint_sets)?;

	let mut new_evalcheck_claims = Vec::new();

	for (eq_ind_challenges, mlecheck_claim, meta) in
		izip!(&eq_ind_challenges, mlecheck_claims, metas)
	{
		let mlecheck_claim = vec![mlecheck_claim];

		let batch_sumcheck_verifier = front_loaded::BatchVerifier::new(
			&reduce_to_regular_sumchecks(&mlecheck_claim)?,
			transcript,
		)?;
		let mut sumcheck_output = batch_sumcheck_verifier.run(transcript)?;

		// Reverse challenges since foldling high-to-low
		sumcheck_output.challenges.reverse();

		let eq_ind_output = eq_ind::verify_sumcheck_outputs(
			ClaimsSortingOrder::AscendingVars,
			&mlecheck_claim,
			eq_ind_challenges,
			sumcheck_output,
		)?;

		let evalcheck_claims =
			sumcheck::make_eval_claims(EvaluationOrder::HighToLow, vec![meta], eq_ind_output)?;
		new_evalcheck_claims.extend(evalcheck_claims)
	}

	Ok(chain!(new_evalcheck_claims.into_iter(), linear_claims.into_iter()).collect::<Vec<_>>())
}
