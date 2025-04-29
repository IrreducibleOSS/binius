// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	BinaryField, PackedField, TowerField,
};
use binius_hash::PseudoCompressionFunction;
use binius_math::{ArithExpr, CompositionPoly, EvaluationOrder};
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize};
use digest::{core_api::BlockSizeUser, Digest, Output};
use itertools::{chain, Itertools};
use tracing::instrument;

use super::{
	channel::{Boundary, OracleOrConst},
	error::{Error, VerificationError},
	exp, ConstraintSystem, Proof,
};
use crate::{
	constraint_system::{
		channel::{Flush, FlushDirection},
		common::{FDomain, FEncode, FExt},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeScheme,
	oracle::{MultilinearOracleSet, OracleId},
	piop,
	protocols::{
		gkr_exp,
		gkr_gpa::{self},
		greedy_evalcheck,
		sumcheck::{self, constraint_set_zerocheck_claim, ZerocheckClaim},
	},
	ring_switch,
	transcript::VerifierTranscript,
};

/// Verifies a proof against a constraint system.
#[instrument("constraint_system::verify", skip_all, level = "debug")]
pub fn verify<U, Tower, Hash, Compress, Challenger_>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	boundaries: &[Boundary<FExt<Tower>>],
	proof: Proof,
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower>,
	Hash: Digest + BlockSizeUser,
	Compress: PseudoCompressionFunction<Output<Hash>, 2> + Default + Sync,
	Challenger_: Challenger + Default,
{
	let ConstraintSystem {
		mut oracles,
		mut table_constraints,
		mut flushes,
		non_zero_oracle_ids,
		max_channel_id,
		mut exponents,
		..
	} = constraint_system.clone();

	// Stable sort constraint sets in ascending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| constraint_set.n_vars);

	let Proof { transcript } = proof;

	let mut transcript = VerifierTranscript::<Challenger_>::new(transcript);
	transcript.observe().write_slice(boundaries);

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
	exponents.sort_by_key(|b| std::cmp::Reverse(b.n_vars(&oracles)));

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
	let permutation_challenges = transcript.sample_vec(max_channel_id + 1);

	flushes.sort_by_key(|flush| flush.channel_id);
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
		EvaluationOrder::LowToHigh,
		[flush_prodcheck_claims, non_zero_prodcheck_claims].concat(),
		&mut transcript,
	)?;

	// Reduce non_zero_final_layer_claims to evalcheck claims
	let prodcheck_eval_claims = gkr_gpa::make_eval_claims(
		chain!(flush_oracle_ids, non_zero_oracle_ids),
		final_layer_claims,
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
		chain!(prodcheck_eval_claims, zerocheck_eval_claims, exp_eval_claims,),
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
	} = ring_switch::verify::<_, Tower, _>(&system, &mut transcript)?;

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

/// For each flush,
/// - if there is a selector $S$, we are taking the Grand product of the composite $1 + S * (-1 + r + F_0 + F_1 s + F_2 s^1 + …)$
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

					if let Some(selector_id) = &flush.selector {
						let got_tower_level = oracles.oracle(*selector_id).tower_level;
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

					let poly = match flush.selector {
						Some(selector_id) => {
							let offset = *permutation_challenge + const_linear_combination + F::ONE;
							let arith_expr_linear = ArithExpr::Const(offset);
							let var_offset = 1; // Var(0) represents the selector column.
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
										+ ArithExpr::Var(offset + var_offset)
											* ArithExpr::Const(coeff)
								},
							);

							// The ArithExpr is of the form 1 + S * linear_factors
							oracles
								.add_named(format!("flush channel_id={channel_id} composite"))
								.composite_mle(
									n_vars,
									iter::once(selector_id).chain(non_const_oracles),
									ArithExpr::Const(F::ONE)
										+ ArithExpr::Var(0) * arith_expr_linear,
								)?
						}
						None => oracles
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
							)?,
					};
					Ok(poly)
				})
				.collect::<Vec<_>>()
		})
		.collect()
}
