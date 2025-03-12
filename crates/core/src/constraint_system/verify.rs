// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::Reverse, iter};

use binius_field::{BinaryField, PackedField, TowerField};
use binius_hash::PseudoCompressionFunction;
use binius_math::{ArithExpr, CompositionPoly, EvaluationOrder};
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize};
use digest::{core_api::BlockSizeUser, Digest, Output};
use itertools::{izip, multiunzip, Itertools};
use tracing::instrument;

use super::{
	channel::Boundary,
	error::{Error, VerificationError},
	ConstraintSystem, Proof,
};
use crate::{
	composition::IndexComposition,
	constraint_system::{
		channel::{Flush, FlushDirection},
		common::{FDomain, FEncode, FExt},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeScheme,
	oracle::{MultilinearOracleSet, OracleId},
	piop,
	polynomial::MultivariatePoly,
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		gkr_gpa,
		gkr_gpa::LayerClaim,
		greedy_evalcheck,
		sumcheck::{
			self, constraint_set_zerocheck_claim, eq_ind::ExtraProduct, zerocheck,
			BatchSumcheckOutput, CompositeSumClaim, SumcheckClaim, ZerocheckClaim,
		},
	},
	ring_switch,
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::VerifierTranscript,
	transparent::eq_ind::EqIndPartialEval,
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
		..
	} = constraint_system.clone();

	// Stable sort constraint sets in descending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| Reverse(constraint_set.n_vars));

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

	// Grand product arguments
	// Grand products for non-zero checks
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
	let flush_selectors = flushes
		.iter()
		.map(|flush| flush.selector)
		.collect::<Vec<_>>();

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
	let mut final_layer_claims = gkr_gpa::batch_verify(
		EvaluationOrder::LowToHigh,
		[flush_prodcheck_claims, non_zero_prodcheck_claims].concat(),
		&mut transcript,
	)?;

	let non_zero_final_layer_claims = final_layer_claims.split_off(flush_oracle_ids.len());
	let flush_final_layer_claims = final_layer_claims;

	// Reduce non_zero_final_layer_claims to evalcheck claims
	let non_zero_prodcheck_eval_claims =
		gkr_gpa::make_eval_claims(non_zero_oracle_ids, non_zero_final_layer_claims)?;

	// Reduce flush_final_layer_claims to sumcheck claims then evalcheck claims
	let (flush_oracle_ids, flush_selectors, flush_final_layer_claims) =
		reorder_for_flushing_by_n_vars(
			&oracles,
			&flush_oracle_ids,
			flush_selectors,
			flush_final_layer_claims,
		);

	let flush_sumcheck_metas = get_flush_dedup_sumcheck_metas(
		&oracles,
		&flush_oracle_ids,
		&flush_selectors,
		&flush_final_layer_claims,
	)?;

	let DedupSumcheckClaims {
		sumcheck_claims,
		gkr_eval_points,
		flush_selectors_unique_by_claim,
		flush_oracle_ids_by_claim,
	} = get_flush_dedup_sumcheck_claims(flush_sumcheck_metas)?;

	let flush_sumcheck_output =
		sumcheck::batch_verify(EvaluationOrder::LowToHigh, &sumcheck_claims, &mut transcript)?;

	let flush_eval_claims = get_post_flush_sumcheck_eval_claims_without_eq(
		&oracles,
		&flush_selectors_unique_by_claim,
		&flush_oracle_ids_by_claim,
		&flush_sumcheck_output,
	)?;

	// Check the eval claim on the transparent eq polynomial
	for (gkr_eval_point, evals) in izip!(gkr_eval_points, flush_sumcheck_output.multilinear_evals) {
		let gkr_eval_point_len = gkr_eval_point.len();
		let eq_ind = EqIndPartialEval::new(gkr_eval_point);

		let sumcheck_challenges_len = flush_sumcheck_output.challenges.len();
		let expected_eval = eq_ind.evaluate(
			&flush_sumcheck_output.challenges[(sumcheck_challenges_len - gkr_eval_point_len)..],
		)?;

		let &actual_eval = evals
			.last()
			.expect("Flush sumcheck composition non-empty by construction");

		if expected_eval != actual_eval {
			return Err(Error::FalseEqEvaluationClaim);
		}
	}

	// Zerocheck
	let (zerocheck_claims, zerocheck_oracle_metas) = table_constraints
		.iter()
		.cloned()
		.map(constraint_set_zerocheck_claim)
		.collect::<Result<Vec<_>, _>>()?
		.into_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	let (max_n_vars, skip_rounds) =
		max_n_vars_and_skip_rounds(&zerocheck_claims, <FDomain<Tower>>::N_BITS);

	let zerocheck_challenges = transcript.sample_vec(max_n_vars - skip_rounds);

	let univariate_cnt = zerocheck_claims
		.partition_point(|zerocheck_claim| zerocheck_claim.n_vars() > max_n_vars - skip_rounds);

	let univariate_output = sumcheck::batch_verify_zerocheck_univariate_round(
		&zerocheck_claims[..univariate_cnt],
		skip_rounds,
		&mut transcript,
	)?;

	let univariate_challenge = univariate_output.univariate_challenge;

	let sumcheck_claims = zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;

	let sumcheck_output = sumcheck::batch_verify_with_start(
		EvaluationOrder::LowToHigh,
		univariate_output.batch_verify_start,
		&sumcheck_claims,
		&mut transcript,
	)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let univariate_cnt =
		zerocheck_claims.partition_point(|claim| claim.n_vars() > max_n_vars - skip_rounds);

	let mut reduction_claims = Vec::with_capacity(univariate_cnt);
	for (claim, univariatized_multilinear_evals) in
		iter::zip(&zerocheck_claims, &zerocheck_output.multilinear_evals)
	{
		let claim_skip_rounds = claim.n_vars().saturating_sub(max_n_vars - skip_rounds);

		let reduction_claim = sumcheck::univariate::univariatizing_reduction_claim(
			claim_skip_rounds,
			univariatized_multilinear_evals,
		)?;

		reduction_claims.push(reduction_claim);
	}

	let univariatizing_output =
		sumcheck::batch_verify(EvaluationOrder::LowToHigh, &reduction_claims, &mut transcript)?;

	let multilinear_zerocheck_output = sumcheck::univariate::verify_sumcheck_outputs(
		&reduction_claims,
		univariate_challenge,
		&zerocheck_output.challenges,
		univariatizing_output,
	)?;

	let zerocheck_eval_claims =
		sumcheck::make_eval_claims(zerocheck_oracle_metas, multilinear_zerocheck_output)?;

	// Evalcheck
	let eval_claims = greedy_evalcheck::verify(
		&mut oracles,
		[non_zero_prodcheck_eval_claims, flush_eval_claims]
			.concat()
			.into_iter()
			.chain(zerocheck_eval_claims),
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
	// We also do back-loaded batching and need to align last skip rounds
	// according to individual claim initial rounds.
	let domain_max_skip_rounds = zerocheck_claims
		.iter()
		.map(|claim| {
			let log_degree = log2_ceil_usize(claim.max_individual_degree());
			max_n_vars - claim.n_vars() + domain_bits.saturating_sub(log_degree)
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
	flushes: &[Flush],
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

pub fn make_flush_oracles<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	flushes: &[Flush],
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
					let first_oracle = flush.oracles.first().ok_or(Error::EmptyFlushOracles)?;
					let n_vars = oracles.n_vars(*first_oracle);
					for &oracle_id in flush.oracles.iter().skip(1) {
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

					let id = oracles
						.add_named(format!("flush channel_id={channel_id}"))
						.linear_combination_with_offset(
							n_vars,
							*permutation_challenge,
							flush
								.oracles
								.iter()
								.copied()
								.zip(mixing_powers.iter().copied()),
						)?;
					Ok(id)
				})
				.collect::<Vec<_>>()
		})
		.collect()
}

#[derive(Debug)]
pub struct FlushSumcheckMeta<F: TowerField> {
	pub composite_sum_claims:
		Vec<CompositeSumClaim<F, IndexComposition<FlushSumcheckComposition, 2>>>,
	pub flush_selectors_unique: Vec<OracleId>,
	pub flush_oracle_ids: Vec<OracleId>,
	pub eval_point: Vec<F>,
}

pub fn get_flush_dedup_sumcheck_metas<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	flush_oracle_ids: &[OracleId],
	flush_selectors: &[OracleId],
	final_layer_claims: &[LayerClaim<F>],
) -> Result<Vec<FlushSumcheckMeta<F>>, Error> {
	let total_flushes = flush_oracle_ids.len();

	debug_assert_eq!(total_flushes, flush_selectors.len());
	debug_assert_eq!(total_flushes, final_layer_claims.len());

	let mut flush_sumcheck_metas = Vec::new();

	let mut begin = 0;
	for end in 1..=total_flushes {
		if end < total_flushes
			&& final_layer_claims[end].eval_point == final_layer_claims[end - 1].eval_point
		{
			continue;
		}

		let eval_point = final_layer_claims[begin].eval_point.clone();
		let n_vars = eval_point.len();

		// deduplicate selector oracles
		let mut flush_selectors_unique = flush_selectors[begin..end].to_vec();
		flush_selectors_unique.sort();
		flush_selectors_unique.dedup();

		let n_selectors = flush_selectors_unique.len();
		let n_multilinears = n_selectors + end - begin;

		let mut composite_sum_claims = Vec::with_capacity(end - begin);

		for (i, (&oracle_id, &flush_selector, layer_claim)) in izip!(
			&flush_oracle_ids[begin..end],
			&flush_selectors[begin..end],
			&final_layer_claims[begin..end]
		)
		.enumerate()
		{
			debug_assert_eq!(n_vars, oracles.n_vars(oracle_id));

			let Some(selector_index) = flush_selectors_unique
				.iter()
				.position(|&flush_selectors_dedup| flush_selectors_dedup == flush_selector)
			else {
				panic!(
					"flush_selectors_unique is a set of unique flush selectors \
                     from a given slice of oracles, search must succeed by construction."
				);
			};

			let composite_sum_claim = CompositeSumClaim {
				composition: IndexComposition::new(
					n_multilinears,
					[i + n_selectors, selector_index],
					FlushSumcheckComposition,
				)?,
				sum: layer_claim.eval,
			};

			composite_sum_claims.push(composite_sum_claim);
		}

		let flush_sumcheck_meta = FlushSumcheckMeta {
			composite_sum_claims,
			flush_selectors_unique,
			flush_oracle_ids: flush_oracle_ids[begin..end].to_vec(),
			eval_point,
		};

		flush_sumcheck_metas.push(flush_sumcheck_meta);

		begin = end;
	}

	Ok(flush_sumcheck_metas)
}

#[derive(Debug)]
pub struct FlushSumcheckComposition;

impl<P: PackedField> CompositionPoly<P> for FlushSumcheckComposition {
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		ArithExpr::Var(0) * ArithExpr::Var(1) + ArithExpr::one() - ArithExpr::Var(1)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		let unmasked_flush = query[0];
		let selector = query[1];

		Ok(selector * unmasked_flush + (P::one() - selector))
	}
}

pub fn get_post_flush_sumcheck_eval_claims_without_eq<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	flush_selectors_unique_by_claim: &[Vec<OracleId>],
	flush_oracle_ids_by_claim: &[Vec<OracleId>],
	sumcheck_output: &BatchSumcheckOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let n_claims = sumcheck_output.multilinear_evals.len();
	debug_assert_eq!(n_claims, flush_oracle_ids_by_claim.len());
	debug_assert_eq!(n_claims, flush_selectors_unique_by_claim.len());

	let max_n_vars = sumcheck_output.challenges.len();

	let mut evalcheck_claims = Vec::new();
	for (flush_oracle_ids, evals, flush_selectors_unique) in izip!(
		flush_oracle_ids_by_claim,
		&sumcheck_output.multilinear_evals,
		flush_selectors_unique_by_claim
	) {
		let n_selectors = flush_selectors_unique.len();
		debug_assert_eq!(evals.len(), n_selectors + flush_oracle_ids.len() + 1);

		for (&flush_selector, &eval) in izip!(flush_selectors_unique, evals) {
			let n_vars = oracles.n_vars(flush_selector);
			let eval_point = sumcheck_output.challenges[max_n_vars - n_vars..].into();

			evalcheck_claims.push(EvalcheckMultilinearClaim {
				id: flush_selector,
				eval_point,
				eval,
			});
		}

		for (&flush_oracle, &eval) in izip!(flush_oracle_ids, &evals[n_selectors..]) {
			let n_vars = oracles.n_vars(flush_oracle);
			let eval_point = sumcheck_output.challenges[max_n_vars - n_vars..].into();

			evalcheck_claims.push(EvalcheckMultilinearClaim {
				id: flush_oracle,
				eval_point,
				eval,
			});
		}
	}

	Ok(evalcheck_claims)
}

pub struct DedupSumcheckClaims<F: TowerField, Composition: CompositionPoly<F>> {
	sumcheck_claims: Vec<SumcheckClaim<F, Composition>>,
	gkr_eval_points: Vec<Vec<F>>,
	flush_selectors_unique_by_claim: Vec<Vec<OracleId>>,
	flush_oracle_ids_by_claim: Vec<Vec<OracleId>>,
}

#[allow(clippy::type_complexity)]
pub fn get_flush_dedup_sumcheck_claims<F: TowerField>(
	flush_sumcheck_metas: Vec<FlushSumcheckMeta<F>>,
) -> Result<DedupSumcheckClaims<F, impl CompositionPoly<F>>, Error> {
	let n_claims = flush_sumcheck_metas.len();
	let mut sumcheck_claims = Vec::with_capacity(n_claims);
	let mut gkr_eval_points = Vec::with_capacity(n_claims);
	let mut flush_oracle_ids_by_claim = Vec::with_capacity(n_claims);
	let mut flush_selectors_unique_by_claim = Vec::with_capacity(n_claims);
	for flush_sumcheck_meta in flush_sumcheck_metas {
		let FlushSumcheckMeta {
			composite_sum_claims,
			flush_selectors_unique,
			flush_oracle_ids,
			eval_point,
		} = flush_sumcheck_meta;

		let composite_sum_claims = composite_sum_claims
			.into_iter()
			.map(|composite_sum_claim| CompositeSumClaim {
				composition: ExtraProduct {
					inner: composite_sum_claim.composition,
				},
				sum: composite_sum_claim.sum,
			})
			.collect::<Vec<_>>();

		let n_vars = eval_point.len();
		let n_multilinears = flush_selectors_unique.len() + flush_oracle_ids.len();
		let sumcheck_claim = SumcheckClaim::new(n_vars, n_multilinears + 1, composite_sum_claims)?;

		sumcheck_claims.push(sumcheck_claim);
		gkr_eval_points.push(eval_point);
		flush_selectors_unique_by_claim.push(flush_selectors_unique);
		flush_oracle_ids_by_claim.push(flush_oracle_ids);
	}

	Ok(DedupSumcheckClaims {
		sumcheck_claims,
		gkr_eval_points,
		flush_selectors_unique_by_claim,
		flush_oracle_ids_by_claim,
	})
}

pub fn reorder_for_flushing_by_n_vars<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	flush_oracle_ids: &[OracleId],
	flush_selectors: Vec<OracleId>,
	flush_final_layer_claims: Vec<LayerClaim<F>>,
) -> (Vec<OracleId>, Vec<usize>, Vec<LayerClaim<F>>) {
	let mut zipped: Vec<_> =
		izip!(flush_oracle_ids.iter().copied(), flush_selectors, flush_final_layer_claims)
			.collect();
	zipped.sort_by_key(|&(id, flush_selector, _)| Reverse((oracles.n_vars(id), flush_selector)));
	multiunzip(zipped)
}
