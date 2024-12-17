// Copyright 2024 Irreducible Inc.

use super::{
	channel::Boundary,
	error::{Error, VerificationError},
	ConstraintSystem, Proof,
};
use crate::{
	composition::IndexComposition,
	constraint_system::{
		channel::{Flush, FlushDirection},
		common::{
			standard_pcs::{self, FRIMerklePCS, FRIMerkleTowerPCS},
			FExt, TowerPCS, TowerPCSFamily,
		},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree_vcs::BinaryMerkleTreeProver,
	oracle::{CommittedBatch, MultilinearOracleSet, OracleId},
	poly_commit::{batch_pcs::BatchPCS, FRIPCS},
	polynomial::MultivariatePoly,
	protocols::{
		evalcheck::EvalcheckMultilinearClaim,
		gkr_gpa,
		gkr_gpa::LayerClaim,
		greedy_evalcheck,
		sumcheck::{
			self, constraint_set_zerocheck_claim,
			zerocheck::{self, ExtraProduct},
			BatchSumcheckOutput, CompositeSumClaim, SumcheckClaim, ZerocheckClaim,
		},
	},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceReader, CanRead, TranscriptReader},
	transparent::{eq_ind::EqIndPartialEval, step_down},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	BinaryField, ExtensionField, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::Hasher;
use binius_math::{ArithExpr, CompositionPolyOS, EvaluationDomainFactory};
use binius_ntt::NTTOptions;
use binius_utils::bail;
use itertools::{izip, multiunzip, Itertools};
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_ceil_usize;
use std::{cmp::Reverse, iter};
use tracing::instrument;

/// Verifies a proof against a constraint system.
#[instrument("constraint_system::verify", skip_all, level = "debug")]
pub fn verify<U, Tower, Digest, DomainFactory, Hash, Compress, Challenger_>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	domain_factory: DomainFactory,
	boundaries: Vec<Boundary<FExt<Tower>>>,
	proof: Proof,
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField<Scalar: TowerField>,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
{
	let pcss = make_standard_pcss::<U, Tower, _, _, Hash, Compress>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory,
	)?;
	verify_with_pcs::<_, _, _, Challenger_, Digest>(constraint_system, boundaries, proof, &pcss)
}

/// Verifies a proof against a constraint system with provided PCSs.
fn verify_with_pcs<U, Tower, PCSFamily, Challenger_, Digest>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	boundaries: Vec<Boundary<FExt<Tower>>>,
	proof: Proof,
	pcss: &[TowerPCS<Tower, U, PCSFamily>],
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	PCSFamily: TowerPCSFamily<Tower, U, Commitment = Digest>,
	Challenger_: Challenger + Default,
	Digest: PackedField<Scalar: TowerField>,
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

	let Proof { transcript, advice } = proof;

	let mut transcript = TranscriptReader::<Challenger_>::new(transcript);
	let mut advice = AdviceReader::new(advice);

	let backend = make_portable_backend();

	let commitments = transcript.read_packed_slice(oracles.n_batches())?;

	// Grand product arguments
	// Grand products for non-zero checks
	let non_zero_products = transcript.read_scalar_slice(non_zero_oracle_ids.len())?;
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
	let flush_counts = flushes.iter().map(|flush| flush.count).collect::<Vec<_>>();

	let flush_products = transcript.read_scalar_slice(flush_oracle_ids.len())?;
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
		[flush_prodcheck_claims, non_zero_prodcheck_claims].concat(),
		&mut transcript,
	)?;

	let non_zero_final_layer_claims = final_layer_claims.split_off(flush_oracle_ids.len());
	let flush_final_layer_claims = final_layer_claims;

	// Reduce non_zero_final_layer_claims to evalcheck claims
	let non_zero_prodcheck_eval_claims =
		gkr_gpa::make_eval_claims(&oracles, non_zero_oracle_ids, non_zero_final_layer_claims)?;

	// Reduce flush_final_layer_claims to sumcheck claims then evalcheck claims
	let (flush_oracle_ids, flush_counts, flush_final_layer_claims) = reorder_for_flushing_by_n_vars(
		&oracles,
		flush_oracle_ids,
		flush_counts,
		flush_final_layer_claims,
	);

	let flush_sumcheck_metas = get_flush_dedup_sumcheck_metas(
		&mut oracles,
		&flush_oracle_ids,
		&flush_counts,
		&flush_final_layer_claims,
	)?;

	let (flush_sumcheck_claims, gkr_eval_points, all_step_down_metas, flush_oracles_by_claim) =
		get_flush_dedup_sumcheck_claims(flush_sumcheck_metas)?;

	let flush_sumcheck_output = sumcheck::batch_verify(&flush_sumcheck_claims, &mut transcript)?;

	let flush_eval_claims = get_post_flush_sumcheck_eval_claims_without_eq(
		&oracles,
		&all_step_down_metas,
		&flush_oracles_by_claim,
		&flush_sumcheck_output,
	)?;

	// Check the eval claim on the transparent eq polynomial
	for (gkr_eval_point, evals) in izip!(gkr_eval_points, flush_sumcheck_output.multilinear_evals) {
		let gkr_eval_point_len = gkr_eval_point.len();
		let eq_ind = EqIndPartialEval::new(gkr_eval_point_len, gkr_eval_point)?;

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
		max_n_vars_and_skip_rounds(&zerocheck_claims, standard_pcs::FDomain::<Tower>::N_BITS);

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

	let univariatizing_output = sumcheck::batch_verify(&reduction_claims, &mut transcript)?;

	let multilinear_zerocheck_output = sumcheck::univariate::verify_sumcheck_outputs(
		&reduction_claims,
		univariate_challenge,
		&zerocheck_output.challenges,
		univariatizing_output,
	)?;

	let zerocheck_eval_claims =
		sumcheck::make_eval_claims(&oracles, zerocheck_oracle_metas, multilinear_zerocheck_output)?;

	// Evalcheck
	let mut pcs_claims = greedy_evalcheck::verify(
		&mut oracles,
		[non_zero_prodcheck_eval_claims, flush_eval_claims]
			.concat()
			.into_iter()
			.chain(zerocheck_eval_claims),
		&mut transcript,
		&mut advice,
	)?;

	pcs_claims.sort_by_key(|(batch_id, _)| *batch_id);

	// Check that we have a PCS claim for each batch, otherwise the constraint system is
	// under-constrained.
	for (i, (batch_id, _)) in pcs_claims.iter().enumerate() {
		if *batch_id != i {
			bail!(Error::UnconstrainedBatch(i));
		}
	}
	if pcs_claims.len() < oracles.n_batches() {
		bail!(Error::UnconstrainedBatch(pcs_claims.len()));
	}

	// Verify PCS proofs
	for ((_batch_id, claim), pcs, commitment) in izip!(pcs_claims, pcss, commitments) {
		pcs.verify_evaluation(
			&mut advice,
			&mut transcript,
			&commitment,
			&claim.eval_point,
			&claim.evals,
			&backend,
		)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
	}

	transcript.finalize()?;
	advice.finalize()?;

	Ok(())
}

#[allow(clippy::type_complexity)]
pub fn make_standard_pcss<U, Tower, Digest, DomainFactory, Hash, Compress>(
	log_inv_rate: usize,
	security_bits: usize,
	oracles: &MultilinearOracleSet<Tower::B128>,
	domain_factory: DomainFactory,
) -> Result<Vec<FRIMerkleTowerPCS<Tower, U, Digest, DomainFactory, Hash, Compress>>, Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField<Scalar: TowerField>,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	PackedType<U, Tower::B128>: PackedTop<Tower> + PackedFieldIndexable,
{
	oracles
		.committed_batches()
		.into_iter()
		.map(|batch| match batch.tower_level {
			0 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B1),
			3 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B8),
			4 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B16),
			5 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B32),
			6 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B64),
			7 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B128),
			_ => Err(Error::CannotCommitTowerLevel {
				tower_level: batch.tower_level,
			}),
		})
		.collect()
}

#[allow(clippy::type_complexity)]
fn make_standard_pcs<U, Tower, F, Digest, DomainFactory, Hash, Compress>(
	log_inv_rate: usize,
	security_bits: usize,
	domain_factory: DomainFactory,
	batch: CommittedBatch,
) -> Result<FRIMerklePCS<Tower, U, F, Digest, DomainFactory, Hash, Compress>, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<F>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower> + ExtensionField<F> + PackedExtension<F>,
	F: TowerField,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField<Scalar: TowerField>,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	PackedType<U, Tower::B128>: PackedTop<Tower> + PackedFieldIndexable,
{
	let merkle_prover = BinaryMerkleTreeProver::<_, Hash, _>::new(Compress::default());
	let log_n_polys = log2_ceil_usize(batch.n_polys);
	let fri_n_vars = batch.n_vars + log_n_polys;
	let fri_pcs = FRIPCS::<
		_,
		standard_pcs::FDomain<Tower>,
		standard_pcs::FEncode<Tower>,
		PackedType<U, Tower::B128>,
		_,
		_,
		_,
	>::with_optimal_arity(
		fri_n_vars,
		log_inv_rate,
		security_bits,
		merkle_prover,
		domain_factory.clone(),
		NTTOptions::default(),
	)
	.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
	let batch_pcs = BatchPCS::new(fri_pcs, batch.n_vars, log_n_polys)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
	Ok(batch_pcs)
}

pub fn max_n_vars_and_skip_rounds<F, Composition>(
	zerocheck_claims: &[ZerocheckClaim<F, Composition>],
	domain_bits: usize,
) -> (usize, usize)
where
	F: TowerField,
	Composition: CompositionPolyOS<F>,
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
	Composition: CompositionPolyOS<F>,
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
	boundaries: Vec<Boundary<F>>,
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
				match flush.direction {
					FlushDirection::Pull => (pull_product * flush_product, push_product),
					FlushDirection::Push => (pull_product, push_product * flush_product),
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
	let mut flush_iter = flushes.iter().peekable();
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
pub struct StepDownMeta {
	pub n_vars: usize,
	pub flush_count: usize,
	pub step_down: step_down::StepDown,
	pub step_down_oracle_id: OracleId,
}

#[derive(Debug)]
pub struct FlushSumcheckMeta<F: TowerField> {
	pub composite_sum_claims:
		Vec<CompositeSumClaim<F, IndexComposition<FlushSumcheckComposition, 2>>>,
	pub step_down_metas: Vec<StepDownMeta>,
	pub flush_oracle_ids: Vec<OracleId>,
	pub eval_point: Vec<F>,
}

pub fn get_flush_dedup_sumcheck_metas<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	flush_oracle_ids: &[OracleId],
	flush_counts: &[usize],
	final_layer_claims: &[LayerClaim<F>],
) -> Result<Vec<FlushSumcheckMeta<F>>, Error> {
	let total_flushes = flush_oracle_ids.len();

	debug_assert_eq!(total_flushes, flush_counts.len());
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

		// deduplicate StepDown transparent multilinears
		let mut step_down_metas: Vec<StepDownMeta> = Vec::new();
		for &flush_count in &flush_counts[begin..end] {
			if step_down_metas
				.iter()
				.all(|meta| meta.flush_count != flush_count)
			{
				let step_down = step_down::StepDown::new(n_vars, flush_count)?;
				let step_down_oracle_id = oracles
					.add_named("selective_flush_step_down")
					.transparent(step_down.clone())?;

				let step_down_meta = StepDownMeta {
					n_vars,
					flush_count,
					step_down,
					step_down_oracle_id,
				};

				step_down_metas.push(step_down_meta);
			}
		}

		let n_step_downs = step_down_metas.len();
		let n_multilinears = n_step_downs + end - begin;

		let mut composite_sum_claims = Vec::with_capacity(end - begin);

		for (i, (&oracle_id, &flush_count, layer_claim)) in izip!(
			&flush_oracle_ids[begin..end],
			&flush_counts[begin..end],
			&final_layer_claims[begin..end]
		)
		.enumerate()
		{
			debug_assert_eq!(n_vars, oracles.n_vars(oracle_id));

			let Some(step_down_index) = step_down_metas
				.iter()
				.position(|meta| meta.flush_count == flush_count)
			else {
				panic!(
					"step_down_metas is a set of unique step down transparent polynomials \
                        from a given slice of oracles, search must succeed by construction."
				);
			};

			let composite_sum_claim = CompositeSumClaim {
				composition: IndexComposition::new(
					n_multilinears,
					[i + n_step_downs, step_down_index],
					FlushSumcheckComposition,
				)?,
				sum: layer_claim.eval,
			};

			composite_sum_claims.push(composite_sum_claim);
		}

		let flush_sumcheck_meta = FlushSumcheckMeta {
			composite_sum_claims,
			step_down_metas,
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

impl<P: PackedField> CompositionPolyOS<P> for FlushSumcheckComposition {
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
		let step_down = query[1];

		Ok(step_down * unmasked_flush + (P::one() - step_down))
	}
}

pub fn get_post_flush_sumcheck_eval_claims_without_eq<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	all_step_down_metas: &[Vec<StepDownMeta>],
	flush_oracles_by_claim: &[Vec<OracleId>],
	sumcheck_output: &BatchSumcheckOutput<F>,
) -> Result<Vec<EvalcheckMultilinearClaim<F>>, Error> {
	let n_claims = sumcheck_output.multilinear_evals.len();
	debug_assert_eq!(n_claims, flush_oracles_by_claim.len());
	debug_assert_eq!(n_claims, all_step_down_metas.len());

	let max_n_vars = sumcheck_output.challenges.len();

	let mut evalcheck_claims = Vec::new();
	for (flush_oracles, evals, step_down_metas) in
		izip!(flush_oracles_by_claim, &sumcheck_output.multilinear_evals, all_step_down_metas)
	{
		let n_step_downs = step_down_metas.len();
		debug_assert_eq!(evals.len(), n_step_downs + flush_oracles.len() + 1);

		for (meta, &eval) in izip!(step_down_metas, evals) {
			let eval_point = sumcheck_output.challenges[max_n_vars - meta.n_vars..].to_vec();

			evalcheck_claims.push(EvalcheckMultilinearClaim {
				poly: oracles.oracle(meta.step_down_oracle_id),
				eval_point,
				eval,
			});
		}

		for (&flush_oracle, &eval) in izip!(flush_oracles, &evals[n_step_downs..]) {
			let n_vars = oracles.n_vars(flush_oracle);
			let eval_point = sumcheck_output.challenges[max_n_vars - n_vars..].to_vec();

			evalcheck_claims.push(EvalcheckMultilinearClaim {
				poly: oracles.oracle(flush_oracle),
				eval_point,
				eval,
			});
		}
	}

	Ok(evalcheck_claims)
}

#[allow(clippy::type_complexity)]
pub fn get_flush_dedup_sumcheck_claims<F: TowerField>(
	flush_sumcheck_metas: Vec<FlushSumcheckMeta<F>>,
) -> Result<
	(
		Vec<SumcheckClaim<F, impl CompositionPolyOS<F>>>,
		Vec<Vec<F>>,
		Vec<Vec<StepDownMeta>>,
		Vec<Vec<OracleId>>,
	),
	Error,
> {
	let n_claims = flush_sumcheck_metas.len();
	let mut sumcheck_claims = Vec::with_capacity(n_claims);
	let mut gkr_eval_points = Vec::with_capacity(n_claims);
	let mut flush_oracles_by_claim = Vec::with_capacity(n_claims);
	let mut all_step_down_metas = Vec::with_capacity(n_claims);
	for flush_sumcheck_meta in flush_sumcheck_metas {
		let FlushSumcheckMeta {
			composite_sum_claims,
			step_down_metas,
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

		let n_vars = step_down_metas
			.first()
			.expect("flush sumcheck does not create empty provers")
			.n_vars;
		let n_multilinears = step_down_metas.len() + flush_oracle_ids.len();

		let sumcheck_claim = SumcheckClaim::new(n_vars, n_multilinears + 1, composite_sum_claims)?;

		sumcheck_claims.push(sumcheck_claim);
		gkr_eval_points.push(eval_point);
		all_step_down_metas.push(step_down_metas);
		flush_oracles_by_claim.push(flush_oracle_ids);
	}

	Ok((sumcheck_claims, gkr_eval_points, all_step_down_metas, flush_oracles_by_claim))
}

pub fn reorder_for_flushing_by_n_vars<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	flush_oracle_ids: Vec<OracleId>,
	flush_counts: Vec<usize>,
	flush_final_layer_claims: Vec<LayerClaim<F>>,
) -> (Vec<OracleId>, Vec<usize>, Vec<LayerClaim<F>>) {
	let mut zipped: Vec<_> =
		izip!(flush_oracle_ids.iter().copied(), flush_counts, flush_final_layer_claims).collect();
	zipped.sort_by_key(|&(id, flush_count, _)| Reverse((oracles.n_vars(id), flush_count)));
	multiunzip(zipped)
}
