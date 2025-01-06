// Copyright 2024 Irreducible Inc.

use std::{cmp::Reverse, env};

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{
	EvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_utils::bail;
use digest::{core_api::BlockSizeUser, Digest, FixedOutputReset, Output};
use itertools::izip;
use p3_symmetric::PseudoCompressionFunction;
use rayon::prelude::*;
use tracing::instrument;

use super::{
	channel::Boundary,
	error::Error,
	verify::{
		get_post_flush_sumcheck_eval_claims_without_eq, make_flush_oracles,
		max_n_vars_and_skip_rounds, reorder_for_flushing_by_n_vars,
	},
	ConstraintSystem, Proof,
};
use crate::{
	constraint_system::{
		common::{FDomain, FEncode, FExt},
		verify::{get_flush_dedup_sumcheck_metas, FlushSumcheckMeta, StepDownMeta},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeProver,
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, OracleId},
	piop,
	protocols::{
		fri::CommitOutput,
		gkr_gpa::{
			self, gpa_sumcheck::prove::GPAProver, GrandProductBatchProveOutput,
			GrandProductWitness, LayerClaim,
		},
		greedy_evalcheck,
		sumcheck::{
			self, constraint_set_zerocheck_claim,
			prove::{SumcheckProver, UnivariateZerocheckProver},
			standard_switchover_heuristic, zerocheck,
		},
	},
	ring_switch,
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceWriter, CanWrite, Proof as ProofWriter, TranscriptWriter},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<U, Tower, FBase, DomainFactory, Hash, Compress, Challenger_, Backend>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	boundaries: Vec<Boundary<FExt<Tower>>>,
	log_inv_rate: usize,
	security_bits: usize,
	mut witness: MultilinearExtensionIndex<U, FExt<Tower>>,
	domain_factory: DomainFactory,
	backend: &Backend,
) -> Result<Proof, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FBase>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower> + ExtensionField<FBase>,
	FBase: TowerField + ExtensionField<FDomain<Tower>> + TryFrom<FExt<Tower>>,
	DomainFactory: EvaluationDomainFactory<FDomain<Tower>>,
	Hash: Digest + BlockSizeUser + FixedOutputReset,
	Compress: PseudoCompressionFunction<Output<Hash>, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	Backend: ComputationBackend,
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
	PackedType<U, FBase>: PackedFieldIndexable
		+ PackedExtension<FDomain<Tower>, PackedSubfield: PackedFieldIndexable>,
{
	tracing::debug!(
		arch = env::consts::ARCH,
		rayon_threads = rayon::current_num_threads(),
		"using computation backend: {backend:?}"
	);

	let mut transcript = TranscriptWriter::<Challenger_>::default();
	let mut advice = AdviceWriter::default();

	let ConstraintSystem {
		mut oracles,
		mut table_constraints,
		mut flushes,
		non_zero_oracle_ids,
		max_channel_id,
	} = constraint_system.clone();

	// Stable sort constraint sets in descending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| Reverse(constraint_set.n_vars));

	// Commit polynomials
	let merkle_prover = BinaryMerkleTreeProver::<_, Hash, _>::new(Compress::default());
	let merkle_scheme = merkle_prover.scheme();

	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(&oracles)?;
	let committed_multilins = piop::collect_committed_witnesses(
		&commit_meta,
		&oracle_to_commit_index,
		&oracles,
		&witness,
	)?;

	let fri_params = piop::make_commit_params_with_optimal_arity::<_, FEncode<Tower>, _>(
		&commit_meta,
		merkle_scheme,
		security_bits,
		log_inv_rate,
	)?;
	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = piop::commit(&fri_params, &merkle_prover, &committed_multilins)?;

	// Observe polynomial commitment
	transcript.write(&commitment);

	// Observe table heights
	for constraint_set in table_constraints.iter() {
		transcript.write_u64(constraint_set.n_vars as u64);
	}

	// Observe boundary values
	transcript.write_u64(boundaries.len() as u64);
	for boundary in boundaries.iter() {
		boundary.write_to(&mut transcript);
	}

	// Grand product arguments
	// Grand products for non-zero checking
	let non_zero_prodcheck_witnesses =
		gkr_gpa::construct_grand_product_witnesses(&non_zero_oracle_ids, &witness)?;
	let non_zero_products =
		gkr_gpa::get_grand_products_from_witnesses(&non_zero_prodcheck_witnesses);
	if non_zero_products
		.iter()
		.any(|count| *count == Tower::B128::zero())
	{
		bail!(Error::Zeros);
	}

	transcript.write_scalar_slice(&non_zero_products);

	let non_zero_prodcheck_claims = gkr_gpa::construct_grand_product_claims(
		&non_zero_oracle_ids,
		&oracles,
		&non_zero_products,
	)?;

	// Grand products for flushing
	let mixing_challenge = transcript.sample();
	let permutation_challenges = transcript.sample_vec(max_channel_id + 1);

	flushes.sort_by_key(|flush| flush.channel_id);
	let flush_oracle_ids =
		make_flush_oracles(&mut oracles, &flushes, mixing_challenge, &permutation_challenges)?;
	let flush_counts = flushes.iter().map(|flush| flush.count).collect::<Vec<_>>();

	make_unmasked_flush_witnesses(&oracles, &mut witness, &flush_oracle_ids)?;
	// there are no oracle ids associated with these flush_witnesses
	let flush_witnesses =
		make_masked_flush_witnesses(&oracles, &witness, &flush_oracle_ids, &flush_counts)?;

	// This is important to do in parallel.
	let flush_prodcheck_witnesses = flush_witnesses
		.into_par_iter()
		.map(GrandProductWitness::new)
		.collect::<Result<Vec<_>, _>>()?;
	let flush_products = gkr_gpa::get_grand_products_from_witnesses(&flush_prodcheck_witnesses);

	transcript.write_scalar_slice(&flush_products);

	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracle_ids, &oracles, &flush_products)?;

	// Prove grand products
	let GrandProductBatchProveOutput {
		mut final_layer_claims,
	} = gkr_gpa::batch_prove::<_, _, FDomain<Tower>, _, _>(
		[flush_prodcheck_witnesses, non_zero_prodcheck_witnesses].concat(),
		&[flush_prodcheck_claims, non_zero_prodcheck_claims].concat(),
		&domain_factory,
		&mut transcript,
		backend,
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

	let (flush_sumcheck_provers, all_step_down_metas, flush_oracles_by_claim) =
		get_flush_sumcheck_provers::<_, _, FDomain<Tower>, _, _>(
			&mut oracles,
			&flush_oracle_ids,
			&flush_counts,
			&flush_final_layer_claims,
			&mut witness,
			&domain_factory,
			backend,
		)?;

	let flush_sumcheck_output =
		sumcheck::prove::batch_prove(flush_sumcheck_provers, &mut transcript)?;

	let flush_eval_claims = get_post_flush_sumcheck_eval_claims_without_eq(
		&oracles,
		&all_step_down_metas,
		&flush_oracles_by_claim,
		&flush_sumcheck_output,
	)?;

	// Zerocheck
	let (zerocheck_claims, zerocheck_oracle_metas) = table_constraints
		.iter()
		.cloned()
		.map(constraint_set_zerocheck_claim)
		.collect::<Result<Vec<_>, _>>()?
		.into_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	let (max_n_vars, skip_rounds) =
		max_n_vars_and_skip_rounds(&zerocheck_claims, FDomain::<Tower>::N_BITS);

	let zerocheck_challenges = transcript.sample_vec(max_n_vars - skip_rounds);

	let switchover_fn = standard_switchover_heuristic(-2);
	let mut univariate_provers = table_constraints
		.into_iter()
		.map(|constraint_set| {
			let skip_challenges = (max_n_vars - constraint_set.n_vars).saturating_sub(skip_rounds);
			sumcheck::prove::constraint_set_zerocheck_prover::<
				U,
				FBase,
				FExt<Tower>,
				FDomain<Tower>,
				_,
			>(
				constraint_set,
				&witness,
				&domain_factory,
				switchover_fn,
				&zerocheck_challenges[skip_challenges..],
				backend,
			)
		})
		.collect::<Result<Vec<_>, _>>()?;

	let univariate_cnt = univariate_provers
		.partition_point(|univariate_prover| univariate_prover.n_vars() > max_n_vars - skip_rounds);

	let univariatized_multilinears = univariate_provers
		.iter()
		.map(|univariate_prover| univariate_prover.multilinears().clone())
		.collect::<Vec<_>>();

	let tail_provers = univariate_provers.split_off(univariate_cnt);
	let tail_regular_zerocheck_provers = tail_provers
		.into_iter()
		.map(|univariate_prover| univariate_prover.into_regular_zerocheck())
		.collect::<Result<Vec<_>, _>>()?;

	let univariate_output = sumcheck::prove::batch_prove_zerocheck_univariate_round(
		univariate_provers,
		skip_rounds,
		&mut transcript,
	)?;

	let univariate_challenge = univariate_output.univariate_challenge;

	let sumcheck_output = sumcheck::prove::batch_prove_with_start(
		univariate_output.batch_prove_start,
		tail_regular_zerocheck_provers,
		&mut transcript,
	)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let mut reduction_claims = Vec::with_capacity(univariate_cnt);
	let mut reduction_provers = Vec::with_capacity(univariate_cnt);

	for (univariatized_multilinear_evals, multilinears) in
		izip!(&zerocheck_output.multilinear_evals, univariatized_multilinears)
	{
		let claim_n_vars = multilinears
			.first()
			.map_or(0, |multilinear| multilinear.n_vars());

		let skip_challenges = (max_n_vars - claim_n_vars).saturating_sub(skip_rounds);
		let challenges = &zerocheck_output.challenges[skip_challenges..];
		let reduced_multilinears =
			sumcheck::prove::reduce_to_skipped_projection(multilinears, challenges, backend)?;

		let claim_skip_rounds = claim_n_vars - challenges.len();
		let reduction_claim = sumcheck::univariate::univariatizing_reduction_claim(
			claim_skip_rounds,
			univariatized_multilinear_evals,
		)?;

		let reduction_prover =
			sumcheck::prove::univariatizing_reduction_prover::<_, FDomain<Tower>, _, _>(
				reduced_multilinears,
				univariatized_multilinear_evals,
				univariate_challenge,
				&domain_factory,
				backend,
			)?;

		reduction_claims.push(reduction_claim);
		reduction_provers.push(reduction_prover);
	}

	let univariatizing_output = sumcheck::prove::batch_prove(reduction_provers, &mut transcript)?;

	let multilinear_zerocheck_output = sumcheck::univariate::verify_sumcheck_outputs(
		&reduction_claims,
		univariate_challenge,
		&zerocheck_output.challenges,
		univariatizing_output,
	)?;

	let zerocheck_eval_claims =
		sumcheck::make_eval_claims(&oracles, zerocheck_oracle_metas, multilinear_zerocheck_output)?;

	// Prove evaluation claims
	let eval_claims = greedy_evalcheck::prove::<_, _, FDomain<Tower>, _, _>(
		&mut oracles,
		&mut witness,
		[non_zero_prodcheck_eval_claims, flush_eval_claims]
			.concat()
			.into_iter()
			.chain(zerocheck_eval_claims),
		switchover_fn,
		&mut transcript,
		&mut advice,
		&domain_factory,
		backend,
	)?;

	// Reduce committed evaluation claims to PIOP sumcheck claims
	let system =
		ring_switch::EvalClaimSystem::new(&commit_meta, oracle_to_commit_index, &eval_claims)?;

	let mut proof_writer = ProofWriter {
		transcript: &mut transcript,
		advice: &mut advice,
	};
	let ring_switch::ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims: piop_sumcheck_claims,
	} = ring_switch::prove::<_, _, _, Tower, _, _, _>(
		&system,
		&committed_multilins,
		&mut proof_writer,
		backend,
	)?;

	// Prove evaluation claims using PIOP compiler
	piop::prove::<_, FDomain<Tower>, _, _, _, _, _, _, _, _, _>(
		&fri_params,
		&merkle_prover,
		domain_factory,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&piop_sumcheck_claims,
		&mut proof_writer,
		&backend,
	)?;

	Ok(Proof {
		transcript: transcript.finalize(),
		advice: advice.finalize(),
	})
}

#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
fn make_unmasked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &mut MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	flush_oracle_ids: &[OracleId],
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
{
	// The function is on the critical path, parallelize.
	let flush_witnesses: Result<Vec<MultilinearWitness<'a, _>>, Error> = flush_oracle_ids
		.par_iter()
		.map(|&oracle_id| {
			let MultilinearPolyOracle::LinearCombination {
				linear_combination: lincom,
				..
			} = oracles.oracle(oracle_id)
			else {
				unreachable!("make_flush_oracles adds linear combination oracles");
			};
			let polys = lincom
				.polys()
				.map(|oracle| witness.get_multilin_poly(oracle.id()))
				.collect::<Result<Vec<_>, _>>()?;

			let packed_len = 1
				<< lincom
					.n_vars()
					.saturating_sub(<PackedType<U, FExt<Tower>>>::LOG_WIDTH);
			let data = (0..packed_len)
				.into_par_iter()
				.map(|i| {
					<PackedType<U, FExt<Tower>>>::from_fn(|j| {
						let index = i << <PackedType<U, FExt<Tower>>>::LOG_WIDTH | j;
						polys.iter().zip(lincom.coefficients()).fold(
							lincom.offset(),
							|sum, (poly, coeff)| {
								sum + poly
									.evaluate_on_hypercube_and_scale(index, coeff)
									.unwrap_or(<FExt<Tower>>::ZERO)
							},
						)
					})
				})
				.collect::<Vec<_>>();
			let lincom_poly = MultilinearExtension::new(lincom.n_vars(), data)
				.expect("data is constructed with the correct length with respect to n_vars");

			Ok(MLEDirectAdapter::from(lincom_poly).upcast_arc_dyn())
		})
		.collect();

	witness.update_multilin_poly(izip!(flush_oracle_ids.iter().copied(), flush_witnesses?))?;
	Ok(())
}

#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
fn make_masked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	flush_oracles: &[OracleId],
	flush_counts: &[usize],
) -> Result<Vec<MultilinearWitness<'a, PackedType<U, FExt<Tower>>>>, Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
{
	// The function is on the critical path, parallelize.
	flush_oracles
		.par_iter()
		.zip(flush_counts.par_iter())
		.map(|(&flush_oracle_id, &flush_count)| {
			let n_vars = oracles.n_vars(flush_oracle_id);
			let packed_len = 1 << n_vars.saturating_sub(<PackedType<U, FExt<Tower>>>::LOG_WIDTH);
			let mut result = vec![<PackedType<U, FExt<Tower>>>::one(); packed_len];

			let poly = witness.get_multilin_poly(flush_oracle_id)?;
			let width = <PackedType<U, FExt<Tower>>>::WIDTH;
			let packed_index = flush_count / width;
			for (i, result_val) in result.iter_mut().take(packed_index).enumerate() {
				for j in 0..width {
					let index = (i << <PackedType<U, FExt<Tower>>>::LOG_WIDTH) | j;
					result_val.set(j, poly.evaluate_on_hypercube(index)?);
				}
			}
			for j in 0..flush_count % width {
				let index = packed_index << <PackedType<U, FExt<Tower>>>::LOG_WIDTH | j;
				result[packed_index].set(j, poly.evaluate_on_hypercube(index)?);
			}

			let masked_poly = MultilinearExtension::new(n_vars, result)
				.expect("data is constructed with the correct length with respect to n_vars");
			Ok(MLEDirectAdapter::from(masked_poly).upcast_arc_dyn())
		})
		.collect()
}

#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
fn get_flush_sumcheck_provers<'a, 'b, U, Tower, FDomain, DomainFactory, Backend>(
	oracles: &mut MultilinearOracleSet<Tower::B128>,
	flush_oracle_ids: &[OracleId],
	flush_counts: &[usize],
	final_layer_claims: &[LayerClaim<Tower::B128>],
	witness: &mut MultilinearExtensionIndex<'a, U, Tower::B128>,
	domain_factory: DomainFactory,
	backend: &'b Backend,
) -> Result<
	(Vec<impl SumcheckProver<Tower::B128> + 'b>, Vec<Vec<StepDownMeta>>, Vec<Vec<OracleId>>),
	Error,
>
where
	U: TowerUnderlier<Tower> + PackScalar<FDomain>,
	Tower: TowerFamily,
	Tower::B128: ExtensionField<FDomain>,
	FDomain: Field,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Backend: ComputationBackend,
	PackedType<U, Tower::B128>: PackedFieldIndexable,
	'a: 'b,
{
	let flush_sumcheck_metas = get_flush_dedup_sumcheck_metas(
		oracles,
		flush_oracle_ids,
		flush_counts,
		final_layer_claims,
	)?;

	let n_claims = flush_sumcheck_metas.len();
	let mut provers = Vec::with_capacity(n_claims);
	let mut flush_oracles_by_claim = Vec::with_capacity(n_claims);
	let mut all_step_down_metas = Vec::with_capacity(n_claims);
	for flush_sumcheck_meta in flush_sumcheck_metas {
		let FlushSumcheckMeta {
			composite_sum_claims,
			step_down_metas,
			flush_oracle_ids,
			eval_point,
		} = flush_sumcheck_meta;

		let mut multilinears = Vec::with_capacity(step_down_metas.len() + flush_oracle_ids.len());

		for meta in &step_down_metas {
			let oracle_id = meta.step_down_oracle_id;

			let step_down_multilinear =
				MLEDirectAdapter::from(meta.step_down.multilinear_extension()?).upcast_arc_dyn();

			witness.update_multilin_poly([(oracle_id, step_down_multilinear)])?;
			multilinears.push(witness.get_multilin_poly(oracle_id)?);
		}

		for &oracle_id in &flush_oracle_ids {
			multilinears.push(witness.get_multilin_poly(oracle_id)?);
		}

		let prover = GPAProver::new(
			multilinears,
			None,
			composite_sum_claims,
			domain_factory.clone(),
			&eval_point,
			backend,
		)?;

		provers.push(prover);
		flush_oracles_by_claim.push(flush_oracle_ids);
		all_step_down_metas.push(step_down_metas);
	}

	Ok((provers, all_step_down_metas, flush_oracles_by_claim))
}
