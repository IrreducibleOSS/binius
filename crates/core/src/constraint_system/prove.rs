// Copyright 2024-2025 Irreducible Inc.

use std::{env, marker::PhantomData};

use binius_compute::{
	alloc::{BumpAllocator, HostBumpAllocator},
	layer::ComputeLayer,
};
use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
	as_packed_field::PackedType,
	linear_transformation::{PackedTransformationFactory, Transformation},
	tower::{PackedTop, ProverTowerFamily, ProverTowerUnderlier},
	underlier::WithUnderlier,
};
use binius_hal::ComputationBackend;
use binius_hash::PseudoCompressionFunction;
use binius_math::{
	CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomainFactory, EvaluationOrder,
	IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_maybe_rayon::prelude::*;
use binius_ntt::SingleThreadedNTT;
use binius_utils::bail;
use digest::{Digest, FixedOutputReset, Output, core_api::BlockSizeUser};
use itertools::{chain, izip};
use tracing::instrument;

use super::{
	ConstraintSystem, Proof,
	channel::Boundary,
	error::Error,
	verify::{make_flush_oracles, max_n_vars_and_skip_rounds},
};
use crate::{
	constraint_system::{
		Flush,
		common::{FDomain, FEncode, FExt, FFastExt},
		exp::{self, reorder_exponents},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeProver,
	oracle::{Constraint, MultilinearOracleSet, MultilinearPolyVariant, OracleId},
	piop,
	protocols::{
		fri::CommitOutput,
		gkr_exp,
		gkr_gpa::{self, GrandProductBatchProveOutput, GrandProductWitness},
		greedy_evalcheck::{self, GreedyEvalcheckProveOutput},
		sumcheck::{
			self, constraint_set_zerocheck_claim, prove::ZerocheckProver,
			standard_switchover_heuristic,
		},
	},
	ring_switch,
	transcript::ProverTranscript,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<U, Tower, Hash, Compress, Challenger_, Backend>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	boundaries: &[Boundary<FExt<Tower>>],
	mut witness: MultilinearExtensionIndex<PackedType<U, FExt<Tower>>>,
	backend: &Backend,
) -> Result<Proof, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	Tower::B128: PackedTop<Tower>,
	Hash: Digest + BlockSizeUser + FixedOutputReset + Send + Sync + Clone,
	Compress: PseudoCompressionFunction<Output<Hash>, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	Backend: ComputationBackend,
	// REVIEW: Consider changing TowerFamily and associated traits to shorten/remove these bounds
	PackedType<U, Tower::B128>: PackedTop<Tower>
		+ PackedFieldIndexable // REVIEW: remove this bound after piop::commit is adjusted
		+ RepackedExtension<PackedType<U, Tower::B8>>
		+ RepackedExtension<PackedType<U, Tower::B16>>
		+ RepackedExtension<PackedType<U, Tower::B32>>
		+ RepackedExtension<PackedType<U, Tower::B64>>
		+ RepackedExtension<PackedType<U, Tower::B128>>
		+ PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
{
	tracing::debug!(
		arch = env::consts::ARCH,
		rayon_threads = binius_maybe_rayon::current_num_threads(),
		"using computation backend: {backend:?}"
	);

	let domain_factory = DefaultEvaluationDomainFactory::<FDomain<Tower>>::default();
	let fast_domain_factory = IsomorphicEvaluationDomainFactory::<FFastExt<Tower>>::default();

	let mut transcript = ProverTranscript::<Challenger_>::new();
	transcript.observe().write_slice(boundaries);

	let ConstraintSystem {
		mut oracles,
		mut table_constraints,
		mut flushes,
		mut exponents,
		non_zero_oracle_ids,
		max_channel_id,
	} = constraint_system.clone();

	reorder_exponents(&mut exponents, &oracles);

	// We must generate multiplication witnesses before committing, as this function
	// adds the committed witnesses for exponentiation results to the witness index.
	let exp_witnesses = exp::make_exp_witnesses::<U, Tower>(&mut witness, &oracles, &exponents)?;

	// Stable sort constraint sets in ascending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| constraint_set.n_vars);

	// Commit polynomials
	let merkle_prover = BinaryMerkleTreeProver::<_, Hash, _>::new(Compress::default());
	let merkle_scheme = merkle_prover.scheme();

	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(&oracles)?;
	let committed_multilins = piop::collect_committed_witnesses::<U, _>(
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
	let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len())?
		.precompute_twiddles()
		.multithreaded();

	let commit_span =
		tracing::info_span!("[phase] Commit", phase = "commit", perfetto_category = "phase.main")
			.entered();
	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = piop::commit(&fri_params, &ntt, &merkle_prover, &committed_multilins)?;
	drop(commit_span);

	// Observe polynomial commitment
	let mut writer = transcript.message();
	writer.write(&commitment);

	// GKR exp
	let exp_challenge = transcript.sample_vec(exp::max_n_vars(&exponents, &oracles));

	let exp_evals = gkr_exp::get_evals_in_point_from_witnesses(&exp_witnesses, &exp_challenge)?
		.into_iter()
		.map(|x| x.into())
		.collect::<Vec<_>>();

	let mut writer = transcript.message();
	writer.write_scalar_slice(&exp_evals);

	let exp_challenge = exp_challenge
		.into_iter()
		.map(|x| x.into())
		.collect::<Vec<_>>();

	let exp_claims = exp::make_claims(&exponents, &oracles, &exp_challenge, &exp_evals)?
		.into_iter()
		.map(|claim| claim.isomorphic())
		.collect::<Vec<_>>();

	let base_exp_output = gkr_exp::batch_prove::<_, _, FFastExt<Tower>, _, _>(
		EvaluationOrder::HighToLow,
		exp_witnesses,
		&exp_claims,
		fast_domain_factory.clone(),
		&mut transcript,
		backend,
	)?
	.isomorphic();

	let exp_eval_claims = exp::make_eval_claims(&exponents, base_exp_output)?;

	// Grand product arguments
	// Grand products for non-zero checking
	let non_zero_fast_witnesses =
		make_fast_unmasked_flush_witnesses::<U, _>(&oracles, &witness, &non_zero_oracle_ids)?;
	let non_zero_prodcheck_witnesses = non_zero_fast_witnesses
		.into_par_iter()
		.map(|(n_vars, evals)| GrandProductWitness::new(n_vars, evals))
		.collect::<Result<Vec<_>, _>>()?;

	let non_zero_products =
		gkr_gpa::get_grand_products_from_witnesses(&non_zero_prodcheck_witnesses);
	if non_zero_products
		.iter()
		.any(|count| *count == Tower::B128::zero())
	{
		bail!(Error::Zeros);
	}

	let mut writer = transcript.message();

	writer.write_scalar_slice(&non_zero_products);

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

	make_masked_flush_witnesses::<U, _>(&oracles, &mut witness, &flush_oracle_ids, &flushes)?;

	// there are no oracle ids associated with these flush_witnesses
	let flush_witnesses =
		make_fast_unmasked_flush_witnesses::<U, _>(&oracles, &witness, &flush_oracle_ids)?;

	// This is important to do in parallel.
	let flush_prodcheck_witnesses = flush_witnesses
		.into_par_iter()
		.map(|(n_vars, evals)| GrandProductWitness::new(n_vars, evals))
		.collect::<Result<Vec<_>, _>>()?;
	let flush_products = gkr_gpa::get_grand_products_from_witnesses(&flush_prodcheck_witnesses);

	transcript.message().write_scalar_slice(&flush_products);

	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracle_ids, &oracles, &flush_products)?;

	// Prove grand products
	let all_gpa_witnesses = [flush_prodcheck_witnesses, non_zero_prodcheck_witnesses].concat();
	let all_gpa_claims = chain!(flush_prodcheck_claims, non_zero_prodcheck_claims)
		.map(|claim| claim.isomorphic())
		.collect::<Vec<_>>();

	let GrandProductBatchProveOutput { final_layer_claims } =
		gkr_gpa::batch_prove::<FFastExt<Tower>, _, FFastExt<Tower>, _, _>(
			EvaluationOrder::HighToLow,
			all_gpa_witnesses,
			&all_gpa_claims,
			&fast_domain_factory,
			&mut transcript,
			backend,
		)?;

	// Apply isomorphism to the layer claims
	let final_layer_claims = final_layer_claims
		.into_iter()
		.map(|layer_claim| layer_claim.isomorphic())
		.collect::<Vec<_>>();

	// Reduce non_zero_final_layer_claims to evalcheck claims
	let prodcheck_eval_claims = gkr_gpa::make_eval_claims(
		chain!(flush_oracle_ids, non_zero_oracle_ids),
		final_layer_claims,
	)?;

	// Zerocheck
	let zerocheck_span = tracing::info_span!(
		"[phase] Zerocheck",
		phase = "zerocheck",
		perfetto_category = "phase.main",
	)
	.entered();

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

	let mut zerocheck_provers = Vec::with_capacity(table_constraints.len());

	for constraint_set in table_constraints {
		let n_vars = constraint_set.n_vars;
		let (constraints, multilinears) =
			sumcheck::prove::split_constraint_set(constraint_set, &witness)?;

		let base_tower_level = chain!(
			multilinears
				.iter()
				.map(|multilinear| 7 - multilinear.log_extension_degree()),
			constraints
				.iter()
				.map(|constraint| constraint.composition.binary_tower_level())
		)
		.max()
		.unwrap_or(0);

		// Per prover zerocheck challenges are justified on the high indexed variables
		let zerocheck_challenges = &zerocheck_challenges[max_n_vars - n_vars.max(skip_rounds)..];
		let domain_factory = domain_factory.clone();

		let constructor =
			ZerocheckProverConstructor::<PackedType<U, FExt<Tower>>, FDomain<Tower>, _, _> {
				constraints,
				multilinears,
				zerocheck_challenges,
				domain_factory,
				backend,
				_fdomain_marker: PhantomData,
			};

		let zerocheck_prover = match base_tower_level {
			0..=3 => constructor.create::<Tower::B8>()?,
			4 => constructor.create::<Tower::B16>()?,
			5 => constructor.create::<Tower::B32>()?,
			6 => constructor.create::<Tower::B64>()?,
			7 => constructor.create::<Tower::B128>()?,
			_ => unreachable!(),
		};

		zerocheck_provers.push(zerocheck_prover);
	}

	let zerocheck_output = sumcheck::prove::batch_prove_zerocheck::<
		FExt<Tower>,
		FDomain<Tower>,
		PackedType<U, FExt<Tower>>,
		_,
		_,
	>(zerocheck_provers, skip_rounds, &mut transcript)?;

	let zerocheck_eval_claims =
		sumcheck::make_zerocheck_eval_claims(zerocheck_oracle_metas, zerocheck_output)?;

	drop(zerocheck_span);

	let evalcheck_span = tracing::info_span!(
		"[phase] Evalcheck",
		phase = "evalcheck",
		perfetto_category = "phase.main"
	)
	.entered();

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		eval_claims,
		memoized_data,
	} = greedy_evalcheck::prove::<_, _, FDomain<Tower>, _, _>(
		&mut oracles,
		&mut witness,
		chain!(prodcheck_eval_claims, zerocheck_eval_claims, exp_eval_claims,),
		standard_switchover_heuristic(-2),
		&mut transcript,
		&domain_factory,
		backend,
	)?;

	// Reduce committed evaluation claims to PIOP sumcheck claims
	let system = ring_switch::EvalClaimSystem::new(
		&oracles,
		&commit_meta,
		&oracle_to_commit_index,
		&eval_claims,
	)?;

	drop(evalcheck_span);

	let ring_switch_span = tracing::info_span!(
		"[phase] Ring Switch",
		phase = "ring_switch",
		perfetto_category = "phase.main"
	)
	.entered();
	let ring_switch::ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims: piop_sumcheck_claims,
	} = ring_switch::prove::<_, _, _, Tower, _, _>(
		&system,
		&committed_multilins,
		&mut transcript,
		memoized_data,
		backend,
	)?;
	drop(ring_switch_span);

	// Prove evaluation claims using PIOP compiler
	let piop_compiler_span = tracing::info_span!(
		"[phase] PIOP Compiler",
		phase = "piop_compiler",
		perfetto_category = "phase.main"
	)
	.entered();
	piop::prove::<_, FDomain<Tower>, _, _, _, _, _, _, _, _, _>(
		&fri_params,
		&ntt,
		&merkle_prover,
		domain_factory,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&piop_sumcheck_claims,
		&mut transcript,
		&backend,
	)?;
	drop(piop_compiler_span);

	let proof = Proof {
		transcript: transcript.finalize(),
	};

	tracing::event!(
		name: "proof_size",
		tracing::Level::INFO,
		counter = true,
		value = proof.get_proof_size() as u64,
		unit = "bytes",
	);

	Ok(proof)
}

// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove_compute_layer<'a, U, Tower, Hash, Compress, Challenger_, Backend, CL>(
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	boundaries: &[Boundary<FExt<Tower>>],
	mut witness: MultilinearExtensionIndex<PackedType<U, FExt<Tower>>>,
	backend: &Backend,
	cl: &'a CL,
	dev_allocator: &'a BumpAllocator<'a, Tower::B128, CL::DevMem>,
	host_allocator: &'a HostBumpAllocator<'a, Tower::B128>,
) -> Result<Proof, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	Tower::B128: PackedTop<Tower>,
	Hash: Digest + BlockSizeUser + FixedOutputReset + Send + Sync + Clone,
	Compress: PseudoCompressionFunction<Output<Hash>, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	Backend: ComputationBackend,
	// REVIEW: Consider changing TowerFamily and associated traits to shorten/remove these bounds
	PackedType<U, Tower::B128>: PackedTop<Tower>
		+ PackedFieldIndexable // REVIEW: remove this bound after piop::commit is adjusted
		+ RepackedExtension<PackedType<U, Tower::B8>>
		+ RepackedExtension<PackedType<U, Tower::B16>>
		+ RepackedExtension<PackedType<U, Tower::B32>>
		+ RepackedExtension<PackedType<U, Tower::B64>>
		+ RepackedExtension<PackedType<U, Tower::B128>>
		+ PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
	CL: ComputeLayer<Tower::B128>,
{
	tracing::debug!(
		arch = env::consts::ARCH,
		rayon_threads = binius_maybe_rayon::current_num_threads(),
		"using computation backend: {backend:?}"
	);

	let domain_factory = DefaultEvaluationDomainFactory::<FDomain<Tower>>::default();
	let fast_domain_factory = IsomorphicEvaluationDomainFactory::<FFastExt<Tower>>::default();

	let mut transcript = ProverTranscript::<Challenger_>::new();
	transcript.observe().write_slice(boundaries);

	let ConstraintSystem {
		mut oracles,
		mut table_constraints,
		mut flushes,
		mut exponents,
		non_zero_oracle_ids,
		max_channel_id,
	} = constraint_system.clone();

	reorder_exponents(&mut exponents, &oracles);

	// We must generate multiplication witnesses before committing, as this function
	// adds the committed witnesses for exponentiation results to the witness index.
	let exp_witnesses = exp::make_exp_witnesses::<U, Tower>(&mut witness, &oracles, &exponents)?;

	// Stable sort constraint sets in ascending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| constraint_set.n_vars);

	// Commit polynomials
	let merkle_prover = BinaryMerkleTreeProver::<_, Hash, _>::new(Compress::default());
	let merkle_scheme = merkle_prover.scheme();

	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(&oracles)?;
	let committed_multilins = piop::collect_committed_witnesses::<U, _>(
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
	let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len())?
		.precompute_twiddles()
		.multithreaded();

	let commit_span =
		tracing::info_span!("[phase] Commit", phase = "commit", perfetto_category = "phase.main")
			.entered();
	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = piop::commit(&fri_params, &ntt, &merkle_prover, &committed_multilins)?;
	drop(commit_span);

	// Observe polynomial commitment
	let mut writer = transcript.message();
	writer.write(&commitment);

	// GKR exp
	let exp_challenge = transcript.sample_vec(exp::max_n_vars(&exponents, &oracles));

	let exp_evals = gkr_exp::get_evals_in_point_from_witnesses(&exp_witnesses, &exp_challenge)?
		.into_iter()
		.map(|x| x.into())
		.collect::<Vec<_>>();

	let mut writer = transcript.message();
	writer.write_scalar_slice(&exp_evals);

	let exp_challenge = exp_challenge
		.into_iter()
		.map(|x| x.into())
		.collect::<Vec<_>>();

	let exp_claims = exp::make_claims(&exponents, &oracles, &exp_challenge, &exp_evals)?
		.into_iter()
		.map(|claim| claim.isomorphic())
		.collect::<Vec<_>>();

	let base_exp_output = gkr_exp::batch_prove::<_, _, FFastExt<Tower>, _, _>(
		EvaluationOrder::HighToLow,
		exp_witnesses,
		&exp_claims,
		fast_domain_factory.clone(),
		&mut transcript,
		backend,
	)?
	.isomorphic();

	let exp_eval_claims = exp::make_eval_claims(&exponents, base_exp_output)?;

	// Grand product arguments
	// Grand products for non-zero checking
	let non_zero_fast_witnesses =
		make_fast_unmasked_flush_witnesses::<U, _>(&oracles, &witness, &non_zero_oracle_ids)?;
	let non_zero_prodcheck_witnesses = non_zero_fast_witnesses
		.into_par_iter()
		.map(|(n_vars, evals)| GrandProductWitness::new(n_vars, evals))
		.collect::<Result<Vec<_>, _>>()?;

	let non_zero_products =
		gkr_gpa::get_grand_products_from_witnesses(&non_zero_prodcheck_witnesses);
	if non_zero_products
		.iter()
		.any(|count| *count == Tower::B128::zero())
	{
		bail!(Error::Zeros);
	}

	let mut writer = transcript.message();

	writer.write_scalar_slice(&non_zero_products);

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

	make_masked_flush_witnesses::<U, _>(&oracles, &mut witness, &flush_oracle_ids, &flushes)?;

	// there are no oracle ids associated with these flush_witnesses
	let flush_witnesses =
		make_fast_unmasked_flush_witnesses::<U, _>(&oracles, &witness, &flush_oracle_ids)?;

	// This is important to do in parallel.
	let flush_prodcheck_witnesses = flush_witnesses
		.into_par_iter()
		.map(|(n_vars, evals)| GrandProductWitness::new(n_vars, evals))
		.collect::<Result<Vec<_>, _>>()?;
	let flush_products = gkr_gpa::get_grand_products_from_witnesses(&flush_prodcheck_witnesses);

	transcript.message().write_scalar_slice(&flush_products);

	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracle_ids, &oracles, &flush_products)?;

	// Prove grand products
	let all_gpa_witnesses = [flush_prodcheck_witnesses, non_zero_prodcheck_witnesses].concat();
	let all_gpa_claims = chain!(flush_prodcheck_claims, non_zero_prodcheck_claims)
		.map(|claim| claim.isomorphic())
		.collect::<Vec<_>>();

	let GrandProductBatchProveOutput { final_layer_claims } =
		gkr_gpa::batch_prove::<FFastExt<Tower>, _, FFastExt<Tower>, _, _>(
			EvaluationOrder::HighToLow,
			all_gpa_witnesses,
			&all_gpa_claims,
			&fast_domain_factory,
			&mut transcript,
			backend,
		)?;

	// Apply isomorphism to the layer claims
	let final_layer_claims = final_layer_claims
		.into_iter()
		.map(|layer_claim| layer_claim.isomorphic())
		.collect::<Vec<_>>();

	// Reduce non_zero_final_layer_claims to evalcheck claims
	let prodcheck_eval_claims = gkr_gpa::make_eval_claims(
		chain!(flush_oracle_ids, non_zero_oracle_ids),
		final_layer_claims,
	)?;

	// Zerocheck
	let zerocheck_span = tracing::info_span!(
		"[phase] Zerocheck",
		phase = "zerocheck",
		perfetto_category = "phase.main",
	)
	.entered();

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

	let mut zerocheck_provers = Vec::with_capacity(table_constraints.len());

	for constraint_set in table_constraints {
		let n_vars = constraint_set.n_vars;
		let (constraints, multilinears) =
			sumcheck::prove::split_constraint_set(constraint_set, &witness)?;

		let base_tower_level = chain!(
			multilinears
				.iter()
				.map(|multilinear| 7 - multilinear.log_extension_degree()),
			constraints
				.iter()
				.map(|constraint| constraint.composition.binary_tower_level())
		)
		.max()
		.unwrap_or(0);

		// Per prover zerocheck challenges are justified on the high indexed variables
		let zerocheck_challenges = &zerocheck_challenges[max_n_vars - n_vars.max(skip_rounds)..];
		let domain_factory = domain_factory.clone();

		let constructor =
			ZerocheckProverConstructor::<PackedType<U, FExt<Tower>>, FDomain<Tower>, _, _> {
				constraints,
				multilinears,
				zerocheck_challenges,
				domain_factory,
				backend,
				_fdomain_marker: PhantomData,
			};

		let zerocheck_prover = match base_tower_level {
			0..=3 => constructor.create::<Tower::B8>()?,
			4 => constructor.create::<Tower::B16>()?,
			5 => constructor.create::<Tower::B32>()?,
			6 => constructor.create::<Tower::B64>()?,
			7 => constructor.create::<Tower::B128>()?,
			_ => unreachable!(),
		};

		zerocheck_provers.push(zerocheck_prover);
	}

	let zerocheck_output = sumcheck::prove::batch_prove_zerocheck::<
		FExt<Tower>,
		FDomain<Tower>,
		PackedType<U, FExt<Tower>>,
		_,
		_,
	>(zerocheck_provers, skip_rounds, &mut transcript)?;

	let zerocheck_eval_claims =
		sumcheck::make_zerocheck_eval_claims(zerocheck_oracle_metas, zerocheck_output)?;

	drop(zerocheck_span);

	let evalcheck_span = tracing::info_span!(
		"[phase] Evalcheck",
		phase = "evalcheck",
		perfetto_category = "phase.main"
	)
	.entered();

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		eval_claims,
		memoized_data,
	} = greedy_evalcheck::prove::<_, _, FDomain<Tower>, _, _>(
		&mut oracles,
		&mut witness,
		chain!(prodcheck_eval_claims, zerocheck_eval_claims, exp_eval_claims,),
		standard_switchover_heuristic(-2),
		&mut transcript,
		&domain_factory,
		backend,
	)?;

	// Reduce committed evaluation claims to PIOP sumcheck claims
	let system = ring_switch::EvalClaimSystem::new(
		&oracles,
		&commit_meta,
		&oracle_to_commit_index,
		&eval_claims,
	)?;

	drop(evalcheck_span);

	let ring_switch_span = tracing::info_span!(
		"[phase] Ring Switch",
		phase = "ring_switch",
		perfetto_category = "phase.main"
	)
	.entered();
	let ring_switch::ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims: piop_sumcheck_claims,
	} = ring_switch::prove::<_, _, _, Tower, _, _>(
		&system,
		&committed_multilins,
		&mut transcript,
		memoized_data,
		backend,
	)?;
	drop(ring_switch_span);

	// Prove evaluation claims using PIOP compiler
	let piop_compiler_span = tracing::info_span!(
		"[phase] PIOP Compiler",
		phase = "piop_compiler",
		perfetto_category = "phase.main"
	)
	.entered();
	piop::prove_compute_layer::<_, _, _, _, _, _, _, _, _>(
		&fri_params,
		&ntt,
		&merkle_prover,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&piop_sumcheck_claims,
		&mut transcript,
		cl,
		dev_allocator,
		host_allocator,
	)?;
	drop(piop_compiler_span);

	let proof = Proof {
		transcript: transcript.finalize(),
	};

	tracing::event!(
		name: "proof_size",
		tracing::Level::INFO,
		counter = true,
		value = proof.get_proof_size() as u64,
		unit = "bytes",
	);

	Ok(proof)
}

type TypeErasedZerocheck<'a, P> = Box<dyn ZerocheckProver<'a, P> + 'a>;

struct ZerocheckProverConstructor<'a, P, FDomain, DomainFactory, Backend>
where
	P: PackedField,
{
	constraints: Vec<Constraint<P::Scalar>>,
	multilinears: Vec<MultilinearWitness<'a, P>>,
	domain_factory: DomainFactory,
	zerocheck_challenges: &'a [P::Scalar],
	backend: &'a Backend,
	_fdomain_marker: PhantomData<FDomain>,
}

impl<'a, P, F, FDomain, DomainFactory, Backend>
	ZerocheckProverConstructor<'a, P, FDomain, DomainFactory, Backend>
where
	F: Field,
	P: PackedField<Scalar = F>,
	FDomain: TowerField,
	DomainFactory: EvaluationDomainFactory<FDomain> + 'a,
	Backend: ComputationBackend,
{
	fn create<FBase>(self) -> Result<TypeErasedZerocheck<'a, P>, Error>
	where
		FBase: TowerField + ExtensionField<FDomain> + TryFrom<F>,
		P: PackedExtension<F, PackedSubfield = P>
			+ PackedExtension<FDomain>
			+ PackedExtension<FBase>,
		F: TowerField,
	{
		let zerocheck_prover =
			sumcheck::prove::constraint_set_zerocheck_prover::<_, _, FBase, _, _, _>(
				self.constraints,
				self.multilinears,
				self.domain_factory,
				self.zerocheck_challenges,
				self.backend,
			)?;

		let type_erased_zerocheck_prover = Box::new(zerocheck_prover) as TypeErasedZerocheck<'a, P>;

		Ok(type_erased_zerocheck_prover)
	}
}

#[instrument(skip_all, level = "debug")]
fn make_masked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &mut MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	flush_oracle_ids: &[OracleId],
	flushes: &[Flush<FExt<Tower>>],
) -> Result<(), Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
{
	let ones = PackedType::<U, FExt<Tower>>::one();
	// The function is on the critical path, parallelize.
	let indices_to_update = izip!(flush_oracle_ids, flushes)
		.map(|(&flush_oracle, flush)| match &oracles[flush_oracle].variant {
			MultilinearPolyVariant::Composite(composite) => {
				let inner_polys = composite.inner();

				let selectors = flush
					.selectors
					.iter()
					.map(|id| witness.get_multilin_poly(*id))
					.collect::<Result<Vec<_>, _>>()?;

				let n_vars = composite.n_vars();

				let log_width = <PackedType<U, FExt<Tower>>>::LOG_WIDTH;

				let len: usize = 1 << n_vars;
				let packed_len: usize = 1 << n_vars.saturating_sub(log_width);

				let inner_c = composite.c();

				let zero_suffixes = count_zero_suffixes(&selectors);

				for (zero_suffix, id, poly) in izip!(&zero_suffixes, inner_polys, selectors) {
					let nonzero_scalars_prefixes = len.saturating_sub(*zero_suffix);

					witness.update_multilin_poly_with_nonzero_scalars_prefixes([(
						*id,
						poly,
						nonzero_scalars_prefixes,
					)])?;
				}

				let polys = inner_polys
					.iter()
					.map(|id| witness.get_multilin_poly(*id))
					.collect::<Result<Vec<_>, _>>()?;

				let max_packed_zero_suffix =
					zero_suffixes.into_iter().max().unwrap_or(0) >> log_width;

				let mut composite_data = (0..packed_len.saturating_sub(max_packed_zero_suffix))
					.into_par_iter()
					.map(|i| {
						let evals = polys
							.iter()
							.map(|poly| {
								<PackedType<U, FExt<Tower>>>::from_fn(|j| {
									let index = i << <PackedType<U, FExt<Tower>>>::LOG_WIDTH | j;
									poly.evaluate_on_hypercube(index).unwrap_or_default()
								})
							})
							.collect::<Vec<_>>();

						inner_c
							.evaluate(&evals)
							.expect("query length is the same as poly length")
					})
					.collect::<Vec<_>>();

				// `ArithExpr::Const(F::ONE) + selector * arith_expr_linear` — so if selector is
				// zero, we fill with ones.
				composite_data.resize(packed_len, ones);

				let composite_poly = MultilinearExtension::new(n_vars, composite_data)
					.expect("data is constructed with the correct length with respect to n_vars");

				Ok((flush_oracle, MLEDirectAdapter::from(composite_poly).upcast_arc_dyn()))
			}
			MultilinearPolyVariant::LinearCombination(lincom) => {
				let polys = lincom
					.polys()
					.map(|id| witness.get_multilin_poly(id))
					.collect::<Result<Vec<_>, _>>()?;

				let packed_len = 1
					<< lincom
						.n_vars()
						.saturating_sub(<PackedType<U, FExt<Tower>>>::LOG_WIDTH);
				let lin_comb_data = (0..packed_len)
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

				let lincom_poly = MultilinearExtension::new(lincom.n_vars(), lin_comb_data)
					.expect("data is constructed with the correct length with respect to n_vars");
				Ok((flush_oracle, MLEDirectAdapter::from(lincom_poly).upcast_arc_dyn()))
			}
			_ => unreachable!("flush_oracles must either be composite or linear combinations"),
		})
		.collect::<Result<Vec<_>, Error>>()?;

	witness.update_multilin_poly(indices_to_update.into_iter())?;
	Ok(())
}

fn count_zero_suffixes<P: PackedField>(polys: &[MultilinearWitness<P>]) -> Vec<usize> {
	let zeros = P::zero();
	polys
		.iter()
		.map(|poly| {
			if let Some(packed_evals) = poly.packed_evals() {
				let mut zero_suffix_len = 0;

				for &packed_evals in packed_evals.iter().rev() {
					if packed_evals != zeros {
						break;
					}
					zero_suffix_len += 1 << (P::LOG_WIDTH + poly.log_extension_degree());
				}
				zero_suffix_len
			} else {
				0
			}
		})
		.collect()
}

#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
fn make_fast_unmasked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	flush_oracles: &[OracleId],
) -> Result<Vec<(usize, Vec<PackedType<U, FFastExt<Tower>>>)>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
{
	let to_fast = Tower::packed_transformation_to_fast();

	// The function is on the critical path, parallelize.
	flush_oracles
		.into_par_iter()
		.map(|&flush_oracle_id| {
			let n_vars = oracles.n_vars(flush_oracle_id);

			let log_width = <PackedType<U, FFastExt<Tower>>>::LOG_WIDTH;

			let poly = witness.get_multilin_poly(flush_oracle_id)?;

			const MAX_SUBCUBE_VARS: usize = 8;
			let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars);
			let subcube_packed_size = 1 << subcube_vars.saturating_sub(log_width);
			let non_const_scalars = 1usize << n_vars;
			let non_const_subcubes = non_const_scalars.div_ceil(1 << subcube_vars);

			let mut fast_ext_result = vec![
				PackedType::<U, FFastExt<Tower>>::one();
				non_const_subcubes * subcube_packed_size
			];

			fast_ext_result
				.par_chunks_exact_mut(subcube_packed_size)
				.enumerate()
				.for_each(|(subcube_index, fast_subcube)| {
					let underliers =
						PackedType::<U, FFastExt<Tower>>::to_underliers_ref_mut(fast_subcube);

					let subcube_evals =
						PackedType::<U, FExt<Tower>>::from_underliers_ref_mut(underliers);
					poly.subcube_evals(subcube_vars, subcube_index, 0, subcube_evals)
						.expect("witness data populated by make_unmasked_flush_witnesses()");

					for underlier in underliers.iter_mut() {
						let src = PackedType::<U, FExt<Tower>>::from_underlier(*underlier);
						let dest = to_fast.transform(&src);
						*underlier = PackedType::<U, FFastExt<Tower>>::to_underlier(dest);
					}
				});

			fast_ext_result.truncate(non_const_scalars);
			Ok((n_vars, fast_ext_result))
		})
		.collect()
}
