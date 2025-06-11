// Copyright 2024-2025 Irreducible Inc.

use std::{collections::HashSet, env, iter, marker::PhantomData};

use binius_compute::{ComputeData, ComputeLayer, alloc::ComputeAllocator, cpu::CpuMemory};
use binius_field::{
	AESTowerField8b, AESTowerField128b, BinaryField, ByteSlicedUnderlier, ExtensionField, Field,
	PackedExtension, PackedField, PackedFieldIndexable, RepackedExtension, TowerField,
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	tower::{PackedTop, ProverTowerFamily, ProverTowerUnderlier},
	underlier::{NumCast, UnderlierWithBitOps, WithUnderlier},
	util::powers,
};
use binius_hal::ComputationBackend;
use binius_hash::{PseudoCompressionFunction, multi_digest::ParallelDigest};
use binius_math::{
	B1, B8, CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomainFactory,
	EvaluationOrder, IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly,
};
use binius_maybe_rayon::prelude::*;
use binius_ntt::SingleThreadedNTT;
use binius_utils::{bail, checked_arithmetics::log2_ceil_usize};
use bytemuck::zeroed_vec;
use digest::{FixedOutputReset, Output, core_api::BlockSizeUser};
use itertools::chain;
use tracing::instrument;

use super::{
	ConstraintSystem, Proof,
	channel::Boundary,
	error::Error,
	verify::{make_flush_oracles, max_n_vars_and_skip_rounds},
};
use crate::{
	constraint_system::{
		Flush, TableSizeSpec,
		channel::OracleOrConst,
		common::{FDomain, FEncode, FExt, FFastExt},
		exp::{self, reorder_exponents},
		verify::augument_flush_po2_step_down,
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeProver,
	oracle::{
		Constraint, ConstraintSetBuilder, MultilinearOracleSet, MultilinearPolyVariant, OracleId,
		SizedConstraintSet,
	},
	piop,
	protocols::{
		evalcheck::{
			ConstraintSetEqIndPoint, EvalPoint, EvalcheckMultilinearClaim,
			subclaims::{MemoizedData, prove_mlecheck_with_switchover},
		},
		fri::CommitOutput,
		gkr_exp,
		gkr_gpa::{self, GrandProductBatchProveOutput, GrandProductWitness},
		greedy_evalcheck::{self, GreedyEvalcheckProveOutput},
		sumcheck::{
			self, constraint_set_zerocheck_claim, immediate_switchover_heuristic,
			prove::ZerocheckProver, standard_switchover_heuristic,
		},
	},
	ring_switch,
	transcript::ProverTranscript,
	transparent::step_down::StepDown,
	witness::{IndexEntry, MultilinearExtensionIndex, MultilinearWitness},
};

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[allow(clippy::too_many_arguments)]
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<
	Hal,
	U,
	Tower,
	Hash,
	Compress,
	Challenger_,
	Backend,
	HostAllocatorType,
	DeviceAllocatorType,
>(
	compute_data: &mut ComputeData<Tower::B128, Hal, HostAllocatorType, DeviceAllocatorType>,
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	constraint_system_digest: &Output<Hash::Digest>,
	boundaries: &[Boundary<FExt<Tower>>],
	table_sizes: &[usize],
	mut witness: MultilinearExtensionIndex<PackedType<U, FExt<Tower>>>,
	backend: &Backend,
) -> Result<Proof, Error>
where
	Hal: ComputeLayer<Tower::B128> + Default,
	U: ProverTowerUnderlier<Tower> + UnderlierWithBitOps + From<u8> + Pod,
	Tower: ProverTowerFamily,
	Tower::B128: binius_math::TowerTop
		+ binius_math::PackedTop
		+ PackedTop<Tower>
		+ From<FFastExt<Tower>>
		+ From<AESTowerField128b>
		+ ExtensionField<B8>,
	Hash: ParallelDigest,
	Hash::Digest: BlockSizeUser + FixedOutputReset + Send + Sync + Clone,
	Compress: PseudoCompressionFunction<Output<Hash::Digest>, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	Backend: ComputationBackend,
	// REVIEW: Consider changing TowerFamily and associated traits to shorten/remove these bounds
	PackedType<U, Tower::B128>: PackedTop<Tower>
		+ PackedFieldIndexable
		// REVIEW: remove this bound after piop::commit is adjusted
		+ RepackedExtension<PackedType<U, Tower::B1>>
		+ RepackedExtension<PackedType<U, Tower::B8>>
		+ RepackedExtension<PackedType<U, Tower::B16>>
		+ RepackedExtension<PackedType<U, Tower::B32>>
		+ RepackedExtension<PackedType<U, Tower::B64>>
		+ RepackedExtension<PackedType<U, Tower::B128>>
		+ PackedTransformationFactory<PackedType<U, Tower::FastB128>>
		+ binius_math::PackedTop,
	PackedType<U, Tower::FastB128>: PackedTransformationFactory<PackedType<U, Tower::B128>>,
	HostAllocatorType: ComputeAllocator<Tower::B128, CpuMemory>,
	DeviceAllocatorType: ComputeAllocator<Tower::B128, Hal::DevMem>,
{
	tracing::debug!(
		arch = env::consts::ARCH,
		rayon_threads = binius_maybe_rayon::current_num_threads(),
		"using computation backend: {backend:?}"
	);

	let domain_factory = DefaultEvaluationDomainFactory::<FDomain<Tower>>::default();
	let fast_domain_factory = IsomorphicEvaluationDomainFactory::<FFastExt<Tower>>::default();
	let aes_domain_factory = IsomorphicEvaluationDomainFactory::<AESTowerField8b>::default();

	let ConstraintSystem {
		mut oracles,
		table_constraints,
		mut flushes,
		mut exponents,
		non_zero_oracle_ids,
		channel_count,
		table_size_specs,
	} = constraint_system.clone();

	if table_sizes.len() != table_size_specs.len() {
		return Err(Error::TableSizesLenMismatch {
			expected: table_size_specs.len(),
			got: table_sizes.len(),
		});
	}
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

	let mut transcript = ProverTranscript::<Challenger_>::new();
	transcript
		.observe()
		.write_slice(constraint_system_digest.as_ref());
	transcript.observe().write_slice(boundaries);
	let mut writer = transcript.message();
	writer.write_slice(table_sizes);

	reorder_exponents(&mut exponents, &oracles);

	let witness_span = tracing::info_span!(
		"[phase] Witness Finalization",
		phase = "witness",
		perfetto_category = "phase.main"
	)
	.entered();

	// We must generate multiplication witnesses before committing, as this function
	// adds the committed witnesses for exponentiation results to the witness index.
	let exp_compute_layer_span = tracing::info_span!(
		"[step] Compute Exponentiation Layers",
		phase = "witness",
		perfetto_category = "phase.sub"
	)
	.entered();
	let exp_witnesses = exp::make_exp_witnesses::<U, Tower>(&mut witness, &oracles, &exponents)?;
	drop(exp_compute_layer_span);

	drop(witness_span);

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
	let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
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

	let exp_span = tracing::info_span!(
		"[phase] Exponentiation",
		phase = "exp",
		perfetto_category = "phase.main"
	)
	.entered();
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
		fast_domain_factory,
		&mut transcript,
		backend,
	)?
	.isomorphic();

	let exp_eval_claims = exp::make_eval_claims(&exponents, base_exp_output)?;
	drop(exp_span);

	// Grand product arguments
	// Grand products for non-zero checking
	let prodcheck_span = tracing::info_span!(
		"[phase] Product Check",
		phase = "prodcheck",
		perfetto_category = "phase.main"
	)
	.entered();

	let nonzero_convert_span = tracing::info_span!(
		"[task] Convert Non-Zero to Fast Field",
		phase = "prodcheck",
		perfetto_category = "task.main"
	)
	.entered();
	let non_zero_fast_witnesses =
		convert_witnesses_to_bytesliced::<U, _>(&witness, &non_zero_oracle_ids)?;
	drop(nonzero_convert_span);

	let nonzero_prodcheck_compute_layer_span = tracing::info_span!(
		"[step] Compute Non-Zero Product Layers",
		phase = "prodcheck",
		perfetto_category = "phase.sub"
	)
	.entered();
	let non_zero_prodcheck_witnesses = non_zero_fast_witnesses
		.into_par_iter()
		.map(|(n_vars, evals)| GrandProductWitness::new(n_vars, evals))
		.collect::<Result<Vec<_>, _>>()?;
	drop(nonzero_prodcheck_compute_layer_span);

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
	let permutation_challenges = transcript.sample_vec(channel_count);

	flushes.retain(|flush| table_sizes[flush.table_id] > 0);
	flushes.sort_by_key(|flush| flush.channel_id);
	let po2_step_down_polys =
		augument_flush_po2_step_down(&mut oracles, &mut flushes, &table_size_specs, table_sizes)?;
	populate_flush_po2_step_down_witnesses::<U, _>(po2_step_down_polys, &mut witness)?;
	let flush_oracle_ids =
		make_flush_oracles(&mut oracles, &flushes, mixing_challenge, &permutation_challenges)?;

	let flush_convert_span = tracing::info_span!(
		"[task] Convert Flushes to Fast Field",
		phase = "prodcheck",
		perfetto_category = "task.main"
	)
	.entered();

	let mut fast_witness = MultilinearExtensionIndex::<
		PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>,
	>::new();

	make_masked_flush_witnesses::<U, _>(
		&oracles,
		&mut witness,
		&mut fast_witness,
		&flush_oracle_ids,
		&flushes,
		mixing_challenge,
		&permutation_challenges,
	)?;

	// there are no oracle ids associated with these flush_witnesses
	let flush_witnesses = convert_witnesses_to_bytesliced::<U, _>(&witness, &flush_oracle_ids)?;
	drop(flush_convert_span);

	let flush_prodcheck_compute_layer_span = tracing::info_span!(
		"[step] Compute Flush Product Layers",
		phase = "prodcheck",
		perfetto_category = "phase.sub"
	)
	.entered();
	let flush_prodcheck_witnesses = flush_witnesses
		.into_par_iter()
		.map(|(n_vars, evals)| GrandProductWitness::new(n_vars, evals))
		.collect::<Result<Vec<_>, _>>()?;
	drop(flush_prodcheck_compute_layer_span);

	let flush_products = gkr_gpa::get_grand_products_from_witnesses(&flush_prodcheck_witnesses);

	transcript.message().write_scalar_slice(&flush_products);

	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracle_ids, &oracles, &flush_products)?;

	// Prove grand products
	let all_gpa_witnesses =
		chain!(flush_prodcheck_witnesses, non_zero_prodcheck_witnesses).collect::<Vec<_>>();
	let all_gpa_claims = chain!(flush_prodcheck_claims, non_zero_prodcheck_claims)
		.map(|claim| claim.isomorphic())
		.collect::<Vec<_>>();

	let GrandProductBatchProveOutput { final_layer_claims } =
		gkr_gpa::batch_prove::<AESTowerField128b, _, AESTowerField8b, _, _>(
			EvaluationOrder::HighToLow,
			all_gpa_witnesses,
			&all_gpa_claims,
			&aes_domain_factory,
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
		chain!(flush_oracle_ids.clone(), non_zero_oracle_ids),
		final_layer_claims,
	)?;

	let mut flush_prodcheck_eval_claims = prodcheck_eval_claims;

	let prodcheck_eval_claims = flush_prodcheck_eval_claims.split_off(flush_oracle_ids.len());

	let flush_eval_claims = reduce_flush_evalcheck_claims::<U, Tower, Challenger_, Backend>(
		flush_prodcheck_eval_claims,
		&oracles,
		fast_witness,
		aes_domain_factory,
		&mut transcript,
		backend,
	)?;

	drop(prodcheck_span);

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
		chain!(flush_eval_claims, prodcheck_eval_claims, zerocheck_eval_claims, exp_eval_claims,),
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

	let hal = compute_data.hal;

	let dev_alloc = &compute_data.dev_alloc;
	let host_alloc = &compute_data.host_alloc;

	let ring_switch::ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims: piop_sumcheck_claims,
	} = ring_switch::prove(
		&system,
		&committed_multilins,
		&mut transcript,
		memoized_data,
		hal,
		dev_alloc,
		host_alloc,
	)?;
	drop(ring_switch_span);

	// Prove evaluation claims using PIOP compiler
	let piop_compiler_span = tracing::info_span!(
		"[phase] PIOP Compiler",
		phase = "piop_compiler",
		perfetto_category = "phase.main"
	)
	.entered();

	piop::prove(
		compute_data,
		&fri_params,
		&ntt,
		&merkle_prover,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		transparent_multilins,
		&piop_sumcheck_claims,
		&mut transcript,
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

fn populate_flush_po2_step_down_witnesses<'a, U, Tower>(
	step_down_polys: Vec<(OracleId, StepDown)>,
	witness: &mut MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
) -> Result<(), Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
{
	for (oracle_id, step_down_poly) in step_down_polys {
		let witness_poly = step_down_poly
			.multilinear_extension::<PackedType<U, Tower::B1>>()?
			.specialize_arc_dyn();
		witness.update_multilin_poly([(oracle_id, witness_poly)])?
	}
	Ok(())
}

#[instrument(skip_all, level = "debug")]
pub fn make_masked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness_index: &mut MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	fast_witness_index: &mut MultilinearExtensionIndex<
		'a,
		PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>,
	>,
	flush_oracle_ids: &[OracleId],
	flushes: &[Flush<FExt<Tower>>],
	mixing_challenge: FExt<Tower>,
	permutation_challenges: &[FExt<Tower>],
) -> Result<(), Error>
where
	U: ProverTowerUnderlier<Tower> + UnderlierWithBitOps + From<u8> + Pod,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: RepackedExtension<PackedType<U, Tower::B1>>,
	ByteSlicedUnderlier<U, 16>:
		PackScalar<B1, Packed: Pod> + PackScalar<AESTowerField128b, Packed: Pod>,
	u8: NumCast<U>,
	PackedType<U, B8>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>,
{
	// TODO: Move me out into a separate function & deduplicate.
	// Count the suffix zeros on all selectors.
	for flush in flushes {
		let fast_selectors =
			convert_1b_witnesses_to_byte_sliced::<U, Tower>(witness_index, &flush.selectors)?;

		for (&selector_id, fast_selector) in flush.selectors.iter().zip(fast_selectors) {
			let selector = witness_index.get_multilin_poly(selector_id)?;
			let zero_suffix_len = count_zero_suffixes(&selector);

			let nonzero_prefix_len = (1 << selector.n_vars()) - zero_suffix_len;
			witness_index.update_multilin_poly_with_nonzero_scalars_prefixes([(
				selector_id,
				selector,
				nonzero_prefix_len,
			)])?;

			fast_witness_index.update_multilin_poly_with_nonzero_scalars_prefixes([(
				selector_id,
				fast_selector,
				nonzero_prefix_len,
			)])?;
		}
	}

	let inner_oracles_id = flushes
		.iter()
		.flat_map(|flush| {
			flush
				.oracles
				.iter()
				.filter_map(|oracle_or_const| match oracle_or_const {
					OracleOrConst::Oracle(oracle_id) => Some(*oracle_id),
					_ => None,
				})
		})
		.collect::<HashSet<_>>();

	let inner_oracles_id = inner_oracles_id.into_iter().collect::<Vec<_>>();

	let fast_inner_oracles =
		convert_witnesses_to_bytesliced::<U, Tower>(witness_index, &inner_oracles_id)?;

	for ((n_vars, witness_data), id) in fast_inner_oracles.into_iter().zip(inner_oracles_id) {
		let fast_witness = MLEDirectAdapter::from(
			MultilinearExtension::new(n_vars, witness_data)
				.expect("witness_data created with correct n_vars"),
		);

		let nonzero_scalars_prefix = witness_index.get_index_entry(id)?.nonzero_scalars_prefix;

		fast_witness_index.update_multilin_poly_with_nonzero_scalars_prefixes([(
			id,
			fast_witness.upcast_arc_dyn(),
			nonzero_scalars_prefix,
		)])?;
	}

	// Find the maximum power of the mixing challenge needed.
	let max_n_mixed = flushes
		.iter()
		.map(|flush| flush.oracles.len())
		.max()
		.unwrap_or_default();
	let mixing_powers = powers(mixing_challenge)
		.take(max_n_mixed)
		.collect::<Vec<_>>();

	// The function is on the critical path, parallelize.
	let indices_to_update = flush_oracle_ids
		.par_iter()
		.zip(flushes)
		.map(|(&flush_oracle, flush)| {
			let n_vars = oracles.n_vars(flush_oracle);

			let const_term = flush
				.oracles
				.iter()
				.copied()
				.zip(mixing_powers.iter())
				.filter_map(|(oracle_or_const, coeff)| match oracle_or_const {
					OracleOrConst::Const { base, .. } => Some(base * coeff),
					_ => None,
				})
				.sum::<FExt<Tower>>();
			let const_term = permutation_challenges[flush.channel_id] + const_term;

			let inner_oracles = flush
				.oracles
				.iter()
				.copied()
				.zip(mixing_powers.iter())
				.filter_map(|(oracle_or_const, &coeff)| match oracle_or_const {
					OracleOrConst::Oracle(oracle_id) => Some((oracle_id, coeff)),
					_ => None,
				})
				.map(|(inner_id, coeff)| {
					let witness = witness_index.get_multilin_poly(inner_id)?;
					Ok((witness, coeff))
				})
				.collect::<Result<Vec<_>, Error>>()?;

			let selector_entries = flush
				.selectors
				.iter()
				.map(|id| witness_index.get_index_entry(*id))
				.collect::<Result<Vec<_>, _>>()?;

			// Get the number of entries before any selector column is fully disabled.
			let selector_prefix_len = selector_entries
				.iter()
				.map(|selector_entry| selector_entry.nonzero_scalars_prefix)
				.min()
				.unwrap_or(1 << n_vars);

			let selectors = selector_entries
				.into_iter()
				.map(|entry| entry.multilin_poly)
				.collect::<Vec<_>>();

			let log_width = <PackedType<U, FExt<Tower>>>::LOG_WIDTH;
			let packed_selector_prefix_len = selector_prefix_len.div_ceil(1 << log_width);

			let mut witness_data = Vec::with_capacity(1 << n_vars.saturating_sub(log_width));
			(0..packed_selector_prefix_len)
				.into_par_iter()
				.map(|i| {
					<PackedType<U, FExt<Tower>>>::from_fn(|j| {
						let index = i << log_width | j;

						// If n_vars < P::LOG_WIDTH, fill the remaining scalars with zeroes.
						if index >= 1 << n_vars {
							return <FExt<Tower>>::ZERO;
						}

						// Compute the product of all selectors at this point
						let selector_off = selectors.iter().any(|selector| {
							let sel_val = selector
								.evaluate_on_hypercube(index)
								.expect("index < 1 << n_vars");
							sel_val.is_zero()
						});

						if selector_off {
							// If any selector is zero, the result is 1
							<FExt<Tower>>::ONE
						} else {
							// Otherwise, compute the linear combination
							let mut inner_oracles_iter = inner_oracles.iter();

							// Handle the first one specially because the mixing power is ONE,
							// unless the first oracle was a constant.
							if let Some((poly, coeff)) = inner_oracles_iter.next() {
								let first_term = if *coeff == FExt::<Tower>::ONE {
									poly.evaluate_on_hypercube(index).expect("index in bounds")
								} else {
									poly.evaluate_on_hypercube_and_scale(index, *coeff)
										.expect("index in bounds")
								};
								inner_oracles_iter.fold(
									const_term + first_term,
									|sum, (poly, coeff)| {
										let scaled_eval = poly
											.evaluate_on_hypercube_and_scale(index, *coeff)
											.expect("index in bounds");
										sum + scaled_eval
									},
								)
							} else {
								const_term
							}
						}
					})
				})
				.collect_into_vec(&mut witness_data);
			witness_data.resize(witness_data.capacity(), PackedType::<U, FExt<Tower>>::one());

			let witness = MLEDirectAdapter::from(
				MultilinearExtension::new(n_vars, witness_data)
					.expect("witness_data created with correct n_vars"),
			);
			// TODO: This is sketchy. The field on witness index is called "nonzero_prefix", but
			// I'm setting it when the suffix is 1, not zero.
			Ok((witness, selector_prefix_len))
		})
		.collect::<Result<Vec<_>, Error>>()?;

	witness_index.update_multilin_poly_with_nonzero_scalars_prefixes(
		iter::zip(flush_oracle_ids, indices_to_update).map(
			|(&oracle_id, (witness, nonzero_scalars_prefix))| {
				(oracle_id, witness.upcast_arc_dyn(), nonzero_scalars_prefix)
			},
		),
	)?;
	Ok(())
}

fn count_zero_suffixes<P: PackedField, M: MultilinearPoly<P>>(poly: &M) -> usize {
	let zeros = P::zero();
	if let Some(packed_evals) = poly.packed_evals() {
		let packed_zero_suffix_len = packed_evals
			.iter()
			.rev()
			.position(|&packed_eval| packed_eval != zeros)
			.unwrap_or(packed_evals.len());

		let log_scalars_per_elem = P::LOG_WIDTH + poly.log_extension_degree();
		if poly.n_vars() < log_scalars_per_elem {
			debug_assert_eq!(packed_evals.len(), 1, "invariant of MultilinearPoly");
			packed_zero_suffix_len << poly.n_vars()
		} else {
			packed_zero_suffix_len << log_scalars_per_elem
		}
	} else {
		0
	}
}

/// Converts specified oracles' witness representations from the base extension field
/// to the fast extension field format for optimized grand product calculations.
///
/// This function processes the provided list of oracle IDs, extracting the corresponding
/// multilinear polynomials from the witness index, and converting their evaluations
/// to the fast field representation. The conversion is performed efficiently using
/// the tower transformation infrastructure.
///
/// # Performance Considerations
/// - This function is optimized for parallel execution as it's on the critical path of the proving
///   system.
///
/// # Arguments
/// * `oracles` - Reference to the multilinear oracle set containing metadata for all oracles
/// * `witness` - Reference to the witness index containing the multilinear polynomial evaluations
/// * `oracle_ids` - Slice of oracle IDs for which to generate fast field representations
///
/// # Returns
/// A vector of tuples, where each tuple contains:
/// - The number of variables in the oracle's multilinear polynomial
/// - A vector of packed field elements representing the polynomial's evaluations in the fast field
///
/// # Errors
/// Returns an error if:
/// - Any oracle ID is invalid or not found in the witness index
/// - Subcube evaluation fails for any polynomial
#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
fn convert_witnesses_to_fast_ext<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	oracle_ids: &[OracleId],
) -> Result<Vec<(usize, Vec<PackedType<U, FFastExt<Tower>>>)>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
{
	let to_fast = Tower::packed_transformation_to_fast();

	// The function is on the critical path, parallelize.
	oracle_ids
		.into_par_iter()
		.map(|&flush_oracle_id| {
			let n_vars = oracles.n_vars(flush_oracle_id);

			let log_width = <PackedType<U, FFastExt<Tower>>>::LOG_WIDTH;

			let IndexEntry {
				multilin_poly: poly,
				nonzero_scalars_prefix,
			} = witness.get_index_entry(flush_oracle_id)?;

			const MAX_SUBCUBE_VARS: usize = 8;
			let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars);
			let subcube_packed_size = 1 << subcube_vars.saturating_sub(log_width);
			let non_const_scalars = nonzero_scalars_prefix;
			let non_const_subcubes = non_const_scalars.div_ceil(1 << subcube_vars);

			let mut fast_ext_result = zeroed_vec(non_const_subcubes * subcube_packed_size);
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

#[allow(clippy::type_complexity)]
fn convert_witnesses_to_bytesliced<'a, U, Tower>(
	witness: &MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	oracle_ids: &[OracleId],
) -> Result<Vec<(usize, Vec<PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>>)>, Error>
where
	U: ProverTowerUnderlier<Tower> + UnderlierWithBitOps + From<u8> + Pod,
	Tower: ProverTowerFamily,
	PackedType<U, B8>: PackedTransformationFactory<PackedType<U, AESTowerField8b>>,
	ByteSlicedUnderlier<U, 16>: PackScalar<AESTowerField128b, Packed: Pod>,
	u8: NumCast<U>,
{
	oracle_ids
		.iter()
		.map(|&flush_oracle_id| {
			let poly = witness.get_multilin_poly(flush_oracle_id)?;

			let log_width = PackedType::<U, FExt<Tower>>::LOG_WIDTH;

			let mut evals = zeroed_vec((1 << poly.n_vars().saturating_sub(log_width)).max(16));

			poly.subcube_evals(
				poly.n_vars(),
				0,
				0,
				&mut evals[0..(1 << poly.n_vars().saturating_sub(log_width))],
			)?;

			Ok((poly.n_vars(), Tower::transform_128b_to_bytesliced(evals.clone())))
		})
		.collect::<Result<Vec<_>, _>>()
}

#[allow(clippy::type_complexity)]
pub fn convert_1b_witnesses_to_fast_ext<'a, U, Tower, FTrg>(
	witness: &MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	ids: &[OracleId],
) -> Result<Vec<MultilinearWitness<'a, PackedType<U, FTrg>>>, Error>
where
	U: ProverTowerUnderlier<Tower> + PackScalar<FTrg>,
	FTrg: TowerField + ExtensionField<Tower::B1>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: RepackedExtension<PackedType<U, Tower::B1>>,
{
	ids.iter()
		.map(|&id| {
			let exp_witness = witness.get_multilin_poly(id)?;

			let packed_evals = exp_witness
				.packed_evals()
				.expect("poly contain packed_evals");

			let packed_evals = PackedType::<U, Tower::B128>::cast_bases(packed_evals);

			MultilinearExtension::new(exp_witness.n_vars(), packed_evals.to_vec())
				.map(|mle| mle.specialize_arc_dyn())
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, _>>()
}

#[instrument(skip_all, name = "flush::reduce_flush_evalcheck_claims")]
fn reduce_flush_evalcheck_claims<
	U,
	Tower: ProverTowerFamily,
	Challenger_,
	Backend: ComputationBackend,
>(
	claims: Vec<EvalcheckMultilinearClaim<FExt<Tower>>>,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness_index: MultilinearExtensionIndex<
		PackedType<ByteSlicedUnderlier<U, 16>, AESTowerField128b>,
	>,
	domain_factory: IsomorphicEvaluationDomainFactory<AESTowerField8b>,
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<Vec<EvalcheckMultilinearClaim<FExt<Tower>>>, Error>
where
	FExt<Tower>: From<AESTowerField128b>,
	AESTowerField128b: From<FExt<Tower>>,
	ByteSlicedUnderlier<U, 16>:
		PackScalar<AESTowerField128b, Packed: Pod> + PackScalar<AESTowerField8b>,
	U: ProverTowerUnderlier<Tower>,
	Challenger_: Challenger + Default,
{
	let mut linear_claims = Vec::new();

	#[allow(clippy::type_complexity)]
	let mut new_mlechecks_constraints: Vec<(
		EvalPoint<AESTowerField128b>,
		ConstraintSetBuilder<AESTowerField128b>,
	)> = Vec::new();

	for claim in &claims {
		match &oracles[claim.id].variant {
			MultilinearPolyVariant::LinearCombination(_) => linear_claims.push(claim.clone()),
			MultilinearPolyVariant::Composite(composite) => {
				let eval_point = claim.eval_point.isomorphic();

				let eval = claim.eval.into();

				let position = new_mlechecks_constraints
					.iter()
					.position(|(ep, _)| *ep == eval_point)
					.unwrap_or(new_mlechecks_constraints.len());

				let oracle_ids = composite.inner().clone();

				let exp = <_ as CompositionPoly<FExt<Tower>>>::expression(composite.c());
				let fast_exp = exp.convert_field::<AESTowerField128b>();

				if let Some((_, constraint_builder)) = new_mlechecks_constraints.get_mut(position) {
					constraint_builder.add_sumcheck(oracle_ids, fast_exp, eval);
				} else {
					let mut new_builder = ConstraintSetBuilder::new();
					new_builder.add_sumcheck(oracle_ids, fast_exp, eval);
					new_mlechecks_constraints.push((eval_point.clone(), new_builder));
				}
			}
			_ => unreachable!(),
		}
	}

	let new_mlechecks = new_mlechecks_constraints
		.into_iter()
		.map(|(ep, builder)| {
			builder
				.build_one(oracles)
				.map(|constraint| ConstraintSetEqIndPoint {
					eq_ind_challenges: ep.clone(),
					constraint_set: constraint,
				})
				.map_err(Error::from)
		})
		.collect::<Result<Vec<_>, Error>>()?;

	let mut memoized_data = MemoizedData::new();

	let mut fast_new_evalcheck_claims = Vec::new();

	for ConstraintSetEqIndPoint {
		eq_ind_challenges,
		constraint_set,
	} in new_mlechecks
	{
		let evalcheck_claims = prove_mlecheck_with_switchover::<_, _, AESTowerField8b, _, _>(
			&witness_index,
			constraint_set,
			eq_ind_challenges,
			&mut memoized_data,
			transcript,
			immediate_switchover_heuristic,
			domain_factory.clone(),
			backend,
		)?;
		fast_new_evalcheck_claims.extend(evalcheck_claims);
	}

	Ok(chain!(
		fast_new_evalcheck_claims
			.into_iter()
			.map(|claim| claim.isomorphic::<FExt<Tower>>()),
		linear_claims.into_iter()
	)
	.collect::<Vec<_>>())
}
