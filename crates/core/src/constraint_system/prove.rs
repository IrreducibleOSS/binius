// Copyright 2024-2025 Irreducible Inc.

use std::{env, iter, marker::PhantomData};

use binius_compute::{ComputeLayer, ComputeMemory, FSliceMut, cpu::CpuMemory};
use binius_field::{
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
	as_packed_field::PackedType,
	linear_transformation::{PackedTransformationFactory, Transformation},
	tower::{PackedTop, ProverTowerFamily, ProverTowerUnderlier},
	underlier::WithUnderlier,
	util::powers,
};
use binius_hal::ComputationBackend;
use binius_hash::{PseudoCompressionFunction, multi_digest::ParallelDigest};
use binius_math::{
	DefaultEvaluationDomainFactory, EvaluationDomainFactory, EvaluationOrder,
	IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_maybe_rayon::prelude::*;
use binius_ntt::SingleThreadedNTT;
use binius_utils::bail;
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
		Flush,
		channel::OracleOrConst,
		common::{FDomain, FEncode, FExt, FFastExt},
		exp::{self, reorder_exponents},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeProver,
	oracle::{Constraint, MultilinearOracleSet, OracleId, SizedConstraintSet},
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
	witness::{IndexEntry, MultilinearExtensionIndex, MultilinearWitness},
};

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[allow(clippy::too_many_arguments)]
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<Hal, U, Tower, Hash, Compress, Challenger_, Backend>(
	hal: &Hal,
	host_mem: <CpuMemory as ComputeMemory<Tower::B128>>::FSliceMut<'_>,
	dev_mem: FSliceMut<'_, Tower::B128, Hal>,
	constraint_system: &ConstraintSystem<FExt<Tower>>,
	log_inv_rate: usize,
	security_bits: usize,
	boundaries: &[Boundary<FExt<Tower>>],
	mut witness: MultilinearExtensionIndex<PackedType<U, FExt<Tower>>>,
	backend: &Backend,
) -> Result<Proof, Error>
where
	Hal: ComputeLayer<Tower::B128> + Default,
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	Tower::B128: binius_math::TowerTop + binius_math::PackedTop + PackedTop<Tower>,
	Hash: ParallelDigest,
	Hash::Digest: BlockSizeUser + FixedOutputReset + Send + Sync + Clone,
	Compress: PseudoCompressionFunction<Output<Hash::Digest>, 2> + Default + Sync,
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
		+ PackedTransformationFactory<PackedType<U, Tower::FastB128>>
		+ binius_math::PackedTop,
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
		table_constraints,
		mut flushes,
		mut exponents,
		non_zero_oracle_ids,
		channel_count,
	} = constraint_system.clone();

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
		.map(|u| {
			// Pick the first oracle and get its n_vars.
			//
			// TODO(pep): I know that this invariant is not guaranteed to hold at this point, but
			//            this is fine and is going away in a follow up where we read the sizes of
			//            tables from the transcript or pass it in the prover.
			let first_oracle_id = u.oracle_ids[0];
			let n_vars = oracles.n_vars(first_oracle_id);
			SizedConstraintSet::new(n_vars, u)
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
		fast_domain_factory.clone(),
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
		convert_witnesses_to_fast_ext::<U, _>(&oracles, &witness, &non_zero_oracle_ids)?;
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

	flushes.sort_by_key(|flush| flush.channel_id);
	let flush_oracle_ids =
		make_flush_oracles(&mut oracles, &flushes, mixing_challenge, &permutation_challenges)?;

	let flush_convert_span = tracing::info_span!(
		"[task] Convert Flushes to Fast Field",
		phase = "prodcheck",
		perfetto_category = "task.main"
	)
	.entered();
	make_masked_flush_witnesses::<U, _>(
		&oracles,
		&mut witness,
		&flush_oracle_ids,
		&flushes,
		mixing_challenge,
		&permutation_challenges,
	)?;

	// there are no oracle ids associated with these flush_witnesses
	let flush_witnesses =
		convert_witnesses_to_fast_ext::<U, _>(&oracles, &witness, &flush_oracle_ids)?;
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
	} = ring_switch::prove(&system, &committed_multilins, &mut transcript, memoized_data)?;
	drop(ring_switch_span);

	// Prove evaluation claims using PIOP compiler
	let piop_compiler_span = tracing::info_span!(
		"[phase] PIOP Compiler",
		phase = "piop_compiler",
		perfetto_category = "phase.main"
	)
	.entered();

	piop::prove(
		hal,
		host_mem,
		dev_mem,
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
pub fn make_masked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness_index: &mut MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	flush_oracle_ids: &[OracleId],
	flushes: &[Flush<FExt<Tower>>],
	mixing_challenge: FExt<Tower>,
	permutation_challenges: &[FExt<Tower>],
) -> Result<(), Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
{
	// TODO: Move me out into a separate function & deduplicate.
	// Count the suffix zeros on all selectors.
	for flush in flushes {
		for &selector_id in &flush.selectors {
			let selector = witness_index.get_multilin_poly(selector_id)?;
			let zero_suffix_len = count_zero_suffixes(&selector);

			let nonzero_prefix_len = (1 << selector.n_vars()) - zero_suffix_len;
			witness_index.update_multilin_poly_with_nonzero_scalars_prefixes([(
				selector_id,
				selector,
				nonzero_prefix_len,
			)])?;
		}
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
