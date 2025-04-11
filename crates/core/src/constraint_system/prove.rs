// Copyright 2024-2025 Irreducible Inc.

use std::{env, marker::PhantomData, slice::from_mut};

use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	underlier::WithUnderlier,
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
};
use binius_hal::ComputationBackend;
use binius_hash::PseudoCompressionFunction;
use binius_math::{
	DefaultEvaluationDomainFactory, EvaluationDomainFactory, EvaluationOrder,
	IsomorphicEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use digest::{core_api::BlockSizeUser, Digest, FixedOutputReset, Output};
use itertools::{chain, izip};
use tracing::instrument;

use super::{
	channel::Boundary,
	error::Error,
	verify::{
		get_post_flush_sumcheck_eval_claims_without_eq, make_flush_oracles,
		max_n_vars_and_skip_rounds, reduce_unmasked_flush_eval_claims,
		reorder_for_flushing_by_n_vars,
	},
	ConstraintSystem, Proof,
};
use crate::{
	constraint_system::{
		common::{FDomain, FEncode, FExt, FFastExt},
		exp,
		verify::{make_flush_sumcheck_metas, FlushSumcheckMeta},
	},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::BinaryMerkleTreeProver,
	oracle::{Constraint, MultilinearOracleSet, MultilinearPolyVariant, OracleId},
	piop,
	protocols::{
		fri::CommitOutput,
		gkr_exp,
		gkr_gpa::{self, GrandProductBatchProveOutput, GrandProductWitness, LayerClaim},
		greedy_evalcheck::{self, GreedyEvalcheckProveOutput},
		sumcheck::{
			self, constraint_set_zerocheck_claim,
			prove::{eq_ind::EqIndSumcheckProverBuilder, SumcheckProver, ZerocheckProver},
			standard_switchover_heuristic,
		},
	},
	ring_switch,
	tower::{PackedTop, ProverTowerFamily, ProverTowerUnderlier},
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

	exponents.sort_by_key(|b| std::cmp::Reverse(b.n_vars(&oracles)));

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
	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = piop::commit(&fri_params, &merkle_prover, &committed_multilins)?;

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
	let non_zero_fast_witnesses = make_fast_masked_flush_witnesses::<U, _>(
		&oracles,
		&witness,
		&non_zero_oracle_ids,
		&vec![None; non_zero_oracle_ids.len()],
	)?;
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
	let flush_selectors = flushes
		.iter()
		.map(|flush| flush.selector)
		.collect::<Vec<_>>();

	make_unmasked_flush_witnesses::<U, _>(&oracles, &mut witness, &flush_oracle_ids)?;
	// there are no oracle ids associated with these flush_witnesses
	let flush_witnesses = make_fast_masked_flush_witnesses::<U, _>(
		&oracles,
		&witness,
		&flush_oracle_ids,
		&flush_selectors,
	)?;

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
			EvaluationOrder::LowToHigh,
			all_gpa_witnesses,
			&all_gpa_claims,
			&fast_domain_factory,
			&mut transcript,
			backend,
		)?;

	// Apply isomorphism to the layer claims
	let mut final_layer_claims = final_layer_claims
		.into_iter()
		.map(|layer_claim| layer_claim.isomorphic())
		.collect::<Vec<_>>();

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
			&flush_selectors,
			flush_final_layer_claims,
		);

	let unmasked_flush_eval_claims = reduce_unmasked_flush_eval_claims(
		&flush_oracle_ids,
		&flush_selectors,
		&flush_final_layer_claims,
	);

	let FlushSumcheckProvers {
		provers,
		flush_selectors_unique_by_claim,
		flush_oracle_ids_by_claim,
	} = get_flush_sumcheck_provers::<U, _, FDomain<Tower>, _, _>(
		&mut oracles,
		&flush_oracle_ids,
		&flush_selectors,
		&flush_final_layer_claims,
		&mut witness,
		&domain_factory,
		backend,
	)?;

	let flush_sumcheck_output = sumcheck::prove::batch_prove(provers, &mut transcript)?;

	let flush_eval_claims = get_post_flush_sumcheck_eval_claims_without_eq(
		&oracles,
		&flush_selectors_unique_by_claim,
		&flush_oracle_ids_by_claim,
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

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		eval_claims,
		memoized_data,
	} = greedy_evalcheck::prove::<_, _, FDomain<Tower>, _, _>(
		&mut oracles,
		&mut witness,
		chain!(
			non_zero_prodcheck_eval_claims,
			unmasked_flush_eval_claims,
			flush_eval_claims,
			zerocheck_eval_claims,
			exp_eval_claims,
		),
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

	// Prove evaluation claims using PIOP compiler
	piop::prove::<_, FDomain<Tower>, _, _, _, _, _, _, _, _>(
		&fri_params,
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

	Ok(Proof {
		transcript: transcript.finalize(),
	})
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
fn make_unmasked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &mut MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	flush_oracle_ids: &[OracleId],
) -> Result<(), Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
{
	// The function is on the critical path, parallelize.
	let flush_witnesses: Result<Vec<MultilinearWitness<'a, _>>, Error> = flush_oracle_ids
		.par_iter()
		.map(|&oracle_id| {
			let MultilinearPolyVariant::LinearCombination(lincom) =
				oracles.oracle(oracle_id).variant
			else {
				unreachable!("make_flush_oracles adds linear combination oracles");
			};
			let polys = lincom
				.polys()
				.map(|id| witness.get_multilin_poly(id))
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
fn make_fast_masked_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &MultilinearExtensionIndex<'a, PackedType<U, FExt<Tower>>>,
	flush_oracles: &[OracleId],
	flush_selectors: &[Option<OracleId>],
) -> Result<Vec<(usize, Vec<PackedType<U, FFastExt<Tower>>>)>, Error>
where
	U: ProverTowerUnderlier<Tower>,
	Tower: ProverTowerFamily,
	PackedType<U, Tower::B128>: PackedTransformationFactory<PackedType<U, Tower::FastB128>>,
{
	let to_fast = Tower::packed_transformation_to_fast();

	// The function is on the critical path, parallelize.
	(flush_oracles, flush_selectors)
		.into_par_iter()
		.map(|(&flush_oracle_id, &flush_selector)| {
			let n_vars = oracles.n_vars(flush_oracle_id);

			let log_width = <PackedType<U, FFastExt<Tower>>>::LOG_WIDTH;
			let width = 1 << log_width;

			let poly = witness.get_multilin_poly(flush_oracle_id)?;
			let selector_index_entry = flush_selector
				.map(|flush_selector| witness.get_index_entry(flush_selector))
				.transpose()?;

			const MAX_SUBCUBE_VARS: usize = 8;
			let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars);
			let subcube_packed_size = 1 << subcube_vars.saturating_sub(log_width);
			let non_const_scalars = selector_index_entry
				.as_ref()
				.map_or(1 << n_vars, |entry| entry.nonzero_scalars_prefix);
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

					if let Some(selector_index_entry) = selector_index_entry.as_ref() {
						let fast_subcube =
							PackedType::<U, FFastExt<Tower>>::from_underliers_ref_mut(underliers);

						let mut ones_mask = PackedType::<U, FExt<Tower>>::default();
						for (i, packed) in fast_subcube.iter_mut().enumerate() {
							selector_index_entry
								.multilin_poly
								.subcube_evals(
									log_width,
									(subcube_index << subcube_vars.saturating_sub(log_width)) | i,
									0,
									from_mut(&mut ones_mask),
								)
								.expect("selector n_vars equals flushed n_vars");

							if ones_mask == PackedField::zero() {
								*packed = PackedField::one();
							} else if ones_mask != PackedField::one() {
								for j in 0..width {
									if ones_mask.get(j) == FExt::<Tower>::ZERO {
										packed.set(j, FFastExt::<Tower>::ONE);
									}
								}
							}
						}
					}
				});

			fast_ext_result.truncate(non_const_scalars);
			Ok((n_vars, fast_ext_result))
		})
		.collect()
}

pub struct FlushSumcheckProvers<Prover> {
	provers: Vec<Prover>,
	flush_oracle_ids_by_claim: Vec<Vec<OracleId>>,
	flush_selectors_unique_by_claim: Vec<Vec<OracleId>>,
}

#[instrument(skip_all, level = "debug")]
fn get_flush_sumcheck_provers<'a, 'b, U, Tower, FDomain, DomainFactory, Backend>(
	oracles: &mut MultilinearOracleSet<Tower::B128>,
	flush_oracle_ids: &[OracleId],
	flush_selectors: &[Option<OracleId>],
	final_layer_claims: &[LayerClaim<Tower::B128>],
	witness: &mut MultilinearExtensionIndex<'a, PackedType<U, Tower::B128>>,
	domain_factory: DomainFactory,
	backend: &'b Backend,
) -> Result<FlushSumcheckProvers<impl SumcheckProver<Tower::B128> + 'b>, Error>
where
	U: ProverTowerUnderlier<Tower> + PackScalar<FDomain>,
	Tower: ProverTowerFamily,
	Tower::B128: ExtensionField<FDomain>,
	FDomain: Field,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Backend: ComputationBackend,
	'a: 'b,
{
	let flush_sumcheck_metas =
		make_flush_sumcheck_metas(oracles, flush_oracle_ids, flush_selectors, final_layer_claims)?;

	let n_claims = flush_sumcheck_metas.len();
	let mut provers = Vec::with_capacity(n_claims);
	let mut flush_oracle_ids_by_claim = Vec::with_capacity(n_claims);
	let mut flush_selectors_unique_by_claim = Vec::with_capacity(n_claims);
	for flush_sumcheck_meta in flush_sumcheck_metas {
		let FlushSumcheckMeta {
			composite_sum_claims,
			flush_selectors_unique,
			flush_oracle_ids,
			eval_point,
		} = flush_sumcheck_meta;

		let mut multilinears =
			Vec::with_capacity(flush_selectors_unique.len() + flush_oracle_ids.len());

		let mut const_suffixes = Vec::with_capacity(multilinears.len());

		for &oracle_id in chain!(&flush_selectors_unique, &flush_oracle_ids) {
			let entry = witness.get_index_entry(oracle_id)?;
			let suffix_len = (1 << entry.multilin_poly.n_vars()) - entry.nonzero_scalars_prefix;
			multilinears.push(entry.multilin_poly);
			const_suffixes.push((Field::ZERO, suffix_len));
		}

		// REVIEW: we extract a type erased multilin from the witness index here,
		//         but we can do better and move the large-field evals (potentially truncated)
		//         directly into this sumcheck, as those are not shared
		let prover = EqIndSumcheckProverBuilder::with_switchover(
			multilinears,
			standard_switchover_heuristic(-2),
			backend,
		)?
		.with_const_suffixes(&const_suffixes)?
		.build(
			EvaluationOrder::LowToHigh,
			&eval_point,
			composite_sum_claims,
			domain_factory.clone(),
		)?;

		provers.push(prover);
		flush_oracle_ids_by_claim.push(flush_oracle_ids);
		flush_selectors_unique_by_claim.push(flush_selectors_unique);
	}

	Ok(FlushSumcheckProvers {
		provers,
		flush_selectors_unique_by_claim,
		flush_oracle_ids_by_claim,
	})
}
