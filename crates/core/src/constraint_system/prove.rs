// Copyright 2024 Irreducible Inc.

use super::{
	error::Error,
	verify::{
		get_flush_sumcheck_composite_sum_claims, get_post_flush_sumcheck_eval_claims,
		make_flush_oracles, make_standard_pcss, max_n_vars_and_skip_rounds,
		reorder_for_flushing_by_n_vars, FlushSumcheckComposition,
	},
	ConstraintSystem, Proof,
};
use crate::{
	constraint_system::common::{standard_pcs, FExt, TowerPCS, TowerPCSFamily},
	fiat_shamir::{CanSample, CanSampleBits, Challenger},
	oracle::{CommittedBatch, CommittedId, MultilinearOracleSet, MultilinearPolyOracle, OracleId},
	poly_commit::PolyCommitScheme,
	protocols::{
		gkr_gpa::{self, GrandProductBatchProveOutput, GrandProductWitness, LayerClaim},
		greedy_evalcheck::{self, GreedyEvalcheckProveOutput},
		sumcheck::{
			self, constraint_set_zerocheck_claim,
			prove::{RegularSumcheckProver, UnivariateZerocheckProver},
			standard_switchover_heuristic, zerocheck,
		},
	},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceWriter, CanWrite, TranscriptWriter},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField, ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable,
	RepackedExtension, TowerField,
};
use binius_hal::ComputationBackend;
use binius_hash::Hasher;
use binius_math::{
	EvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
};
use binius_utils::bail;
use itertools::izip;
use p3_symmetric::PseudoCompressionFunction;
use rayon::prelude::*;
use std::{cmp::Reverse, env, fmt::Debug};
use tracing::instrument;

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<U, Tower, FBase, Digest, DomainFactory, Hash, Compress, Challenger_, Backend>(
	constraint_system: &ConstraintSystem<Tower::B128>,
	log_inv_rate: usize,
	security_bits: usize,
	witness: MultilinearExtensionIndex<U, Tower::B128>,
	domain_factory: DomainFactory,
	backend: &Backend,
) -> Result<Proof, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FBase>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower> + ExtensionField<FBase>,
	FBase: TowerField + ExtensionField<Tower::B8> + TryFrom<Tower::B128>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField<Scalar: TowerField>,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	Backend: ComputationBackend + Debug,
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
	PackedType<U, FBase>:
		PackedFieldIndexable + PackedExtension<Tower::B8, PackedSubfield: PackedFieldIndexable>,
{
	tracing::debug!(
		arch = env::consts::ARCH,
		rayon_threads = rayon::current_num_threads(),
		"using computation backend: {backend:?}"
	);

	let pcss = make_standard_pcss::<U, Tower, _, _, Hash, Compress>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory.clone(),
	)?;
	prove_with_pcs::<U, Tower, FBase, Tower::B8, _, _, Challenger_, Digest, _>(
		constraint_system,
		witness,
		&pcss,
		domain_factory,
		backend,
	)
}

/// Generates a proof that a witness satisfies a constraint system with provided PCSs.
#[allow(clippy::type_complexity)]
#[instrument(skip_all, level = "debug")]
fn prove_with_pcs<
	U,
	Tower,
	FBase,
	FDomain,
	PCSFamily,
	DomainFactory,
	Challenger_,
	Digest,
	Backend,
>(
	constraint_system: &ConstraintSystem<Tower::B128>,
	mut witness: MultilinearExtensionIndex<U, Tower::B128>,
	pcss: &[TowerPCS<Tower, U, PCSFamily>],
	domain_factory: DomainFactory,
	backend: &Backend,
) -> Result<Proof, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FDomain> + PackScalar<FBase>,
	Tower: TowerFamily,
	Tower::B128: ExtensionField<FBase> + ExtensionField<FDomain>,
	FBase: TowerField + ExtensionField<FDomain> + TryFrom<Tower::B128>,
	FDomain: TowerField,
	PCSFamily: TowerPCSFamily<Tower, U, Commitment = Digest>,
	PCSFamily::Committed: Send,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Challenger_: Challenger + Default,
	Digest: PackedField<Scalar: TowerField>,
	Backend: ComputationBackend,
	PackedType<U, Tower::B128>: PackedTop<Tower>
		+ PackedFieldIndexable
		// Required for ZerocheckProver
		+ RepackedExtension<PackedType<U, Tower::B128>>,
	PackedType<U, FBase>:
		PackedFieldIndexable + PackedExtension<FDomain, PackedSubfield: PackedFieldIndexable>,
{
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

	// Commit polynomials in parallel.
	let (commitments, committeds) = constraint_system
		.oracles
		.committed_batches()
		.into_par_iter()
		.zip(pcss.into_par_iter())
		.map(|(batch, pcs)| match pcs {
			TowerPCS::B1(pcs) => {
				tower_pcs_commit::<_, Tower::B1, _, _>(pcs, batch, &oracles, &witness)
			}
			TowerPCS::B8(pcs) => {
				tower_pcs_commit::<_, Tower::B8, _, _>(pcs, batch, &oracles, &witness)
			}
			TowerPCS::B16(pcs) => {
				tower_pcs_commit::<_, Tower::B16, _, _>(pcs, batch, &oracles, &witness)
			}
			TowerPCS::B32(pcs) => {
				tower_pcs_commit::<_, Tower::B32, _, _>(pcs, batch, &oracles, &witness)
			}
			TowerPCS::B64(pcs) => {
				tower_pcs_commit::<_, Tower::B64, _, _>(pcs, batch, &oracles, &witness)
			}
			TowerPCS::B128(pcs) => {
				tower_pcs_commit::<_, Tower::B128, _, _>(pcs, batch, &oracles, &witness)
			}
		})
		.collect::<Result<Vec<_>, _>>()?
		.into_par_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	// Observe polynomial commitments
	transcript.write_packed_slice(&commitments);

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
	} = gkr_gpa::batch_prove(
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

	let (flush_sumcheck_provers, flush_transparents_ids) = get_flush_sumcheck_provers(
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

	let flush_eval_claims = get_post_flush_sumcheck_eval_claims(
		&oracles,
		&flush_oracle_ids,
		&flush_transparents_ids,
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
		max_n_vars_and_skip_rounds(&zerocheck_claims, standard_pcs::FDomain::<Tower>::N_BITS);

	let zerocheck_challenges = transcript.sample_vec(max_n_vars - skip_rounds);

	let switchover_fn = standard_switchover_heuristic(-2);
	let mut univariate_provers = table_constraints
		.into_iter()
		.map(|constraint_set| {
			let skip_challenges = (max_n_vars - constraint_set.n_vars).saturating_sub(skip_rounds);
			sumcheck::prove::constraint_set_zerocheck_prover::<U, FBase, Tower::B128, FDomain, _>(
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

		let reduction_prover = sumcheck::prove::univariatizing_reduction_prover(
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
	let GreedyEvalcheckProveOutput {
		same_query_claims: mut pcs_claims,
	} = greedy_evalcheck::prove(
		&mut oracles,
		&mut witness,
		[non_zero_prodcheck_eval_claims, flush_eval_claims]
			.concat()
			.into_iter()
			.chain(zerocheck_eval_claims),
		switchover_fn,
		&mut transcript,
		&mut advice,
		domain_factory,
		backend,
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
	let batches = constraint_system.oracles.committed_batches();
	for ((_batch_id, claim), pcs, batch, committed) in izip!(pcs_claims, pcss, batches, committeds)
	{
		match pcs {
			TowerPCS::B1(pcs) => tower_pcs_open::<_, Tower::B1, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut advice,
				&mut transcript,
				backend,
			),
			TowerPCS::B8(pcs) => tower_pcs_open::<_, Tower::B8, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut advice,
				&mut transcript,
				backend,
			),
			TowerPCS::B16(pcs) => tower_pcs_open::<_, Tower::B16, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut advice,
				&mut transcript,
				backend,
			),
			TowerPCS::B32(pcs) => tower_pcs_open::<_, Tower::B32, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut advice,
				&mut transcript,
				backend,
			),
			TowerPCS::B64(pcs) => tower_pcs_open::<_, Tower::B64, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut advice,
				&mut transcript,
				backend,
			),
			TowerPCS::B128(pcs) => tower_pcs_open::<_, Tower::B128, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut advice,
				&mut transcript,
				backend,
			),
		}?
	}

	Ok(Proof {
		transcript: transcript.finalize(),
		advice: advice.finalize(),
	})
}

fn tower_pcs_commit<U, F, FExt, PCS>(
	pcs: &PCS,
	batch: CommittedBatch,
	oracles: &MultilinearOracleSet<FExt>,
	witness: &MultilinearExtensionIndex<U, FExt>,
) -> Result<(PCS::Commitment, PCS::Committed), Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FExt>,
	F: TowerField,
	FExt: TowerField + ExtensionField<F>,
	PCS: PolyCommitScheme<PackedType<U, F>, FExt>,
{
	// Precondition
	assert_eq!(batch.tower_level, F::TOWER_LEVEL);

	let mles = (0..batch.n_polys)
		.map(|i| {
			let oracle = oracles.committed_oracle(CommittedId {
				batch_id: batch.id,
				index: i,
			});
			let MultilinearPolyOracle::Committed { oracle_id, .. } = oracle else {
				panic!("MultilinearOracleSet::committed_oracle returned a non-committed oracle");
			};
			witness.get::<F>(oracle_id)
		})
		.collect::<Result<Vec<_>, _>>()?;
	pcs.commit(&mles)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))
}

#[allow(clippy::too_many_arguments)]
fn tower_pcs_open<U, F, FExt, PCS, Transcript, Backend>(
	pcs: &PCS,
	batch: CommittedBatch,
	oracles: &MultilinearOracleSet<FExt>,
	witness: &MultilinearExtensionIndex<U, FExt>,
	committed: PCS::Committed,
	eval_point: &[FExt],
	advice: &mut AdviceWriter,
	mut transcript: Transcript,
	backend: &Backend,
) -> Result<(), Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FExt>,
	F: TowerField,
	FExt: TowerField + ExtensionField<F>,
	PCS: PolyCommitScheme<PackedType<U, F>, FExt>,
	Transcript: CanSample<FExt> + CanSampleBits<usize> + CanWrite,
	Backend: ComputationBackend,
{
	// Precondition
	assert_eq!(batch.tower_level, F::TOWER_LEVEL);

	let mles = (0..batch.n_polys)
		.map(|i| {
			let oracle = oracles.committed_oracle(CommittedId {
				batch_id: batch.id,
				index: i,
			});
			let MultilinearPolyOracle::Committed { oracle_id, .. } = oracle else {
				panic!("MultilinearOracleSet::committed_oracle returned a non-committed oracle");
			};
			witness.get::<F>(oracle_id)
		})
		.collect::<Result<Vec<_>, _>>()?;
	pcs.prove_evaluation(advice, &mut transcript, &committed, &mles, eval_point, backend)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))
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

type FlushSumcheckProver<'b, U, Tower, FDomain, Backend> = RegularSumcheckProver<
	'b,
	FDomain,
	PackedType<U, <Tower as TowerFamily>::B128>,
	FlushSumcheckComposition,
	MultilinearWitness<'b, PackedType<U, <Tower as TowerFamily>::B128>>,
	Backend,
>;
type FlushSumcheckProverWithTransparentsIds<'b, U, Tower, FDomain, Backend> =
	(Vec<FlushSumcheckProver<'b, U, Tower, FDomain, Backend>>, Vec<(OracleId, OracleId)>);

#[instrument(skip_all, level = "debug")]
fn get_flush_sumcheck_provers<'a, 'b, U, Tower, FDomain, DomainFactory, Backend>(
	oracles: &mut MultilinearOracleSet<Tower::B128>,
	flush_oracle_ids: &[OracleId],
	flush_counts: &[usize],
	final_layer_claims: &[LayerClaim<Tower::B128>],
	witness: &mut MultilinearExtensionIndex<'a, U, Tower::B128>,
	domain_factory: DomainFactory,
	backend: &'b Backend,
) -> Result<FlushSumcheckProverWithTransparentsIds<'b, U, Tower, FDomain, Backend>, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FDomain>,
	Tower: TowerFamily,
	Tower::B128: ExtensionField<FDomain>,
	FDomain: Field,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Backend: ComputationBackend,
	'a: 'b,
{
	let (composite_sum_claims, step_down_info, eq_ind_info) =
		get_flush_sumcheck_composite_sum_claims::<Tower::B128>(
			oracles,
			flush_oracle_ids,
			flush_counts,
			final_layer_claims,
		)?;

	// The function is on the critical path, parallelize.
	let eq_ind_adapters = eq_ind_info
		.into_par_iter()
		.map(|(eq_ind_id, eq_ind)| -> Result<_, Error> {
			let eq_ind_multilinear_extension = eq_ind.multilinear_extension(backend)?;
			let eq_ind_adapter =
				MLEDirectAdapter::from(eq_ind_multilinear_extension).upcast_arc_dyn();
			Ok((eq_ind_id, eq_ind_adapter))
		})
		.collect::<Result<Vec<_>, Error>>()?;
	let results: Result<Vec<_>, Error> = izip!(
		flush_oracle_ids.iter(),
		composite_sum_claims.into_iter(),
		step_down_info.into_iter(),
		eq_ind_adapters.into_iter()
	)
	.map(
		|(
			&flush_oracle_id,
			composite_sum_claim,
			(step_down_id, step_down),
			(eq_ind_id, eq_ind_adapter),
		)| {
			let step_down_multilinear =
				MLEDirectAdapter::from(step_down.multilinear_extension()?).upcast_arc_dyn();

			witness.update_multilin_poly([
				(step_down_id, step_down_multilinear),
				(eq_ind_id, eq_ind_adapter),
			])?;

			let prover = RegularSumcheckProver::new(
				vec![
					witness.get_multilin_poly(flush_oracle_id)?,
					witness.get_multilin_poly(step_down_id)?,
					witness.get_multilin_poly(eq_ind_id)?,
				],
				[composite_sum_claim],
				domain_factory.clone(),
				standard_switchover_heuristic(0), // what should this be?
				backend,
			)?;

			Ok((prover, (step_down_id, eq_ind_id)))
		},
	)
	.collect();

	Ok(results?.into_iter().unzip())
}
