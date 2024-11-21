// Copyright 2024 Irreducible Inc.

use super::{
	error::Error,
	verify::{make_flush_oracles, make_standard_pcss, max_n_vars_and_skip_rounds},
	ConstraintSystem, Proof, ProofGenericPCS,
};
use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	constraint_system::common::{standard_pcs, FExt, TowerPCS, TowerPCSFamily},
	fiat_shamir::Challenger,
	oracle::{CommittedBatch, CommittedId, MultilinearOracleSet, MultilinearPolyOracle, OracleId},
	poly_commit::PolyCommitScheme,
	protocols::{
		gkr_gpa::{self, GrandProductBatchProveOutput, GrandProductWitness},
		greedy_evalcheck::{self, GreedyEvalcheckProveOutput},
		sumcheck::{
			self, constraint_set_zerocheck_claim, prove::UnivariateZerocheckProver,
			standard_switchover_heuristic, zerocheck,
		},
	},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceWriter, CanWrite, TranscriptWriter},
	witness::MultilinearExtensionIndex,
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
use std::cmp::Reverse;
use tracing::instrument;

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<U, Tower, FBase, Digest, DomainFactory, Hash, Compress, Challenger_, Backend>(
	constraint_system: &ConstraintSystem<PackedType<U, Tower::B128>, PackedType<U, FBase>>,
	log_inv_rate: usize,
	security_bits: usize,
	witness: MultilinearExtensionIndex<U, Tower::B128>,
	domain_factory: DomainFactory,
	backend: &Backend,
) -> Result<Proof<Tower::B128, Digest>, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FBase>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower> + ExtensionField<FBase>,
	FBase: TowerField + ExtensionField<Tower::B8>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField<Scalar: TowerField>,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	Backend: ComputationBackend,
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
	PackedType<U, FBase>:
		PackedFieldIndexable + PackedExtension<Tower::B8, PackedSubfield: PackedFieldIndexable>,
{
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
	constraint_system: &ConstraintSystem<PackedType<U, Tower::B128>, PackedType<U, FBase>>,
	mut witness: MultilinearExtensionIndex<U, Tower::B128>,
	pcss: &[TowerPCS<Tower, U, PCSFamily>],
	domain_factory: DomainFactory,
	backend: &Backend,
) -> Result<ProofGenericPCS<Tower::B128, PCSFamily::Commitment>, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FDomain> + PackScalar<FBase>,
	Tower: TowerFamily,
	Tower::B128: ExtensionField<FBase> + ExtensionField<FDomain>,
	FBase: TowerField + ExtensionField<FDomain>,
	FDomain: TowerField,
	PCSFamily: TowerPCSFamily<Tower, U, Commitment = Digest>,
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
		mut table_constraints_base,
		mut flushes,
		non_zero_oracle_ids,
		max_channel_id,
	} = constraint_system.clone();

	let non_zero_prodcheck_witnesses =
		gkr_gpa::construct_grand_product_witnesses(&non_zero_oracle_ids, &witness)?;
	let non_zero_products =
		gkr_gpa::get_grand_products_from_witnesses(&non_zero_prodcheck_witnesses);
	let non_zero_prodcheck_claims = gkr_gpa::construct_grand_product_claims(
		&non_zero_oracle_ids,
		&oracles,
		&non_zero_products,
	)?;

	if non_zero_products
		.iter()
		.any(|count| *count == Tower::B128::zero())
	{
		bail!(Error::Zeros);
	}

	transcript.observe_slice(&non_zero_products);

	// Stable sort constraint sets in descending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| Reverse(constraint_set.n_vars));
	table_constraints_base.sort_by_key(|constraint_set_base| Reverse(constraint_set_base.n_vars));

	// Stable sort flushes by channel ID.
	flushes.sort_by_key(|flush| flush.channel_id);

	// Commit polynomials
	let (commitments, committeds) = constraint_system
		.oracles
		.committed_batches()
		.into_iter()
		.zip(pcss)
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
		.into_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	// Observe polynomial commitments
	transcript.observe_slice(&commitments);

	// Channel balancing argument
	let mixing_challenge = transcript.sample();
	let permutation_challenges = transcript.sample_vec(max_channel_id + 1);

	// Grand product arguments
	let flush_oracles =
		make_flush_oracles(&mut oracles, &flushes, mixing_challenge, &permutation_challenges)?;
	let flush_prodcheck_witnesses = make_flush_witnesses(&oracles, &witness, &flush_oracles)?;
	let flush_products = gkr_gpa::get_grand_products_from_witnesses(&flush_prodcheck_witnesses);
	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracles, &oracles, &flush_products)?;

	transcript.observe_slice(&flush_products);

	let prodcheck_witnesses = [flush_prodcheck_witnesses, non_zero_prodcheck_witnesses].concat();
	let prodcheck_claims = [flush_prodcheck_claims, non_zero_prodcheck_claims].concat();

	let GrandProductBatchProveOutput {
		final_layer_claims,
		proof: prodcheck_proof,
	} = gkr_gpa::batch_prove(
		prodcheck_witnesses,
		&prodcheck_claims,
		&domain_factory,
		&mut transcript,
		backend,
	)?;
	let prodcheck_eval_claims = gkr_gpa::make_eval_claims(
		&oracles,
		[flush_oracles, non_zero_oracle_ids].concat(),
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

	let (max_n_vars, skip_rounds) =
		max_n_vars_and_skip_rounds(&zerocheck_claims, standard_pcs::FDomain::<Tower>::N_BITS);

	let zerocheck_challenges = transcript.sample_vec(max_n_vars - skip_rounds);

	let switchover_fn = standard_switchover_heuristic(-2);
	let mut univariate_provers = izip!(table_constraints_base, table_constraints)
		.map(|(constraint_set_base, constraint_set)| {
			let skip_challenges = (max_n_vars - constraint_set.n_vars).saturating_sub(skip_rounds);
			sumcheck::prove::constraint_set_zerocheck_prover::<U, FBase, Tower::B128, FDomain, _>(
				constraint_set_base,
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

	let (univariate_output, zerocheck_univariate_proof) =
		sumcheck::prove::batch_prove_zerocheck_univariate_round(
			univariate_provers,
			skip_rounds,
			&mut transcript,
		)?;

	let univariate_challenge = univariate_output.univariate_challenge;

	let (sumcheck_output, zerocheck_proof) = sumcheck::prove::batch_prove_with_start(
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
		let reduced_multilinears = sumcheck::prove::reduce_to_skipped_projection(
			multilinears,
			&zerocheck_output.challenges,
			backend,
		)?;

		let reduction_claim = sumcheck::univariate::univariatizing_reduction_claim(
			skip_rounds,
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

	let (univariatizing_output, univariatizing_proof) =
		sumcheck::prove::batch_prove(reduction_provers, &mut transcript)?;

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
		proof: greedy_evalcheck_proof,
	} = greedy_evalcheck::prove(
		&mut oracles,
		&mut witness,
		prodcheck_eval_claims
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

	Ok(ProofGenericPCS {
		commitments,
		flush_products,
		non_zero_products,
		prodcheck_proof,
		zerocheck_univariate_proof,
		zerocheck_proof,
		univariatizing_proof,
		greedy_evalcheck_proof,
		transcript: transcript.finalize(),
		advice: advice.finalize(),
	})
}

#[allow(clippy::type_complexity)]
fn make_flush_witnesses<'a, U, Tower>(
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	witness: &MultilinearExtensionIndex<'a, U, FExt<Tower>>,
	flush_oracles: &[OracleId],
) -> Result<Vec<GrandProductWitness<'a, PackedType<U, FExt<Tower>>>>, Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
{
	flush_oracles
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

			let witness =
				GrandProductWitness::new(MLEDirectAdapter::from(lincom_poly).upcast_arc_dyn())?;
			Ok(witness)
		})
		.collect()
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
	Transcript: CanObserve<PCS::Commitment>
		+ CanObserve<FExt>
		+ CanSample<FExt>
		+ CanSampleBits<usize>
		+ CanWrite,
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
