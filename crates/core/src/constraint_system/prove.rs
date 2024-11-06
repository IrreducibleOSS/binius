// Copyright 2024 Irreducible Inc.

use super::{error::Error, verify::make_standard_pcss, ConstraintSystem, Proof, ProofGenericPCS};
use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	constraint_system::common::{TowerPCS, TowerPCSFamily},
	merkle_tree::MerkleCap,
	oracle::{CommittedBatch, CommittedId, MultilinearOracleSet, MultilinearPolyOracle},
	poly_commit::PolyCommitScheme,
	protocols::{
		greedy_evalcheck,
		greedy_evalcheck::GreedyEvalcheckProveOutput,
		sumcheck,
		sumcheck::{constraint_set_zerocheck_claim, standard_switchover_heuristic, zerocheck},
	},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, PackedField, PackedFieldIndexable, RepackedExtension, TowerField,
};
use binius_hal::ComputationBackend;
use binius_hash::Hasher;
use binius_math::EvaluationDomainFactory;
use binius_utils::bail;
use itertools::izip;
use p3_symmetric::PseudoCompressionFunction;
use std::cmp::Reverse;
use tracing::instrument;

/// Generates a proof that a witness satisfies a constraint system with the standard FRI PCS.
#[instrument("constraint_system::prove", skip_all, level = "debug")]
pub fn prove<U, Tower, Digest, DomainFactory, Hash, Compress, Challenger, Backend>(
	constraint_system: &ConstraintSystem<PackedType<U, Tower::B128>>,
	log_inv_rate: usize,
	security_bits: usize,
	witness: MultilinearExtensionIndex<U, Tower::B128>,
	domain_factory: DomainFactory,
	challenger: Challenger,
	backend: &Backend,
) -> Result<Proof<Tower::B128, Digest, Hash, Compress>, Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger: CanObserve<MerkleCap<Digest>>
		+ CanObserve<Tower::B128>
		+ CanSample<Tower::B128>
		+ CanSampleBits<usize>,
	Backend: ComputationBackend,
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
{
	let pcss = make_standard_pcss::<U, Tower, _, _, _, _>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory.clone(),
	)?;
	prove_with_pcs::<U, Tower, Tower::B8, _, _, _, _>(
		constraint_system,
		witness,
		&pcss,
		domain_factory,
		challenger,
		backend,
	)
}

/// Generates a proof that a witness satisfies a constraint system with provided PCSs.
#[allow(clippy::type_complexity)]
fn prove_with_pcs<U, Tower, FDomain, PCSFamily, DomainFactory, Challenger, Backend>(
	constraint_system: &ConstraintSystem<PackedType<U, Tower::B128>>,
	mut witness: MultilinearExtensionIndex<U, Tower::B128>,
	pcss: &[TowerPCS<Tower, U, PCSFamily>],
	domain_factory: DomainFactory,
	mut challenger: Challenger,
	backend: &Backend,
) -> Result<ProofGenericPCS<Tower::B128, PCSFamily::Commitment, PCSFamily::Proof>, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<FDomain>,
	Tower: TowerFamily,
	Tower::B128: ExtensionField<FDomain>,
	FDomain: TowerField,
	PCSFamily: TowerPCSFamily<Tower, U>,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Challenger: CanObserve<PCSFamily::Commitment>
		+ CanObserve<Tower::B128>
		+ CanSample<Tower::B128>
		+ CanSampleBits<usize>,
	Backend: ComputationBackend,
	PackedType<U, Tower::B128>: PackedTop<Tower>
		+ PackedFieldIndexable
		// Required for ZerocheckProver
		+ RepackedExtension<PackedType<U, Tower::B128>>,
{
	let ConstraintSystem {
		mut oracles,
		mut table_constraints,
		max_channel_id,
		..
	} = constraint_system.clone();

	if max_channel_id != 0 {
		todo!("multiset matching using grand-product argument");
	}

	// Stable sort constraint sets in descending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| Reverse(constraint_set.n_vars));

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
	challenger.observe_slice(&commitments);

	// Zerocheck
	let (zerocheck_claims, zerocheck_oracle_metas) = table_constraints
		.iter()
		.cloned()
		.map(constraint_set_zerocheck_claim)
		.collect::<Result<Vec<_>, _>>()?
		.into_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	let n_zerocheck_challenges = zerocheck_claims
		.iter()
		.map(|claim| claim.n_vars())
		.max()
		.unwrap_or(0);
	let zerocheck_challenges = challenger.sample_vec(n_zerocheck_challenges);

	let switchover_fn = standard_switchover_heuristic(-2);
	let provers = table_constraints
		.into_iter()
		.map(|constraint_set| {
			let skip_rounds = n_zerocheck_challenges - constraint_set.n_vars;
			sumcheck::prove::constraint_set_zerocheck_prover::<U, Tower::B128, Tower::B128, _, _>(
				constraint_set.clone(),
				constraint_set,
				&witness,
				&domain_factory,
				switchover_fn,
				&zerocheck_challenges[skip_rounds..],
				backend,
			)?
			.into_regular_zerocheck()
		})
		.collect::<Result<Vec<_>, _>>()?;

	let (sumcheck_output, zerocheck_proof) =
		sumcheck::prove::batch_prove(provers, &mut challenger)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_claims =
		sumcheck::make_eval_claims(&oracles, zerocheck_oracle_metas, zerocheck_output)?;

	// Prove evaluation claims
	let GreedyEvalcheckProveOutput {
		same_query_claims: mut pcs_claims,
		proof: greedy_evalcheck_proof,
	} = greedy_evalcheck::prove(
		&mut oracles,
		&mut witness,
		evalcheck_claims,
		switchover_fn,
		&mut challenger,
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
	let pcs_proofs = izip!(pcs_claims, pcss, batches, committeds)
		.map(|((_batch_id, claim), pcs, batch, committed)| match pcs {
			TowerPCS::B1(pcs) => tower_pcs_open::<_, Tower::B1, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut challenger,
				backend,
			),
			TowerPCS::B8(pcs) => tower_pcs_open::<_, Tower::B8, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut challenger,
				backend,
			),
			TowerPCS::B16(pcs) => tower_pcs_open::<_, Tower::B16, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut challenger,
				backend,
			),
			TowerPCS::B32(pcs) => tower_pcs_open::<_, Tower::B32, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut challenger,
				backend,
			),
			TowerPCS::B64(pcs) => tower_pcs_open::<_, Tower::B64, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut challenger,
				backend,
			),
			TowerPCS::B128(pcs) => tower_pcs_open::<_, Tower::B128, _, _, _, _>(
				pcs,
				batch,
				&oracles,
				&witness,
				committed,
				&claim.eval_point,
				&mut challenger,
				backend,
			),
		})
		.collect::<Result<Vec<_>, _>>()?;

	Ok(ProofGenericPCS {
		commitments,
		zerocheck_proof,
		greedy_evalcheck_proof,
		pcs_proofs,
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
fn tower_pcs_open<U, F, FExt, PCS, Challenger, Backend>(
	pcs: &PCS,
	batch: CommittedBatch,
	oracles: &MultilinearOracleSet<FExt>,
	witness: &MultilinearExtensionIndex<U, FExt>,
	committed: PCS::Committed,
	eval_point: &[FExt],
	mut challenger: Challenger,
	backend: &Backend,
) -> Result<PCS::Proof, Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FExt>,
	F: TowerField,
	FExt: TowerField + ExtensionField<F>,
	PCS: PolyCommitScheme<PackedType<U, F>, FExt>,
	Challenger:
		CanObserve<PCS::Commitment> + CanObserve<FExt> + CanSample<FExt> + CanSampleBits<usize>,
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
	pcs.prove_evaluation(&mut challenger, &committed, &mles, eval_point, backend)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))
}
