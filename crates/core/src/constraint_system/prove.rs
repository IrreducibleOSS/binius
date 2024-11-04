// Copyright 2024 Irreducible Inc.

use super::{error::Error, verify::make_standard_pcss, ConstraintSystem, Proof, ProofGenericPCS};
use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	merkle_tree::MerkleCap,
	oracle::CommittedId,
	poly_commit::PolyCommitScheme,
	protocols::{
		greedy_evalcheck,
		greedy_evalcheck::GreedyEvalcheckProveOutput,
		sumcheck,
		sumcheck::{constraint_set_zerocheck_claim, standard_switchover_heuristic, zerocheck},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField, BinaryField1b, ExtensionField, PackedExtension, PackedField, PackedFieldIndexable,
	TowerField,
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
pub fn prove<
	U,
	FExt,
	FDomain,
	FEncode,
	Digest,
	DomainFactory,
	Hash,
	Compress,
	Challenger,
	Backend,
>(
	constraint_system: &ConstraintSystem<PackedType<U, FExt>>,
	log_inv_rate: usize,
	security_bits: usize,
	witness: MultilinearExtensionIndex<U, FExt>,
	domain_factory: DomainFactory,
	challenger: Challenger,
	backend: &Backend,
) -> Result<Proof<FExt, Digest, Hash, Compress>, Error>
where
	U: UnderlierType
		+ PackScalar<BinaryField1b>
		+ PackScalar<FDomain>
		+ PackScalar<FEncode>
		+ PackScalar<FExt>,
	FExt: TowerField
		+ PackedField<Scalar = FExt>
		+ ExtensionField<FDomain>
		+ ExtensionField<FEncode>
		+ PackedExtension<BinaryField1b>
		+ PackedExtension<FEncode>,
	FDomain: TowerField,
	FEncode: BinaryField,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Digest: PackedField,
	Hash: Hasher<FExt, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger:
		CanObserve<MerkleCap<Digest>> + CanObserve<FExt> + CanSample<FExt> + CanSampleBits<usize>,
	Backend: ComputationBackend,
	PackedType<U, FExt>: PackedFieldIndexable,
{
	let pcss = make_standard_pcss::<_, _, FEncode, PackedType<U, FExt>, _, _, _, _>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory.clone(),
	)?;
	prove_with_pcs(constraint_system, witness, &pcss, domain_factory, challenger, backend)
}

/// Generates a proof that a witness satisfies a constraint system with provided PCSs.
fn prove_with_pcs<U, FExt, FDomain, PCS, DomainFactory, Challenger, Backend>(
	constraint_system: &ConstraintSystem<PackedType<U, FExt>>,
	mut witness: MultilinearExtensionIndex<U, FExt>,
	pcss: &[PCS],
	domain_factory: DomainFactory,
	mut challenger: Challenger,
	backend: &Backend,
) -> Result<ProofGenericPCS<FExt, PCS::Commitment, PCS::Proof>, Error>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<FDomain> + PackScalar<FExt>,
	FExt: TowerField + ExtensionField<FDomain>,
	FDomain: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, FExt>,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Challenger:
		CanObserve<PCS::Commitment> + CanObserve<FExt> + CanSample<FExt> + CanSampleBits<usize>,
	Backend: ComputationBackend,
	PackedType<U, FExt>: PackedFieldIndexable,
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
	let batch_mles = constraint_system
		.oracles
		.committed_batches()
		.into_iter()
		.map(|batch| {
			(0..batch.n_polys)
				.map(|i| {
					let oracle_id = oracles.committed_oracle_id(CommittedId {
						batch_id: batch.id,
						index: i,
					});
					witness.get::<BinaryField1b>(oracle_id)
				})
				.collect::<Result<Vec<_>, _>>()
		})
		.collect::<Result<Vec<_>, _>>()?;

	let (commitments, committeds) = batch_mles
		.iter()
		.zip(pcss)
		.map(|(mles, pcs)| {
			pcs.commit(mles)
				.map_err(|err| Error::PolyCommitError(Box::new(err)))
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
			sumcheck::prove::constraint_set_zerocheck_prover::<_, FExt, FExt, _, _>(
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
	let batch_mles = constraint_system
		.oracles
		.committed_batches()
		.into_iter()
		.map(|batch| {
			(0..batch.n_polys)
				.map(|i| {
					let oracle_id = oracles.committed_oracle_id(CommittedId {
						batch_id: batch.id,
						index: i,
					});
					witness.get::<BinaryField1b>(oracle_id)
				})
				.collect::<Result<Vec<_>, _>>()
		})
		.collect::<Result<Vec<_>, _>>()?;

	let pcs_proofs = izip!(pcs_claims, pcss, batch_mles, committeds)
		.map(|((_batch_id, claim), pcs, mles, committed)| {
			pcs.prove_evaluation(&mut challenger, &committed, &mles, &claim.eval_point, backend)
				.map_err(|err| Error::PolyCommitError(Box::new(err)))
		})
		.collect::<Result<Vec<_>, _>>()?;

	Ok(ProofGenericPCS {
		commitments,
		zerocheck_proof,
		greedy_evalcheck_proof,
		pcs_proofs,
	})
}
