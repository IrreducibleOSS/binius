// Copyright 2024 Irreducible Inc.

use super::{
	error::{Error, VerificationError},
	ConstraintSystem, Proof, ProofGenericPCS,
};
use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	merkle_tree::{MerkleCap, MerkleTreeVCS},
	oracle::MultilinearOracleSet,
	poly_commit::{batch_pcs::BatchPCS, PolyCommitScheme, FRIPCS},
	protocols::{
		greedy_evalcheck,
		sumcheck::{self, constraint_set_zerocheck_claim, zerocheck},
	},
};
use binius_field::{
	BinaryField, BinaryField1b, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::Hasher;
use binius_math::EvaluationDomainFactory;
use binius_ntt::NTTOptions;
use binius_utils::bail;
use itertools::izip;
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_ceil_usize;
use std::cmp::Reverse;
use tracing::instrument;

/// Verifies a proof against a constraint system.
#[instrument("constraint_system::verify", skip_all, level = "debug")]
pub fn verify<FExt, FDomain, FEncode, P, Digest, DomainFactory, Hash, Compress, Challenger>(
	constraint_system: &ConstraintSystem<P>,
	log_inv_rate: usize,
	security_bits: usize,
	domain_factory: DomainFactory,
	proof: Proof<FExt, Digest, Hash, Compress>,
	challenger: Challenger,
) -> Result<(), Error>
where
	FExt: TowerField
		+ PackedField<Scalar = FExt>
		+ ExtensionField<FEncode>
		+ ExtensionField<FDomain>
		+ PackedExtension<BinaryField1b>
		+ PackedExtension<FEncode>,
	FDomain: Field,
	FEncode: BinaryField,
	P: PackedFieldIndexable<Scalar = FExt>
		+ PackedExtension<BinaryField1b>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>,
	Digest: PackedField,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Hash: Hasher<FExt, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger:
		CanObserve<MerkleCap<Digest>> + CanObserve<FExt> + CanSample<FExt> + CanSampleBits<usize>,
{
	let pcss = make_standard_pcss::<_, _, FEncode, P, _, _, _, _>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory,
	)?;
	verify_with_pcs(constraint_system, proof, &pcss, challenger)
}

/// Verifies a proof against a constraint system with provided PCSs.
fn verify_with_pcs<FExt, P1b, P, PCS, Challenger>(
	constraint_system: &ConstraintSystem<P>,
	proof: ProofGenericPCS<FExt, PCS::Commitment, PCS::Proof>,
	pcss: &[PCS],
	mut challenger: Challenger,
) -> Result<(), Error>
where
	FExt: TowerField,
	P1b: PackedField<Scalar = BinaryField1b>,
	P: PackedField<Scalar = FExt> + PackedExtension<BinaryField1b>,
	PCS: PolyCommitScheme<P1b, FExt>,
	Challenger:
		CanObserve<PCS::Commitment> + CanObserve<FExt> + CanSample<FExt> + CanSampleBits<usize>,
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

	let ProofGenericPCS {
		commitments,
		zerocheck_proof,
		greedy_evalcheck_proof,
		pcs_proofs,
	} = proof;

	let backend = make_portable_backend();

	if commitments.len() != oracles.n_batches() {
		return Err(VerificationError::IncorrectNumberOfCommitments.into());
	}

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

	let sumcheck_claims = zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;
	let sumcheck_output =
		sumcheck::batch_verify(&sumcheck_claims, zerocheck_proof, &mut challenger)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	// Evalcheck
	let evalcheck_claims =
		sumcheck::make_eval_claims(&oracles, zerocheck_oracle_metas, zerocheck_output)?;

	let mut pcs_claims = greedy_evalcheck::verify(
		&mut oracles,
		evalcheck_claims,
		greedy_evalcheck_proof,
		&mut challenger,
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
	for ((_batch_id, claim), pcs, commitment, proof) in
		izip!(pcs_claims, pcss, commitments, pcs_proofs)
	{
		pcs.verify_evaluation(
			&mut challenger,
			&commitment,
			&claim.eval_point,
			proof,
			&claim.evals,
			&backend,
		)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
	}

	Ok(())
}

pub type StandardPCS<FExt, FDomain, FEncode, P, Digest, DomainFactory, Hash, Compress> = BatchPCS<
	<P as PackedExtension<BinaryField1b>>::PackedSubfield,
	FExt,
	FRIPCS<
		BinaryField1b,
		FDomain,
		FEncode,
		P,
		DomainFactory,
		MerkleTreeVCS<FExt, Digest, Hash, Compress>,
	>,
>;

#[allow(clippy::type_complexity)]
pub fn make_standard_pcss<FExt, FDomain, FEncode, P, Digest, DomainFactory, Hash, Compress>(
	log_inv_rate: usize,
	security_bits: usize,
	oracles: &MultilinearOracleSet<FExt>,
	domain_factory: DomainFactory,
) -> Result<Vec<StandardPCS<FExt, FDomain, FEncode, P, Digest, DomainFactory, Hash, Compress>>, Error>
where
	FExt: TowerField
		+ PackedField<Scalar = FExt>
		+ ExtensionField<FEncode>
		+ ExtensionField<FDomain>
		+ PackedExtension<BinaryField1b>
		+ PackedExtension<FEncode>,
	FDomain: Field,
	FEncode: BinaryField,
	P: PackedFieldIndexable<Scalar = FExt>
		+ PackedExtension<BinaryField1b>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>,
	Digest: PackedField,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	Hash: Hasher<FExt, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
{
	// TODO: Set Merkle cap height
	let make_merkle_vcs =
		|log_len| MerkleTreeVCS::<FExt, _, Hash, _>::new(log_len, 0, Compress::default());
	oracles
		.committed_batches()
		.into_iter()
		.map(|batch| {
			let log_n_polys = log2_ceil_usize(batch.n_polys);
			let fri_n_vars = batch.n_vars + log_n_polys;
			let fri_pcs = FRIPCS::<_, _, FEncode, P, _, _>::with_optimal_arity(
				fri_n_vars,
				log_inv_rate,
				security_bits,
				make_merkle_vcs,
				domain_factory.clone(),
				NTTOptions::default(),
			)
			.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
			let batch_pcs = BatchPCS::new(fri_pcs, batch.n_vars, log_n_polys)
				.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
			Ok::<_, Error>(batch_pcs)
		})
		.collect()
}
