// Copyright 2024 Irreducible Inc.

use super::{
	error::{Error, VerificationError},
	ConstraintSystem, Proof, ProofGenericPCS,
};
use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	constraint_system::common::{
		standard_pcs,
		standard_pcs::{FRIMerklePCS, FRIMerkleTowerPCS},
		FExt, TowerPCS, TowerPCSFamily,
	},
	merkle_tree::{MerkleCap, MerkleTreeVCS},
	oracle::{CommittedBatch, MultilinearOracleSet},
	poly_commit::{batch_pcs::BatchPCS, FRIPCS},
	protocols::{
		greedy_evalcheck,
		sumcheck::{self, constraint_set_zerocheck_claim, zerocheck},
	},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	ExtensionField, PackedExtension, PackedField, PackedFieldIndexable, RepackedExtension,
	TowerField,
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
pub fn verify<U, Tower, Digest, DomainFactory, Hash, Compress, Challenger>(
	constraint_system: &ConstraintSystem<PackedType<U, FExt<Tower>>>,
	log_inv_rate: usize,
	security_bits: usize,
	domain_factory: DomainFactory,
	proof: Proof<FExt<Tower>, Digest, Hash, Compress>,
	challenger: Challenger,
) -> Result<(), Error>
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
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
{
	let pcss = make_standard_pcss::<U, Tower, _, _, _, _>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory,
	)?;
	verify_with_pcs(constraint_system, proof, &pcss, challenger)
}

/// Verifies a proof against a constraint system with provided PCSs.
fn verify_with_pcs<U, Tower, PCSFamily, Challenger>(
	constraint_system: &ConstraintSystem<PackedType<U, FExt<Tower>>>,
	proof: ProofGenericPCS<FExt<Tower>, PCSFamily::Commitment, PCSFamily::Proof>,
	pcss: &[TowerPCS<Tower, U, PCSFamily>],
	mut challenger: Challenger,
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	PCSFamily: TowerPCSFamily<Tower, U>,
	Challenger: CanObserve<PCSFamily::Commitment>
		+ CanObserve<Tower::B128>
		+ CanSample<Tower::B128>
		+ CanSampleBits<usize>,
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

#[allow(clippy::type_complexity)]
pub fn make_standard_pcss<U, Tower, Digest, DomainFactory, Hash, Compress>(
	log_inv_rate: usize,
	security_bits: usize,
	oracles: &MultilinearOracleSet<Tower::B128>,
	domain_factory: DomainFactory,
) -> Result<Vec<FRIMerkleTowerPCS<Tower, U, Digest, DomainFactory, Hash, Compress>>, Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	PackedType<U, Tower::B128>: PackedTop<Tower> + PackedFieldIndexable,
{
	oracles
		.committed_batches()
		.into_iter()
		.map(|batch| match batch.tower_level {
			0 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B1),
			3 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B8),
			4 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B16),
			5 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B32),
			6 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B64),
			7 => make_standard_pcs::<U, Tower, _, _, _, _, _>(
				log_inv_rate,
				security_bits,
				domain_factory.clone(),
				batch,
			)
			.map(TowerPCS::B128),
			_ => Err(Error::CannotCommitTowerLevel {
				tower_level: batch.tower_level,
			}),
		})
		.collect()
}

#[allow(clippy::type_complexity)]
fn make_standard_pcs<U, Tower, F, Digest, DomainFactory, Hash, Compress>(
	log_inv_rate: usize,
	security_bits: usize,
	domain_factory: DomainFactory,
	batch: CommittedBatch,
) -> Result<FRIMerklePCS<Tower, U, F, Digest, DomainFactory, Hash, Compress>, Error>
where
	U: TowerUnderlier<Tower> + PackScalar<F>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower> + ExtensionField<F> + PackedExtension<F>,
	F: TowerField,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	PackedType<U, Tower::B128>: PackedTop<Tower> + PackedFieldIndexable,
{
	// TODO: Set Merkle cap height
	let make_merkle_vcs =
		|log_len| MerkleTreeVCS::<Tower::B128, _, Hash, _>::new(log_len, 0, Compress::default());
	let log_n_polys = log2_ceil_usize(batch.n_polys);
	let fri_n_vars = batch.n_vars + log_n_polys;
	let fri_pcs = FRIPCS::<
		_,
		standard_pcs::FDomain<Tower>,
		standard_pcs::FEncode<Tower>,
		PackedType<U, Tower::B128>,
		_,
		_,
	>::with_optimal_arity(
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
	Ok(batch_pcs)
}
