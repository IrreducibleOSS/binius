// Copyright 2024 Irreducible Inc.

use super::{
	error::{Error, VerificationError},
	ConstraintSystem, Proof, ProofGenericPCS,
};
use crate::{
	challenger::{CanObserve, CanSample},
	constraint_system::{
		channel::{Flush, FlushDirection},
		common::{
			standard_pcs,
			standard_pcs::{FRIMerklePCS, FRIMerkleTowerPCS},
			FExt, TowerPCS, TowerPCSFamily,
		},
	},
	fiat_shamir::Challenger,
	merkle_tree::{MerkleCap, MerkleTreeVCS},
	oracle::{CommittedBatch, MultilinearOracleSet, OracleId},
	poly_commit::{batch_pcs::BatchPCS, FRIPCS},
	protocols::{
		gkr_gpa, greedy_evalcheck,
		sumcheck::{self, constraint_set_zerocheck_claim, zerocheck},
	},
	tower::{PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceReader, TranscriptReader},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	ExtensionField, Field, PackedExtension, PackedField, PackedFieldIndexable, RepackedExtension,
	TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::Hasher;
use binius_math::EvaluationDomainFactory;
use binius_ntt::NTTOptions;
use binius_utils::bail;
use itertools::{izip, Itertools};
use p3_symmetric::PseudoCompressionFunction;
use p3_util::log2_ceil_usize;
use std::cmp::Reverse;
use tracing::instrument;

/// Verifies a proof against a constraint system.
#[instrument("constraint_system::verify", skip_all, level = "debug")]
pub fn verify<U, Tower, Digest, DomainFactory, Hash, Compress, Challenger_>(
	constraint_system: &ConstraintSystem<PackedType<U, FExt<Tower>>>,
	log_inv_rate: usize,
	security_bits: usize,
	domain_factory: DomainFactory,
	proof: Proof<FExt<Tower>, Digest, Hash, Compress>,
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B128: PackedTop<Tower>,
	DomainFactory: EvaluationDomainFactory<Tower::B8>,
	Digest: PackedField<Scalar: TowerField>,
	Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
	Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
	Challenger_: Challenger + Default,
	PackedType<U, Tower::B128>:
		PackedTop<Tower> + PackedFieldIndexable + RepackedExtension<PackedType<U, Tower::B128>>,
{
	let pcss = make_standard_pcss::<U, Tower, _, _, _, _>(
		log_inv_rate,
		security_bits,
		&constraint_system.oracles,
		domain_factory,
	)?;
	verify_with_pcs::<_, _, _, Challenger_, Digest>(constraint_system, proof, &pcss)
}

/// Verifies a proof against a constraint system with provided PCSs.
fn verify_with_pcs<U, Tower, PCSFamily, Challenger_, Digest>(
	constraint_system: &ConstraintSystem<PackedType<U, FExt<Tower>>>,
	proof: ProofGenericPCS<FExt<Tower>, PCSFamily::Commitment, PCSFamily::Proof>,
	pcss: &[TowerPCS<Tower, U, PCSFamily>],
) -> Result<(), Error>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	PCSFamily: TowerPCSFamily<Tower, U, Commitment = MerkleCap<Digest>>,
	Challenger_: Challenger + Default,
	Digest: PackedField<Scalar: TowerField>,
{
	let ConstraintSystem {
		mut oracles,
		mut table_constraints,
		mut flushes,
		non_zero_oracle_ids,
		max_channel_id,
	} = constraint_system.clone();

	if !non_zero_oracle_ids.is_empty() {
		todo!("non-zero oracles are not supported yet");
	}

	// Stable sort constraint sets in descending order by number of variables.
	table_constraints.sort_by_key(|constraint_set| Reverse(constraint_set.n_vars));

	// Stable sort flushes by channel ID.
	flushes.sort_by_key(|flush| flush.channel_id);

	let ProofGenericPCS {
		commitments,
		flush_products,
		prodcheck_proof,
		zerocheck_proof,
		greedy_evalcheck_proof,
		pcs_proofs,
		transcript,
		advice,
	} = proof;

	let mut transcript = TranscriptReader::<Challenger_>::new(transcript);
	let advice = AdviceReader::new(advice);

	let backend = make_portable_backend();

	if commitments.len() != oracles.n_batches() {
		return Err(VerificationError::IncorrectNumberOfCommitments.into());
	}

	// Observe polynomial commitments
	transcript.observe_slice(&commitments);

	// Channel balancing argument
	let mixing_challenge = transcript.sample();
	// TODO(cryptographers): Find a way to sample less randomness
	let permutation_challenges = transcript.sample_vec(max_channel_id + 1);

	// Grand product arguments
	let flush_oracles =
		make_flush_oracles(&mut oracles, &flushes, mixing_challenge, &permutation_challenges)?;
	let prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracles, &oracles, &flush_products)?;
	let final_layer_claims =
		gkr_gpa::batch_verify(prodcheck_claims, prodcheck_proof, &mut transcript)?;
	let prodcheck_eval_claims =
		gkr_gpa::make_eval_claims(&oracles, flush_oracles, final_layer_claims)?;

	verify_channels_balance(&flushes, &flush_products)?;

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
	let zerocheck_challenges = transcript.sample_vec(n_zerocheck_challenges);

	let sumcheck_claims = zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;
	let sumcheck_output =
		sumcheck::batch_verify(&sumcheck_claims, zerocheck_proof, &mut transcript)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let zerocheck_eval_claims =
		sumcheck::make_eval_claims(&oracles, zerocheck_oracle_metas, zerocheck_output)?;

	// Evalcheck
	let mut pcs_claims = greedy_evalcheck::verify(
		&mut oracles,
		prodcheck_eval_claims
			.into_iter()
			.chain(zerocheck_eval_claims),
		greedy_evalcheck_proof,
		&mut transcript,
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
			&mut transcript,
			&commitment,
			&claim.eval_point,
			proof,
			&claim.evals,
			&backend,
		)
		.map_err(|err| Error::PolyCommitError(Box::new(err)))?;
	}

	transcript.finalize()?;
	advice.finalize()?;

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

fn verify_channels_balance<F: Field>(flushes: &[Flush], flush_products: &[F]) -> Result<(), Error> {
	if flush_products.len() != flushes.len() {
		return Err(VerificationError::IncorrectNumberOfFlushProducts.into());
	}

	let mut flush_iter = flushes
		.iter()
		.zip(flush_products.iter().copied())
		.peekable();
	while let Some((flush, _)) = flush_iter.peek() {
		let channel_id = flush.channel_id;
		let (pull_product, push_product) = flush_iter
			.peeking_take_while(|(flush, _)| flush.channel_id == channel_id)
			.fold((F::ONE, F::ONE), |(pull_product, push_product), (flush, flush_product)| {
				match flush.direction {
					FlushDirection::Pull => (pull_product * flush_product, push_product),
					FlushDirection::Push => (pull_product, push_product * flush_product),
				}
			});
		if pull_product != push_product {
			return Err(VerificationError::ChannelUnbalanced { id: channel_id }.into());
		}
	}

	Ok(())
}

pub fn make_flush_oracles<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	flushes: &[Flush],
	mixing_challenge: F,
	permutation_challenges: &[F],
) -> Result<Vec<OracleId>, Error> {
	let mut mixing_powers = vec![F::ONE];
	let mut flush_iter = flushes.iter().peekable();
	permutation_challenges
		.iter()
		.enumerate()
		.flat_map(|(channel_id, permutation_challenge)| {
			flush_iter
				.peeking_take_while(|flush| flush.channel_id == channel_id)
				.map(|flush| {
					// Check that all flushed oracles have the same number of variables
					let first_oracle = flush.oracles.first().ok_or(Error::EmptyFlushOracles)?;
					let n_vars = oracles.n_vars(*first_oracle);
					for &oracle_id in flush.oracles.iter().skip(1) {
						let oracle_n_vars = oracles.n_vars(oracle_id);
						if oracle_n_vars != n_vars {
							return Err(Error::ChannelFlushNvarsMismatch {
								expected: n_vars,
								got: oracle_n_vars,
							});
						}
					}

					// Compute powers of the mixing challenge
					while mixing_powers.len() < flush.oracles.len() {
						let last_power = *mixing_powers.last().expect(
							"mixing_powers is initialized with one element; \
								mixing_powers never shrinks; \
								thus, it must not be empty",
						);
						mixing_powers.push(last_power * mixing_challenge);
					}

					let id = oracles.add_linear_combination_with_offset(
						n_vars,
						*permutation_challenge,
						flush
							.oracles
							.iter()
							.copied()
							.zip(mixing_powers.iter().copied()),
					)?;
					Ok(id)
				})
				.collect::<Vec<_>>()
		})
		.collect()
}
