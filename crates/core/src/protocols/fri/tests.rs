// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::{new_hasher_challenger, CanObserve, CanSample, CanSampleBits},
	linear_code::LinearCode,
	merkle_tree::MerkleTreeVCS,
	protocols::fri::{
		self, common::calculate_fold_arities, CommitOutput, FRIFolder, FRIVerifier, FoldRoundOutput,
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{
	arch::OptimalUnderlier128b,
	as_packed_field::{PackScalar, PackedType},
	underlier::{Divisible, UnderlierType},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField8b, ExtensionField, PackedExtension,
	PackedField, PackedFieldIndexable,
};
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use binius_ntt::NTTOptions;
use rand::prelude::*;
use std::iter::repeat_with;

fn test_commit_prove_verify_success<U, F, FA>(
	log_dimension: usize,
	log_inv_rate: usize,
	log_batch_size: usize,
	log_vcs_vector_lens: &[usize],
	log_final_dimension: usize,
) where
	U: UnderlierType + PackScalar<F> + PackScalar<FA> + PackScalar<BinaryField8b> + Divisible<u8>,
	F: BinaryField + ExtensionField<FA> + ExtensionField<BinaryField8b>,
	F: PackedField<Scalar = F>
		+ PackedExtension<BinaryField8b, PackedSubfield: PackedFieldIndexable>,
	FA: BinaryField,
	PackedType<U, F>: PackedFieldIndexable,
	PackedType<U, FA>: PackedFieldIndexable,
{
	let mut rng = StdRng::seed_from_u64(0);

	let committed_rs_code_packed = ReedSolomonCode::<PackedType<U, FA>>::new(
		log_dimension,
		log_inv_rate,
		NTTOptions::default(),
	)
	.unwrap();
	let final_rs_code =
		ReedSolomonCode::<F>::new(log_final_dimension, log_inv_rate, NTTOptions::default())
			.unwrap();

	let n_test_queries = 3;
	let n_round_commitments = log_vcs_vector_lens.len();
	let folding_arities = calculate_fold_arities(
		log_dimension + log_inv_rate,
		log_final_dimension + log_inv_rate,
		log_vcs_vector_lens.iter().copied(),
		log_batch_size,
	)
	.unwrap();

	let make_merkle_vcs = |log_len| {
		MerkleTreeVCS::<F, _, GroestlHasher<_>, _>::new(
			log_len,
			0,
			GroestlDigestCompression::<BinaryField8b>::default(),
		)
	};

	let merkle_vcs = make_merkle_vcs(committed_rs_code_packed.log_len());
	let merkle_round_vcss = log_vcs_vector_lens
		.iter()
		.map(|&log_len| make_merkle_vcs(log_len))
		.collect::<Vec<_>>();

	// Generate a random message
	let msg = repeat_with(|| <PackedType<U, F>>::random(&mut rng))
		.take(committed_rs_code_packed.dim() << log_batch_size >> <PackedType<U, F>>::LOG_WIDTH)
		.collect::<Vec<_>>();

	// Prover commits the message
	let CommitOutput {
		commitment: codeword_commitment,
		committed: codeword_committed,
		codeword,
	} = fri::commit_interleaved(&committed_rs_code_packed, log_batch_size, &merkle_vcs, &msg).unwrap();

	let mut challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	challenger.observe(codeword_commitment.clone());

	// Run the prover to generate the proximity proof
	let committed_rs_code = ReedSolomonCode::<FA>::new(
		committed_rs_code_packed.log_dim(),
		committed_rs_code_packed.log_inv_rate(),
		NTTOptions::default(),
	)
	.unwrap();

	let mut round_prover = FRIFolder::new(
		&committed_rs_code,
		&final_rs_code,
		log_batch_size,
		<PackedType<U, F>>::unpack_scalars(&codeword),
		&merkle_vcs,
		&merkle_round_vcss,
		&codeword_committed,
	)
	.unwrap();

	let mut prover_challenger = challenger.clone();
	let mut round_commitments = Vec::with_capacity(round_prover.n_rounds());
	for _i in 0..round_prover.n_rounds() {
		let challenge = prover_challenger.sample();
		let fold_round_output = round_prover.execute_fold_round(challenge).unwrap();
		match fold_round_output {
			FoldRoundOutput::NoCommitment => {}
			FoldRoundOutput::Commitment(round_commitment) => {
				prover_challenger.observe(round_commitment.clone());
				round_commitments.push(round_commitment);
			}
		}
	}

	let (final_message, query_prover) = round_prover.finalize().unwrap();
	prover_challenger.observe_slice(&final_message);

	let query_proofs = repeat_with(|| {
		let index = prover_challenger.sample_bits(committed_rs_code.log_len());
		query_prover.prove_query(index)
	})
	.take(n_test_queries)
	.collect::<Result<Vec<_>, _>>()
	.unwrap();

	// Now run the verifier
	let mut verifier_challenger = challenger.clone();
	let mut verifier_challenges = Vec::with_capacity(committed_rs_code.log_dim());

	assert_eq!(round_commitments.len(), n_round_commitments);
	for (i, commitment) in round_commitments.iter().enumerate() {
		verifier_challenges.append(&mut verifier_challenger.sample_vec(folding_arities[i]));
		verifier_challenger.observe(commitment.clone());
	}

	verifier_challenges
		.append(&mut verifier_challenger.sample_vec(*folding_arities.last().unwrap()));
	verifier_challenger.observe_slice(&final_message);

	let verifier = FRIVerifier::new(
		&committed_rs_code,
		&final_rs_code,
		log_batch_size,
		&merkle_vcs,
		&merkle_round_vcss,
		&codeword_commitment,
		&round_commitments,
		&verifier_challenges,
		final_message,
	)
	.unwrap();

	assert_eq!(query_proofs.len(), n_test_queries);
	for query_proof in query_proofs {
		let index = verifier_challenger.sample_bits(committed_rs_code.log_len());
		verifier.verify_query(index, query_proof).unwrap();
	}
}

#[test]
fn test_commit_prove_verify_success_128b_full() {
	binius_utils::rayon::adjust_thread_pool();

	// This tests the case where we have a round commitment for every round
	let log_dimension = 8;
	let log_inv_rate = 2;
	let log_final_dim = 0;
	let minimum = log_inv_rate + log_final_dim + 1;
	let maximum = log_inv_rate + log_dimension - 1;

	let log_vcs_vector_lens = (minimum..=maximum).rev().collect::<Vec<_>>();

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&log_vcs_vector_lens,
		log_final_dim,
	);
}

#[test]
#[ignore]
fn test_commit_prove_verify_success_128b_nontrivial_final_dim() {
	binius_utils::rayon::adjust_thread_pool();

	let log_dimension = 8;
	let log_inv_rate = 2;
	let log_final_dim = 2;
	let minimum = log_inv_rate + log_final_dim + 1;
	let maximum = log_inv_rate + log_dimension - 1;

	let log_vcs_vector_lens = (minimum..=maximum).rev().collect::<Vec<_>>();

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&log_vcs_vector_lens,
		log_final_dim,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_higher_arity() {
	let log_dimension = 8;
	let log_inv_rate = 2;
	let log_commit_dims = [5, 3, 2];
	let log_final_dim = 0;
	let log_vcs_vector_lens = log_commit_dims.map(|dim| dim + log_inv_rate);

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&log_vcs_vector_lens,
		log_final_dim,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let log_commit_dims = [5, 3, 1];
	let log_final_message_dimension = 0;
	let log_vcs_vector_lens = log_commit_dims.map(|dim| dim + log_inv_rate);

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&log_vcs_vector_lens,
		log_final_message_dimension,
	);
}
