// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::{new_hasher_challenger, CanObserve, CanSample, CanSampleBits},
	linear_code::LinearCode,
	merkle_tree::MerkleTreeVCS,
	protocols::fri::{self, CommitOutput, FRIFolder, FRIVerifier, FoldRoundOutput},
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
use std::{collections::HashSet, iter::repeat_with};

fn make_folding_arities(
	log_len: usize,
	n_rounds: usize,
	log_vcs_vector_lens: &[usize],
) -> Vec<usize> {
	let folding_rounds = log_vcs_vector_lens
		.iter()
		.map(|&log_vector_len| log_len - 1 - log_vector_len)
		.chain(std::iter::once(n_rounds - 1))
		.collect::<Vec<_>>();
	(0..folding_rounds.len())
		.map(|i| {
			let prev = if i == 0 { 0 } else { folding_rounds[i - 1] + 1 };
			let curr = folding_rounds[i] + 1;
			curr - prev
		})
		.collect()
}

fn test_commit_prove_verify_success<U, F, FA>(
	log_dimension: usize,
	log_inv_rate: usize,
	log_vcs_vector_lens: &[usize],
	final_message_dimension: usize,
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
		ReedSolomonCode::<F>::new(final_message_dimension, log_inv_rate, NTTOptions::default())
			.unwrap();

	let n_test_queries = 1;
	let n_round_commitments = log_vcs_vector_lens.len();
	let folding_arities =
		make_folding_arities(log_dimension + log_inv_rate, log_dimension, log_vcs_vector_lens);

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
		.take(committed_rs_code_packed.dim() / <PackedType<U, F>>::WIDTH)
		.collect::<Vec<_>>();

	// Prover commits the message
	let CommitOutput {
		commitment: codeword_commitment,
		committed: codeword_committed,
		codeword,
	} = fri::commit_message(&committed_rs_code_packed, &merkle_vcs, &msg).unwrap();

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
		<PackedType<U, F>>::unpack_scalars(&codeword),
		&merkle_vcs,
		&merkle_round_vcss,
		&codeword_committed,
	)
	.unwrap();

	let mut prover_challenger = challenger.clone();
	let mut round_commitments = Vec::with_capacity(round_prover.n_rounds() - 1);
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
	for (query_rd, commitment) in round_commitments.iter().enumerate() {
		verifier_challenges.append(&mut verifier_challenger.sample_vec(folding_arities[query_rd]));
		verifier_challenger.observe(commitment.clone());
	}

	verifier_challenges
		.append(&mut verifier_challenger.sample_vec(*folding_arities.last().unwrap()));
	verifier_challenger.observe_slice(&final_message);

	let verifier = FRIVerifier::new(
		&committed_rs_code,
		&final_rs_code,
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

fn generate_random_decreasing_sequence(
	minimum: usize,
	maximum: usize,
	length: usize,
) -> Vec<usize> {
	assert!(maximum >= minimum + length);

	let mut rng = thread_rng();
	let mut numbers = HashSet::new();

	while numbers.len() < length {
		let num = rng.gen_range(minimum..=maximum);
		numbers.insert(num);
	}

	let mut result: Vec<usize> = numbers.into_iter().collect();
	result.sort_unstable_by(|a, b| b.cmp(a)); // Sort in descending order
	result
}

#[test]
fn test_commit_prove_verify_success_128b_simple() {
	// This tests the case where we have a round commitment for every round
	let log_dimension = 8;
	let log_inv_rate = 2;
	let final_message_dimension = 0;
	let minimum = log_inv_rate + final_message_dimension + 1;
	let maximum = log_inv_rate + log_dimension - 1;

	let log_vcs_vector_lens = (minimum..=maximum).rev().collect::<Vec<_>>();

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		&log_vcs_vector_lens,
		final_message_dimension,
	);
}

#[test]
fn test_commit_prove_verify_success_128b() {
	let log_dimension = 7;
	let log_inv_rate = 2;
	let n_commitments = 3;
	let final_message_dimension = 2;
	let log_vcs_vector_lens = generate_random_decreasing_sequence(
		log_inv_rate + final_message_dimension + 1,
		log_inv_rate + log_dimension - 1,
		n_commitments,
	);

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		&log_vcs_vector_lens,
		final_message_dimension,
	);
}
