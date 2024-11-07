// Copyright 2024 Irreducible Inc.

use crate::{
	challenger::{new_hasher_challenger, CanObserve, CanSample},
	linear_code::LinearCode,
	merkle_tree::MerkleTreeVCS,
	protocols::fri::{
		self, to_par_scalar_small_chunks, CommitOutput, FRIFolder, FRIParams, FRIVerifier,
		FoldRoundOutput,
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{
	arch::{packed_64::PackedBinaryField4x16b, OptimalUnderlier128b},
	as_packed_field::{PackScalar, PackedType},
	underlier::{Divisible, UnderlierType},
	BinaryField, BinaryField128b, BinaryField16b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedBinaryField16x16b, PackedExtension, PackedField, PackedFieldIndexable,
};
use binius_hal::{make_portable_backend, ComputationBackendExt};
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use binius_math::MultilinearExtension;
use binius_ntt::NTTOptions;
use rand::prelude::*;
use rayon::prelude::ParallelIterator;
use std::{iter, iter::repeat_with, vec};

use super::to_par_scalar_big_chunks;

fn test_commit_prove_verify_success<U, F, FA>(
	log_dimension: usize,
	log_inv_rate: usize,
	log_batch_size: usize,
	log_vcs_vector_lens: &[usize],
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

	let make_merkle_vcs = |log_len| {
		MerkleTreeVCS::<F, _, GroestlHasher<_>, _>::new(
			log_len,
			0,
			GroestlDigestCompression::<BinaryField8b>::default(),
		)
	};

	let mut merkle_vcss_iter = log_vcs_vector_lens
		.iter()
		.copied()
		.chain(iter::once(log_inv_rate))
		.map(make_merkle_vcs);
	let merkle_vcs = merkle_vcss_iter.next().expect("non-empty");
	let merkle_round_vcss = merkle_vcss_iter.collect::<Vec<_>>();

	let committed_rs_code =
		ReedSolomonCode::<FA>::new(log_dimension, log_inv_rate, NTTOptions::default()).unwrap();

	let n_test_queries = 3;
	let params = FRIParams::new(
		committed_rs_code,
		log_batch_size,
		merkle_vcs,
		merkle_round_vcss,
		n_test_queries,
	)
	.unwrap();

	let n_round_commitments = log_vcs_vector_lens.len();

	// Generate a random message
	let msg = repeat_with(|| <PackedType<U, F>>::random(&mut rng))
		.take(committed_rs_code_packed.dim() << log_batch_size >> <PackedType<U, F>>::LOG_WIDTH)
		.collect::<Vec<_>>();

	// Prover commits the message
	let CommitOutput {
		commitment: codeword_commitment,
		committed: codeword_committed,
		codeword,
	} = fri::commit_interleaved(
		&committed_rs_code_packed,
		log_batch_size,
		params.codeword_vcs(),
		&msg,
	)
	.unwrap();

	let mut challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	challenger.observe(codeword_commitment.clone());

	// Run the prover to generate the proximity proof
	let mut round_prover =
		FRIFolder::new(&params, <PackedType<U, F>>::unpack_scalars(&codeword), &codeword_committed)
			.unwrap();

	let mut prover_challenger = challenger.clone();
	let mut round_commitments = Vec::with_capacity(params.n_oracles());
	for _i in 0..params.n_fold_rounds() {
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

	let fri_proof = round_prover.finish_proof(prover_challenger).unwrap();

	// Now run the verifier
	let mut verifier_challenger = challenger.clone();
	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());

	assert_eq!(round_commitments.len(), n_round_commitments);
	for (i, commitment) in round_commitments.iter().enumerate() {
		verifier_challenges.append(&mut verifier_challenger.sample_vec(params.fold_arities()[i]));
		verifier_challenger.observe(commitment.clone());
	}

	verifier_challenges.append(&mut verifier_challenger.sample_vec(params.n_final_challenges()));

	assert_eq!(verifier_challenges.len(), params.n_fold_rounds());

	// check c == t(r'_0, ..., r'_{\ell-1})
	// note that the prover is claiming that the final_message is [c]
	let backend = make_portable_backend();
	let eval_query = backend
		.multilinear_query::<F>(&verifier_challenges)
		.unwrap();
	// recall that msg, the message the prover commits to, is (the evaluations on the Boolean hypercube of) a multilinear polynomial.
	let multilin = MultilinearExtension::from_values_slice(&msg).unwrap();
	let computed_eval = multilin.evaluate(&eval_query).unwrap();

	let verifier =
		FRIVerifier::new(&params, &codeword_commitment, &round_commitments, &verifier_challenges)
			.unwrap();

	let final_fri_value = verifier.verify(fri_proof, verifier_challenger).unwrap();
	assert_eq!(computed_eval, final_fri_value);
}

#[test]
fn test_commit_prove_verify_success_128b_full() {
	binius_utils::rayon::adjust_thread_pool();

	// This tests the case where we have a round commitment for every round
	let log_dimension = 8;
	let log_inv_rate = 2;
	let log_final_dim = 1;
	let minimum = log_inv_rate + log_final_dim + 1;
	let maximum = log_inv_rate + log_dimension - 1;

	let log_vcs_vector_lens = (minimum..=maximum).rev().collect::<Vec<_>>();

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&log_vcs_vector_lens,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_higher_arity() {
	let log_dimension = 8;
	let log_inv_rate = 2;
	let log_commit_dims = [5, 3, 2];
	let log_vcs_vector_lens = log_commit_dims.map(|dim| dim + log_inv_rate);

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&log_vcs_vector_lens,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let log_commit_dims = [5, 3, 2];
	let log_vcs_vector_lens = log_commit_dims.map(|dim| dim + log_inv_rate);

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&log_vcs_vector_lens,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved_packed() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let log_commit_dims = [5, 3, 2];
	let log_vcs_vector_lens = log_commit_dims.map(|dim| dim + log_inv_rate);

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField32b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&log_vcs_vector_lens,
	);
}

#[test]
fn test_commit_prove_verify_success_without_folding() {
	let log_dimension = 4;
	let log_inv_rate = 2;
	let log_batch_size = 2;

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&[],
	);
}

#[test]
fn test_parallel_iterator_for_commitments() {
	// Compare results for small and large chunk sizes to ensure that theyre identical
	let data: Vec<_> = (0..64).map(BinaryField16b::from).collect();

	let mut data_packed_4 = vec![];

	for i in 0..64 / 4 {
		let mut scalars = vec![];
		for j in 0..4 {
			scalars.push(data[4 * i + j]);
		}

		data_packed_4.push(PackedBinaryField4x16b::from_scalars(scalars));
	}

	let mut data_packed_16 = vec![];

	for i in 0..64 / 16 {
		let mut scalars = vec![];
		for j in 0..16 {
			scalars.push(data[16 * i + j]);
		}

		data_packed_16.push(PackedBinaryField16x16b::from_scalars(scalars));
	}

	let packing_smaller_than_chunk = to_par_scalar_big_chunks(&data_packed_4, 8);

	let packing_bigger_than_chunk = to_par_scalar_small_chunks(&data_packed_16, 8);

	let collected_smaller: Vec<_> = packing_smaller_than_chunk
		.map(|inner| {
			let result: Vec<_> = inner.collect();
			result
		})
		.collect();

	let collected_bigger: Vec<_> = packing_bigger_than_chunk
		.map(|inner| {
			let result: Vec<_> = inner.collect();
			result
		})
		.collect();

	assert_eq!(collected_smaller, collected_bigger);
}
