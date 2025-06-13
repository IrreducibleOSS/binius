// Copyright 2024-2025 Irreducible Inc.

use std::{iter::repeat_with, vec};

use binius_compute::{ComputeData, ComputeHolder, cpu::layer::CpuLayerHolder};
use binius_field::{
	BinaryField, BinaryField16b, BinaryField32b, BinaryField128b, ExtensionField,
	PackedBinaryField16x16b, PackedField, TowerField,
	arch::{OptimalUnderlier128b, packed_64::PackedBinaryField4x16b},
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
};
use binius_hal::{ComputationBackendExt, make_portable_backend};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{MultilinearExtension, MultilinearQuery, TowerTop, fold_right};
use binius_maybe_rayon::prelude::ParallelIterator;
use binius_ntt::{AdditiveNTT, NTTShape, SingleThreadedNTT, fri::fold_interleaved};
use binius_utils::checked_arithmetics::log2_strict_usize;
use bytemuck::zeroed_vec;
use proptest::prelude::*;
use rand::prelude::*;

use super::to_par_scalar_big_chunks;
use crate::{
	fiat_shamir::{CanSample, HasherChallenger},
	merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver},
	protocols::fri::{
		self, CommitOutput, FRIFolder, FRIParams, FRIVerifier, FoldRoundOutput,
		to_par_scalar_small_chunks,
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
	transcript::ProverTranscript,
};

proptest! {
	#[test]
	fn test_fri_compatible_ntt_domains(log_dim in 0..8usize, arity in 0..4usize) {
		test_help_fri_compatible_ntt_domains(log_dim, arity);
	}
}

fn test_help_fri_compatible_ntt_domains(log_dim: usize, arity: usize) {
	let ntt = SingleThreadedNTT::<BinaryField32b>::new(32).unwrap();

	let msg = repeat_with(|| BinaryField32b::random(&mut thread_rng()))
		.take(1 << (log_dim + arity))
		.collect::<Vec<_>>();
	let challenges = repeat_with(|| BinaryField32b::random(&mut thread_rng()))
		.take(arity)
		.collect::<Vec<_>>();

	let query = MultilinearQuery::expand(&challenges).into_expansion();

	// Fold the message using regular folding.
	let mut folded_msg = zeroed_vec(1 << log_dim);
	fold_right::<BinaryField32b, BinaryField32b>(
		&msg,
		log_dim + arity,
		&query,
		arity,
		&mut folded_msg,
	)
	.unwrap();

	// Encode the message over the large domain.
	let mut codeword = msg;
	ntt.forward_transform(
		&mut codeword,
		NTTShape {
			log_y: log_dim + arity,
			..Default::default()
		},
		0,
		0,
		0,
	)
	.unwrap();

	// Fold the encoded message using FRI folding.
	let folded_codeword = fold_interleaved(&ntt, &codeword, &challenges, log_dim + arity, 0);

	// Encode the folded message.
	ntt.forward_transform(
		&mut folded_msg,
		NTTShape {
			log_y: log_dim,
			..Default::default()
		},
		0,
		0,
		0,
	)
	.unwrap();

	// Check that folding and encoding commute.
	assert_eq!(folded_codeword, folded_msg);
}

fn test_commit_prove_verify_success<U, F, FA>(
	log_dimension: usize,
	log_inv_rate: usize,
	log_batch_size: usize,
	arities: &[usize],
) where
	U: UnderlierType + PackScalar<F> + PackScalar<FA>,
	F: TowerField + ExtensionField<FA> + PackedField<Scalar = F> + TowerTop,
	FA: BinaryField,
	PackedType<U, F>: PackedField,
	PackedType<U, FA>: PackedField,
{
	let mut rng = StdRng::seed_from_u64(0);

	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);

	let committed_rs_code = ReedSolomonCode::<FA>::new(log_dimension, log_inv_rate).unwrap();

	let n_test_queries = 3;
	let params =
		FRIParams::new(committed_rs_code, log_batch_size, arities.to_vec(), n_test_queries)
			.unwrap();

	let committed_rs_code = ReedSolomonCode::<FA>::new(log_dimension, log_inv_rate).unwrap();
	let ntt = SingleThreadedNTT::new(params.rs_code().log_len()).unwrap();

	let n_round_commitments = arities.len();

	// Generate a random message
	let msg = repeat_with(|| <PackedType<U, F>>::random(&mut rng))
		.take(committed_rs_code.dim() << log_batch_size >> <PackedType<U, F>>::LOG_WIDTH)
		.collect::<Vec<_>>();

	// Prover commits the message
	let CommitOutput {
		commitment: mut codeword_commitment,
		committed: codeword_committed,
		codeword,
	} = fri::commit_interleaved(&committed_rs_code, &params, &ntt, &merkle_prover, &msg).unwrap();

	let mut compute_holder = CpuLayerHolder::<F>::new(1 << 10, 1 << 20);
	let ComputeData { hal, dev_alloc, .. } = compute_holder.to_data();

	// Run the prover to generate the proximity proof
	let mut round_prover =
		FRIFolder::new(hal, &params, &ntt, &merkle_prover, &codeword, &codeword_committed).unwrap();

	let mut prover_challenger = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	prover_challenger.message().write(&codeword_commitment);
	let mut round_commitments = Vec::with_capacity(params.n_oracles());
	for _i in 0..params.n_fold_rounds() {
		let challenge = prover_challenger.sample();
		let fold_round_output = round_prover
			.execute_fold_round(&dev_alloc, challenge)
			.unwrap();
		match fold_round_output {
			FoldRoundOutput::NoCommitment => {}
			FoldRoundOutput::Commitment(round_commitment) => {
				prover_challenger.message().write(&round_commitment);
				round_commitments.push(round_commitment);
			}
		}
	}

	round_prover.finish_proof(&mut prover_challenger).unwrap();
	// Now run the verifier
	let mut verifier_challenger = prover_challenger.into_verifier();
	codeword_commitment = verifier_challenger.message().read().unwrap();
	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());

	assert_eq!(round_commitments.len(), n_round_commitments);
	for (i, commitment) in round_commitments.iter().enumerate() {
		verifier_challenges.append(&mut verifier_challenger.sample_vec(params.fold_arities()[i]));
		let mut _commitment = *commitment;
		_commitment = verifier_challenger.message().read().unwrap();
	}

	verifier_challenges.append(&mut verifier_challenger.sample_vec(params.n_final_challenges()));

	assert_eq!(verifier_challenges.len(), params.n_fold_rounds());

	// check c == t(r'_0, ..., r'_{\ell-1})
	// note that the prover is claiming that the final_message is [c]
	let backend = make_portable_backend();
	let eval_query = backend
		.multilinear_query::<F>(&verifier_challenges)
		.unwrap();
	// recall that msg, the message the prover commits to, is (the evaluations on the Boolean
	// hypercube of) a multilinear polynomial.
	let multilin = MultilinearExtension::from_values_slice(&msg).unwrap();
	let computed_eval = multilin.evaluate(&eval_query).unwrap();

	let verifier = FRIVerifier::new(
		&params,
		merkle_prover.scheme(),
		&codeword_commitment,
		&round_commitments,
		&verifier_challenges,
	)
	.unwrap();

	let mut cloned_verifier_challenger = verifier_challenger.clone();

	let terminate_codeword_len =
		1 << (params.n_final_challenges() + params.rs_code().log_inv_rate());

	let mut advice = verifier_challenger.decommitment();
	let terminate_codeword: Vec<F> = advice.read_scalar_slice(terminate_codeword_len).unwrap();

	let log_batch_size =
		log2_strict_usize(terminate_codeword.len()).saturating_sub(params.rs_code().log_inv_rate());

	let (commitment, tree) = merkle_prover
		.commit(&terminate_codeword, 1 << log_batch_size)
		.unwrap();

	// Ensure that the terminate_codeword commitment is correct
	let last_round_commitment = round_commitments.last().unwrap_or(&codeword_commitment);
	assert_eq!(*last_round_commitment, commitment.root);

	// Verify that the Merkle tree has exactly inv_rate leaves.
	assert_eq!(tree.log_len, params.rs_code().log_inv_rate());

	let final_fri_value = verifier.verify(&mut cloned_verifier_challenger).unwrap();
	assert_eq!(computed_eval, final_fri_value);
}

#[test]
fn test_commit_prove_verify_success_128b_full() {
	binius_utils::rayon::adjust_thread_pool();

	// This tests the case where we have a round commitment for every round
	let log_dimension = 8;
	let log_final_dimension = 1;
	let log_inv_rate = 2;
	let arities = vec![1; log_dimension - log_final_dimension];

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_higher_arity() {
	let log_dimension = 8;
	let log_inv_rate = 2;
	let arities = [3, 2, 1];

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		0,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let arities = [3, 2, 1];

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved_packed() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let arities = [3, 2, 1];

	test_commit_prove_verify_success::<OptimalUnderlier128b, BinaryField128b, BinaryField16b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&arities,
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
	// Compare results for small and large chunk sizes to ensure that they're identical
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
