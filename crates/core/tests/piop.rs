// Copyright 2025 Irreducible Inc.

use binius_compute::{ComputeHolder, cpu::layer::CpuLayerHolder};
use binius_compute_test_utils::piop::commit_prove_verify;
use binius_core::{merkle_tree::BinaryMerkleTreeProver, piop::CommitMeta};
use binius_field::PackedBinaryField2x128b;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{B8, B16, B128};

#[test]
fn test_with_one_poly() {
	let commit_meta = CommitMeta::with_vars([4]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 1;
	let log_inv_rate = 1;
	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 14, 1 << 22);

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _, _>(
		&mut compute_holder.to_data(),
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_without_opening_claims() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 0;
	let log_inv_rate = 1;
	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 14, 1 << 22);

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _, _>(
		&mut compute_holder.to_data(),
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_with_one_n_vars() {
	let commit_meta = CommitMeta::with_vars([4, 4]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 1;
	let log_inv_rate = 1;
	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 14, 1 << 22);

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _, _>(
		&mut compute_holder.to_data(),
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_commit_prove_verify_extreme_rate() {
	let commit_meta = CommitMeta::with_vars([3, 3, 5, 6]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 8;
	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 14, 1 << 22);

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _, _>(
		&mut compute_holder.to_data(),
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_commit_prove_verify_small() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 1;
	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 14, 1 << 22);

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _, _>(
		&mut compute_holder.to_data(),
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_commit_prove_verify() {
	let commit_meta = CommitMeta::with_vars([6, 6, 8, 9]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 1;
	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 14, 1 << 22);

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _, _>(
		&mut compute_holder.to_data(),
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}
