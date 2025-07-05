// Copyright 2025 Irreducible Inc.

use binius_compute_test_utils::layer::{
	test_generic_fri_fold, test_generic_kernel_add, test_generic_single_inner_product,
	test_generic_single_inner_product_using_kernel_accumulator, test_generic_single_left_fold,
	test_generic_single_right_fold, test_generic_single_tensor_expand,
};
use binius_fast_compute::layer::FastCpuLayerHolder;
use binius_field::{
	BinaryField16b, BinaryField128b, PackedBinaryField1x128b, PackedBinaryField2x128b,
	PackedBinaryField4x128b, tower::CanonicalTowerFamily,
};

#[test]
fn test_exec_single_tensor_expand() {
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	test_generic_single_tensor_expand(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (n_vars + 1), 1 << n_vars),
		n_vars,
	);
}

#[test]
fn test_exec_single_left_fold() {
	type F = BinaryField16b;
	type F2 = BinaryField128b;
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	test_generic_single_left_fold::<F, F2, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (n_vars + 1), 1 << n_vars),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_right_fold() {
	type F = BinaryField16b;
	type F2 = BinaryField128b;
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	test_generic_single_right_fold::<F, F2, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (n_vars + 1), 1 << n_vars),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_inner_product() {
	type F2 = BinaryField16b;
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	test_generic_single_inner_product::<F2, _, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (n_vars + 2), 1 << (n_vars + 1)),
		n_vars,
	);
}

#[test]
fn test_exec_single_inner_product_using_kernel_accumulator() {
	type F = BinaryField128b;
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	test_generic_single_inner_product_using_kernel_accumulator::<F, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (n_vars + 2), 1 << (n_vars + 1)),
		n_vars,
	);
}

#[test]
fn test_exec_fri_fold_non_zero_log_batch() {
	type F = BinaryField128b;
	type FSub = BinaryField16b;
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	let log_batch_size = 4;
	let log_fold_challenges = 2;
	test_generic_fri_fold::<F, FSub, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(
			1 << (log_len + log_batch_size + 2),
			1 << (log_len + log_batch_size + 1),
		),
		log_len,
		log_batch_size,
		log_fold_challenges,
	);
}

#[test]
fn test_exec_fri_fold_zero_log_batch() {
	type F = BinaryField128b;
	type FSub = BinaryField16b;
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	let log_batch_size = 0;
	let log_fold_challenges = 2;
	test_generic_fri_fold::<F, FSub, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(
			1 << (log_len + log_batch_size + 2),
			1 << (log_len + log_batch_size + 1),
		),
		log_len,
		log_batch_size,
		log_fold_challenges,
	);
}

#[test]
fn test_exec_kernel_add() {
	type F = BinaryField128b;
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	test_generic_kernel_add::<F, _, _>(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_extrapolate_line_128b() {
	type P = PackedBinaryField1x128b;
	let log_len = 10;
	binius_compute_test_utils::layer::test_extrapolate_line(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_extrapolate_line_256b() {
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	binius_compute_test_utils::layer::test_extrapolate_line(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_extrapolate_line_512b() {
	type P = PackedBinaryField4x128b;
	let log_len = 10;
	binius_compute_test_utils::layer::test_extrapolate_line(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_compute_composite() {
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	binius_compute_test_utils::layer::test_generic_compute_composite(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_map_kernels() {
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	binius_compute_test_utils::layer::test_map_kernels(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_pairwise_product_reduce_single_round() {
	type P = PackedBinaryField4x128b;
	let log_len = 1;
	binius_compute_test_utils::layer::test_generic_pairwise_product_reduce(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_pairwise_product_reduce() {
	type P = PackedBinaryField4x128b;
	let log_len = 8;
	binius_compute_test_utils::layer::test_generic_pairwise_product_reduce(
		FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}
