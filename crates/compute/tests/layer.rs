// Copyright 2025 Irreducible Inc.

use binius_compute::cpu::layer::CpuLayerHolder;
use binius_compute_test_utils::layer::{
	test_generic_fri_fold, test_generic_kernel_add, test_generic_map_with_multilinear_evaluations,
	test_generic_multiple_multilinear_evaluations, test_generic_single_inner_product,
	test_generic_single_inner_product_using_kernel_accumulator, test_generic_single_left_fold,
	test_generic_single_right_fold, test_generic_single_tensor_expand,
};
use binius_math::{B16, B32, B128};

#[test]
fn test_exec_single_tensor_expand() {
	let n_vars = 8;
	test_generic_single_tensor_expand(
		CpuLayerHolder::<B128>::new(1 << (n_vars + 1), 1 << n_vars),
		n_vars,
	);
}

#[test]
fn test_exec_single_left_fold() {
	type F = B16;
	type F2 = B128;
	let n_vars = 8;
	test_generic_single_left_fold::<F, F2, _, _>(
		CpuLayerHolder::<B128>::new(1 << (n_vars + 1), 1 << n_vars),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_right_fold() {
	type F = B16;
	type F2 = B128;
	let n_vars = 8;
	test_generic_single_right_fold::<F, F2, _, _>(
		CpuLayerHolder::<B128>::new(1 << (n_vars + 1), 1 << n_vars),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_inner_product() {
	type F2 = B16;
	let n_vars = 8;
	test_generic_single_inner_product::<F2, _, _, _>(
		CpuLayerHolder::<B128>::new(1 << (n_vars + 2), 1 << (n_vars + 1)),
		n_vars,
	);
}

#[test]
fn test_exec_multiple_multilinear_evaluations() {
	type F1 = B16;
	type F2 = B32;
	let n_vars = 8;
	test_generic_multiple_multilinear_evaluations::<F1, F2, _, _, _>(
		CpuLayerHolder::<B128>::new(1 << (n_vars + 2), 1 << (n_vars + 1)),
		n_vars,
	);
}

#[test]
fn test_exec_map_with_mle_evaluations() {
	let n_vars = 8;
	test_generic_map_with_multilinear_evaluations(
		CpuLayerHolder::<B128>::new(3 << n_vars, 3 << (n_vars + 1)),
		n_vars,
	);
}

#[test]
fn test_exec_single_inner_product_using_kernel_accumulator() {
	let n_vars = 8;
	test_generic_single_inner_product_using_kernel_accumulator::<B128, _, _>(
		CpuLayerHolder::<B128>::new(1 << (n_vars + 2), 1 << (n_vars + 1)),
		n_vars,
	);
}

#[test]
fn test_exec_fri_fold_non_zero_log_batch() {
	type F = B128;
	type FSub = B16;

	let log_len = 10;
	let log_batch_size = 4;
	let log_fold_challenges = 2;
	test_generic_fri_fold::<F, FSub, _, _>(
		CpuLayerHolder::<B128>::new(
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
	type F = B128;
	type FSub = B16;

	let log_len = 10;
	let log_batch_size = 0;
	let log_fold_challenges = 2;
	test_generic_fri_fold::<F, FSub, _, _>(
		CpuLayerHolder::<B128>::new(
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
	let log_len = 10;
	test_generic_kernel_add::<B128, _, _>(
		CpuLayerHolder::<B128>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_extrapolate_line() {
	let log_len = 10;
	binius_compute_test_utils::layer::test_extrapolate_line(
		CpuLayerHolder::<B128>::new(1 << (log_len + 4), 1 << (log_len + 3)),
		log_len,
	);
}

#[test]
fn test_map_kernels() {
	let log_len = 10;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (log_len + 3)];
	binius_compute_test_utils::layer::test_map_kernels(&compute, &mut device_memory, log_len);
}
