// Copyright 2025 Irreducible Inc.

use binius_compute::cpu::CpuLayer;
use binius_compute_test_utils::layer::{
	test_generic_fri_fold, test_generic_kernel_add, test_generic_map_with_multilinear_evaluations,
	test_generic_multiple_multilinear_evaluations, test_generic_single_inner_product,
	test_generic_single_inner_product_using_kernel_accumulator, test_generic_single_left_fold,
	test_generic_single_right_fold, test_generic_single_tensor_expand,
};
use binius_field::{BinaryField16b, BinaryField128b, Field};
use binius_math::{B16, B32, B128};

#[test]
fn test_exec_single_tensor_expand() {
	let n_vars = 8;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << n_vars];
	test_generic_single_tensor_expand(compute, &mut device_memory, n_vars);
}

#[test]
fn test_exec_single_left_fold() {
	type F = BinaryField16b;
	type F2 = BinaryField128b;
	let n_vars = 8;
	let mut device_memory = vec![F2::ZERO; 1 << n_vars];
	let compute = <CpuLayer<B128>>::default();
	test_generic_single_left_fold::<F, F2, _>(
		&compute,
		device_memory.as_mut_slice(),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_right_fold() {
	let n_vars = 8;
	let mut device_memory = vec![B128::ZERO; 1 << n_vars];
	let compute = <CpuLayer<B128>>::default();
	test_generic_single_right_fold::<B16, _, _>(
		&compute,
		device_memory.as_mut_slice(),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_inner_product() {
	let n_vars = 8;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (n_vars + 1)];
	test_generic_single_inner_product::<B16, _, _>(compute, &mut device_memory, n_vars);
}

#[test]
fn test_exec_multiple_multilinear_evaluations() {
	let n_vars = 8;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (n_vars + 1)];
	test_generic_multiple_multilinear_evaluations::<B16, B32, _, _>(
		compute,
		&mut device_memory,
		n_vars,
	);
}

#[test]
fn test_exec_map_with_mle_evaluations() {
	let n_vars = 8;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 3 << n_vars];
	test_generic_map_with_multilinear_evaluations(compute, &mut device_memory, n_vars);
}

#[test]
fn test_exec_single_inner_product_using_kernel_accumulator() {
	let n_vars = 8;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (n_vars + 1)];
	test_generic_single_inner_product_using_kernel_accumulator(compute, &mut device_memory, n_vars);
}

#[test]
fn test_exec_fri_fold_non_zero_log_batch() {
	let log_len = 10;
	let log_batch_size = 4;
	let log_fold_challenges = 2;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (log_len + log_batch_size + 1)];
	test_generic_fri_fold::<_, B16, _>(
		compute,
		&mut device_memory,
		log_len,
		log_batch_size,
		log_fold_challenges,
	);
}

#[test]
fn test_exec_fri_fold_zero_log_batch() {
	let log_len = 10;
	let log_batch_size = 0;
	let log_fold_challenges = 2;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (log_len + log_batch_size + 1)];
	test_generic_fri_fold::<_, B16, _>(
		compute,
		&mut device_memory,
		log_len,
		log_batch_size,
		log_fold_challenges,
	);
}

#[test]
fn test_exec_kernel_add() {
	let log_len = 10;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (log_len + 3)];
	test_generic_kernel_add(compute, &mut device_memory, log_len);
}

#[test]
fn test_extrapolate_line() {
	let log_len = 10;
	let compute = <CpuLayer<B128>>::default();
	let mut device_memory = vec![B128::ZERO; 1 << (log_len + 3)];
	binius_compute_test_utils::layer::test_extrapolate_line(&compute, &mut device_memory, log_len);
}
