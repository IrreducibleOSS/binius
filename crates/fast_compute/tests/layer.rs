// Copyright 2025 Irreducible Inc.

use binius_compute_test_utils::layer::{
	test_generic_fri_fold, test_generic_kernel_add, test_generic_single_inner_product,
	test_generic_single_inner_product_using_kernel_accumulator, test_generic_single_left_fold,
	test_generic_single_right_fold, test_generic_single_tensor_expand,
};
use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
use binius_field::{
	BinaryField16b, BinaryField128b, PackedBinaryField1x128b, PackedBinaryField2x128b,
	PackedBinaryField4x128b, PackedField, tower::CanonicalTowerFamily,
};

#[test]
fn test_exec_single_tensor_expand() {
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << n_vars];
	test_generic_single_tensor_expand(
		compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		n_vars,
	);
}

#[test]
fn test_exec_single_left_fold() {
	type F = BinaryField16b;
	type F2 = BinaryField128b;
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	let mut device_memory = vec![P::zero(); 1 << (n_vars - P::LOG_WIDTH)];
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	test_generic_single_left_fold::<F, F2, _>(
		&compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
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
	let mut device_memory = vec![P::zero(); 1 << (n_vars - P::LOG_WIDTH)];
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	test_generic_single_right_fold::<F, F2, _>(
		&compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		n_vars / 2,
		n_vars / 8,
	);
}

#[test]
fn test_exec_single_inner_product() {
	type F2 = BinaryField16b;
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (n_vars + 1 - P::LOG_WIDTH)];
	test_generic_single_inner_product::<F2, _, _>(
		compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		n_vars,
	);
}

#[test]
fn test_exec_single_inner_product_using_kernel_accumulator() {
	type F = BinaryField128b;
	type P = PackedBinaryField2x128b;
	let n_vars = 16;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (n_vars + 1 - P::LOG_WIDTH)];
	test_generic_single_inner_product_using_kernel_accumulator::<F, _>(
		compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
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
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + log_batch_size + 1 - P::LOG_WIDTH)];
	test_generic_fri_fold::<F, FSub, _>(
		compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
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
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + log_batch_size + 1 - P::LOG_WIDTH)];
	test_generic_fri_fold::<F, FSub, _>(
		compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
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
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + 3 - P::LOG_WIDTH)];
	test_generic_kernel_add::<F, _>(
		compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		log_len,
	);
}

#[test]
fn test_extrapolate_line_128b() {
	type P = PackedBinaryField1x128b;
	let log_len = 10;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + 3 - P::LOG_WIDTH)];
	binius_compute_test_utils::layer::test_extrapolate_line(
		&compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		log_len,
	);
}

#[test]
fn test_extrapolate_line_256b() {
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + 3 - P::LOG_WIDTH)];
	binius_compute_test_utils::layer::test_extrapolate_line(
		&compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		log_len,
	);
}

#[test]
fn test_extrapolate_line_512b() {
	type P = PackedBinaryField4x128b;
	let log_len = 10;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + 3 - P::LOG_WIDTH)];
	binius_compute_test_utils::layer::test_extrapolate_line(
		&compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		log_len,
	);
}

#[test]
fn test_compute_composite() {
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + 3 - P::LOG_WIDTH)];
	binius_compute_test_utils::layer::test_generic_compute_composite(
		&compute,
		PackedMemorySliceMut::new_slice(&mut device_memory),
		log_len,
	);
}
