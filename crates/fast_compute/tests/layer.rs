// Copyright 2025 Irreducible Inc.

use binius_compute_test_utils::layer::{
	test_generic_single_inner_product, test_generic_single_tensor_expand,
};
use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
use binius_field::{
	BinaryField16b, PackedBinaryField2x128b, PackedField, tower::CanonicalTowerFamily,
};

#[test]
fn test_exec_single_tensor_expand() {
	type P = PackedBinaryField2x128b;
	let n_vars = 8;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << n_vars];
	test_generic_single_tensor_expand(
		compute,
		PackedMemorySliceMut::new(&mut device_memory),
		n_vars,
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
		PackedMemorySliceMut::new(&mut device_memory),
		n_vars,
	);
}

#[test]
fn test_extrapolate_line() {
	type P = PackedBinaryField2x128b;
	let log_len = 10;
	let compute = <FastCpuLayer<CanonicalTowerFamily, P>>::default();
	let mut device_memory = vec![P::zero(); 1 << (log_len + 3 - P::LOG_WIDTH)];
	binius_compute_test_utils::layer::test_extrapolate_line(
		&compute,
		PackedMemorySliceMut::new(&mut device_memory),
		log_len,
	);
}
