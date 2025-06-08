// Copyright 2025 Irreducible Inc.

use binius_compute::{
	ComputeLayer, ComputeMemory,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
	cpu::CpuLayer,
};
use binius_field::Field;
use binius_math::B128;
use rand::{SeedableRng, rngs::StdRng};

use super::ProductCircuitLayers;

fn test_basic_prover_product_is_correct<Hal: ComputeLayer<B128>>(
	hal: &Hal,
	dev_mem: &BumpAllocator<B128, Hal::DevMem>,
	host_mem: &HostBumpAllocator<B128>,
) {
	let mut rng = StdRng::seed_from_u64(0);

	// Generate a random multilinear polynomial with 10 variables
	let n_vars = 10;
	let evals = host_mem.alloc(1 << n_vars).unwrap();
	evals.fill_with(|| <B128 as Field>::random(&mut rng));

	let mut dev_evals = dev_mem.alloc(1 << n_vars).unwrap();
	hal.copy_h2d(evals, &mut dev_evals).unwrap();

	let layers =
		ProductCircuitLayers::compute(Hal::DevMem::as_const(&dev_evals), hal, dev_mem, host_mem)
			.unwrap();

	let expected_product = evals.iter().copied().product();
	assert_eq!(layers.product(), expected_product);
}

#[test]
fn test_basic_prover_product_is_correct_on_cpu() {
	let hal = CpuLayer::<B128>::default();
	let mut dev_mem = vec![B128::default(); 2048];
	let mut host_mem = vec![B128::default(); 2048];
	test_basic_prover_product_is_correct(
		&hal,
		&BumpAllocator::new(dev_mem.as_mut_slice()),
		&BumpAllocator::new(host_mem.as_mut_slice()),
	);
}
