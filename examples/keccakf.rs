// Copyright 2024 Ulvetanna Inc.

use binius_examples::keccakf;
use binius_hal::make_portable_backend;
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};

fn main() {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let _guard = init_tracing().expect("failed to initialize tracing");

	// Values below 14 are rejected by `find_proof_size_optimal_pcs()`.
	let log_size = get_log_trace_size().unwrap_or(14);
	let log_inv_rate = 1;
	let backend = make_portable_backend();

	keccakf::run_prove_verify(log_size, log_inv_rate, SECURITY_BITS, backend)
		.expect("failed to run prove-verify");
}
