pub fn make_best_backend() -> impl binius_hal::ComputationBackend {
	let backend = binius_hal::cpu::CpuBackend;
	#[cfg(feature = "linerate-backend")]
	let backend = binius_linerate_backend::make_linerate_backend(backend);
	backend
}
