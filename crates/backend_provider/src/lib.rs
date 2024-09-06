use binius_hal::cpu::CpuBackend;
pub use binius_hal::HalSlice;

pub fn make_best_backend() -> impl binius_hal::ComputationBackend {
	let backend = CpuBackend;
	#[cfg(feature = "linerate-backend")]
	let backend = binius_linerate_backend::make_linerate_backend(backend);
	backend
}
