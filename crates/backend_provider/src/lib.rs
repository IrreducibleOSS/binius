pub fn make_best_backend() -> impl binius_hal::ComputationBackend {
	let backend = binius_hal::cpu::CpuBackend;
	#[cfg(feature = "linerate-backend")]
	let backend = binius_linerate_backend::make_linerate_backend(backend);
	backend
}

pub use binius_hal::HalVecTrait;
#[cfg(feature = "linerate-backend")]
pub type HalVec<P> = binius_linerate_backend::HalVec<binius_hal::cpu::CpuBackend,P>;
#[cfg(not(feature = "linerate-backend"))]
pub type HalVec<P> = <binius_hal::cpu::CpuBackend as binius_hal::ComputationBackend>::Vec<P>;
