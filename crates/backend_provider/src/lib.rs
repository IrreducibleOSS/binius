use binius_hal::cpu::CpuBackend;
pub use binius_hal::HalSlice;

pub fn make_best_backend() -> BestBackend {
	let backend = CpuBackend;
	#[cfg(feature = "linerate-backend")]
	let backend = binius_linerate::LinerateBackend::<CpuBackend>::new(backend);
	backend
}

#[cfg(feature = "linerate-backend")]
pub type BestBackend = binius_linerate::LinerateBackend<CpuBackend>;
#[cfg(not(feature = "linerate-backend"))]
pub type BestBackend = CpuBackend;
