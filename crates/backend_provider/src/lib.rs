use binius_hal::cpu::CpuBackend;
pub use binius_hal::HalSlice;

pub fn make_portable_backend() -> CpuBackend {
	CpuBackend
}

pub fn make_backend() -> Backend {
	let backend = make_portable_backend();
	#[cfg(feature = "linerate-backend")]
	let backend = binius_linerate::LinerateBackend::<CpuBackend>::new(backend);
	backend
}

#[cfg(feature = "linerate-backend")]
pub type Backend = binius_linerate::LinerateBackend<CpuBackend>;
#[cfg(not(feature = "linerate-backend"))]
pub type Backend = CpuBackend;
