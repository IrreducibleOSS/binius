// Copyright 2024 Ulvetanna Inc.

use binius_hal::cpu::CpuBackend;
pub use binius_hal::HalSlice;

pub fn make_portable_backend() -> CpuBackend {
	CpuBackend
}

pub fn make_backend() -> Backend {
	let backend = make_portable_backend();
	#[cfg(feature = "linerate-backend")]
	let backend = linerate_binius_backend::LinerateBackend::<CpuBackend>::new(backend);
	backend
}

#[cfg(feature = "linerate-backend")]
pub type Backend = linerate_binius_backend::LinerateBackend<CpuBackend>;
#[cfg(not(feature = "linerate-backend"))]
pub type Backend = CpuBackend;
