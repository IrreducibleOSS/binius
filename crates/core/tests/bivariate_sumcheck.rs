// Copyright 2025 Irreducible Inc.

use binius_compute::cpu::layer::CpuLayerHolder;
use binius_core_test_utils::instantiate_bivariate_sumcheck_tests;
use binius_fast_compute::layer::FastCpuLayerHolder;
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_math::B128;

mod cpu_layer {
	use super::*;

	instantiate_bivariate_sumcheck_tests!(CpuLayerHolder<B128>);
}

mod fast_cpu_layer {
	use super::*;

	instantiate_bivariate_sumcheck_tests!(
		FastCpuLayerHolder::<CanonicalTowerFamily, PackedType<OptimalUnderlier, B128>>
	);
}
