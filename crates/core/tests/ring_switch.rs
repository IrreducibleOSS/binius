// Copyright 2024-2025 Irreducible Inc.

use binius_core::merkle_tree::BinaryMerkleTreeProver;
use binius_core_test_utils::{instantiate_ring_switch_tests, ring_switch::make_test_oracle_set};
use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::B128;

mod cpu_tests {
	use binius_compute::cpu::layer::CpuLayerHolder;

	use super::*;

	instantiate_ring_switch_tests!(CpuLayerHolder<B128>);
}

mod fast_cpu_tests {
	use binius_fast_compute::layer::FastCpuLayerHolder;
	use binius_field::tower::CanonicalTowerFamily;

	use super::*;

	instantiate_ring_switch_tests!(
		FastCpuLayerHolder::<CanonicalTowerFamily, PackedType<OptimalUnderlier128b, B128>>
	);
}
