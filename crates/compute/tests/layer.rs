// Copyright 2025 Irreducible Inc.

use binius_compute::cpu::{CpuLayer, layer::CpuLayerHolder};
use binius_compute_test_utils::instantiate_compute_layer_tests;
use binius_field::tower::CanonicalTowerFamily;
use binius_math::B128;

instantiate_compute_layer_tests!(CanonicalTowerFamily, CpuLayer<B128>, CpuLayerHolder<B128>);
