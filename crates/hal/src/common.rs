// Copyright 2025 Irreducible Inc.

use binius_field::{
	PackedField,
	underlier::{UnderlierType, WithUnderlier},
};

// A kibibyte per multilinear seems like a reasonable compromise.
pub const MAX_SRC_SUBCUBE_LOG_BITS: usize = 13;

// A heuristic to determine the optimal subcube size for sumcheck calc / fold stages.
//
// Rough idea is as follows: we want subcubes small enough to create parallelization
// opportunities, while big enough to amortize dynamic dispatch and leverage L1 caches.
//
// Top to bottom, heuristics are:
//   - keep working set at constant bit size ...
//   - accounting for pre-switchover rounds reading from a larger working set (inner_product_vars)
//   - try to minimize wasted effort (max_subcube_src_vars)
//   - do not allow subcubes to be smaller than packing width (P::LOG_WIDTH) ...
//   - unless the multilinear is smaller than a single packed field (max_total_vars)
pub fn subcube_vars_for_bits<P: PackedField>(
	max_subcube_src_bits: usize,
	max_subcube_src_vars: usize,
	inner_product_vars: usize,
	max_total_vars: usize,
) -> usize {
	let scalar_log_bits = <P::Scalar as WithUnderlier>::Underlier::LOG_BITS;
	let src_vars = max_subcube_src_vars.min(max_subcube_src_bits - scalar_log_bits);

	src_vars
		.saturating_sub(inner_product_vars)
		.max(P::LOG_WIDTH)
		.min(max_total_vars)
}
