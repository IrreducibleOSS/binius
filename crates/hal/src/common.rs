// Copyright 2025 Irreducible Inc.

use binius_field::{
	underlier::{UnderlierType, WithUnderlier},
	PackedField,
};

pub(crate) const MAX_SRC_SUBCUBE_LOG_BITS: usize = 13;

// TODO: document
pub(crate) fn subcube_vars_for_bits<P: PackedField>(
	max_subcube_src_bits: usize,
	max_subcube_src_vars: usize,
	inner_product_vars: usize,
	max_total_vars: usize,
) -> (usize, usize) {
	let scalar_log_bits = <P::Scalar as WithUnderlier>::Underlier::LOG_BITS;
	let src_vars = max_subcube_src_vars.min(max_subcube_src_bits - scalar_log_bits);
	let subcube_vars = src_vars
		.saturating_sub(inner_product_vars)
		.max(P::LOG_WIDTH)
		.min(max_total_vars);

	(subcube_vars, 1 << subcube_vars.saturating_sub(P::LOG_WIDTH))
}
