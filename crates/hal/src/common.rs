// Copyright 2025 Irreducible Inc.

use binius_field::{
	underlier::{UnderlierType, WithUnderlier},
	PackedField,
};

pub(crate) const MAX_SRC_SUBCUBE_LOG_BITS: usize = 13;

// TODO: document
pub(crate) fn subcube_vars_for_bits<P: PackedField>(
	subcube_src_bits: usize,
	inner_product_vars: usize,
	max_vars: usize,
) -> (usize, usize) {
	let subcube_vars = (subcube_src_bits - <P::Scalar as WithUnderlier>::Underlier::LOG_BITS)
		.saturating_sub(inner_product_vars)
		.max(P::LOG_WIDTH)
		.min(max_vars);

	(subcube_vars, 1 << subcube_vars.saturating_sub(P::LOG_WIDTH))
}
