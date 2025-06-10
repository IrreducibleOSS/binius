// Copyright 2025 Irreducible Inc.

use binius_field::{
	Field, PackedBinaryField2x128b, PackedBinaryField256x1b, PackedField, TowerField,
	arch::OptimalUnderlier256b, tower::CanonicalTowerFamily,
};
use binius_math::{B1, B128, MLEDirectAdapter, MLEEmbeddingAdapter, MultilinearExtension};

use crate::{
	constraint_system::{
		channel::{Flush, FlushDirection, OracleOrConst},
		prove::make_masked_flush_witnesses,
	},
	oracle::MultilinearOracleSet,
	witness::MultilinearExtensionIndex,
};

#[test]
// Test that [make_masked_flush_witnesses] does not fail when n_vars < P::LOG_WIDTH
fn test_make_masked_flush_witnesses_handles_small_n_vars() {
	type P = PackedBinaryField2x128b;
	type F = B128;

	let mut oracles = MultilinearOracleSet::<F>::new();
	let mut witness = MultilinearExtensionIndex::<P>::new();

	let poly_id = oracles.add_committed(0, F::TOWER_LEVEL);
	let eval = P::from_scalars(vec![F::ONE]);
	let mle = MultilinearExtension::new(0, vec![eval]).unwrap();
	let poly = MLEDirectAdapter::from(mle).upcast_arc_dyn();

	let selector_id = oracles.add_committed(0, B1::TOWER_LEVEL);
	let eval = PackedBinaryField256x1b::from_scalars(vec![B1::ONE]);
	let mle = MultilinearExtension::new(0, vec![eval]).unwrap();
	let selector = MLEEmbeddingAdapter::<_, P>::from(mle).upcast_arc_dyn();

	witness
		.update_multilin_poly(vec![(selector_id, selector.clone()), (poly_id, poly.clone())])
		.unwrap();

	let flush = Flush {
		table_id: 0,
		log_values_per_row: 0,
		oracles: vec![OracleOrConst::<F>::Oracle(poly_id)],
		channel_id: 0,
		direction: FlushDirection::Push,
		selectors: vec![selector_id],
		multiplicity: 1,
	};

	make_masked_flush_witnesses::<OptimalUnderlier256b, CanonicalTowerFamily>(
		&oracles,
		&mut witness,
		&[poly_id],
		&[flush],
		F::ONE,
		&[F::ONE],
	)
	.unwrap();
}
