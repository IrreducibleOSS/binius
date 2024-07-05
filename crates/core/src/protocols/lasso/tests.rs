// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::{CommittedBatchSpec, CommittedId, MultilinearOracleSet},
	polynomial::MultilinearExtension,
	protocols::lasso::{prove, verify, LassoBatch, LassoClaim, LassoWitness},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::PackedType, underlier::WithUnderlier, BinaryField128b, BinaryField16b,
	BinaryField64b, PackedBinaryField128x1b, PackedFieldIndexable, TowerField,
};

#[test]
fn test_prove_verify_interaction() {
	type F = BinaryField128b;
	type E = BinaryField64b;
	type C = BinaryField16b;
	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let n_vars = 10;

	// Setup witness

	let e_log_width = <PackedType<U, E>>::LOG_WIDTH;
	assert!(n_vars > e_log_width);
	let mut t_values = vec![PackedType::<U, E>::default(); 1 << (n_vars - e_log_width)];
	let mut u_values = vec![PackedType::<U, E>::default(); 1 << (n_vars - e_log_width)];

	for (i, t) in PackedType::<U, E>::unpack_scalars_mut(t_values.as_mut_slice())
		.iter_mut()
		.enumerate()
	{
		*t = E::new((i % 137) as u64);
	}

	for (i, u) in PackedType::<U, E>::unpack_scalars_mut(u_values.as_mut_slice())
		.iter_mut()
		.enumerate()
	{
		*u = E::new(((i >> 4) % 137) as u64);
	}

	let u_to_t_mapping = (0..)
		.map(|x| (x >> 4) % 137)
		.take(1 << n_vars)
		.collect::<Vec<_>>();

	let t_polynomial = MultilinearExtension::from_values(t_values)
		.unwrap()
		.specialize_arc_dyn();
	let u_polynomial = MultilinearExtension::from_values(u_values)
		.unwrap()
		.specialize_arc_dyn();

	let witness =
		LassoWitness::<PackedType<U, F>, _>::new(t_polynomial, u_polynomial, u_to_t_mapping)
			.unwrap();

	// Setup claim
	let mut oracles = MultilinearOracleSet::<F>::new();

	let lookup_batch = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: 2,
		tower_level: E::TOWER_LEVEL,
	});

	let t_oracle = oracles.committed_oracle(CommittedId {
		batch_id: lookup_batch,
		index: 0,
	});

	let u_oracle = oracles.committed_oracle(CommittedId {
		batch_id: lookup_batch,
		index: 1,
	});

	let lasso_batch = LassoBatch::new_in::<C, _>(&mut oracles, n_vars);

	let claim = LassoClaim::new(t_oracle, u_oracle).unwrap();

	// PROVER
	let witness_index = MultilinearExtensionIndex::new();

	let _prove_output =
		prove::<C, U, F, F, _>(&mut oracles.clone(), witness_index, &claim, witness, &lasso_batch)
			.unwrap();

	// VERIFIER
	let _verified_reduced_claim =
		verify::<C, _>(&mut oracles.clone(), &claim, &lasso_batch).unwrap();
}
