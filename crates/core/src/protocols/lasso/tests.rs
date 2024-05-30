use crate::{
	oracle::{CommittedBatchSpec, CommittedId, MultilinearOracleSet},
	polynomial::MultilinearExtension,
	protocols::lasso::{prove, verify, LassoBatch, LassoClaim, LassoWitness},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	BinaryField128b, BinaryField16b, BinaryField64b, PackedBinaryField128x1b,
	PackedBinaryField8x16b, TowerField,
};

#[test]
fn test_prove_verify_interaction() {
	type F = BinaryField128b;
	type E = BinaryField64b;
	type C = BinaryField16b;
	type PC = PackedBinaryField8x16b;
	type PB = PackedBinaryField128x1b;

	let n_vars = 10;

	// Setup witness

	let t_values = (0..)
		.map(|x| E::new(x % 137))
		.take(1 << n_vars)
		.collect::<Vec<_>>();
	let u_values = (0..)
		.map(|x| E::new((x >> 4) % 137))
		.take(1 << n_vars)
		.collect::<Vec<_>>();
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

	let witness = LassoWitness::<F, _>::new(t_polynomial, u_polynomial, u_to_t_mapping).unwrap();

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
	let mut witness_index = MultilinearWitnessIndex::new();

	let _prove_output = prove::<PC, PB, F, F, _>(
		&mut oracles.clone(),
		&mut witness_index,
		&claim,
		witness,
		&lasso_batch,
	)
	.unwrap();

	// VERIFIER
	let _verified_reduced_claim =
		verify::<C, _>(&mut oracles.clone(), &claim, &lasso_batch).unwrap();
}
