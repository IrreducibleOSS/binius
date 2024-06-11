// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::{CommittedBatchSpec, CommittedId, MultilinearOracleSet},
	polynomial::MultilinearExtension,
	protocols::msetcheck::{prove, verify, MsetcheckClaim, MsetcheckWitness},
	witness::{MultilinearWitness, MultilinearWitnessIndex},
};
use binius_field::{
	BinaryField128b, BinaryField16b, BinaryField32b, BinaryField64b, ExtensionField, Field,
	TowerField,
};
use std::iter::{successors, Step};

fn create_polynomial<F: Field + Step, FW>(
	n_vars: usize,
	stride: usize,
	reversed: bool,
) -> MultilinearWitness<'static, FW>
where
	FW: ExtensionField<F>,
{
	let mut values = successors(Some(F::ZERO), |&pred| F::forward_checked(pred, stride))
		.take(1 << n_vars)
		.collect::<Vec<_>>();

	if reversed {
		values.reverse();
	}

	MultilinearExtension::from_values(values)
		.unwrap()
		.specialize_arc_dyn()
}

#[test]
fn test_prove_verify_interaction() {
	type F = BinaryField128b;
	type F1 = BinaryField16b;
	type F2 = BinaryField32b;
	type F3 = BinaryField64b;
	let n_vars = 10;

	// Setup witness
	let t1_polynomial = create_polynomial::<F1, F>(n_vars, 13, false);
	let u1_polynomial = create_polynomial::<F1, F>(n_vars, 13, true);

	let t2_polynomial = create_polynomial::<F2, F>(n_vars, 19, false);
	let u2_polynomial = create_polynomial::<F2, F>(n_vars, 19, true);

	let t3_polynomial = create_polynomial::<F3, F>(n_vars, 29, false);
	let u3_polynomial = create_polynomial::<F3, F>(n_vars, 29, true);

	let t_polynomials = [
		t1_polynomial.clone(),
		t2_polynomial.clone(),
		t3_polynomial.clone(),
	];
	let u_polynomials = [
		u1_polynomial.clone(),
		u2_polynomial.clone(),
		u3_polynomial.clone(),
	];

	let witness = MsetcheckWitness::new(t_polynomials, u_polynomials).unwrap();

	// Setup claim
	let mut oracles = MultilinearOracleSet::<F>::new();

	let round_1_batch_1_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: 2,
		tower_level: F1::TOWER_LEVEL,
	});

	let round_1_batch_2_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: 2,
		tower_level: F2::TOWER_LEVEL,
	});

	let round_1_batch_3_id = oracles.add_committed_batch(CommittedBatchSpec {
		n_vars,
		n_polys: 2,
		tower_level: F3::TOWER_LEVEL,
	});

	let t1_oracle = oracles.committed_oracle(CommittedId {
		batch_id: round_1_batch_1_id,
		index: 0,
	});
	let u1_oracle = oracles.committed_oracle(CommittedId {
		batch_id: round_1_batch_1_id,
		index: 1,
	});

	let t2_oracle = oracles.committed_oracle(CommittedId {
		batch_id: round_1_batch_2_id,
		index: 0,
	});
	let u2_oracle = oracles.committed_oracle(CommittedId {
		batch_id: round_1_batch_2_id,
		index: 1,
	});

	let t3_oracle = oracles.committed_oracle(CommittedId {
		batch_id: round_1_batch_3_id,
		index: 0,
	});
	let u3_oracle = oracles.committed_oracle(CommittedId {
		batch_id: round_1_batch_3_id,
		index: 1,
	});

	let t_oracles = [t1_oracle, t2_oracle, t3_oracle];
	let u_oracles = [u1_oracle, u2_oracle, u3_oracle];

	let claim = MsetcheckClaim::new(t_oracles, u_oracles).unwrap();

	// challenges
	let gamma = F::new(0x123);
	let alpha = F::new(0x346);

	// PROVER
	let mut witness_index = MultilinearWitnessIndex::new();

	let prove_output =
		prove(&mut oracles.clone(), &mut witness_index, &claim, witness, gamma, Some(alpha))
			.unwrap();

	// VERIFIER
	let verified_reduced_claim = verify(&mut oracles.clone(), &claim, gamma, Some(alpha)).unwrap();

	// Consistency checks
	let alpha2 = alpha * alpha;

	for i in 0..1 << n_vars {
		let vt1 = t1_polynomial.evaluate_on_hypercube(i).unwrap();
		let vt2 = t2_polynomial.evaluate_on_hypercube(i).unwrap();
		let vt3 = t3_polynomial.evaluate_on_hypercube(i).unwrap();
		let actual_eval = gamma + vt1 + alpha * vt2 + alpha2 * vt3;
		let witness_eval = prove_output
			.prodcheck_witness
			.t_polynomial
			.evaluate_on_hypercube(i)
			.unwrap();
		assert_eq!(actual_eval, witness_eval);
	}

	for i in 0..1 << n_vars {
		let vt1 = u1_polynomial.evaluate_on_hypercube(i).unwrap();
		let vt2 = u2_polynomial.evaluate_on_hypercube(i).unwrap();
		let vt3 = u3_polynomial.evaluate_on_hypercube(i).unwrap();
		let actual_eval = gamma + vt1 + alpha * vt2 + alpha2 * vt3;
		let witness_eval = prove_output
			.prodcheck_witness
			.u_polynomial
			.evaluate_on_hypercube(i)
			.unwrap();
		assert_eq!(actual_eval, witness_eval);
	}

	assert_eq!(verified_reduced_claim.t_oracle.n_vars(), n_vars);
	assert_eq!(verified_reduced_claim.u_oracle.n_vars(), n_vars);
}
