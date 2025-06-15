// Copyright 2024-2025 Irreducible Inc.

use binius_compute::{
	ComputeData, ComputeHolder,
	cpu::{CpuLayer, layer::CpuLayerHolder},
};
use binius_compute_test_utils::ring_switch::{
	check_eval_point_consistency, commit_prove_verify_piop, generate_multilinears,
	make_test_oracle_set, setup_test_eval_claims,
};
use binius_core::{
	fiat_shamir::HasherChallenger,
	merkle_tree::BinaryMerkleTreeProver,
	oracle::MultilinearOracleSet,
	piop,
	protocols::evalcheck::subclaims::MemoizedData,
	ring_switch::{EvalClaimSystem, ReducedClaim, ReducedWitness, prove, verify},
	transcript::ProverTranscript,
	witness::MultilinearWitness,
};
use binius_field::{
	arch::OptimalUnderlier128b,
	as_packed_field::{PackScalar, PackedType},
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{B128, TowerTop, TowerUnderlier};
use rand::prelude::*;

fn with_test_instance_from_oracles<U, F, R>(
	mut rng: R,
	oracles: &MultilinearOracleSet<F>,
	func: impl FnOnce(R, EvalClaimSystem<F>, Vec<MultilinearWitness<PackedType<U, F>>>),
) where
	U: TowerUnderlier + PackScalar<F>,
	F: TowerTop,
	R: Rng,
{
	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(oracles).unwrap();

	let witness_index = generate_multilinears::<U, F>(&mut rng, oracles);
	let witnesses = piop::collect_committed_witnesses::<U, _>(
		&commit_meta,
		&oracle_to_commit_index,
		oracles,
		&witness_index,
	)
	.unwrap();

	let eval_claims = setup_test_eval_claims::<U, _>(&mut rng, oracles, &witness_index);

	// Finish setting up the test case
	let system =
		EvalClaimSystem::new(oracles, &commit_meta, &oracle_to_commit_index, &eval_claims).unwrap();
	check_eval_point_consistency(&system);

	func(rng, system, witnesses)
}

#[test]
fn test_prove_verify_claim_reduction_with_naive_validation() {
	type U = OptimalUnderlier128b;
	type F = B128;

	let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 7, 1 << 12);

	let ComputeData {
		hal,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	let rng = StdRng::seed_from_u64(0);
	let oracles = make_test_oracle_set();

	with_test_instance_from_oracles::<U, F, _>(rng, &oracles, |_rng, system, witnesses| {
		let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();

		let ReducedWitness {
			transparents: transparent_witnesses,
			sumcheck_claims: prover_sumcheck_claims,
		} = prove(&system, &witnesses, &mut proof, MemoizedData::new(), hal, &dev_alloc, &host_alloc)
			.unwrap();

		let mut proof = proof.into_verifier();
		let ReducedClaim {
			transparents: _,
			sumcheck_claims: verifier_sumcheck_claims,
		} = verify(&system, &mut proof).unwrap();

		assert_eq!(prover_sumcheck_claims, verifier_sumcheck_claims);

		piop::validate_sumcheck_witness(
			&witnesses,
			&transparent_witnesses,
			&prover_sumcheck_claims,
			hal,
		)
		.unwrap();
	});
}

#[test]
fn test_prove_verify_piop_integration() {
	type U = OptimalUnderlier128b;
	type F = B128;

	let oracles = make_test_oracle_set();
	let log_inv_rate = 2;
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);

	commit_prove_verify_piop::<U, F, _, _, CpuLayer<F>, CpuLayerHolder<F>>(
		&merkle_prover,
		&oracles,
		log_inv_rate,
		CpuLayerHolder::new,
	);
}
