// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::Ordering, iter::repeat_with};

use binius_field::{
	arch::OptimalUnderlier128b,
	as_packed_field::{PackScalar, PackedType},
	tower::{CanonicalTowerFamily, PackedTop, TowerFamily, TowerUnderlier},
	underlier::UnderlierType,
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{
	DefaultEvaluationDomainFactory, MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly,
	MultilinearQuery,
};
use binius_ntt::SingleThreadedNTT;
use binius_utils::{DeserializeBytes, SerializeBytes};
use rand::prelude::*;

use super::{
	common::EvalClaimSystem,
	prove,
	verify::{verify, ReducedClaim},
};
use crate::{
	fiat_shamir::HasherChallenger,
	merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
	oracle::{MultilinearOracleSet, MultilinearPolyVariant, OracleId},
	piop,
	protocols::{
		evalcheck::{subclaims::MemoizedData, EvalcheckMultilinearClaim},
		fri::CommitOutput,
	},
	ring_switch::prove::ReducedWitness,
	transcript::ProverTranscript,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};

type FExt<Tower> = <Tower as TowerFamily>::B128;

const SECURITY_BITS: usize = 32;

fn generate_multilinear<U, F, FExt>(
	mut rng: impl Rng,
	n_vars: usize,
) -> MultilinearWitness<'static, PackedType<U, FExt>>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FExt>,
	F: Field,
	FExt: ExtensionField<F>,
{
	let data = repeat_with(|| <PackedType<U, F>>::random(&mut rng))
		.take(1 << n_vars.saturating_sub(<PackedType<U, F>>::LOG_WIDTH))
		.collect::<Vec<_>>();
	let mle = MultilinearExtension::new(n_vars, data).unwrap();
	MLEEmbeddingAdapter::from(mle).upcast_arc_dyn()
}

fn generate_multilinears<U, Tower>(
	mut rng: impl Rng,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
) -> MultilinearExtensionIndex<PackedType<U, FExt<Tower>>>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
{
	let mut witness_index = MultilinearExtensionIndex::new();

	for oracle in oracles.iter() {
		if matches!(oracle.variant, MultilinearPolyVariant::Committed) {
			let n_vars = oracle.n_vars();
			let witness = match oracle.binary_tower_level() {
				0 => generate_multilinear::<U, Tower::B1, FExt<Tower>>(&mut rng, n_vars),
				3 => generate_multilinear::<U, Tower::B8, FExt<Tower>>(&mut rng, n_vars),
				4 => generate_multilinear::<U, Tower::B16, FExt<Tower>>(&mut rng, n_vars),
				5 => generate_multilinear::<U, Tower::B32, FExt<Tower>>(&mut rng, n_vars),
				6 => generate_multilinear::<U, Tower::B64, FExt<Tower>>(&mut rng, n_vars),
				7 => generate_multilinear::<U, Tower::B128, FExt<Tower>>(&mut rng, n_vars),
				_ => panic!("unsupported tower level"),
			};
			witness_index
				.update_multilin_poly([(oracle.id(), witness)])
				.unwrap();
		}
	}

	witness_index
}

fn random_eval_point<F: Field>(mut rng: impl Rng, n_vars: usize) -> Vec<F> {
	repeat_with(|| F::random(&mut rng)).take(n_vars).collect()
}

fn make_eval_claim<U, F>(
	oracle_id: OracleId,
	eval_point: Vec<F>,
	witness_index: &MultilinearExtensionIndex<PackedType<U, F>>,
) -> EvalcheckMultilinearClaim<F>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	let witness = witness_index.get_multilin_poly(oracle_id).unwrap();
	let query = MultilinearQuery::expand(&eval_point);
	let eval = witness.evaluate(query.to_ref()).unwrap();
	EvalcheckMultilinearClaim {
		id: oracle_id,
		eval_point: eval_point.into(),
		eval,
	}
}

fn check_eval_point_consistency<F: Field>(system: &EvalClaimSystem<F>) {
	for (i, claim_desc) in system.sumcheck_claim_descs.iter().enumerate() {
		let prefix_desc_idx = system.eval_claim_to_prefix_desc_index[i];
		let prefix_desc = &system.prefix_descs[prefix_desc_idx];
		let suffix_desc = &system.suffix_descs[claim_desc.suffix_desc_idx];
		assert_eq!(prefix_desc.kappa(), suffix_desc.kappa);

		let eval_point = &*system.sumcheck_claim_descs[i].eval_claim.eval_point;
		if suffix_desc.suffix.is_empty() {
			assert_eq!(&prefix_desc.prefix[..eval_point.len()], eval_point);
		} else {
			assert_eq!(
				&[prefix_desc.prefix.clone(), suffix_desc.suffix.to_vec()].concat(),
				eval_point
			);
		}
	}
}

fn setup_test_eval_claims<U, F>(
	mut rng: impl Rng,
	oracles: &MultilinearOracleSet<F>,
	witness_index: &MultilinearExtensionIndex<PackedType<U, F>>,
) -> Vec<EvalcheckMultilinearClaim<F>>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	let max_n_vars = oracles
		.iter()
		.filter(|oracle| matches!(oracle.variant, MultilinearPolyVariant::Committed))
		.map(|oracle| oracle.n_vars())
		.max()
		.unwrap();
	let eval_points = repeat_with(|| random_eval_point(&mut rng, max_n_vars))
		.take(2)
		.collect::<Vec<_>>();

	let mut eval_claims = Vec::new();
	for oracle in oracles.iter() {
		if !matches!(oracle.variant, MultilinearPolyVariant::Committed) {
			continue;
		}

		for eval_point in &eval_points {
			match oracle.n_vars().cmp(&eval_point.len()) {
				Ordering::Less => {
					// Create both back-loaded and front-loaded claims to test both shared prefixes
					// and suffixes.
					eval_claims.push(make_eval_claim::<U, F>(
						oracle.id(),
						eval_point[..oracle.n_vars()].to_vec(),
						witness_index,
					));
					eval_claims.push(make_eval_claim::<U, F>(
						oracle.id(),
						eval_point[eval_point.len() - oracle.n_vars()..].to_vec(),
						witness_index,
					));
				}
				Ordering::Equal => {
					eval_claims.push(make_eval_claim::<U, F>(
						oracle.id(),
						eval_point.clone(),
						witness_index,
					));
				}
				_ => panic!("eval_point does not have enough coordinates"),
			}
		}
	}
	eval_claims
}

fn with_test_instance_from_oracles<U, Tower, R>(
	mut rng: R,
	oracles: &MultilinearOracleSet<Tower::B128>,
	func: impl FnOnce(
		R,
		EvalClaimSystem<Tower::B128>,
		Vec<MultilinearWitness<PackedType<U, Tower::B128>>>,
	),
) where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	R: Rng,
{
	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(oracles).unwrap();

	let witness_index = generate_multilinears::<U, Tower>(&mut rng, oracles);
	let witnesses = piop::collect_committed_witnesses::<U, Tower::B128>(
		&commit_meta,
		&oracle_to_commit_index,
		oracles,
		&witness_index,
	)
	.unwrap();

	let eval_claims = setup_test_eval_claims::<U, Tower::B128>(&mut rng, oracles, &witness_index);

	// Finish setting up the test case
	let system =
		EvalClaimSystem::new(oracles, &commit_meta, &oracle_to_commit_index, &eval_claims).unwrap();
	check_eval_point_consistency(&system);

	func(rng, system, witnesses)
}

fn make_test_oracle_set<F: TowerField>() -> MultilinearOracleSet<F> {
	let mut oracles = MultilinearOracleSet::new();

	// This first one ensures that not all oracles are added in ascending order by number of packed
	// coefficients.
	oracles.add_committed(10, 3);
	oracles.add_committed(8, 3);
	oracles.add_committed(8, 5);
	oracles.add_committed(4, 3); // data is exactly one packed field element
	oracles.add_committed(2, 3); // data is less than one packed field element
	oracles.add_committed(10, 5);
	oracles
}

#[test]
fn test_prove_verify_claim_reduction_with_naive_validation() {
	type U = OptimalUnderlier128b;
	type Tower = CanonicalTowerFamily;

	let rng = StdRng::seed_from_u64(0);
	let oracles = make_test_oracle_set();

	with_test_instance_from_oracles::<U, Tower, _>(rng, &oracles, |_rng, system, witnesses| {
		let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();

		let backend = make_portable_backend();
		let ReducedWitness {
			transparents: transparent_witnesses,
			sumcheck_claims: prover_sumcheck_claims,
		} = prove::<_, _, _, Tower, _, _>(
			&system,
			&witnesses,
			&mut proof,
			MemoizedData::new(),
			&backend,
		)
		.unwrap();

		let mut proof = proof.into_verifier();
		let ReducedClaim {
			transparents: _,
			sumcheck_claims: verifier_sumcheck_claims,
		} = verify::<_, Tower, _>(&system, &mut proof).unwrap();

		assert_eq!(prover_sumcheck_claims, verifier_sumcheck_claims);

		piop::validate_sumcheck_witness(
			&witnesses,
			&transparent_witnesses,
			&prover_sumcheck_claims,
		)
		.unwrap();
	});
}

fn commit_prove_verify_piop<U, Tower, MTScheme, MTProver>(
	merkle_prover: &MTProver,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	log_inv_rate: usize,
) where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	PackedType<U, FExt<Tower>>: PackedFieldIndexable,
	FExt<Tower>: PackedTop<Tower>,
	MTScheme: MerkleTreeScheme<FExt<Tower>, Digest: SerializeBytes + DeserializeBytes>,
	MTProver: MerkleTreeProver<FExt<Tower>, Scheme = MTScheme>,
{
	let mut rng = StdRng::seed_from_u64(0);
	let merkle_scheme = merkle_prover.scheme();

	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(oracles).unwrap();

	let fri_params = piop::make_commit_params_with_optimal_arity::<_, Tower::B32, _>(
		&commit_meta,
		merkle_scheme,
		SECURITY_BITS,
		log_inv_rate,
	)
	.unwrap();
	let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len()).unwrap();

	let witness_index = generate_multilinears::<U, Tower>(&mut rng, oracles);
	let committed_multilins = piop::collect_committed_witnesses::<U, FExt<Tower>>(
		&commit_meta,
		&oracle_to_commit_index,
		oracles,
		&witness_index,
	)
	.unwrap();

	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = piop::commit(&fri_params, &ntt, merkle_prover, &committed_multilins).unwrap();

	let eval_claims = setup_test_eval_claims::<U, FExt<Tower>>(&mut rng, oracles, &witness_index);

	// Finish setting up the test case
	let system =
		EvalClaimSystem::new(oracles, &commit_meta, &oracle_to_commit_index, &eval_claims).unwrap();
	check_eval_point_consistency(&system);

	let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	proof.message().write(&commitment);

	let backend = make_portable_backend();
	let ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims,
	} = prove::<_, _, _, Tower, _, _>(
		&system,
		&committed_multilins,
		&mut proof,
		MemoizedData::new(),
		&backend,
	)
	.unwrap();

	let domain_factory = DefaultEvaluationDomainFactory::<Tower::B8>::default();
	piop::prove(
		&fri_params,
		&ntt,
		merkle_prover,
		domain_factory,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&sumcheck_claims,
		&mut proof,
		&backend,
	)
	.unwrap();

	let mut proof = proof.into_verifier();
	let commitment = proof.message().read().unwrap();

	let ReducedClaim {
		transparents,
		sumcheck_claims,
	} = verify::<_, Tower, _>(&system, &mut proof).unwrap();

	piop::verify(
		&commit_meta,
		merkle_scheme,
		&fri_params,
		&commitment,
		&transparents,
		&sumcheck_claims,
		&mut proof,
	)
	.unwrap();
}

#[test]
fn test_prove_verify_piop_integration() {
	type U = OptimalUnderlier128b;
	type Tower = CanonicalTowerFamily;

	let oracles = make_test_oracle_set();
	let log_inv_rate = 2;
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);

	commit_prove_verify_piop::<U, Tower, _, _>(&merkle_prover, &oracles, log_inv_rate);
}
