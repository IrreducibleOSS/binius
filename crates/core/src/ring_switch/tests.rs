// Copyright 2024 Irreducible, Inc

use std::{
	cmp::Ordering,
	iter::{repeat_with, Step},
};

use binius_field::{
	arch::OptimalUnderlier128b,
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use binius_math::{
	DefaultEvaluationDomainFactory, MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly,
	MultilinearQuery,
};
use groestl_crypto::Groestl256;
use rand::prelude::*;

use super::{
	common::EvalClaimSystem,
	prove,
	verify::{verify, ReducedClaim},
};
use crate::{
	fiat_shamir::HasherChallenger,
	merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
	oracle::{MultilinearOracleSet, MultilinearPolyOracle},
	piop,
	protocols::{evalcheck::EvalcheckMultilinearClaim, fri::CommitOutput},
	ring_switch::prove::ReducedWitness,
	tower::{CanonicalTowerFamily, PackedTop, TowerFamily, TowerUnderlier},
	transcript::{AdviceWriter, CanRead, CanWrite, Proof, TranscriptWriter},
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
		.take(1 << (n_vars - <PackedType<U, F>>::LOG_WIDTH))
		.collect::<Vec<_>>();
	let mle = MultilinearExtension::new(n_vars, data).unwrap();
	MLEEmbeddingAdapter::from(mle).upcast_arc_dyn()
}

fn generate_multilinears<U, Tower>(
	mut rng: impl Rng,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
) -> MultilinearExtensionIndex<U, FExt<Tower>>
where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
{
	let mut witness_index = MultilinearExtensionIndex::new();

	for oracle in oracles.iter() {
		if let MultilinearPolyOracle::Committed {
			oracle_id: id,
			n_vars,
			tower_level,
			..
		} = oracle
		{
			let witness = match tower_level {
				0 => generate_multilinear::<U, Tower::B1, FExt<Tower>>(&mut rng, n_vars),
				3 => generate_multilinear::<U, Tower::B8, FExt<Tower>>(&mut rng, n_vars),
				4 => generate_multilinear::<U, Tower::B16, FExt<Tower>>(&mut rng, n_vars),
				5 => generate_multilinear::<U, Tower::B32, FExt<Tower>>(&mut rng, n_vars),
				6 => generate_multilinear::<U, Tower::B64, FExt<Tower>>(&mut rng, n_vars),
				7 => generate_multilinear::<U, Tower::B128, FExt<Tower>>(&mut rng, n_vars),
				_ => panic!("unsupported tower level"),
			};
			witness_index.update_multilin_poly([(id, witness)]).unwrap();
		}
	}

	witness_index
}

fn random_eval_point<F: Field>(mut rng: impl Rng, n_vars: usize) -> Vec<F> {
	repeat_with(|| F::random(&mut rng)).take(n_vars).collect()
}

fn make_eval_claim<U, F>(
	oracle: &MultilinearPolyOracle<F>,
	eval_point: Vec<F>,
	witness_index: &MultilinearExtensionIndex<U, F>,
) -> EvalcheckMultilinearClaim<F>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
{
	let witness = witness_index.get_multilin_poly(oracle.id()).unwrap();
	let query = MultilinearQuery::expand(&eval_point);
	let eval = witness.evaluate(query.to_ref()).unwrap();
	EvalcheckMultilinearClaim {
		poly: oracle.clone(),
		eval_point,
		eval,
	}
}

fn check_eval_point_consistency<F: Field>(system: &EvalClaimSystem<F>) {
	for (i, claim_desc) in system.sumcheck_claim_descs.iter().enumerate() {
		let prefix_desc_idx = system.eval_claim_to_prefix_desc_index[i];
		let prefix_desc = &system.prefix_descs[prefix_desc_idx];
		let suffix_desc = &system.suffix_descs[claim_desc.suffix_desc_idx];
		assert_eq!(prefix_desc.kappa(), suffix_desc.kappa);
		assert_eq!(
			[prefix_desc.prefix.clone(), suffix_desc.suffix.to_vec()].concat(),
			system.sumcheck_claim_descs[i].eval_claim.eval_point
		);
	}
}

fn setup_test_eval_claims<U, F>(
	mut rng: impl Rng,
	oracles: &MultilinearOracleSet<F>,
	witness_index: &MultilinearExtensionIndex<U, F>,
) -> Vec<EvalcheckMultilinearClaim<F>>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	let max_n_vars = oracles
		.iter()
		.filter(|oracle| matches!(oracle, MultilinearPolyOracle::Committed { .. }))
		.map(|oracle| oracle.n_vars())
		.max()
		.unwrap();
	let eval_points = repeat_with(|| random_eval_point(&mut rng, max_n_vars))
		.take(2)
		.collect::<Vec<_>>();

	let mut eval_claims = Vec::new();
	for oracle in oracles.iter() {
		if !matches!(oracle, MultilinearPolyOracle::Committed { .. }) {
			continue;
		}

		for eval_point in &eval_points {
			match oracle.n_vars().cmp(&eval_point.len()) {
				Ordering::Less => {
					// Create both back-loaded and front-loaded claims to test both shared prefixes
					// and suffixes.
					eval_claims.push(make_eval_claim(
						&oracle,
						eval_point[..oracle.n_vars()].to_vec(),
						witness_index,
					));
					eval_claims.push(make_eval_claim(
						&oracle,
						eval_point[eval_point.len() - oracle.n_vars()..].to_vec(),
						witness_index,
					));
				}
				Ordering::Equal => {
					eval_claims.push(make_eval_claim(&oracle, eval_point.clone(), witness_index));
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
	let witnesses = piop::collect_committed_witnesses(
		&commit_meta,
		&oracle_to_commit_index,
		oracles,
		&witness_index,
	)
	.unwrap();

	let eval_claims = setup_test_eval_claims(&mut rng, oracles, &witness_index);

	// Finish setting up the test case
	let system = EvalClaimSystem::new(&commit_meta, oracle_to_commit_index, &eval_claims).unwrap();
	check_eval_point_consistency(&system);

	func(rng, system, witnesses)
}

fn make_test_oracle_set<F: TowerField>() -> MultilinearOracleSet<F> {
	let mut oracles = MultilinearOracleSet::new();

	// This first one ensures that not all oracles are added in ascending order by number of packed
	// coefficients.
	let batch_3_0 = oracles.add_committed_batch(10, 3);
	let _ = oracles.add_committed(batch_3_0);

	let batch_3_1 = oracles.add_committed_batch(8, 3);
	let _ = oracles.add_committed(batch_3_1);

	let batch_5_0 = oracles.add_committed_batch(8, 5);
	let _ = oracles.add_committed(batch_5_0);

	let batch_5_1 = oracles.add_committed_batch(10, 5);
	let _ = oracles.add_committed(batch_5_1);

	oracles
}

#[test]
fn test_prove_verify_claim_reduction_with_naive_validation() {
	type U = OptimalUnderlier128b;
	type Tower = CanonicalTowerFamily;

	let rng = StdRng::seed_from_u64(0);
	let oracles = make_test_oracle_set();

	with_test_instance_from_oracles::<U, Tower, _>(rng, &oracles, |_rng, system, witnesses| {
		let mut proof = Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};

		let backend = make_portable_backend();
		let ReducedWitness {
			transparents: transparent_witnesses,
			sumcheck_claims: prover_sumcheck_claims,
		} = prove::<_, _, _, Tower, _, _, _>(&system, &witnesses, &mut proof, &backend).unwrap();

		let mut proof = proof.into_verifier();
		let ReducedClaim {
			transparents: _,
			sumcheck_claims: verifier_sumcheck_claims,
		} = verify::<_, Tower, _, _>(&system, &mut proof).unwrap();

		assert_eq!(prover_sumcheck_claims, verifier_sumcheck_claims);

		piop::validate_sumcheck_witness(
			&witnesses,
			&transparent_witnesses,
			&prover_sumcheck_claims,
		)
		.unwrap();
	});
}

fn commit_prove_verify_piop<U, Tower, MTScheme, MTProver, Digest>(
	merkle_prover: &MTProver,
	oracles: &MultilinearOracleSet<FExt<Tower>>,
	log_inv_rate: usize,
) where
	U: TowerUnderlier<Tower>,
	Tower: TowerFamily,
	Tower::B8: Step,
	PackedType<U, FExt<Tower>>: PackedFieldIndexable,
	FExt<Tower>: PackedTop<Tower>,
	MTScheme: MerkleTreeScheme<FExt<Tower>, Digest = Digest, Proof = Vec<Digest>>,
	MTProver: MerkleTreeProver<FExt<Tower>, Scheme = MTScheme>,
	Digest: PackedField<Scalar: TowerField>,
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

	let witness_index = generate_multilinears::<U, Tower>(&mut rng, oracles);
	let committed_multilins = piop::collect_committed_witnesses(
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
	} = piop::commit(&fri_params, merkle_prover, &committed_multilins).unwrap();

	let eval_claims = setup_test_eval_claims(&mut rng, oracles, &witness_index);

	// Finish setting up the test case
	let system = EvalClaimSystem::new(&commit_meta, oracle_to_commit_index, &eval_claims).unwrap();
	check_eval_point_consistency(&system);

	let mut proof = Proof {
		transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
		advice: AdviceWriter::default(),
	};
	proof.transcript.write_packed(commitment);

	let backend = make_portable_backend();
	let ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims,
	} = prove::<_, _, _, Tower, _, _, _>(&system, &committed_multilins, &mut proof, &backend).unwrap();

	let domain_factory = DefaultEvaluationDomainFactory::<Tower::B8>::default();
	piop::prove(
		&fri_params,
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
	let commitment = proof.transcript.read_packed().unwrap();

	let ReducedClaim {
		transparents,
		sumcheck_claims,
	} = verify::<_, Tower, _, _>(&system, &mut proof).unwrap();

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
	let merkle_prover =
		BinaryMerkleTreeProver::<_, GroestlHasher<_>, _>::new(GroestlDigestCompression::default());

	commit_prove_verify_piop::<U, Tower, _, _, _>(&merkle_prover, &oracles, log_inv_rate);
}
