// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::Ordering, iter::repeat_with};

use binius_compute::{ComputeHolder, ComputeLayer};
use binius_core::{
	fiat_shamir::HasherChallenger,
	merkle_tree::{MerkleTreeProver, MerkleTreeScheme},
	oracle::{MultilinearOracleSet, OracleId},
	piop,
	protocols::{
		evalcheck::{EvalcheckMultilinearClaim, subclaims::MemoizedData},
		fri::{CommitOutput, FRISoundnessParams},
	},
	ring_switch::{EvalClaimSystem, ReducedClaim, ReducedWitness, prove, verify},
	transcript::ProverTranscript,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	ExtensionField, Field, PackedField, PackedFieldIndexable, TowerField,
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
};
use binius_hash::groestl::Groestl256;
use binius_math::{
	B1, B8, B16, B32, B64, B128, MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly,
	MultilinearQuery, PackedTop, TowerTop, TowerUnderlier,
};
use binius_ntt::SingleThreadedNTT;
use binius_utils::{DeserializeBytes, SerializeBytes};
use rand::prelude::*;

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

pub fn generate_multilinears<U, F>(
	mut rng: impl Rng,
	oracles: &MultilinearOracleSet<F>,
) -> MultilinearExtensionIndex<PackedType<U, F>>
where
	U: TowerUnderlier + PackScalar<F>,
	F: TowerTop,
{
	let mut witness_index = MultilinearExtensionIndex::new();

	for oracle in oracles.polys() {
		if oracle.variant.is_committed() {
			let n_vars = oracle.n_vars();
			let witness = match oracle.binary_tower_level() {
				0 => generate_multilinear::<U, B1, F>(&mut rng, n_vars),
				3 => generate_multilinear::<U, B8, F>(&mut rng, n_vars),
				4 => generate_multilinear::<U, B16, F>(&mut rng, n_vars),
				5 => generate_multilinear::<U, B32, F>(&mut rng, n_vars),
				6 => generate_multilinear::<U, B64, F>(&mut rng, n_vars),
				7 => generate_multilinear::<U, B128, F>(&mut rng, n_vars),
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

pub fn check_eval_point_consistency<F: Field>(system: &EvalClaimSystem<F>) {
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

pub fn setup_test_eval_claims<U, F>(
	mut rng: impl Rng,
	oracles: &MultilinearOracleSet<F>,
	witness_index: &MultilinearExtensionIndex<PackedType<U, F>>,
) -> Vec<EvalcheckMultilinearClaim<F>>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	let max_n_vars = oracles
		.polys()
		.filter(|oracle| oracle.variant.is_committed())
		.map(|oracle| oracle.n_vars())
		.max()
		.unwrap();
	let eval_points = repeat_with(|| random_eval_point(&mut rng, max_n_vars))
		.take(2)
		.collect::<Vec<_>>();

	let mut eval_claims = Vec::new();
	for oracle in oracles.polys() {
		if !oracle.variant.is_committed() {
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

pub fn make_test_oracle_set<F: TowerField>() -> MultilinearOracleSet<F> {
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

pub fn commit_prove_verify_piop<U, F, MTScheme, MTProver, Hal, HalHolder>(
	merkle_prover: &MTProver,
	oracles: &MultilinearOracleSet<F>,
	log_inv_rate: usize,
	create_hal_holder: impl FnOnce(usize, usize) -> HalHolder,
) where
	U: TowerUnderlier + PackScalar<F>,
	PackedType<U, F>: PackedFieldIndexable + PackedTop,
	F: TowerTop + PackedTop<Scalar = F>,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes + DeserializeBytes>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
	Hal: ComputeLayer<F>,
	HalHolder: ComputeHolder<F, Hal>,
{
	let mut rng = StdRng::seed_from_u64(0);
	let merkle_scheme = merkle_prover.scheme();

	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(oracles).unwrap();

	let fri_soundness_params = FRISoundnessParams::new(SECURITY_BITS, log_inv_rate);
	let fri_params = piop::make_commit_params_with_optimal_arity::<_, B32, _>(
		&commit_meta,
		merkle_scheme,
		&fri_soundness_params,
	)
	.unwrap();
	let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace()).unwrap();

	let witness_index = generate_multilinears::<U, _>(&mut rng, oracles);
	let committed_multilins = piop::collect_committed_witnesses::<U, _>(
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

	let eval_claims = setup_test_eval_claims::<U, _>(&mut rng, oracles, &witness_index);

	// Finish setting up the test case
	let system =
		EvalClaimSystem::new(oracles, &commit_meta, &oracle_to_commit_index, &eval_claims).unwrap();
	check_eval_point_consistency(&system);

	let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	proof.message().write(&commitment);

	let ReducedWitness {
		transparents: transparent_multilins,
		sumcheck_claims,
	} = prove(&system, &committed_multilins, &mut proof, MemoizedData::new()).unwrap();

	let host_mem_size_committed = committed_multilins.len();
	let dev_mem_size_committed = committed_multilins
		.iter()
		.map(|multilin| 1 << (multilin.n_vars() + 1))
		.sum::<usize>();

	let host_mem_size_transparent = transparent_multilins.len();
	let dev_mem_size_transparent = transparent_multilins
		.iter()
		.map(|multilin| 1 << (multilin.n_vars() + 1))
		.sum::<usize>();

	let mut compute_holder = create_hal_holder(
		host_mem_size_committed + host_mem_size_transparent,
		dev_mem_size_committed + dev_mem_size_transparent,
	);

	piop::prove(
		&mut compute_holder.to_data(),
		&fri_params,
		&ntt,
		merkle_prover,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&sumcheck_claims,
		&mut proof,
	)
	.unwrap();

	let mut proof = proof.into_verifier();
	let commitment = proof.message().read().unwrap();

	let ReducedClaim {
		transparents,
		sumcheck_claims,
	} = verify(&system, &mut proof).unwrap();

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
