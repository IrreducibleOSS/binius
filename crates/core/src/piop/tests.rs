// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::cpu::CpuLayer;
use binius_field::{
	BinaryField, Field, PackedBinaryField2x128b, PackedExtension, PackedField, PackedFieldIndexable,
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{
	B8, B16, B128, MLEDirectAdapter, MultilinearExtension, MultilinearPoly, TowerTop,
};
use binius_ntt::SingleThreadedNTT;
use binius_utils::{DeserializeBytes, SerializeBytes};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{
	PIOPSumcheckClaim, prove,
	prove::commit,
	verify,
	verify::{CommitMeta, make_commit_params_with_optimal_arity},
};
use crate::{
	fiat_shamir::HasherChallenger,
	merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
	polynomial::MultivariatePoly,
	protocols::fri::CommitOutput,
	transcript::ProverTranscript,
	transparent,
};

const SECURITY_BITS: usize = 32;

fn generate_multilin<P>(n_vars: usize, mut rng: impl Rng) -> MultilinearExtension<P>
where
	P: PackedField,
{
	MultilinearExtension::new(
		n_vars,
		repeat_with(|| P::random(&mut rng))
			.take(1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.collect(),
	)
	.unwrap()
}

fn generate_multilins<P>(
	n_multilins_by_vars: &[usize],
	mut rng: impl Rng,
) -> Vec<MultilinearExtension<P>>
where
	P: PackedField,
{
	n_multilins_by_vars
		.iter()
		.enumerate()
		.flat_map(|(n_vars, &n_multilins)| {
			repeat_with(|| generate_multilin(n_vars, &mut rng))
				.take(n_multilins)
				.collect::<Vec<_>>()
		})
		.collect()
}

fn make_sumcheck_claims<F, P, M>(
	committed_multilins: &[M],
	transparent_multilins: &[M],
) -> Vec<PIOPSumcheckClaim<F>>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P>,
{
	let mut sumcheck_claims = Vec::new();
	for (i, committed_multilin) in committed_multilins.iter().enumerate() {
		for (j, transparent_multilin) in transparent_multilins.iter().enumerate() {
			if committed_multilin.n_vars() == transparent_multilin.n_vars() {
				let n_vars = committed_multilin.n_vars();
				let sum = (0..1 << n_vars)
					.map(|v| {
						let committed_eval = committed_multilin.evaluate_on_hypercube(v).unwrap();
						let transparent_eval =
							transparent_multilin.evaluate_on_hypercube(v).unwrap();
						committed_eval * transparent_eval
					})
					.sum();
				sumcheck_claims.push(PIOPSumcheckClaim {
					n_vars,
					committed: i,
					transparent: j,
					sum,
				});
			}
		}
	}
	sumcheck_claims
}

fn commit_prove_verify<FDomain, FEncode, F, P, MTScheme>(
	commit_meta: &CommitMeta,
	n_transparents: usize,
	merkle_prover: &impl MerkleTreeProver<F, Scheme = MTScheme>,
	log_inv_rate: usize,
) where
	FDomain: BinaryField,
	FEncode: BinaryField,
	F: TowerTop,
	P: PackedFieldIndexable<Scalar = F>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>
		+ PackedExtension<F, PackedSubfield = P>,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes + DeserializeBytes>,
{
	let merkle_scheme = merkle_prover.scheme();

	let fri_params = make_commit_params_with_optimal_arity::<_, FEncode, _>(
		commit_meta,
		merkle_scheme,
		SECURITY_BITS,
		log_inv_rate,
	)
	.unwrap();
	let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);

	let committed_multilins = generate_multilins::<P>(commit_meta.n_multilins_by_vars(), &mut rng)
		.into_iter()
		.map(MLEDirectAdapter::from)
		.collect::<Vec<_>>();
	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = commit(&fri_params, &ntt, merkle_prover, &committed_multilins).unwrap();

	let transparent_multilins_by_vars = commit_meta
		.n_multilins_by_vars()
		.iter()
		.map(|&n_committed| if n_committed == 0 { 0 } else { n_transparents })
		.collect::<Vec<_>>();

	let transparent_mles = generate_multilins::<P>(&transparent_multilins_by_vars, &mut rng);
	let transparent_multilins = transparent_mles
		.iter()
		.map(|mle| MLEDirectAdapter::from(mle.clone()))
		.collect::<Vec<_>>();

	let sumcheck_claims =
		make_sumcheck_claims(&committed_multilins, transparent_multilins.as_slice());

	let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	proof.message().write(&commitment);

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

	let hal = CpuLayer::<F>::default();
	let mut host_mem = vec![F::ZERO; host_mem_size_committed + host_mem_size_transparent];
	let mut dev_mem = vec![F::ZERO; dev_mem_size_committed + dev_mem_size_transparent - 1];
	prove(
		&hal,
		&mut host_mem,
		&mut dev_mem,
		&fri_params,
		&ntt,
		merkle_prover,
		commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&sumcheck_claims,
		&mut proof,
	)
	.unwrap();

	let mut proof = proof.into_verifier();

	let transparent_polys = transparent_mles
		.iter()
		.map(|mle| {
			transparent::MultilinearExtensionTransparent::<P, P>::from_values_and_mu(
				mle.evals().to_vec(),
				mle.n_vars(),
			)
			.unwrap()
		})
		.collect::<Vec<_>>();
	let transparent_polys = transparent_polys
		.iter()
		.map(|poly| poly as &dyn MultivariatePoly<F>)
		.collect::<Vec<_>>();

	let commitment = proof.message().read().unwrap();
	verify(
		commit_meta,
		merkle_scheme,
		&fri_params,
		&commitment,
		&transparent_polys,
		&sumcheck_claims,
		&mut proof,
	)
	.unwrap();
}

#[test]
fn test_commit_meta_total_vars() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	assert_eq!(commit_meta.total_vars(), 8);

	let commit_meta = CommitMeta::with_vars([4, 4, 6, 6, 6, 7]);
	assert_eq!(commit_meta.total_vars(), 9);
}

#[test]
fn test_with_one_poly() {
	let commit_meta = CommitMeta::with_vars([4]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 1;
	let log_inv_rate = 1;

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _>(
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_without_opening_claims() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 0;
	let log_inv_rate = 1;

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _>(
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_with_one_n_vars() {
	let commit_meta = CommitMeta::with_vars([4, 4]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 1;
	let log_inv_rate = 1;

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _>(
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_commit_prove_verify_extreme_rate() {
	let commit_meta = CommitMeta::with_vars([3, 3, 5, 6]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 8;

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _>(
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_commit_prove_verify_small() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _>(
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}

#[test]
fn test_commit_prove_verify() {
	let commit_meta = CommitMeta::with_vars([6, 6, 8, 9]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<B8, B16, B128, PackedBinaryField2x128b, _>(
		&commit_meta,
		n_transparents,
		&merkle_prover,
		log_inv_rate,
	);
}
