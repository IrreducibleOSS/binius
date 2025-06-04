// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::{
	ComputeLayer, ComputeMemory, FSlice, SizedSlice,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
	cpu::CpuLayer,
};
use binius_field::{
	BinaryField, BinaryField8b, BinaryField16b, Field, PackedBinaryField2x128b, PackedExtension,
	PackedField, PackedFieldIndexable,
	tower::{CanonicalTowerFamily, TowerFamily},
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{MLEDirectAdapter, MultilinearExtension, MultilinearPoly};
use binius_ntt::SingleThreadedNTT;
use binius_utils::{DeserializeBytes, SerializeBytes, checked_arithmetics::strict_log_2};
use bytemuck::zeroed_vec;
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

fn make_sumcheck_claims<'a, F, P, M, Hal: ComputeLayer<F>>(
	committed_multilins: &[M],
	transparent_multilins: &[FSlice<'a, F, Hal>],
	hal: &Hal,
) -> Vec<PIOPSumcheckClaim<F>>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P>,
{
	let mut sumcheck_claims = Vec::new();
	for (i, committed_multilin) in committed_multilins.iter().enumerate() {
		for (j, transparent_multilin) in transparent_multilins.iter().enumerate() {
			if committed_multilin.n_vars() == strict_log_2(transparent_multilin.len()).unwrap_or(0)
			{
				let n_vars = committed_multilin.n_vars();

				let mut transparent_evals = zeroed_vec(transparent_multilin.len());

				hal.copy_d2h(*transparent_multilin, &mut transparent_evals)
					.unwrap();

				let sum = (0..1 << n_vars)
					.map(|v| {
						let committed_eval = committed_multilin.evaluate_on_hypercube(v).unwrap();
						committed_eval * transparent_evals[v]
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

fn commit_prove_verify<FDomain, FEncode, P, MTScheme, Tower>(
	commit_meta: &CommitMeta,
	n_transparents: usize,
	merkle_prover: &impl MerkleTreeProver<Tower::B128, Scheme = MTScheme>,
	log_inv_rate: usize,
) where
	FDomain: BinaryField,
	FEncode: BinaryField,
	P: PackedField<Scalar = Tower::B128>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>
		+ PackedExtension<Tower::B128, PackedSubfield = P>
		+ PackedFieldIndexable<Scalar = Tower::B128>,
	MTScheme: MerkleTreeScheme<Tower::B128, Digest: SerializeBytes + DeserializeBytes>,
	Tower: TowerFamily + Default,
{
	let merkle_scheme = merkle_prover.scheme();

	let hal = CpuLayer::<Tower>::default();

	let mut host_mem: Vec<Tower::B128> = zeroed_vec(1 << 7);
	let mut dev_mem: Vec<Tower::B128> = zeroed_vec(1 << 12);

	let host_alloc = HostBumpAllocator::new(&mut host_mem);
	let dev_alloc = BumpAllocator::<_, <CpuLayer<Tower> as ComputeLayer<Tower::B128>>::DevMem>::new(
		&mut dev_mem,
	);

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
		.map(|mle| {
			let mut buffer = dev_alloc.alloc(1 << mle.n_vars()).unwrap();

			let evals = P::iter_slice(mle.evals()).collect::<Vec<_>>();

			hal.copy_h2d(&evals, &mut buffer).unwrap();
			<CpuLayer<Tower> as ComputeLayer<Tower::B128>>::DevMem::into_const(buffer)
		})
		.collect::<Vec<_>>();

	let sumcheck_claims =
		make_sumcheck_claims(&committed_multilins, transparent_multilins.as_slice(), &hal);

	let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	proof.message().write(&commitment);

	prove(
		&hal,
		&dev_alloc,
		&host_alloc,
		&fri_params,
		&ntt,
		merkle_prover,
		commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		transparent_multilins,
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
		.map(|poly| poly as &dyn MultivariatePoly<Tower::B128>)
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

	commit_prove_verify::<
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
		_,
		CanonicalTowerFamily,
	>(&commit_meta, n_transparents, &merkle_prover, log_inv_rate);
}

#[test]
fn test_without_opening_claims() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 0;
	let log_inv_rate = 1;

	commit_prove_verify::<
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
		_,
		CanonicalTowerFamily,
	>(&commit_meta, n_transparents, &merkle_prover, log_inv_rate);
}

#[test]
fn test_with_one_n_vars() {
	let commit_meta = CommitMeta::with_vars([4, 4]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 1;
	let log_inv_rate = 1;

	commit_prove_verify::<
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
		_,
		CanonicalTowerFamily,
	>(&commit_meta, n_transparents, &merkle_prover, log_inv_rate);
}

#[test]
fn test_commit_prove_verify_extreme_rate() {
	let commit_meta = CommitMeta::with_vars([3, 3, 5, 6]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 8;

	commit_prove_verify::<
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
		_,
		CanonicalTowerFamily,
	>(&commit_meta, n_transparents, &merkle_prover, log_inv_rate);
}

#[test]
fn test_commit_prove_verify_small() {
	let commit_meta = CommitMeta::with_vars([4, 4, 6, 7]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
		_,
		CanonicalTowerFamily,
	>(&commit_meta, n_transparents, &merkle_prover, log_inv_rate);
}

#[test]
fn test_commit_prove_verify() {
	let commit_meta = CommitMeta::with_vars([6, 6, 8, 9]);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let n_transparents = 2;
	let log_inv_rate = 1;

	commit_prove_verify::<
		BinaryField8b,
		BinaryField16b,
		PackedBinaryField2x128b,
		_,
		CanonicalTowerFamily,
	>(&commit_meta, n_transparents, &merkle_prover, log_inv_rate);
}
