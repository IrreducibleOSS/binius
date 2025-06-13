// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::{
	ComputeHolder, ComputeLayer, ComputeMemory, FSlice, SizedSlice, alloc::ComputeAllocator,
	cpu::CpuMemory,
};
use binius_core::{
	fiat_shamir::HasherChallenger,
	merkle_tree::{MerkleTreeProver, MerkleTreeScheme},
	piop::{
		CommitMeta, PIOPSumcheckClaim, commit, make_commit_params_with_optimal_arity, prove, verify,
	},
	polynomial::MultivariatePoly,
	protocols::fri::CommitOutput,
	transcript::ProverTranscript,
	transparent,
};
use binius_field::{BinaryField, Field, PackedExtension, PackedField, PackedFieldIndexable};
use binius_hash::groestl::Groestl256;
use binius_math::{MLEDirectAdapter, MultilinearExtension, MultilinearPoly, TowerTop};
use binius_ntt::SingleThreadedNTT;
use binius_utils::{DeserializeBytes, SerializeBytes, checked_arithmetics::checked_log_2};
use rand::{Rng, SeedableRng, rngs::StdRng};

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

fn make_sumcheck_claims<'a, F, P, M, Hal: ComputeLayer<F>, HostAllocatorType>(
	committed_multilins: &[M],
	transparent_multilins: &[FSlice<'a, F, Hal>],
	hal: &Hal,
	host_alloc: &HostAllocatorType,
) -> Vec<PIOPSumcheckClaim<F>>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P>,
	HostAllocatorType: ComputeAllocator<F, CpuMemory>,
{
	let mut sumcheck_claims = Vec::new();
	for (i, committed_multilin) in committed_multilins.iter().enumerate() {
		for (j, transparent_multilin) in transparent_multilins.iter().enumerate() {
			if committed_multilin.n_vars() == checked_log_2(transparent_multilin.len()) {
				let n_vars = committed_multilin.n_vars();

				let transparent_evals = host_alloc.alloc(transparent_multilin.len()).unwrap();

				hal.copy_d2h(*transparent_multilin, transparent_evals)
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

pub fn commit_prove_verify<FDomain, FEncode, F, P, MTScheme, HAL, ComputeHolderType>(
	mut compute_holder: ComputeHolderType,
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
	HAL: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, HAL>,
{
	let mut compute_data = compute_holder.to_data();

	let compute_data_ref = &mut compute_data;

	let hal = compute_data_ref.hal;

	let dev_alloc = &compute_data_ref.dev_alloc;
	let host_alloc = &compute_data_ref.host_alloc;

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
		.map(|mle| {
			let mut buffer = dev_alloc.alloc(1 << mle.n_vars()).unwrap();

			let evals = P::iter_slice(mle.evals()).collect::<Vec<_>>();

			hal.copy_h2d(&evals, &mut buffer).unwrap();
			HAL::DevMem::to_const(buffer)
		})
		.collect::<Vec<_>>();

	let sumcheck_claims = make_sumcheck_claims(
		&committed_multilins,
		transparent_multilins.as_slice(),
		hal,
		host_alloc,
	);

	let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	proof.message().write(&commitment);

	// If this unwraps on an out-of-memory error, allocate more above (tests are assumed to not
	// require so much memory)
	prove(
		compute_data_ref,
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
