// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::{
	ComputeLayer,
	alloc::{BumpAllocator, HostBumpAllocator},
};
use binius_core::{
	fiat_shamir::{Challenger, HasherChallenger},
	merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
	piop::{
		CommitMeta, PIOPSumcheckClaim, commit, make_commit_params_with_optimal_arity, prove,
		prove_compute_layer, verify,
	},
	polynomial::MultivariatePoly,
	protocols::fri::{CommitOutput, FRIParams},
	transcript::ProverTranscript,
	transparent,
};
use binius_field::{BinaryField, Field, PackedExtension, PackedField, tower::TowerFamily};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{
	DefaultEvaluationDomainFactory, EvaluationDomainFactory, MLEDirectAdapter,
	MultilinearExtension, MultilinearPoly,
};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use binius_utils::SerializeBytes;
use rand::{Rng, SeedableRng, rngs::StdRng};

const SECURITY_BITS: usize = 32;

pub struct ComputeLayerInfo<'a, F, CL>
where
	F: Field,
	CL: ComputeLayer<F>,
{
	pub compute_layer: &'a CL,
	pub host_allocator: &'a HostBumpAllocator<'a, F>,
	pub dev_allocator: &'a BumpAllocator<'a, F, CL::DevMem>,
}

pub fn commit_prove_verify_generic<'a, T, FDomain, FEncode, P, CL>(
	commit_meta_vars: Vec<usize>,
	n_transparents: usize,
	log_inv_rate: usize,
	compute_layer: Option<ComputeLayerInfo<'a, T::B128, CL>>,
) where
	T: TowerFamily,
	FDomain: BinaryField,
	FEncode: BinaryField,
	P: PackedField<Scalar = T::B128>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>
		+ PackedExtension<T::B128, PackedSubfield = P>,
	CL: ComputeLayer<T::B128>,
{
	let commit_meta = CommitMeta::with_vars(commit_meta_vars);
	let merkle_prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
	let merkle_scheme = merkle_prover.scheme();

	let fri_params = make_commit_params_with_optimal_arity::<_, FEncode, _>(
		&commit_meta,
		merkle_scheme,
		SECURITY_BITS,
		log_inv_rate,
	)
	.unwrap();
	let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len()).unwrap();

	let mut rng = StdRng::seed_from_u64(0);

	let committed_multilins = generate_multilins::<P>(commit_meta.n_multilins_by_vars(), &mut rng)
		.into_iter()
		.map(MLEDirectAdapter::from)
		.collect::<Vec<_>>();
	let CommitOutput {
		commitment,
		committed,
		codeword,
	} = commit(&fri_params, &ntt, &merkle_prover, &committed_multilins).unwrap();

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

	let domain_factory = DefaultEvaluationDomainFactory::<FDomain>::default();
	run_prove::<T, _, _, _, _, _, _, _, _, _, _>(
		&fri_params,
		&ntt,
		&merkle_prover,
		domain_factory,
		&commit_meta,
		committed,
		&codeword,
		&committed_multilins,
		&transparent_multilins,
		&sumcheck_claims,
		&mut proof,
		compute_layer,
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
		.map(|poly| poly as &dyn MultivariatePoly<T::B128>)
		.collect::<Vec<_>>();

	let commitment = proof.message().read().unwrap();
	verify(
		&commit_meta,
		merkle_scheme,
		&fri_params,
		&commitment,
		&transparent_polys,
		&sumcheck_claims,
		&mut proof,
	)
	.unwrap();
}

#[allow(clippy::too_many_arguments)]
fn run_prove<
	'a,
	T,
	FDomain,
	FEncode,
	P,
	M,
	NTT,
	DomainFactory,
	MTScheme,
	MTProver,
	Challenger_,
	CL,
>(
	fri_params: &FRIParams<T::B128, FEncode>,
	ntt: &NTT,
	merkle_prover: &MTProver,
	domain_factory: DomainFactory,
	commit_meta: &CommitMeta,
	committed: MTProver::Committed,
	codeword: &[P],
	committed_multilins: &[M],
	transparent_multilins: &[M],
	claims: &[PIOPSumcheckClaim<T::B128>],
	transcript: &mut ProverTranscript<Challenger_>,
	compute_layer_info: Option<ComputeLayerInfo<'a, T::B128, CL>>,
) -> Result<(), binius_core::piop::Error>
where
	T: TowerFamily,
	FDomain: Field,
	FEncode: BinaryField,
	P: PackedField<Scalar = T::B128>
		+ PackedExtension<T::B128, PackedSubfield = P>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>,
	M: MultilinearPoly<P> + Send + Sync,
	NTT: AdditiveNTT<FEncode> + Sync,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	MTScheme: MerkleTreeScheme<T::B128, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<T::B128, Scheme = MTScheme>,
	Challenger_: Challenger,
	CL: ComputeLayer<T::B128>,
{
	let backend = make_portable_backend();
	match compute_layer_info {
		None => prove(
			fri_params,
			ntt,
			merkle_prover,
			domain_factory,
			commit_meta,
			committed,
			codeword,
			committed_multilins,
			transparent_multilins,
			claims,
			transcript,
			&backend,
		),
		Some(ComputeLayerInfo {
			compute_layer,
			host_allocator,
			dev_allocator,
		}) => prove_compute_layer(
			fri_params,
			ntt,
			merkle_prover,
			commit_meta,
			committed,
			codeword,
			committed_multilins,
			transparent_multilins,
			claims,
			transcript,
			compute_layer,
			dev_allocator,
			host_allocator,
		),
	}
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
