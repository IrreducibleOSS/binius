// Copyright 2024-2025 Irreducible Inc.

use std::{
	iter::{self, repeat_with},
	sync::Arc,
};

use binius_field::{
	arch::{OptimalUnderlier128b, OptimalUnderlier512b},
	as_packed_field::{PackScalar, PackedType},
	packed::set_packed_slice,
	underlier::UnderlierType,
	BinaryField, BinaryField128b, BinaryField32b, BinaryField8b, ExtensionField, Field,
	PackedBinaryField1x128b, PackedBinaryField4x32b, PackedExtension, PackedField,
	RepackedExtension, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend, ComputationBackendExt};
use binius_math::{
	ArithExpr, CompositionPolyOS, EvaluationDomainFactory, IsomorphicEvaluationDomainFactory,
	MLEEmbeddingAdapter, MultilinearExtension, MultilinearPoly, MultilinearQuery,
};
use binius_maybe_rayon::{current_num_threads, prelude::*};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use groestl_crypto::Groestl256;
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{
	common::CompositeSumClaim,
	front_loaded::BatchVerifier as FrontLoadedBatchVerifier,
	prove::{
		batch_prove, front_loaded::BatchProver as FrontLoadedBatchProver, RegularSumcheckProver,
	},
	verify::batch_verify,
	BatchSumcheckOutput, SumcheckClaim,
};
use crate::{
	composition::index_composition,
	fiat_shamir::{CanSample, HasherChallenger},
	polynomial::{IdentityCompositionPoly, MultilinearComposite},
	protocols::{
		sumcheck::prove::SumcheckProver,
		test_utils::{AddOneComposition, TestProductComposition},
	},
	transcript::ProverTranscript,
};

#[derive(Debug, Clone)]
struct PowerComposition {
	exponent: usize,
}

impl<P: PackedField> CompositionPolyOS<P> for PowerComposition {
	fn n_vars(&self) -> usize {
		1
	}

	fn degree(&self) -> usize {
		self.exponent
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		ArithExpr::Var(0).pow(self.exponent as _)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		Ok(iter::repeat(query[0]).take(self.exponent).product())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

fn generate_random_multilinears<P>(
	mut rng: impl Rng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtension<P>>
where
	P: PackedField,
{
	repeat_with(|| {
		let mut values = repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars.saturating_sub(P::LOG_WIDTH)))
			.collect::<Vec<_>>();
		if n_vars < P::WIDTH {
			for i in n_vars..P::WIDTH {
				set_packed_slice(&mut values, i, P::Scalar::ZERO);
			}
		}

		MultilinearExtension::new(n_vars, values).unwrap()
	})
	.take(n_multilinears)
	.collect()
}

fn compute_composite_sum<P, M, Composition>(
	multilinears: &[M],
	composition: Composition,
) -> P::Scalar
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPolyOS<P>,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears.iter() {
		assert_eq!(multilinear.n_vars(), n_vars);
	}

	let multilinears = multilinears.iter().collect::<Vec<_>>();
	let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
	(0..(1 << n_vars))
		.into_par_iter()
		.map(|j| witness.evaluate_on_hypercube(j).unwrap())
		.sum()
}

fn test_prove_verify_product_helper<U, F, FDomain, FExt>(
	n_vars: usize,
	n_multilinears: usize,
	switchover_rd: usize,
) where
	U: UnderlierType + PackScalar<F> + PackScalar<FDomain> + PackScalar<FExt>,
	F: Field,
	FDomain: BinaryField,
	FExt: TowerField + ExtensionField<F> + ExtensionField<FDomain>,
	BinaryField128b: From<FExt> + Into<FExt>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let multilins =
		generate_random_multilinears::<PackedType<U, F>>(&mut rng, n_vars, n_multilinears)
			.into_iter()
			.map(MLEEmbeddingAdapter::<_, PackedType<U, FExt>, _>::from)
			.collect::<Vec<_>>();
	let product_composition = TestProductComposition::new(n_multilinears);
	let composition = AddOneComposition::new(product_composition);
	let sum = compute_composite_sum(&multilins, &composition);

	let claim = SumcheckClaim::new(
		n_vars,
		n_multilinears,
		vec![CompositeSumClaim {
			composition: &composition,
			sum,
		}],
	)
	.unwrap();

	let backend = make_portable_backend();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
	let prover = RegularSumcheckProver::<FDomain, _, _, _, _>::new(
		multilins.iter().collect(),
		[CompositeSumClaim {
			composition: &composition,
			sum,
		}],
		domain_factory,
		move |_| switchover_rd,
		&backend,
	)
	.unwrap();

	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let prover_reduced_claims =
		batch_prove(vec![prover], &mut prover_transcript).expect("failed to prove sumcheck");

	let prover_sample = CanSample::<FExt>::sample(&mut prover_transcript);
	let mut verifier_transcript = prover_transcript.into_verifier();
	let verifier_reduced_claims = batch_verify(&[claim], &mut verifier_transcript).unwrap();

	// Check that challengers are in the same state
	assert_eq!(prover_sample, CanSample::<FExt>::sample(&mut verifier_transcript));
	verifier_transcript.finalize().unwrap();

	assert_eq!(verifier_reduced_claims, prover_reduced_claims);

	// Verify that the evaluation claims are correct
	let BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	} = verifier_reduced_claims;

	let eval_point = &challenges;
	assert_eq!(multilinear_evals.len(), 1);
	assert_eq!(multilinear_evals[0].len(), n_multilinears);

	// Verify the reduced multilinear evaluations are correct
	let multilin_query = backend.multilinear_query(eval_point).unwrap();
	for (multilinear, &expected) in iter::zip(multilins, multilinear_evals[0].iter()) {
		assert_eq!(multilinear.evaluate(multilin_query.to_ref()).unwrap(), expected);
	}
}

#[test]
fn test_sumcheck_prove_verify_interaction_basic() {
	for n_vars in 2..8 {
		for n_multilinears in 1..4 {
			for switchover_rd in 0..=n_vars / 2 {
				test_prove_verify_product_helper::<
					OptimalUnderlier128b,
					BinaryField32b,
					BinaryField8b,
					BinaryField128b,
				>(n_vars, n_multilinears, switchover_rd);
			}
		}
	}
}

/// For small numbers of variables, the [`test_prove_verify_interaction_basic'] test may have so
/// few vertices to process that each vertex is processed on a separate thread. This ensures that
/// each Rayon task processes more than one vertex and that accumulation is handled correctly in
/// that case.
#[test]
fn test_prove_verify_interaction_pigeonhole_cores() {
	let n_threads = current_num_threads();
	let n_vars = log2_ceil_usize(n_threads) + 1;
	for n_multilinears in 1..4 {
		for switchover_rd in 0..=n_vars / 2 {
			test_prove_verify_product_helper::<
				OptimalUnderlier128b,
				BinaryField32b,
				BinaryField8b,
				BinaryField128b,
			>(n_vars, n_multilinears, switchover_rd);
		}
	}
}

#[test]
fn test_sumcheck_prove_verify_with_nontrivial_packing() {
	let n_vars = 8;
	let n_multilinears = 3;
	let switchover_rd = 3;

	// Using a 512-bit underlier with a 128-bit extension field means the packed field will have a
	// non-trivial packing width of 4.
	test_prove_verify_product_helper::<
		OptimalUnderlier512b,
		BinaryField32b,
		BinaryField8b,
		BinaryField128b,
	>(n_vars, n_multilinears, switchover_rd);
}

#[derive(Clone)]
struct TestSumcheckClaimShape {
	n_vars: usize,
	degree: usize,
}

fn make_test_sumcheck<'a, F, FDomain, P, PExt, Backend>(
	claim_shape: &TestSumcheckClaimShape,
	mut rng: impl Rng,
	domain_factory: impl EvaluationDomainFactory<FDomain>,
	backend: &'a Backend,
) -> (
	Vec<MultilinearExtension<P>>,
	SumcheckClaim<F, impl CompositionPolyOS<F> + Clone + 'static>,
	impl SumcheckProver<F> + 'a,
)
where
	F: Field + ExtensionField<P::Scalar> + ExtensionField<FDomain>,
	FDomain: Field,
	P: PackedField,
	PExt: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = PExt>
		+ RepackedExtension<P>
		+ PackedExtension<FDomain>,
	Backend: ComputationBackend,
{
	let TestSumcheckClaimShape {
		n_vars,
		degree: max_degree,
	} = claim_shape.clone();
	let mles = generate_random_multilinears::<P>(&mut rng, n_vars, 3);
	let multilins = mles
		.clone()
		.into_iter()
		.map(MLEEmbeddingAdapter::<_, PExt, _>::from)
		.collect::<Vec<_>>();

	let mut claim_composite_sums =
		Vec::<CompositeSumClaim<F, Arc<dyn CompositionPolyOS<F>>>>::new();
	let mut prover_composite_sums =
		Vec::<CompositeSumClaim<F, Arc<dyn CompositionPolyOS<PExt>>>>::new();

	if max_degree >= 1 {
		let identity_composition =
			index_composition(&[0, 1, 2], [0], IdentityCompositionPoly).unwrap();
		let identity_sum = compute_composite_sum(&multilins, &identity_composition);
		claim_composite_sums.push(CompositeSumClaim {
			composition: Arc::new(identity_composition.clone()),
			sum: identity_sum,
		});
		prover_composite_sums.push(CompositeSumClaim {
			composition: Arc::new(identity_composition),
			sum: identity_sum,
		});
	}
	if max_degree >= 3 {
		let product_composition = TestProductComposition::new(3);
		let product_sum = compute_composite_sum(&multilins, &product_composition);
		claim_composite_sums.push(CompositeSumClaim {
			composition: Arc::new(product_composition.clone()),
			sum: product_sum,
		});
		prover_composite_sums.push(CompositeSumClaim {
			composition: Arc::new(product_composition),
			sum: product_sum,
		});
	}

	let power_composition = index_composition(
		&[0, 1, 2],
		[1],
		PowerComposition {
			exponent: max_degree,
		},
	)
	.unwrap();
	let power_sum = compute_composite_sum(&multilins, &power_composition);
	claim_composite_sums.push(CompositeSumClaim {
		composition: Arc::new(power_composition.clone()),
		sum: power_sum,
	});
	prover_composite_sums.push(CompositeSumClaim {
		composition: Arc::new(power_composition),
		sum: power_sum,
	});

	let claim = SumcheckClaim::new(n_vars, 3, claim_composite_sums).unwrap();

	let prover = RegularSumcheckProver::<FDomain, _, _, _, _>::new(
		multilins,
		prover_composite_sums,
		domain_factory.clone(),
		|_| (n_vars / 2).max(1),
		backend,
	)
	.unwrap();

	(mles, claim, prover)
}

fn prove_verify_batch(claim_shapes: &[TestSumcheckClaimShape]) {
	type P = PackedBinaryField4x32b;
	type FDomain = BinaryField8b;
	type FE = BinaryField128b;
	type PE = PackedBinaryField1x128b;

	let mut rng = StdRng::seed_from_u64(0);

	let backend = make_portable_backend();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();

	let mut mles = Vec::with_capacity(claim_shapes.len());
	let mut claims = Vec::with_capacity(claim_shapes.len());
	let mut provers = Vec::with_capacity(claim_shapes.len());
	for claim_shape in claim_shapes {
		let (mles_i, claim, prover) = make_test_sumcheck::<FE, FDomain, P, PE, _>(
			claim_shape,
			&mut rng,
			&domain_factory,
			&backend,
		);
		mles.push(mles_i);
		claims.push(claim);
		provers.push(prover);
	}

	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let prover_output =
		batch_prove(provers, &mut prover_transcript).expect("failed to prove sumcheck");

	let prover_sample = CanSample::<FE>::sample(&mut prover_transcript);

	let mut verifier_transcript = prover_transcript.into_verifier();
	let verifier_output = batch_verify(&claims, &mut verifier_transcript).unwrap();

	assert_eq!(prover_output, verifier_output);

	// Check that challengers are in the same state
	assert_eq!(prover_sample, CanSample::<FE>::sample(&mut verifier_transcript));
}

#[test]
fn test_prove_verify_batch() {
	prove_verify_batch(&[
		TestSumcheckClaimShape {
			n_vars: 8,
			degree: 3,
		},
		TestSumcheckClaimShape {
			n_vars: 6,
			degree: 3,
		},
		TestSumcheckClaimShape {
			n_vars: 2,
			degree: 3,
		},
	]);
}

#[test]
fn test_prove_verify_batch_constant_polys() {
	prove_verify_batch(&[
		TestSumcheckClaimShape {
			n_vars: 2,
			degree: 3,
		},
		TestSumcheckClaimShape {
			n_vars: 0,
			degree: 3,
		},
	]);
}

fn prove_verify_batch_front_loaded(claim_shapes: &[TestSumcheckClaimShape]) {
	type P = PackedBinaryField4x32b;
	type FDomain = BinaryField8b;
	type FE = BinaryField128b;
	type PE = PackedBinaryField1x128b;

	let mut rng = StdRng::seed_from_u64(0);

	let backend = make_portable_backend();
	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();

	let mut mles = Vec::with_capacity(claim_shapes.len());
	let mut claims = Vec::with_capacity(claim_shapes.len());
	let mut provers = Vec::with_capacity(claim_shapes.len());
	for claim_shape in claim_shapes {
		let (mles_i, claim, prover) = make_test_sumcheck::<FE, FDomain, P, PE, _>(
			claim_shape,
			&mut rng,
			&domain_factory,
			&backend,
		);
		mles.push(mles_i);
		claims.push(claim);
		provers.push(prover);
	}

	let n_rounds = claim_shapes
		.iter()
		.map(|claim_shape| claim_shape.n_vars)
		.max()
		.unwrap_or(0);

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let mut batch_prover = FrontLoadedBatchProver::new(provers, &mut transcript).unwrap();
	for _ in 0..n_rounds {
		batch_prover
			.send_round_proof(&mut transcript.message())
			.unwrap();
		let challenge = transcript.sample();
		batch_prover.receive_challenge(challenge).unwrap();
	}
	batch_prover.finish(&mut transcript.message()).unwrap();

	let mut transcript = transcript.into_verifier();
	let mut challenges = Vec::with_capacity(n_rounds);
	let mut multilinear_evals = Vec::with_capacity(claims.len());

	let mut verifier = FrontLoadedBatchVerifier::new(&claims, &mut transcript).unwrap();
	for _ in 0..n_rounds {
		let mut writer = transcript.message();
		while let Some(claim_multilinear_evals) = verifier.try_finish_claim(&mut writer).unwrap() {
			multilinear_evals.push(claim_multilinear_evals);
		}
		verifier.receive_round_proof(&mut writer).unwrap();

		let challenge = transcript.sample();
		verifier.finish_round(challenge).unwrap();
		challenges.push(challenge);
	}

	let mut writer = transcript.message();
	while let Some(claim_multilinear_evals) = verifier.try_finish_claim(&mut writer).unwrap() {
		multilinear_evals.push(claim_multilinear_evals);
	}
	verifier.finish().unwrap();

	assert_eq!(multilinear_evals.len(), claims.len());

	for (claim_shape, mles_i, multilinear_evals_i) in izip!(claim_shapes, mles, multilinear_evals) {
		let TestSumcheckClaimShape { n_vars, .. } = claim_shape.clone();
		let query = MultilinearQuery::<PE>::expand(&challenges[..n_vars]);
		for (mle, eval) in iter::zip(mles_i, multilinear_evals_i) {
			assert_eq!(mle.evaluate(&query).unwrap(), eval);
		}
	}
}

#[test]
fn test_prove_verify_batch_front_loaded() {
	prove_verify_batch_front_loaded(&[
		TestSumcheckClaimShape {
			n_vars: 0,
			degree: 2,
		},
		TestSumcheckClaimShape {
			n_vars: 2,
			degree: 4,
		},
		TestSumcheckClaimShape {
			n_vars: 6,
			degree: 3,
		},
		TestSumcheckClaimShape {
			n_vars: 8,
			degree: 1,
		},
	]);
}
