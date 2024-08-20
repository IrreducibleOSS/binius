// Copyright 2024 Ulvetanna Inc.

use super::{
	common::CompositeSumClaim,
	prove::{batch_prove, RegularSumcheckProver},
	verify::batch_verify,
	BatchSumcheckOutput, SumcheckClaim,
};
use crate::{
	challenger::{new_hasher_challenger, CanSample},
	polynomial::{
		composition::index_composition, CompositionPoly, Error as PolynomialError,
		IdentityCompositionPoly, IsomorphicEvaluationDomainFactory, MultilinearComposite,
		MultilinearExtension, MultilinearExtensionSpecialized, MultilinearPoly, MultilinearQuery,
	},
	protocols::test_utils::TestProductComposition,
};
use binius_field::{
	BinaryField128b, BinaryField32b, BinaryField8b, ExtensionField, Field, PackedField,
};
use binius_hash::GroestlHasher;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::{current_num_threads, prelude::*};
use std::{iter, iter::repeat_with, sync::Arc};

#[derive(Debug, Clone)]
struct SquareComposition;

impl<P: PackedField> CompositionPoly<P> for SquareComposition {
	fn n_vars(&self) -> usize {
		1
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		// Square each scalar value in the given packed value.
		Ok(query[0].square())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

fn generate_random_multilinears<P, PE>(
	mut rng: impl Rng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtensionSpecialized<P, PE>>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	repeat_with(|| {
		let values = repeat_with(|| P::random(&mut rng))
			.take(1 << (n_vars - P::LOG_WIDTH))
			.collect::<Vec<_>>();
		MultilinearExtension::from_values(values)
			.unwrap()
			.specialize()
	})
	.take(n_multilinears)
	.collect()
}

fn compute_composite_sum<F, P, M, Composition>(multilinears: &[M], composition: Composition) -> F
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P>,
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

fn test_prove_verify_product_helper(n_vars: usize, n_multilinears: usize, switchover_rd: usize) {
	type F = BinaryField32b;
	type FDomain = BinaryField8b;
	type FE = BinaryField128b;
	let mut rng = StdRng::seed_from_u64(0);

	let multilins = generate_random_multilinears::<F, FE>(&mut rng, n_vars, n_multilinears);
	let composition = TestProductComposition::new(n_multilinears);
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

	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
	let prover = RegularSumcheckProver::<FDomain, _, _, _>::new(
		multilins.iter().collect(),
		[CompositeSumClaim {
			composition: &composition,
			sum,
		}],
		domain_factory,
		move |_| switchover_rd,
	)
	.unwrap();

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

	let mut prover_challenger = challenger.clone();
	let (prover_reduced_claims, proof) =
		batch_prove(vec![prover], &mut prover_challenger).expect("failed to prove sumcheck");

	let mut verifier_challenger = challenger.clone();
	let verifier_reduced_claims = batch_verify(&[claim], proof, &mut verifier_challenger).unwrap();

	// Check that challengers are in the same state
	assert_eq!(
		CanSample::<FE>::sample(&mut prover_challenger),
		CanSample::<FE>::sample(&mut verifier_challenger)
	);

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
	let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
	for (multilinear, &expected) in iter::zip(multilins, multilinear_evals[0].iter()) {
		assert_eq!(multilinear.evaluate(&multilin_query).unwrap(), expected);
	}
}

#[test]
fn test_sumcheck_prove_verify_interaction_basic() {
	for n_vars in 2..8 {
		for n_multilinears in 1..4 {
			for switchover_rd in 1..=n_vars / 2 {
				test_prove_verify_product_helper(n_vars, n_multilinears, switchover_rd);
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
		for switchover_rd in 1..=n_vars / 2 {
			test_prove_verify_product_helper(n_vars, n_multilinears, switchover_rd);
		}
	}
}

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField32b;
	type FDomain = BinaryField8b;
	type FE = BinaryField128b;

	let mut rng = StdRng::seed_from_u64(0);

	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
	let (claims, provers) = [8, 6, 4]
		.map(|n_vars| {
			let multilins = generate_random_multilinears::<F, FE>(&mut rng, n_vars, 3);

			let identity_composition =
				Arc::new(index_composition(&[0, 1, 2], [0], IdentityCompositionPoly).unwrap())
					as Arc<dyn CompositionPoly<FE>>;

			let square_composition =
				Arc::new(index_composition(&[0, 1, 2], [1], SquareComposition).unwrap())
					as Arc<dyn CompositionPoly<FE>>;

			let product_composition = Arc::new(TestProductComposition::new(3));

			let identity_sum = compute_composite_sum(&multilins, &identity_composition);
			let square_sum = compute_composite_sum(&multilins, &square_composition);
			let product_sum = compute_composite_sum(&multilins, &product_composition);

			let claim = SumcheckClaim::new(
				n_vars,
				3,
				vec![
					CompositeSumClaim {
						composition: identity_composition,
						sum: identity_sum,
					},
					CompositeSumClaim {
						composition: square_composition,
						sum: square_sum,
					},
					CompositeSumClaim {
						composition: product_composition,
						sum: product_sum,
					},
				],
			)
			.unwrap();

			let prover = RegularSumcheckProver::<FDomain, _, _, _>::new(
				multilins,
				claim.composite_sums().iter().cloned(),
				domain_factory.clone(),
				|_| (n_vars / 2).max(1),
			)
			.unwrap();

			(claim, prover)
		})
		.into_iter()
		.unzip::<_, _, Vec<_>, Vec<_>>();

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

	let mut prover_challenger = challenger.clone();
	let (prover_output, proof) =
		batch_prove(provers, &mut prover_challenger).expect("failed to prove sumcheck");

	let mut verifier_challenger = challenger.clone();
	let verifier_output = batch_verify(&claims, proof, &mut verifier_challenger).unwrap();

	assert_eq!(prover_output, verifier_output);

	// Check that challengers are in the same state
	assert_eq!(
		CanSample::<FE>::sample(&mut prover_challenger),
		CanSample::<FE>::sample(&mut verifier_challenger)
	);
}
