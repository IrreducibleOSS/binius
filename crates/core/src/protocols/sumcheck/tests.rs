// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::HashChallenger,
	oracle::{CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearExtension, MultilinearExtensionSpecialized, MultilinearQuery,
	},
	protocols::{
		sumcheck::{batch_prove, batch_verify, prove, verify, SumcheckClaim, SumcheckProver},
		test_utils::{transform_poly, TestProductComposition},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	BinaryField128b, BinaryField128bPolyval, BinaryField32b, ExtensionField, Field, PackedField,
	TowerField,
};
use binius_hash::GroestlHasher;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};
use rayon::current_num_threads;
use std::iter::repeat_with;

fn generate_poly_and_sum_helper<F, FE>(
	rng: &mut StdRng,
	n_vars: usize,
	n_multilinears: usize,
) -> (
	MultilinearComposite<
		FE,
		TestProductComposition,
		MultilinearExtensionSpecialized<'static, F, FE>,
	>,
	F,
)
where
	F: Field,
	FE: ExtensionField<F>,
{
	let composition = TestProductComposition::new(n_multilinears);
	let multilinears = repeat_with(|| {
		let values = repeat_with(|| Field::random(&mut *rng))
			.take(1 << n_vars)
			.collect::<Vec<F>>();
		MultilinearExtension::from_values(values).unwrap()
	})
	.take(<_ as CompositionPoly<FE>>::n_vars(&composition))
	.collect::<Vec<_>>();

	let poly = MultilinearComposite::<FE, _, _>::new(
		n_vars,
		composition,
		multilinears
			.iter()
			.map(|multilin| multilin.clone().specialize())
			.collect(),
	)
	.unwrap();

	// Get the sum
	let sum = (0..1 << n_vars)
		.map(|i| {
			let mut prod = F::ONE;
			(0..n_multilinears).for_each(|j| {
				prod *= multilinears[j].packed_evaluate_on_hypercube(i).unwrap();
			});
			prod
		})
		.sum::<F>();

	(poly, sum)
}

fn test_prove_verify_interaction_helper(
	n_vars: usize,
	n_multilinears: usize,
	switchover_rd: usize,
) {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	let mut rng = StdRng::seed_from_u64(0);

	let (poly, sum) = generate_poly_and_sum_helper::<F, FE>(&mut rng, n_vars, n_multilinears);
	let sumcheck_witness = poly.clone();

	// Setup Claim
	let mut oracles = MultilinearOracleSet::new();
	let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		round_id: 0,
		n_vars,
		n_polys: n_multilinears,
		tower_level: F::TOWER_LEVEL,
	});
	let h = (0..n_multilinears)
		.map(|i| oracles.committed_oracle(CommittedId { batch_id, index: i }))
		.collect();
	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();

	let sumcheck_claim = SumcheckClaim {
		sum: sum.into(),
		poly: composite_poly,
		zerocheck_challenges: None,
	};

	// Setup evaluation domain
	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	let final_prove_output =
		prove(&sumcheck_claim, sumcheck_witness, &domain, challenger.clone(), |_| switchover_rd)
			.expect("failed to prove sumcheck");

	let final_verify_output =
		verify(&sumcheck_claim, final_prove_output.sumcheck_proof, challenger.clone())
			.expect("failed to verify sumcheck proof");

	assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
	assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
	assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
	assert!(final_prove_output.evalcheck_claim.is_random_point);
	assert_eq!(final_verify_output.poly.n_vars(), n_vars);

	// Verify that the evalcheck claim is correct
	let eval_point = &final_verify_output.eval_point;
	let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
	let actual = poly.evaluate(&multilin_query).unwrap();
	assert_eq!(actual, final_verify_output.eval);
}

fn test_prove_verify_interaction_with_monomial_basis_conversion_helper(
	n_vars: usize,
	n_multilinears: usize,
) {
	type F = BinaryField128b;
	type OF = BinaryField128bPolyval;
	let mut rng = StdRng::seed_from_u64(0);

	let (poly, sum) = generate_poly_and_sum_helper::<F, F>(&mut rng, n_vars, n_multilinears);

	let prover_poly = MultilinearComposite::new(
		n_vars,
		poly.composition.clone(),
		poly.multilinears
			.iter()
			.map(|multilin| {
				transform_poly::<_, OF>(multilin.as_ref().to_ref())
					.unwrap()
					.specialize::<OF>()
			})
			.collect(),
	)
	.unwrap();

	let operating_witness = prover_poly;

	// CLAIM
	let mut oracles = MultilinearOracleSet::new();
	let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		round_id: 0,
		n_vars,
		n_polys: n_multilinears,
		tower_level: F::TOWER_LEVEL,
	});
	let h = (0..n_multilinears)
		.map(|i| oracles.committed_oracle(CommittedId { batch_id, index: i }))
		.collect();
	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();
	let poly_oracle = composite_poly;

	let sumcheck_claim = SumcheckClaim {
		sum,
		poly: poly_oracle,
		zerocheck_challenges: None,
	};

	// Setup evaluation domain
	let domain = EvaluationDomain::<OF>::new_isomorphic::<F>(n_multilinears + 1).unwrap();

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let switchover_fn = |_| 3;
	let final_prove_output =
		prove(&sumcheck_claim, operating_witness, &domain, challenger.clone(), switchover_fn)
			.expect("failed to prove sumcheck");

	let final_verify_output =
		verify(&sumcheck_claim, final_prove_output.sumcheck_proof, challenger.clone())
			.expect("failed to verify sumcheck proof");

	assert_eq!(final_prove_output.evalcheck_claim.eval, final_verify_output.eval);
	assert_eq!(final_prove_output.evalcheck_claim.eval_point, final_verify_output.eval_point);
	assert_eq!(final_prove_output.evalcheck_claim.poly.n_vars(), n_vars);
	assert!(final_prove_output.evalcheck_claim.is_random_point);
	assert_eq!(final_verify_output.poly.n_vars(), n_vars);

	// Verify that the evalcheck claim is correct
	let eval_point = &final_verify_output.eval_point;
	let multilin_query = MultilinearQuery::with_full_query(eval_point).unwrap();
	let actual = poly.evaluate(&multilin_query).unwrap();
	assert_eq!(actual, final_verify_output.eval);
}

#[test]
fn test_sumcheck_prove_verify_interaction_basic() {
	for n_vars in 2..8 {
		for n_multilinears in 1..4 {
			for switchover_rd in 1..=n_vars / 2 {
				test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
			}
		}
	}
}

#[test]
fn test_prove_verify_interaction_pigeonhole_cores() {
	let n_threads = current_num_threads();
	let n_vars = log2_ceil_usize(n_threads) + 1;
	for n_multilinears in 1..4 {
		for switchover_rd in 1..=n_vars / 2 {
			test_prove_verify_interaction_helper(n_vars, n_multilinears, switchover_rd);
		}
	}
}

#[test]
fn test_prove_verify_interaction_with_monomial_basis_conversion_basic() {
	for n_vars in 2..8 {
		for n_multilinears in 1..4 {
			test_prove_verify_interaction_with_monomial_basis_conversion_helper(
				n_vars,
				n_multilinears,
			);
		}
	}
}

#[test]
fn test_prove_verify_interaction_with_monomial_basis_conversion_pigeonhole_cores() {
	let n_threads = current_num_threads();
	let n_vars = log2_ceil_usize(n_threads) + 1;
	for n_multilinears in 1..6 {
		test_prove_verify_interaction_with_monomial_basis_conversion_helper(n_vars, n_multilinears);
	}
}

#[derive(Debug, Clone)]
struct SquareComposition;

impl<F: Field> CompositionPoly<F> for SquareComposition {
	fn n_vars(&self) -> usize {
		1
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		Ok(query[0].square())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField32b;
	type FE = BinaryField128b;

	let mut rng = StdRng::seed_from_u64(0);

	let mut oracles = MultilinearOracleSet::<FE>::new();
	let mut witness_index = MultilinearWitnessIndex::<FE>::new();

	let batch_ids = [4, 6, 8].map(|n_vars| {
		oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 0,
			n_vars,
			n_polys: 1,
			tower_level: F::TOWER_LEVEL,
		})
	});

	let multilin_oracles =
		batch_ids.map(|batch_id| oracles.committed_oracle(CommittedId { batch_id, index: 0 }));

	let composites = multilin_oracles.clone().map(|poly| {
		CompositePolyOracle::new(poly.n_vars(), vec![poly], SquareComposition).unwrap()
	});

	let poly0 = MultilinearExtension::from_values(
		repeat_with(|| <F as Field>::random(&mut rng))
			.take(1 << 4)
			.collect(),
	)
	.unwrap();
	let poly1 = MultilinearExtension::from_values(
		repeat_with(|| <F as Field>::random(&mut rng))
			.take(1 << 6)
			.collect(),
	)
	.unwrap();
	let poly2 = MultilinearExtension::from_values(
		repeat_with(|| <F as Field>::random(&mut rng))
			.take(1 << 8)
			.collect(),
	)
	.unwrap();

	witness_index.set(multilin_oracles[0].id(), poly0.specialize_arc_dyn());
	witness_index.set(multilin_oracles[1].id(), poly1.specialize_arc_dyn());
	witness_index.set(multilin_oracles[2].id(), poly2.specialize_arc_dyn());

	let witnesses = composites.clone().map(|oracle| {
		MultilinearComposite::new(
			oracle.n_vars(),
			SquareComposition,
			oracle
				.inner_polys()
				.into_iter()
				.map(|multilin_oracle| witness_index.get(multilin_oracle.id()).unwrap())
				.collect(),
		)
		.unwrap()
	});

	let composite_sums = witnesses
		.iter()
		.map(|composite_witness| {
			(0..1 << composite_witness.n_vars())
				.map(|i| composite_witness.evaluate_on_hypercube(i).unwrap())
				.sum()
		})
		.collect::<Vec<_>>();

	let sumcheck_claims = composites
		.into_iter()
		.zip(composite_sums)
		.map(|(poly, sum)| SumcheckClaim {
			poly,
			sum,
			zerocheck_challenges: None,
		})
		.collect::<Vec<_>>();

	let domain = EvaluationDomain::new(3).unwrap();

	let mut witness_iter = witnesses.into_iter();
	let prover0 = SumcheckProver::new(
		&domain,
		sumcheck_claims[0].clone(),
		witness_iter.next().unwrap(),
		|_| 3,
	)
	.unwrap();
	let prover1 = SumcheckProver::new(
		&domain,
		sumcheck_claims[1].clone(),
		witness_iter.next().unwrap(),
		|_| 4,
	)
	.unwrap();
	let prover2 = SumcheckProver::new(
		&domain,
		sumcheck_claims[2].clone(),
		witness_iter.next().unwrap(),
		|_| 5,
	)
	.unwrap();

	// Setup evaluation domain
	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

	let prove_output = batch_prove([prover0, prover1, prover2], challenger.clone()).unwrap();
	let proof = prove_output.proof;
	assert_eq!(proof.rounds.len(), 8);

	let _evalcheck_claims =
		batch_verify(sumcheck_claims.iter().cloned(), proof, challenger.clone()).unwrap();
}
