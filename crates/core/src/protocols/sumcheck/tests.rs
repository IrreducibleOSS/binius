// Copyright 2024 Ulvetanna Inc.

use crate::{
	challenger::new_hasher_challenger,
	oracle::{CompositePolyOracle, MultilinearOracleSet},
	polynomial::{
		CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearExtension,
		MultilinearExtensionSpecialized, MultilinearQuery,
	},
	protocols::{
		sumcheck::{batch_prove, batch_verify, prove, verify, SumcheckClaim},
		test_utils::{transform_poly, TestProductComposition},
	},
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	underlier::WithUnderlier, BinaryField128b, BinaryField128bPolyval, BinaryField32b,
	ExtensionField, Field, PackedBinaryField4x128b, PackedField, TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::GroestlHasher;
use binius_math::IsomorphicEvaluationDomainFactory;
use binius_utils::checked_arithmetics::checked_int_div;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, SeedableRng};
use rayon::current_num_threads;
use std::iter::repeat_with;

fn generate_poly_and_sum_helper<P, PE>(
	rng: &mut StdRng,
	n_vars: usize,
	n_multilinears: usize,
) -> (
	MultilinearComposite<PE, TestProductComposition, MultilinearExtensionSpecialized<P, PE>>,
	P::Scalar,
)
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	let composition = TestProductComposition::new(n_multilinears);
	let multilinears = repeat_with(|| {
		let values = repeat_with(|| PackedField::random(&mut *rng))
			.take(checked_int_div(1 << n_vars, P::WIDTH))
			.collect::<Vec<P>>();
		MultilinearExtension::from_values(values).unwrap()
	})
	.take(<_ as CompositionPoly<PE>>::n_vars(&composition))
	.collect::<Vec<_>>();

	let poly = MultilinearComposite::<PE, _, _>::new(
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
			let mut prod = P::Scalar::ONE;
			(0..n_multilinears).for_each(|j| {
				prod *= multilinears[j].evaluate_on_hypercube(i).unwrap();
			});
			prod
		})
		.sum();

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
	let batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let h = (0..n_multilinears)
		.map(|_| {
			let id = oracles.add_committed(batch_id);
			oracles.oracle(id)
		})
		.collect();
	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();

	let sumcheck_claim = SumcheckClaim {
		sum: sum.into(),
		poly: composite_poly,
	};

	// Setup evaluation domain
	let domain_factory = IsomorphicEvaluationDomainFactory::<F>::default();
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let backend = make_portable_backend();

	let final_prove_output = prove::<_, _, BinaryField32b, _, _>(
		&sumcheck_claim,
		sumcheck_witness,
		domain_factory,
		move |_| switchover_rd,
		challenger.clone(),
		backend.clone(),
	)
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
	let multilin_query = MultilinearQuery::with_full_query(eval_point, backend.clone()).unwrap();
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
				transform_poly::<_, OF, _>(multilin.as_ref().to_ref())
					.unwrap()
					.specialize::<OF>()
			})
			.collect(),
	)
	.unwrap();

	let operating_witness = prover_poly;

	// CLAIM
	let mut oracles = MultilinearOracleSet::new();
	let batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
	let h = (0..n_multilinears)
		.map(|_| {
			let id = oracles.add_committed(batch_id);
			oracles.oracle(id)
		})
		.collect();
	let composite_poly =
		CompositePolyOracle::new(n_vars, h, TestProductComposition::new(n_multilinears)).unwrap();
	let poly_oracle = composite_poly;

	let sumcheck_claim = SumcheckClaim {
		sum,
		poly: poly_oracle,
	};

	// Setup evaluation domain
	let domain_factory = IsomorphicEvaluationDomainFactory::<F>::default();
	let backend = make_portable_backend();

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let switchover_fn = |_| 3;
	let final_prove_output = prove::<_, OF, OF, _, _>(
		&sumcheck_claim,
		operating_witness,
		domain_factory,
		switchover_fn,
		challenger.clone(),
		backend.clone(),
	)
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
	let multilin_query = MultilinearQuery::with_full_query(eval_point, backend).unwrap();
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

#[test]
fn test_prove_verify_batch() {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	type U = <FE as WithUnderlier>::Underlier;

	let mut rng = StdRng::seed_from_u64(0);

	let mut oracles = MultilinearOracleSet::<FE>::new();
	let mut witness_index = MultilinearExtensionIndex::<U, FE>::new();

	let multilin_oracles = [4, 6, 8].map(|n_vars| {
		let batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
		let id = oracles.add_committed(batch_id);
		oracles.oracle(id)
	});

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

	witness_index
		.update_multilin_poly(vec![
			(multilin_oracles[0].id(), poly0.specialize_arc_dyn()),
			(multilin_oracles[1].id(), poly1.specialize_arc_dyn()),
			(multilin_oracles[2].id(), poly2.specialize_arc_dyn()),
		])
		.unwrap();

	let witnesses = composites.clone().map(|oracle| {
		MultilinearComposite::new(
			oracle.n_vars(),
			SquareComposition,
			oracle
				.inner_polys()
				.into_iter()
				.map(|multilin_oracle| {
					witness_index
						.get_multilin_poly(multilin_oracle.id())
						.unwrap()
				})
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
		.map(|(poly, sum)| SumcheckClaim { poly, sum })
		.collect::<Vec<_>>();

	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();

	// Setup evaluation domain
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let backend = make_portable_backend();

	let prove_output = batch_prove::<_, _, BinaryField32b, _, _>(
		sumcheck_claims.clone().into_iter().zip(witnesses),
		domain_factory,
		|_| 5,
		challenger.clone(),
		backend,
	)
	.unwrap();
	let proof = prove_output.proof;
	assert_eq!(proof.rounds.len(), 8);

	let _evalcheck_claims =
		batch_verify(sumcheck_claims.iter().cloned(), proof, challenger.clone()).unwrap();
}

#[test]
fn test_packed_sumcheck() {
	type F = BinaryField32b;
	type FE = BinaryField128b;
	type PE = PackedBinaryField4x128b;
	type U = <PE as WithUnderlier>::Underlier;
	let mut rng = StdRng::seed_from_u64(0);

	let mut oracles = MultilinearOracleSet::<FE>::new();
	let mut witness_index = MultilinearExtensionIndex::<U, FE>::new();

	let multilin_oracles = [4, 6, 8].map(|n_vars| {
		let batch_id = oracles.add_committed_batch(n_vars, F::TOWER_LEVEL);
		let id = oracles.add_committed(batch_id);
		oracles.oracle(id)
	});

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

	witness_index
		.update_multilin_poly(vec![
			(multilin_oracles[0].id(), poly0.specialize_arc_dyn()),
			(multilin_oracles[1].id(), poly1.specialize_arc_dyn()),
			(multilin_oracles[2].id(), poly2.specialize_arc_dyn()),
		])
		.unwrap();

	let witnesses = composites.clone().map(|oracle| {
		MultilinearComposite::new(
			oracle.n_vars(),
			SquareComposition,
			oracle
				.inner_polys()
				.into_iter()
				.map(|multilin_oracle| {
					witness_index
						.get_multilin_poly(multilin_oracle.id())
						.unwrap()
				})
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
		.map(|(poly, sum)| SumcheckClaim { poly, sum })
		.collect::<Vec<_>>();

	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();

	// Setup evaluation domain
	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

	let backend = make_portable_backend();
	let prove_output = batch_prove::<_, PE, BinaryField32b, _, _>(
		sumcheck_claims.clone().into_iter().zip(witnesses),
		domain_factory,
		|_| 5,
		challenger.clone(),
		backend,
	)
	.unwrap();
	let proof = prove_output.proof;
	assert_eq!(proof.rounds.len(), 8);

	let _evalcheck_claims =
		batch_verify(sumcheck_claims.iter().cloned(), proof, challenger.clone()).unwrap();
}
