// Copyright 2024 Ulvetanna Inc.
#![feature(step_trait)]

use binius_core::{
	challenger::new_hasher_challenger,
	oracle::{CompositePolyOracle, MultilinearOracleSet},
	polynomial::{
		CompositionPoly, IsomorphicEvaluationDomainFactory, MultilinearComposite,
		MultilinearExtension, MultilinearPoly,
	},
	protocols::{
		sumcheck::{prove, Error as SumcheckError, SumcheckClaim},
		test_utils::{transform_poly, TestProductComposition},
	},
};
use binius_field::{
	BinaryField128b, BinaryField128bPolyval, BinaryField1b, BinaryField32b, BinaryField8b,
	ExtensionField, Field, PackedField, TowerField,
};
use binius_hash::GroestlHasher;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;
use std::{
	fmt::Debug,
	iter::{repeat_with, Step},
	mem,
	sync::Arc,
};

fn sumcheck_128b_over_1b(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField1b, BinaryField128b>(c, "Sumcheck 128b over 1b", 8)
}

fn sumcheck_128b_over_8b(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField8b, BinaryField128b>(c, "Sumcheck 128b over 8b", 7)
}

fn sumcheck_128b_tower_basis(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField128b, BinaryField128b>(
		c,
		"Sumcheck 128b tower basis",
		1,
	)
}

fn sumcheck_128b_tower_basis_32b_domain(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField128b, BinaryField32b>(
		c,
		"Sumcheck 128b tower basis, 32b domain",
		1,
	)
}

fn sumcheck_128b_tower_basis_8b_domain(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField128b, BinaryField8b>(
		c,
		"Sumcheck 128b tower basis, 8b domain",
		1,
	)
}

fn sumcheck_128b_with_switchover<P, DomainField>(c: &mut Criterion, id: &str, switchover: usize)
where
	P: PackedField + Debug,
	DomainField: Field + Step + Debug,
	BinaryField128b: ExtensionField<P::Scalar> + ExtensionField<DomainField>,
{
	type FTower = BinaryField128b;

	let n_multilinears = 3;
	let composition = TestProductComposition::new(n_multilinears);

	let domain_factory = IsomorphicEvaluationDomainFactory::<DomainField>::default();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group(id);
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		let composition_n_vars = <_ as CompositionPoly<FTower>>::n_vars(&composition);
		group.throughput(Throughput::Bytes(
			(n * composition_n_vars * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| P::random(&mut rng))
					.take((1 << n_vars) / P::WIDTH)
					.collect::<Vec<_>>();
				MultilinearExtension::from_values(values)
					.unwrap()
					.specialize::<FTower>()
			})
			.take(composition_n_vars)
			.collect::<Vec<_>>();
			let poly =
				MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();

			let sumcheck_claim = make_sumcheck_claim(&poly).unwrap();
			let sumcheck_witness = poly.clone();
			let prove_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			b.iter(|| {
				prove::<FTower, FTower, DomainField, _>(
					&sumcheck_claim,
					sumcheck_witness.clone(),
					domain_factory.clone(),
					move |_| switchover,
					prove_challenger.clone(),
				)
			});
		});
	}
}

fn sumcheck_128b_monomial_basis(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let n_multilinears = 3;
	let composition = TestProductComposition::new(n_multilinears);

	let domain_factory = IsomorphicEvaluationDomainFactory::<FTower>::default();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Sumcheck 128b monomial basis (A * B * C)");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		let composition_n_vars = <_ as CompositionPoly<FTower>>::n_vars(&composition);
		group.throughput(Throughput::Bytes(
			(n * composition_n_vars * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| Field::random(&mut rng))
					.take(1 << n_vars)
					.collect::<Vec<FTower>>();
				MultilinearExtension::from_values(values).unwrap()
			})
			.take(composition_n_vars)
			.collect::<Vec<_>>();

			let poly = MultilinearComposite::new(
				n_vars,
				composition.clone(),
				multilinears
					.iter()
					.map(|multilin| multilin.to_ref().specialize())
					.collect(),
			)
			.unwrap();

			let prover_poly = MultilinearComposite::new(
				n_vars,
				composition.clone(),
				multilinears
					.iter()
					.map(|multilin| {
						transform_poly::<_, FPolyval, _>(multilin.to_ref())
							.unwrap()
							.specialize()
					})
					.collect(),
			)
			.unwrap();

			let sumcheck_claim = make_sumcheck_claim(&poly).unwrap();
			let prove_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			b.iter(|| {
				prove::<FTower, FPolyval, FPolyval, _>(
					&sumcheck_claim,
					prover_poly.clone(),
					domain_factory.clone(),
					|_| 1,
					prove_challenger.clone(),
				)
			});
		});
	}
}

fn sumcheck_128b_monomial_basis_with_arc(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let n_multilinears = 3;
	let composition = TestProductComposition::new(n_multilinears);

	let domain_factory = IsomorphicEvaluationDomainFactory::<FTower>::default();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Sumcheck 128b monomial basis with Arc (A * B * C)");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		let composition_n_vars = <_ as CompositionPoly<FTower>>::n_vars(&composition);
		group.throughput(Throughput::Bytes(
			(n * composition_n_vars * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| Field::random(&mut rng))
					.take(1 << n_vars)
					.collect::<Vec<FTower>>();
				MultilinearExtension::from_values(values).unwrap()
			})
			.take(composition_n_vars)
			.collect::<Vec<_>>();

			let poly = MultilinearComposite::new(
				n_vars,
				composition.clone(),
				multilinears
					.iter()
					.map(|multilin| multilin.to_ref().specialize::<FTower>())
					.collect(),
			)
			.unwrap();
			let prover_poly = MultilinearComposite::new(
				n_vars,
				composition.clone(),
				multilinears
					.iter()
					.map(|multilin| {
						transform_poly::<_, FPolyval, _>(multilin.to_ref())
							.unwrap()
							.specialize()
					})
					.collect(),
			)
			.unwrap();

			let multilinears = prover_poly
				.multilinears
				.into_iter()
				.map(|multilin| multilin.upcast_arc_dyn())
				.collect::<Vec<_>>();

			let prover_poly: MultilinearComposite<
				FPolyval,
				_,
				Arc<dyn MultilinearPoly<FPolyval> + Send + Sync>,
			> = MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();

			let sumcheck_claim = make_sumcheck_claim(&poly).unwrap();
			let prove_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			b.iter(|| {
				prove::<_, FPolyval, FPolyval, _>(
					&sumcheck_claim,
					prover_poly.clone(),
					domain_factory.clone(),
					|_| 1,
					prove_challenger.clone(),
				)
			});
		});
	}
}

/// Given a sumcheck witness poly, make sumcheck claim
pub fn make_sumcheck_claim<F, C, M>(
	poly: &MultilinearComposite<F, C, M>,
) -> Result<SumcheckClaim<F>, SumcheckError>
where
	F: TowerField,
	M: MultilinearPoly<F>,
	C: CompositionPoly<F> + Clone + 'static,
{
	// Setup poly_oracle
	let mut oracles = MultilinearOracleSet::new();
	let batch_id = oracles.add_committed_batch(poly.n_vars(), F::TOWER_LEVEL);
	let inner = (0..poly.n_multilinears())
		.map(|_| {
			let id = oracles.add_committed(batch_id);
			oracles.oracle(id)
		})
		.collect();

	let composite_poly =
		CompositePolyOracle::new(poly.n_vars(), inner, poly.composition.clone()).unwrap();

	// Calculate sum
	let degree = poly.composition.degree();
	if degree == 0 {
		return Err(SumcheckError::PolynomialDegreeIsZero);
	}

	let mut evals = vec![F::ZERO; poly.n_multilinears()];
	let sum = (0..1 << poly.n_vars())
		.map(|i| {
			for (evals_i, multilin) in evals.iter_mut().zip(&poly.multilinears) {
				*evals_i = multilin.evaluate_on_hypercube(i).unwrap();
			}
			poly.composition.evaluate(&evals).unwrap()
		})
		.sum();

	let sumcheck_claim = SumcheckClaim {
		poly: composite_poly,
		sum,
	};

	Ok(sumcheck_claim)
}

criterion_group!(
	sumcheck,
	sumcheck_128b_tower_basis,
	sumcheck_128b_monomial_basis,
	sumcheck_128b_monomial_basis_with_arc,
	sumcheck_128b_over_1b,
	sumcheck_128b_over_8b,
	sumcheck_128b_tower_basis_32b_domain,
	sumcheck_128b_tower_basis_8b_domain
);
criterion_main!(sumcheck);
