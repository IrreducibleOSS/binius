use binius::{
	challenger::HashChallenger,
	field::{
		BinaryField128b, BinaryField128bPolyval, BinaryField1b, BinaryField8b, ExtensionField,
		Field, PackedField, TowerField,
	},
	hash::GroestlHasher,
	oracle::{
		CommittedBatchSpec, CommittedId, CompositePolyOracle, MultilinearOracleSet,
		MultivariatePolyOracle,
	},
	polynomial::{
		CompositionPoly, EvaluationDomain, MultilinearComposite, MultilinearExtension,
		MultilinearPoly,
	},
	protocols::{
		sumcheck::{Error as SumcheckError, SumcheckClaim},
		test_utils::{
			full_prove_with_operating_field, full_prove_with_switchover, transform_poly,
			TestProductComposition,
		},
	},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;
use std::{fmt::Debug, iter::repeat_with, mem, sync::Arc};

fn sumcheck_128b_over_1b(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField1b>(c, "Sumcheck 128b over 1b", 8)
}

fn sumcheck_128b_over_8b(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField8b>(c, "Sumcheck 128b over 8b", 7)
}

fn sumcheck_128b_tower_basis(c: &mut Criterion) {
	sumcheck_128b_with_switchover::<BinaryField128b>(c, "Sumcheck 128b tower basis", 1)
}

fn sumcheck_128b_with_switchover<P>(c: &mut Criterion, id: &str, switchover: usize)
where
	P: PackedField + Debug,
	BinaryField128b: ExtensionField<P::Scalar>,
{
	type FTower = BinaryField128b;

	let n_multilinears = 3;
	let composition: Arc<dyn CompositionPoly<FTower>> =
		Arc::new(TestProductComposition::new(n_multilinears));

	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group(id);
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		group.throughput(Throughput::Bytes(
			(n * composition.n_vars() * mem::size_of::<FTower>()) as u64,
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
			.take(composition.n_vars())
			.collect::<Vec<_>>();
			let poly =
				MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();

			let sumcheck_claim = make_sumcheck_claim(&poly).unwrap();
			let sumcheck_witness = poly.clone();
			let prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

			b.iter(|| {
				full_prove_with_switchover(
					&sumcheck_claim,
					sumcheck_witness.clone(),
					&domain,
					prove_challenger.clone(),
					switchover,
				)
			});
		});
	}
}

fn sumcheck_128b_monomial_basis(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let n_multilinears = 3;
	let composition: Arc<dyn CompositionPoly<FTower>> =
		Arc::new(TestProductComposition::new(n_multilinears));
	let prover_composition: Arc<dyn CompositionPoly<FPolyval>> =
		Arc::new(TestProductComposition::new(n_multilinears));

	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Sumcheck 128b monomial basis (A * B * C)");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		group.throughput(Throughput::Bytes(
			(n * composition.n_vars() * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| Field::random(&mut rng))
					.take(1 << n_vars)
					.collect::<Vec<FTower>>();
				MultilinearExtension::from_values(values).unwrap()
			})
			.take(composition.n_vars())
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
				prover_composition.clone(),
				multilinears
					.iter()
					.map(|multilin| {
						transform_poly::<_, FPolyval>(multilin.to_ref())
							.unwrap()
							.specialize()
					})
					.collect(),
			)
			.unwrap();

			let sumcheck_claim = make_sumcheck_claim(&poly).unwrap();
			let prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

			b.iter(|| {
				full_prove_with_operating_field(
					&sumcheck_claim,
					poly.clone(),
					prover_poly.clone(),
					&domain,
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
	let composition: Arc<dyn CompositionPoly<FTower>> =
		Arc::new(TestProductComposition::new(n_multilinears));
	let prover_composition: Arc<dyn CompositionPoly<FPolyval>> =
		Arc::new(TestProductComposition::new(n_multilinears));

	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Sumcheck 128b monomial basis with Arc (A * B * C)");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		group.throughput(Throughput::Bytes(
			(n * composition.n_vars() * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| Field::random(&mut rng))
					.take(1 << n_vars)
					.collect::<Vec<FTower>>();
				MultilinearExtension::from_values(values).unwrap()
			})
			.take(composition.n_vars())
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
				prover_composition.clone(),
				multilinears
					.iter()
					.map(|multilin| {
						transform_poly::<_, FPolyval>(multilin.to_ref())
							.unwrap()
							.specialize()
					})
					.collect(),
			)
			.unwrap();

			let multilinears = prover_poly
				.multilinears
				.into_iter()
				.map(|multilin| {
					Arc::new(multilin) as Arc<dyn MultilinearPoly<FPolyval> + Send + Sync>
				})
				.collect::<Vec<_>>();
			let prover_poly: MultilinearComposite<
				FPolyval,
				Arc<dyn MultilinearPoly<FPolyval> + Send + Sync>,
			> = MultilinearComposite::new(n_vars, prover_composition.clone(), multilinears).unwrap();

			let sumcheck_claim = make_sumcheck_claim(&poly).unwrap();
			let prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

			b.iter(|| {
				full_prove_with_operating_field(
					&sumcheck_claim,
					poly.clone(),
					prover_poly.clone(),
					&domain,
					prove_challenger.clone(),
				)
			});
		});
	}
}

/// Given a sumcheck witness and a domain, make sumcheck claim
/// REQUIRES: Composition is the product composition
pub fn make_sumcheck_claim<F, M>(
	poly: &MultilinearComposite<F, M>,
) -> Result<SumcheckClaim<F>, SumcheckError>
where
	F: TowerField,
	M: MultilinearPoly<F>,
{
	// Setup poly_oracle
	let mut oracles = MultilinearOracleSet::new();
	let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
		round_id: 0,
		n_vars: poly.n_vars(),
		n_polys: poly.n_multilinears(),
		tower_level: F::TOWER_LEVEL,
	});
	let inner = (0..poly.n_multilinears())
		.map(|index| oracles.committed_oracle(CommittedId { batch_id, index }))
		.collect::<Vec<_>>();
	let composite_poly =
		CompositePolyOracle::new(poly.n_vars(), inner, poly.composition.clone()).unwrap();
	let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);

	// Calculate sum
	let degree = poly.composition.degree();
	if degree == 0 {
		return Err(SumcheckError::PolynomialDegreeIsZero);
	}

	let mut evals = vec![F::ZERO; poly.n_multilinears()];
	let sum = (0..1 << poly.n_vars())
		.map(|i| {
			for (evals_i, multilin) in evals.iter_mut().zip(poly.iter_multilinear_polys()) {
				*evals_i = multilin.evaluate_on_hypercube(i).unwrap();
			}
			poly.composition.evaluate_packed(&evals).unwrap()
		})
		.sum();

	let sumcheck_claim = SumcheckClaim {
		poly: poly_oracle,
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
	sumcheck_128b_over_8b
);
criterion_main!(sumcheck);
