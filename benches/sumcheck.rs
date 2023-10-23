use binius::{
	challenger::HashChallenger,
	field::{BinaryField128b, BinaryField128bPolyval, BinaryField8b, Field},
	hash::GroestlHasher,
	polynomial::{
		Error as PolynomialError, EvaluationDomain, MultilinearComposite, MultilinearPoly,
		MultivariatePoly,
	},
	protocols::sumcheck::{self, SumcheckWitness},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::thread_rng;
use std::{iter::repeat_with, mem, sync::Arc};

#[derive(Debug)]
struct ProductMultivariate {
	n_vars: usize,
}

impl<F: Field> MultivariatePoly<F, F> for ProductMultivariate {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.n_vars
	}

	fn evaluate_on_hypercube(&self, index: usize) -> Result<F, PolynomialError> {
		assert!(log2(index) < self.n_vars);
		if index == (1 << self.n_vars) - 1 {
			Ok(F::ONE)
		} else {
			Ok(F::ZERO)
		}
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		assert_eq!(query.len(), self.n_vars);
		Ok(query.iter().product())
	}

	fn evaluate_ext(&self, query: &[F]) -> Result<F, PolynomialError> {
		assert_eq!(query.len(), self.n_vars);
		Ok(query.iter().product())
	}
}

fn sumcheck_128b_monomial_basis(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let composition: Arc<dyn MultivariatePoly<FPolyval, FPolyval>> =
		Arc::new(ProductMultivariate { n_vars: 2 });

	let domain = EvaluationDomain::new(vec![FTower::ZERO, FTower::ONE, FTower::new(2)]).unwrap();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Sumcheck 128b");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		group.throughput(Throughput::Bytes(
			(n * composition.n_vars() * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| Field::random(&mut rng))
					.take(1 << n_vars)
					.collect::<Vec<FPolyval>>();
				MultilinearPoly::from_values(values).unwrap()
			})
			.take(composition.n_vars())
			.collect::<Vec<_>>();
			let poly =
				MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();

			let mut challenger = <HashChallenger<BinaryField8b, GroestlHasher>>::new();

			let sumcheck_witness = SumcheckWitness { polynomial: &poly };
			b.iter(|| sumcheck::prove::prove(sumcheck_witness, &domain, &mut challenger).unwrap());
		});
	}
}

pub fn log2(v: usize) -> usize {
	63 - (v as u64).leading_zeros() as usize
}

criterion_group!(sumcheck, sumcheck_128b_monomial_basis);
criterion_main!(sumcheck);
