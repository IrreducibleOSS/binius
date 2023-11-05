use binius::{
	challenger::HashChallenger,
	field::{BinaryField128b, BinaryField128bPolyval, ExtensionField, Field},
	hash::GroestlHasher,
	iopoly::{CompositePoly, MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearPoly,
	},
	protocols::sumcheck::{
		self, Error as SumcheckError, SumcheckClaim, SumcheckProveOutput, SumcheckWitness,
	},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use p3_challenger::CanSample;
use rand::thread_rng;
use std::{iter::repeat_with, mem, sync::Arc};

fn sumcheck_128b_monomial_basis(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let composition: Arc<dyn CompositionPoly<FTower, FTower>> =
		Arc::new(TestProductComposition::new(2));
	let prover_composition: Arc<dyn CompositionPoly<FPolyval, FPolyval>> =
		Arc::new(TestProductComposition::new(2));

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
			let poly = MultilinearComposite::new(n_vars, prover_composition.clone(), multilinears)
				.unwrap();
			let sumcheck_witness = SumcheckWitness {
				polynomial: poly.clone(),
			};

			let inner: Vec<MultilinearPolyOracle<'_, FTower>> = vec![
				MultilinearPolyOracle::Committed {
					id: 0,
					n_vars: poly.n_vars(),
				},
				MultilinearPolyOracle::Committed {
					id: 1,
					n_vars: poly.n_vars(),
				},
			];
			let composite_poly =
				CompositePoly::new(poly.n_vars(), inner, composition.clone()).unwrap();
			let f_oracle = MultivariatePolyOracle::Composite(composite_poly);
			let sumcheck_claim = make_sumcheck_claim(f_oracle, sumcheck_witness.clone()).unwrap();
			let mut challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
			let challenges = challenger.sample_vec(n_vars);

			b.iter(|| {
				full_prove_wrapper(
					&sumcheck_claim,
					sumcheck_witness.clone(),
					&domain,
					challenges.clone(),
				)
			});
		});
	}
}

/// Given a sumcheck witness and a domain, make sumcheck claim
/// REQUIRES: f_oracle corresponds to the sumcheck_witness polynomial
pub fn make_sumcheck_claim<'a, F, OF>(
	poly_oracle: MultivariatePolyOracle<'a, F>,
	sumcheck_witness: SumcheckWitness<'a, OF>,
) -> Result<SumcheckClaim<'a, F>, SumcheckError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let poly = sumcheck_witness.polynomial;
	let degree = poly.composition.degree();
	if degree == 0 {
		return Err(SumcheckError::PolynomialDegreeIsZero);
	}

	let n_multilinears = poly.composition.n_vars();

	let mut multilinear_evals = vec![OF::ZERO; n_multilinears];
	let mut total_sum = OF::ZERO;

	for i in 0..1 << (poly.n_vars()) {
		for (j, multilin) in poly.iter_multilinear_polys().enumerate() {
			multilinear_evals[j] = multilin.evaluate_on_hypercube(i)?;
		}
		total_sum += poly.composition.evaluate(&multilinear_evals)?;
	}

	let sum = total_sum.into();
	let sumcheck_claim = SumcheckClaim {
		poly: poly_oracle,
		sum,
	};

	Ok(sumcheck_claim)
}

#[derive(Debug)]
pub struct TestProductComposition {
	arity: usize,
}

impl TestProductComposition {
	pub fn new(arity: usize) -> Self {
		Self { arity }
	}
}

impl<F, FE> CompositionPoly<F, FE> for TestProductComposition
where
	F: Field,
	FE: ExtensionField<F>,
{
	fn n_vars(&self) -> usize {
		self.arity
	}

	fn degree(&self) -> usize {
		self.arity
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		let n_vars = self.arity;
		assert_eq!(query.len(), n_vars);
		Ok(query.iter().product())
	}

	fn evaluate_ext(&self, query: &[FE]) -> Result<FE, PolynomialError> {
		CompositionPoly::<FE, FE>::evaluate(self, query)
	}
}

fn full_prove_wrapper<'a, F: Field, OF: Field + Into<F> + From<F>>(
	claim: &'a SumcheckClaim<F>,
	witness: SumcheckWitness<'a, OF>,
	domain: &'a EvaluationDomain<F>,
	challenges: Vec<F>,
) -> SumcheckProveOutput<'a, F, OF> {
	let mut current_witness = witness.clone();

	let n_vars = claim.poly.n_vars();
	assert_eq!(witness.polynomial.n_vars(), n_vars);
	assert_eq!(challenges.len(), n_vars);
	assert!(n_vars > 0);
	let prove_round_output =
		sumcheck::prove::prove_first_round(claim.clone(), current_witness, domain).unwrap();
	current_witness = prove_round_output.sumcheck_witness;
	let mut current_partial_proof = prove_round_output.sumcheck_proof;
	let mut prev_rd_reduced_claim = prove_round_output.sumcheck_reduced_claim;

	#[allow(clippy::needless_range_loop)]
	for i in 0..n_vars {
		let partial_prove_round_output = sumcheck::prove::prove_later_rounds(
			claim,
			current_witness,
			domain,
			current_partial_proof,
			challenges[i],
			prev_rd_reduced_claim,
		)
		.unwrap();
		current_witness = partial_prove_round_output.sumcheck_witness;
		current_partial_proof = partial_prove_round_output.sumcheck_proof;
		prev_rd_reduced_claim = partial_prove_round_output.sumcheck_reduced_claim;
	}
	sumcheck::prove::prove_final(claim, witness, current_partial_proof, &prev_rd_reduced_claim)
		.unwrap()
}

pub fn log2(v: usize) -> usize {
	63 - (v as u64).leading_zeros() as usize
}

criterion_group!(sumcheck, sumcheck_128b_monomial_basis);
criterion_main!(sumcheck);
