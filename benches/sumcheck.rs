use binius::{
	challenger::HashChallenger,
	field::{
		BinaryField, BinaryField128b, BinaryField128bPolyval, BinaryField8b, ExtensionField, Field,
	},
	hash::GroestlHasher,
	iopoly::{CompositePoly, MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::{
		CompositionPoly, Error as PolynomialError, EvaluationDomain, MultilinearComposite,
		MultilinearPoly,
	},
	protocols::sumcheck::{
		prove::{
			prove_at_switchover, prove_before_switchover, prove_final, prove_first_round,
			prove_first_round_with_operating_field, prove_later_round_with_operating_field,
			prove_post_switchover,
		},
		setup_first_round_claim, Error as SumcheckError, SumcheckClaim, SumcheckProveOutput,
		SumcheckWitness,
	},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use p3_challenger::{CanObserve, CanSample};
use rand::thread_rng;
use std::{iter::repeat_with, mem, sync::Arc};

fn sumcheck_8b_128b(c: &mut Criterion) {
	type F = BinaryField8b;
	type FE = BinaryField128b;

	let n_multilinears = 3;
	let composition = Arc::new(TestProductComposition::new(n_multilinears));
	let composition_nvars = 3;
	let domain = EvaluationDomain::<FE>::new(n_multilinears + 1).unwrap();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Small Sumcheck 8b_128b");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		group.throughput(Throughput::Bytes((n * composition_nvars * mem::size_of::<FE>()) as u64));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = repeat_with(|| {
				let values = repeat_with(|| Field::random(&mut rng))
					.take(1 << n_vars)
					.collect::<Vec<F>>();
				MultilinearPoly::from_values(values).unwrap()
			})
			.take(composition_nvars)
			.collect::<Vec<_>>();
			let poly =
				MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();
			let sumcheck_witness = SumcheckWitness {
				polynomial: poly.clone(),
			};

			let sumcheck_claim =
				make_sumcheck_claim(n_vars, n_multilinears, sumcheck_witness.clone()).unwrap();
			let prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

			b.iter(|| {
				full_prove_wrapper(
					&sumcheck_claim,
					sumcheck_witness.clone(),
					&domain,
					prove_challenger.clone(),
				)
			});
		});
	}
}

fn classic_sumcheck_128b_monomial_basis(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let n_multilinears = 3;
	let composition: Arc<dyn CompositionPoly<FTower>> =
		Arc::new(TestProductComposition::new(n_multilinears));
	let prover_composition: Arc<dyn CompositionPoly<FPolyval>> =
		Arc::new(TestProductComposition::new(n_multilinears));

	let domain = EvaluationDomain::new(n_multilinears + 1).unwrap();

	let mut rng = thread_rng();

	let mut group = c.benchmark_group("Sumcheck 128b monomial basis");
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
				MultilinearPoly::from_values(values).unwrap()
			})
			.take(composition.n_vars())
			.collect::<Vec<_>>();
			let poly =
				MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();
			let prover_poly: MultilinearComposite<'_, FPolyval, FPolyval> =
				transform_poly(&poly, prover_composition.clone()).unwrap();

			let sumcheck_witness = SumcheckWitness {
				polynomial: poly.clone(),
			};
			let operating_witness = SumcheckWitness {
				polynomial: prover_poly,
			};

			let sumcheck_claim =
				make_sumcheck_claim(n_vars, n_multilinears, sumcheck_witness.clone()).unwrap();
			let prove_challenger = <HashChallenger<_, GroestlHasher<_>>>::new();

			b.iter(|| {
				full_classic_prove_wrapper(
					&sumcheck_claim,
					sumcheck_witness.clone(),
					operating_witness.clone(),
					&domain,
					prove_challenger.clone(),
				)
			});
		});
	}
}

/// Given a sumcheck witness and a domain, make sumcheck claim
/// REQUIRES: Composition is the product composition
pub fn make_sumcheck_claim<F, FE>(
	n_vars: usize,
	n_multilinears: usize,
	sumcheck_witness: SumcheckWitness<'_, F, FE>,
) -> Result<SumcheckClaim<F>, SumcheckError>
where
	F: BinaryField,
	FE: ExtensionField<F>,
{
	// Setup poly_oracle
	let composition = Arc::new(TestProductComposition::new(n_multilinears));
	let inner = vec![
		MultilinearPolyOracle::Committed {
			id: 0,
			n_vars,
			tower_level: F::TOWER_LEVEL,
		},
		MultilinearPolyOracle::Committed {
			id: 1,
			n_vars,
			tower_level: F::TOWER_LEVEL,
		},
		MultilinearPolyOracle::Committed {
			id: 2,
			n_vars,
			tower_level: F::TOWER_LEVEL,
		},
	];
	let composite_poly = CompositePoly::new(n_vars, inner, composition.clone()).unwrap();
	let poly_oracle = MultivariatePolyOracle::Composite(composite_poly);

	// Calculate sum
	let poly = sumcheck_witness.polynomial;
	let degree = poly.composition.degree();
	if degree == 0 {
		return Err(SumcheckError::PolynomialDegreeIsZero);
	}

	let n_multilinears = poly.composition.n_vars();

	let sum = (0..1 << n_vars)
		.map(|i| {
			let mut prod = F::ONE;
			(0..n_multilinears).for_each(|j| {
				prod *= poly.multilinears[j].evaluate_on_hypercube(i).unwrap();
			});
			prod
		})
		.sum();

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

impl<F> CompositionPoly<F> for TestProductComposition
where
	F: Field,
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
}

fn full_prove_wrapper<'a, F: Field, FE: ExtensionField<F>, CH: CanSample<FE> + CanObserve<FE>>(
	claim: &'a SumcheckClaim<F>,
	witness: SumcheckWitness<'a, F, FE>,
	domain: &'a EvaluationDomain<FE>,
	mut prove_challenger: CH,
) -> SumcheckProveOutput<'a, F, FE> {
	let n_vars = claim.poly.n_vars();
	let switchover = n_vars / 2 - 1;

	// Setup Round Claim
	let mut rd_claim = setup_first_round_claim(claim);

	// FIRST ROUND
	let mut prove_rd_output =
		prove_first_round(claim, witness.clone(), domain, switchover).unwrap();
	prove_challenger.observe_slice(&prove_rd_output.current_proof.rounds[0].coeffs);

	// BEFORE SWITCHOVER
	#[allow(clippy::needless_range_loop)]
	for i in 1..switchover {
		(prove_rd_output, rd_claim) = prove_before_switchover(
			claim,
			rd_claim,
			prove_challenger.sample(),
			prove_rd_output,
			domain,
		)
		.unwrap();
		prove_challenger.observe_slice(&prove_rd_output.current_proof.rounds[i].coeffs);
	}

	// AT SWITCHOVER
	let (mut prove_rd_output, mut rd_claim) =
		prove_at_switchover(claim, rd_claim, prove_challenger.sample(), prove_rd_output, domain)
			.unwrap();
	prove_challenger.observe_slice(&prove_rd_output.current_proof.rounds[switchover].coeffs);

	// AFTER SWITCHOVER
	#[allow(clippy::needless_range_loop)]
	for i in switchover + 1..n_vars {
		(prove_rd_output, rd_claim) = prove_post_switchover(
			claim,
			rd_claim,
			prove_challenger.sample(),
			prove_rd_output,
			domain,
		)
		.unwrap();
		prove_challenger.observe_slice(&prove_rd_output.current_proof.rounds[i].coeffs);
	}

	let sumcheck_proof = prove_rd_output.current_proof;
	let final_prove_output =
		prove_final(claim, witness, sumcheck_proof, rd_claim, prove_challenger.sample(), domain)
			.unwrap();
	final_prove_output
}

fn full_classic_prove_wrapper<
	'a,
	F: Field,
	OF: Field + Into<F> + From<F>,
	CH: CanObserve<F> + CanSample<F>,
>(
	claim: &'a SumcheckClaim<F>,
	witness: SumcheckWitness<'a, F, F>,
	operating_witness: SumcheckWitness<'a, OF, OF>,
	domain: &'a EvaluationDomain<F>,
	mut prove_challenger: CH,
) -> SumcheckProveOutput<'a, F, F> {
	let n_vars = claim.poly.n_vars();
	assert_eq!(n_vars, witness.polynomial.n_vars());

	// Setup Round Claim
	let mut rd_claim = setup_first_round_claim(claim);

	let mut prove_rd_output =
		prove_first_round_with_operating_field(claim, operating_witness, domain).unwrap();
	prove_challenger.observe_slice(&prove_rd_output.current_proof.rounds[0].coeffs);

	#[allow(clippy::needless_range_loop)]
	for i in 1..n_vars {
		(prove_rd_output, rd_claim) = prove_later_round_with_operating_field(
			claim,
			rd_claim,
			prove_challenger.sample(),
			prove_rd_output,
			domain,
		)
		.unwrap();
		prove_challenger.observe_slice(&prove_rd_output.current_proof.rounds[i].coeffs);
	}
	let sumcheck_proof = prove_rd_output.current_proof;
	let final_prove_output =
		prove_final(claim, witness, sumcheck_proof, rd_claim, prove_challenger.sample(), domain)
			.unwrap();
	final_prove_output
}

fn transform_poly<F, OF>(
	poly: &MultilinearComposite<F, F>,
	replacement_composition: Arc<dyn CompositionPoly<OF>>,
) -> Result<MultilinearComposite<'static, OF, OF>, PolynomialError>
where
	F: Field,
	OF: Field + From<F> + Into<F>,
{
	let multilinears = poly
		.iter_multilinear_polys()
		.map(|multilin| {
			let values = multilin
				.evals()
				.iter()
				.cloned()
				.map(OF::from)
				.collect::<Vec<_>>();
			MultilinearPoly::from_values(values)
		})
		.collect::<Result<Vec<_>, _>>()?;
	let ret = MultilinearComposite::new(poly.n_vars(), replacement_composition, multilinears)?;
	Ok(ret)
}

criterion_group!(sumcheck, classic_sumcheck_128b_monomial_basis, sumcheck_8b_128b);
criterion_main!(sumcheck);
