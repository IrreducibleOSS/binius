// Copyright 2024 Ulvetanna Inc.
#![feature(step_trait)]

use binius_core::{
	challenger::new_hasher_challenger,
	oracle::{CompositePolyOracle, MultilinearOracleSet},
	polynomial::{CompositionPoly, MultilinearComposite, MultilinearExtension, MultilinearPoly},
	protocols::{
		test_utils::{transform_poly, TestProductComposition},
		zerocheck::{prove, Error as ZerocheckError, ZerocheckClaim},
	},
	Step,
};
use binius_field::{
	BinaryField128b, BinaryField128bPolyval, BinaryField1b, BinaryField32b, BinaryField8b,
	ExtensionField, Field, PackedExtension, PackedField, TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::GroestlHasher;
use binius_math::IsomorphicEvaluationDomainFactory;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::ThreadRng, thread_rng};
use std::{fmt::Debug, mem, sync::Arc};

fn zerocheck_128b_over_1b(c: &mut Criterion) {
	zerocheck_128b_with_switchover::<BinaryField1b, BinaryField128b>(c, "Zerocheck 128b over 1b", 8)
}

fn zerocheck_128b_over_8b(c: &mut Criterion) {
	zerocheck_128b_with_switchover::<BinaryField8b, BinaryField128b>(c, "Zerocheck 128b over 8b", 7)
}

fn zerocheck_128b_tower_basis(c: &mut Criterion) {
	zerocheck_128b_with_switchover::<BinaryField128b, BinaryField128b>(
		c,
		"Zerocheck 128b tower basis, 128b domain",
		1,
	)
}

fn zerocheck_128b_tower_basis_32b_domain(c: &mut Criterion) {
	zerocheck_128b_with_switchover::<BinaryField128b, BinaryField32b>(
		c,
		"Zerocheck 128b tower basis, 32b domain",
		1,
	)
}

fn zerocheck_128b_tower_basis_8b_domain(c: &mut Criterion) {
	zerocheck_128b_with_switchover::<BinaryField128b, BinaryField8b>(
		c,
		"Zerocheck 128b tower basis, 8b domain",
		1,
	)
}

// Helper function that makes n_multilinears MultilinearExtensions in such a way that over
// the product composition, any hypercube evaluation will be zero.
fn make_multilinears<P: PackedField>(
	rng: &mut ThreadRng,
	n_vars: usize,
	n_multilinears: usize,
) -> Vec<MultilinearExtension<P>> {
	if (1 << n_vars) % P::WIDTH != 0 {
		panic!("(1 << n_vars) must be divisible by P::WIDTH");
	}

	let n_packed_values = (1 << n_vars) / P::WIDTH;
	let mut multilinears = Vec::with_capacity(n_multilinears);
	for j in 0..n_multilinears {
		let mut values = vec![P::default(); n_packed_values];
		for i in 0..(1 << n_vars) {
			if i % n_multilinears != j {
				let packed_idx = i / P::WIDTH;
				let scalar_idx = i % P::WIDTH;
				if i % n_multilinears != j {
					values[packed_idx].set(scalar_idx, <P::Scalar as Field>::random(&mut *rng));
				}
			}
		}
		let multilinear = MultilinearExtension::from_values(values).unwrap();
		multilinears.push(multilinear);
	}
	multilinears
}

fn zerocheck_128b_with_switchover<P, FS>(c: &mut Criterion, id: &str, switchover: usize)
where
	P: PackedField + Debug,
	FS: Field + Step + Debug,
	BinaryField128b: ExtensionField<P::Scalar> + ExtensionField<FS> + PackedExtension<FS>,
{
	type FTower = BinaryField128b;

	let n_multilinears = 3;
	let composition = TestProductComposition::new(n_multilinears);

	let domain_factory = IsomorphicEvaluationDomainFactory::<FS>::default();
	let backend = make_portable_backend();

	let mut rng = thread_rng();
	let mixing_challenge = <BinaryField128b as Field>::random(&mut rng);

	let mut group = c.benchmark_group(id);
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		let composition_n_vars = <_ as CompositionPoly<FTower>>::n_vars(&composition);
		group.throughput(Throughput::Bytes(
			(n * composition_n_vars * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = make_multilinears::<P>(&mut rng, n_vars, composition_n_vars);
			let multilinears = multilinears
				.into_iter()
				.map(|m| m.specialize_arc_dyn::<FTower>())
				.collect::<Vec<_>>();

			let poly =
				MultilinearComposite::new(n_vars, composition.clone(), multilinears).unwrap();

			let zerocheck_claim = make_zerocheck_claim(&poly).unwrap();
			let zerocheck_witness = poly;
			let prove_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			b.iter(|| {
				prove::<FTower, FTower, FS, _, _>(
					&zerocheck_claim,
					zerocheck_witness.clone(),
					domain_factory.clone(),
					move |_| switchover,
					mixing_challenge,
					prove_challenger.clone(),
					backend.clone(),
				)
			});
		});
	}
}

fn zerocheck_128b_monomial_basis(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let n_multilinears = 3;
	let composition = TestProductComposition::new(n_multilinears);

	let domain_factory = IsomorphicEvaluationDomainFactory::<FTower>::default();
	let backend = make_portable_backend();

	let mut rng = thread_rng();
	let mixing_challenge = <BinaryField128b as Field>::random(&mut rng);

	let mut group = c.benchmark_group("Zerocheck 128b monomial basis (A * B * C)");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		let composition_n_vars = <_ as CompositionPoly<FTower>>::n_vars(&composition);
		group.throughput(Throughput::Bytes(
			(n * composition_n_vars * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = make_multilinears::<FTower>(&mut rng, n_vars, composition_n_vars);

			let poly = MultilinearComposite::new(
				n_vars,
				composition.clone(),
				multilinears
					.iter()
					.map(|multilin| multilin.to_ref().specialize())
					.collect(),
			)
			.unwrap();

			let zerocheck_witness = MultilinearComposite::new(
				n_vars,
				composition.clone(),
				multilinears
					.iter()
					.map(|multilin| {
						transform_poly::<_, FPolyval, _>(multilin.to_ref())
							.unwrap()
							.specialize_arc_dyn()
					})
					.collect(),
			)
			.unwrap();

			let zerocheck_claim = make_zerocheck_claim(&poly).unwrap();
			let prove_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			b.iter(|| {
				prove::<FTower, FPolyval, FPolyval, _, _>(
					&zerocheck_claim,
					zerocheck_witness.clone(),
					domain_factory.clone(),
					|_| 1,
					mixing_challenge,
					prove_challenger.clone(),
					backend.clone(),
				)
			});
		});
	}
}

fn zerocheck_128b_monomial_basis_with_arc(c: &mut Criterion) {
	type FTower = BinaryField128b;
	type FPolyval = BinaryField128bPolyval;

	let n_multilinears = 3;
	let composition = TestProductComposition::new(n_multilinears);

	let domain_factory = IsomorphicEvaluationDomainFactory::<FTower>::default();
	let backend = make_portable_backend();

	let mut rng = thread_rng();
	let mixing_challenge = <BinaryField128b as Field>::random(&mut rng);

	let mut group = c.benchmark_group("Zerocheck 128b monomial basis with Arc (A * B * C)");
	for &n_vars in [13, 14, 15, 16].iter() {
		let n = 1 << n_vars;
		let composition_n_vars = <_ as CompositionPoly<FTower>>::n_vars(&composition);
		group.throughput(Throughput::Bytes(
			(n * composition_n_vars * mem::size_of::<FTower>()) as u64,
		));
		group.bench_with_input(BenchmarkId::from_parameter(n_vars), &n_vars, |b, &n_vars| {
			let multilinears = make_multilinears::<FTower>(&mut rng, n_vars, composition_n_vars);

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

			let zerocheck_claim = make_zerocheck_claim(&poly).unwrap();
			let prove_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			b.iter(|| {
				prove::<_, FPolyval, FPolyval, _, _>(
					&zerocheck_claim,
					prover_poly.clone(),
					domain_factory.clone(),
					|_| 1,
					mixing_challenge,
					prove_challenger.clone(),
					backend.clone(),
				)
			});
		});
	}
}

/// Given a zerocheck witness, make zerocheck claim
pub fn make_zerocheck_claim<F, C, M>(
	poly: &MultilinearComposite<F, C, M>,
) -> Result<ZerocheckClaim<F>, ZerocheckError>
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

	let zerocheck_claim = ZerocheckClaim {
		poly: composite_poly,
	};

	Ok(zerocheck_claim)
}

criterion_group!(
	zerocheck,
	zerocheck_128b_tower_basis,
	zerocheck_128b_monomial_basis,
	zerocheck_128b_monomial_basis_with_arc,
	zerocheck_128b_over_1b,
	zerocheck_128b_over_8b,
	zerocheck_128b_tower_basis_32b_domain,
	zerocheck_128b_tower_basis_8b_domain
);
criterion_main!(zerocheck);
