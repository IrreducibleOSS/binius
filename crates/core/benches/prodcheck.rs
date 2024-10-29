// Copyright 2024 Irreducible Inc.

use binius_core::{
	challenger::{new_hasher_challenger, IsomorphicChallenger},
	protocols::gkr_gpa::{self, GrandProductClaim, GrandProductWitness},
};
use binius_field::{
	arch::packed_polyval_128::PackedBinaryPolyval1x128b, BinaryField128b, BinaryField128bPolyval,
	PackedField,
};
use binius_hal::make_portable_backend;
use binius_hash::GroestlHasher;
use binius_math::{IsomorphicEvaluationDomainFactory, MultilinearExtension};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, SeedableRng};
use std::iter::repeat_with;

// Creates T(x), a multilinear with evaluations over the n-dimensional boolean hypercube
fn create_numerator<P: PackedField>(n_vars: usize) -> MultilinearExtension<P> {
	let mut rng = StdRng::seed_from_u64(0);
	let values = repeat_with(|| P::random(&mut rng))
		.take(1 << n_vars)
		.collect::<Vec<P>>();

	MultilinearExtension::from_values(values).unwrap()
}

fn bench_polyval(c: &mut Criterion) {
	type FDomain = BinaryField128b;

	type P = PackedBinaryPolyval1x128b;
	type FW = BinaryField128bPolyval;
	let mut group = c.benchmark_group("prodcheck");
	let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();

	for n in [12, 16, 20] {
		group.throughput(Throughput::Bytes(
			((1 << n) * std::mem::size_of::<BinaryField128b>()) as u64,
		));
		group.bench_function(format!("n_vars={n}"), |bench| {
			let prover_challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

			// Setup witness
			let numerator = create_numerator::<P>(n);

			let gpa_witness =
				GrandProductWitness::<P>::new(numerator.specialize_arc_dyn()).unwrap();

			let product: FW = FW::from(gpa_witness.grand_product_evaluation());
			let gpa_claim = GrandProductClaim { n_vars: n, product };
			let backend = make_portable_backend();

			bench.iter(|| {
				let mut challenger_clone = prover_challenger.clone();
				let mut iso_challenger =
					IsomorphicChallenger::<BinaryField128b, _, FW>::new(&mut challenger_clone);
				gkr_gpa::batch_prove::<FW, P, FW, _, _>(
					[gpa_witness.clone()],
					&[gpa_claim.clone()],
					domain_factory.clone(),
					&mut iso_challenger,
					&backend,
				)
			});
		});
	}
	group.finish()
}

criterion_main!(prodcheck);
criterion_group!(prodcheck, bench_polyval);
