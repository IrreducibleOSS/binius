// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::{
	ComputeData, ComputeHolder, ComputeLayer, ComputeMemory, ops::eq_ind_partial_eval,
};
use binius_core::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::HasherChallenger,
	polynomial::MultilinearComposite,
	protocols::sumcheck::{
		CompositeSumClaim, EqIndSumcheckClaim, batch_prove,
		v3::bivariate_mlecheck::BivariateMLEcheckProver,
	},
	transcript::ProverTranscript,
};
use binius_fast_compute::{
	layer::{FastCpuLayer, FastCpuLayerHolder},
	memory::PackedMemorySlice,
};
use binius_field::{
	ByteSlicedAES32x128b, Field,
	tower::{AESTowerFamily, PackedTop, TowerFamily},
};
use binius_hash::groestl::Groestl256;
use binius_math::{MLEDirectAdapter, MultilinearExtension, MultilinearPoly, MultilinearQuery};
use criterion::{Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn bench_sumcheck_v3<T: TowerFamily, P: PackedTop<T>>(
	c: &mut Criterion,
	field: &str,
	n_vars: usize,
) {
	let mut rng = rand::rng();
	let multilins = repeat_with(|| {
		let values = repeat_with(|| P::random(&mut rng))
			.take(1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.collect::<Vec<_>>();
		MLEDirectAdapter::from(MultilinearExtension::new(n_vars, values).unwrap())
	})
	.take(2)
	.collect::<Vec<_>>();
	let challenge = repeat_with(|| <T::B128 as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let bivariate_composition = BivariateProduct::default();

	let witness =
		MultilinearComposite::new(n_vars, bivariate_composition, multilins.clone()).unwrap();

	let query = MultilinearQuery::expand(&challenge);
	let sum = witness.evaluate(&query).unwrap();

	let composite_sum_claim = CompositeSumClaim {
		composition: IndexComposition::new(2, [0, 1], bivariate_composition).unwrap(),
		sum,
	};

	let multilins = multilins
		.iter()
		.map(|mle| PackedMemorySlice::new_slice(mle.packed_evals().unwrap()))
		.collect::<Vec<_>>();
	let claim =
		EqIndSumcheckClaim::<T::B128, _>::new(n_vars, multilins.len(), vec![composite_sum_claim])
			.unwrap();
	let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let mut group = c.benchmark_group(format!("Sumcheck/{field}"));

	let mut compute_holder =
		FastCpuLayerHolder::new(1 << (n_vars + 8), 1 << (n_vars + 8 - P::LOG_WIDTH));

	group.bench_function(format!("n_vars={n_vars}"), |b| {
		// Move dev_alloc and host_alloc outside the closure so their lifetimes are sufficient
		b.iter(|| {
			let ComputeData {
				hal,
				dev_alloc,
				host_alloc,
				..
			} = compute_holder.to_data();

			let eq_ind_partial_evals =
				eq_ind_partial_eval(hal, &dev_alloc, &challenge[..n_vars.saturating_sub(1)])
					.unwrap();

			let prover = BivariateMLEcheckProver::new(
				hal,
				&dev_alloc,
				&host_alloc,
				&claim,
				multilins.clone(),
				<FastCpuLayer<T, P> as ComputeLayer<T::B128>>::DevMem::as_const(
					&eq_ind_partial_evals,
				),
				challenge.clone(),
			)
			.unwrap();

			batch_prove(vec![prover], &mut prover_transcript).unwrap();
		});
	});
}

fn sumcheck_v3(c: &mut Criterion) {
	bench_sumcheck_v3::<AESTowerFamily, ByteSlicedAES32x128b>(c, "ByteSlicedAES32x128b", 18);
}

// This enables profiling with pprof and flamegraph.
//
// Run
// ```
// export RUSTFLAGS="-Ctarget-cpu=native"
// cargo bench --bench big_field_sumcheck -- --profile-time=10
// ```
// to generate a flamegraph.svg file.
//
// The file can be found in target/criterion/big_field_sumcheck/*/profile/flamegraph.svg.
//
// I recommend opening the SVG file in a browser (I use Firefox).
//
// The flamegraph could be more useful by disabling the "rayon" feature in Cargo.toml. (But the
// real benchmark does use multithreading.)
criterion_group! {
	name = sumcheck_benches;
	config = Criterion::default().sample_size(10)
		.with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
	targets = sumcheck_v3
}
criterion_main!(sumcheck_benches);
