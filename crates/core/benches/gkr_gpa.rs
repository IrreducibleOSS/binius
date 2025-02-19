// Copyright 2025 Irreducible Inc.

use binius_core::{
	constraint_system::{
		channel::{Flush, FlushDirection},
		prover::{make_fast_masked_flush_witnesses, make_unmasked_flush_witnesses},
		verifier::make_flush_oracles,
	},
	fiat_shamir::HasherChallenger,
	oracle::{MultilinearOracleSet, OracleId},
	protocols::{
		gkr_gpa,
		gkr_gpa::{GrandProductClaim, GrandProductWitness},
	},
	tower::{CanonicalTowerFamily, ProverTowerFamily},
	transcript::ProverTranscript,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, packed::set_packed_slice, BinaryField128b,
	BinaryField128bPolyval, PackedField, TowerField,
};
use binius_hal::make_portable_backend;
use binius_math::{IsomorphicEvaluationDomainFactory, MultilinearExtension};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bytemuck::cast_slice_mut;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use groestl_crypto::Groestl256;
use rand::{seq::SliceRandom, thread_rng, RngCore};

type U = OptimalUnderlier;
type F = BinaryField128b;
type FFast = BinaryField128bPolyval;

// We do not deal with boundaries here.
// Assume we are working on a single channel (channel_id = 0)
// Since the flushes are powers of 2 we are disregarding selector for flushes

fn gen_poly_with_selectors(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	counts: &[usize],
	global_values: &[F],
) -> Vec<OracleId> {
	type P = PackedType<U, F>;
	let (oracles, _) = counts
		.iter()
		.fold((Vec::new(), 0), |(mut new_oracles, sum), &count| {
			let n_vars = log2_ceil_usize(count);
			let oracle_id = oracles.add_committed(n_vars, F::TOWER_LEVEL);

			let mut witness_data = vec![P::default(); 1 << n_vars];
			for i in 0..count {
				set_packed_slice(&mut witness_data, i, global_values[sum + i]);
			}
			let witness = MultilinearExtension::new(n_vars, witness_data)
				.unwrap()
				.specialize_arc_dyn();
			witness_index
				.update_multilin_poly([(oracle_id, witness)])
				.unwrap();

			new_oracles.push(oracle_id);
			(new_oracles, sum + count)
		});
	oracles
}

fn create_balancing_flushes(
	flushes: &mut Vec<Flush>,
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	log_size: usize,
	channel_id: usize,
	mut counts: Vec<usize>,
) {
	assert_eq!(counts.iter().sum::<usize>(), 1 << log_size);

	let mut rng = thread_rng();
	let mut values = vec![F::default(); 1 << log_size];
	rng.fill_bytes(cast_slice_mut(&mut values));

	// Create pushes along with selectors.
	for oracle_id in gen_poly_with_selectors(oracles, witness_index, &counts, &values) {
		flushes.push(Flush {
			channel_id,
			oracles: vec![oracle_id],
			direction: FlushDirection::Push,
			selector: usize::MAX,
			multiplicity: 1,
		});
	}

	// Create pulls along with their selectors.
	// Simply shuffle the counts for easy generation
	let mut rng = thread_rng();
	counts.shuffle(&mut rng);

	for oracle_id in gen_poly_with_selectors(oracles, witness_index, &counts, &values) {
		flushes.push(Flush {
			channel_id,
			oracles: vec![oracle_id],
			direction: FlushDirection::Pull,
			selector: usize::MAX,
			multiplicity: 1,
		});
	}
}

// For a given log_size the flushes are of the form
// 2^{start_range} + 2{start_range} + 2^{start_rage+1} + ... + 2^{stop_range} = 2^{log_size}
#[allow(clippy::type_complexity)]
fn create_gkr_gpa(
	oracles: &mut MultilinearOracleSet<F>,
	witness_index: &mut MultilinearExtensionIndex<U, F>,
	start_range: usize,
	stop_range: usize,
	log_size: usize,
) -> (Vec<GrandProductWitness<PackedType<U, FFast>>>, Vec<GrandProductClaim<FFast>>) {
	assert!(stop_range > start_range);
	assert_eq!(
		(start_range..=stop_range).map(|x| 1 << x).sum::<usize>() + (1 << start_range),
		1 << log_size
	);
	let mut flushes = Vec::new();

	let mut counts = vec![1 << start_range; stop_range - start_range + 2];
	counts
		.iter_mut()
		.enumerate()
		.skip(1)
		.for_each(|(i, count)| {
			*count = 1 << (i + start_range - 1);
		});

	create_balancing_flushes(&mut flushes, oracles, witness_index, log_size, 0, counts.clone());

	let mut rng = thread_rng();
	let mixing_challenge: F = F::random(&mut rng);
	let permutation_challenge = [F::random(&mut rng)];
	let flush_oracle_id =
		make_flush_oracles(oracles, &flushes, mixing_challenge, &permutation_challenge).unwrap();

	make_unmasked_flush_witnesses::<_, CanonicalTowerFamily>(
		oracles,
		witness_index,
		&flush_oracle_id,
	)
	.unwrap();

	let flush_witnesses = make_fast_masked_flush_witnesses::<_, CanonicalTowerFamily>(
		oracles,
		witness_index,
		&flush_oracle_id,
		None,
	)
	.unwrap();

	let flush_prodcheck_witnesses = flush_witnesses
		.into_iter()
		.map(GrandProductWitness::new)
		.collect::<Result<Vec<_>, _>>()
		.unwrap();
	let flush_products = gkr_gpa::get_grand_products_from_witnesses(&flush_prodcheck_witnesses);

	let flush_prodcheck_claims =
		gkr_gpa::construct_grand_product_claims(&flush_oracle_id, oracles, &flush_products)
			.unwrap()
			.into_iter()
			.map(|claim| claim.isomorphic())
			.collect::<Vec<_>>();

	(flush_prodcheck_witnesses, flush_prodcheck_claims)
}

const FLUSH_START: usize = 18;
const FLUSH_END: usize = 23;
const LOG_SIZE: usize = 24;

fn bench_gkr_gpa(c: &mut Criterion) {
	type FFastExt<Tower> = <Tower as ProverTowerFamily>::FastB128;

	let mut oracles = MultilinearOracleSet::<BinaryField128b>::new();
	let mut witness_index = MultilinearExtensionIndex::<U, F>::new();
	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
	let fast_domain_factory =
		IsomorphicEvaluationDomainFactory::<FFastExt<CanonicalTowerFamily>>::default();
	let backend = make_portable_backend();

	let (flush_prodcheck_witnesses, flush_prodcheck_claims) =
		create_gkr_gpa(&mut oracles, &mut witness_index, FLUSH_START, FLUSH_END, LOG_SIZE);

	let mut group = c.benchmark_group(format!("Flush products log_size {}", LOG_SIZE));
	group.throughput(Throughput::Elements(1 << LOG_SIZE as u64));
	group.bench_function("grk_gpa", |b| {
		b.iter_batched(
			|| flush_prodcheck_witnesses.clone(),
			|flush_prodcheck_witnesses| {
				gkr_gpa::batch_prove::<
					FFastExt<CanonicalTowerFamily>,
					_,
					FFastExt<CanonicalTowerFamily>,
					_,
					_,
				>(
					flush_prodcheck_witnesses,
					&flush_prodcheck_claims,
					&fast_domain_factory,
					&mut transcript,
					&backend,
				)
				.unwrap();
			},
			BatchSize::SmallInput,
		);
	});
	group.finish();
}

criterion_main!(poly_commit);
criterion_group! {name = poly_commit;
				config = Criterion::default().sample_size(50);
				targets = bench_gkr_gpa
}
