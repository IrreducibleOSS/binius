use binius_core::{
	merkle_tree::BinaryMerkleTreeProver,
	oracle::MultilinearOracleSet,
	piop,
	protocols::{fri, fri::FRIParams},
	reed_solomon::reed_solomon::ReedSolomonCode,
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, packed::set_packed_slice, BinaryField128b,
	BinaryField32b, PackedField, TowerField,
};
use binius_hash::compress::Groestl256ByteCompression;
use binius_math::{MultilinearExtension, MultilinearPoly};
use binius_ntt::{NTTOptions, ThreadingSettings};
use criterion::{criterion_group, criterion_main, Criterion};
use groestl_crypto::Groestl256;
use itertools::Itertools;
use rand::thread_rng;

type U = OptimalUnderlier;
type F = BinaryField128b;

fn merge_multilins<P, M>(multilins: &[M], message_buffer: &mut [P])
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	let mut packed_offset = 0;
	let mut mle_iter = multilins.iter().rev();

	// First copy all the polynomials where the number of elements is a multiple of the packing
	// width.
	let get_n_packed_vars = |mle: &M| mle.n_vars() - mle.log_extension_degree();
	for mle in mle_iter.peeking_take_while(|mle| get_n_packed_vars(mle) >= P::LOG_WIDTH) {
		let evals = mle
			.packed_evals()
			.expect("guaranteed by function precondition");
		message_buffer[packed_offset..packed_offset + evals.len()].copy_from_slice(evals);
		packed_offset += evals.len();
	}

	// Now copy scalars from the remaining multilinears, which have too few elements to copy full
	// packed elements.
	let mut scalar_offset = packed_offset << P::LOG_WIDTH;
	for mle in mle_iter {
		let evals = mle
			.packed_evals()
			.expect("guaranteed by function precondition");
		let packed_eval = evals[0];
		for i in 0..1 << mle.n_vars() {
			set_packed_slice(message_buffer, scalar_offset, packed_eval.get(i));
			scalar_offset += 1;
		}
	}
}

const LOG_SIZE: usize = 20;

fn bench_poly_commit(c: &mut Criterion) {
	let (fri_params, merkle_prover, committed_multilins) = create_poly_commit(LOG_SIZE);
	c.bench_function(format!("PolyCommit log_size {}", LOG_SIZE).as_str(), |b| {
		b.iter(|| {
			let rs_code = ReedSolomonCode::new(
				fri_params.rs_code().log_dim(),
				fri_params.rs_code().log_inv_rate(),
				&NTTOptions {
					precompute_twiddles: true,
					thread_settings: ThreadingSettings::SingleThreaded,
				},
			)
			.unwrap();
			fri::commit_interleaved_with(&rs_code, &fri_params, &merkle_prover, |message_buffer| {
				merge_multilins(&committed_multilins, message_buffer)
			})
			.unwrap();
		});
	});
}

#[allow(clippy::type_complexity)]
fn create_poly_commit(
	log_size: usize,
) -> (
	FRIParams<BinaryField128b, BinaryField32b>,
	BinaryMerkleTreeProver<BinaryField128b, Groestl256, Groestl256ByteCompression>,
	Vec<MultilinearWitness<'static, PackedType<U, BinaryField128b>>>,
) {
	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 100;

	let mut oracles = MultilinearOracleSet::<BinaryField128b>::new();
	let mut witness_index = MultilinearExtensionIndex::<U, F>::new();

	let mut rng = thread_rng();
	let len = 1usize << log_size.saturating_sub(<PackedType<U, F>>::LOG_WIDTH);
	{
		let poly_id = oracles.add_committed(log_size, BinaryField128b::TOWER_LEVEL);
		let mut poly_data = vec![PackedType::<U, BinaryField128b>::default(); len];
		poly_data
			.iter_mut()
			.for_each(|x| *x = PackedType::<U, BinaryField128b>::random(&mut rng));

		let witness = MultilinearExtension::new(log_size, poly_data)
			.unwrap()
			.specialize_arc_dyn();
		witness_index
			.update_multilin_poly([(poly_id, witness)])
			.unwrap();
	}

	let merkle_prover = BinaryMerkleTreeProver::new(Groestl256ByteCompression);
	let merkle_scheme = merkle_prover.scheme();

	let (commit_meta, oracle_to_commit_index) = piop::make_oracle_commit_meta(&oracles).unwrap();
	let committed_multilins = piop::collect_committed_witnesses(
		&commit_meta,
		&oracle_to_commit_index,
		&oracles,
		&witness_index,
	)
	.unwrap();

	let fri_params = piop::make_commit_params_with_optimal_arity::<_, _, _>(
		&commit_meta,
		merkle_scheme,
		SECURITY_BITS,
		LOG_INV_RATE,
	)
	.unwrap();

	(fri_params, merkle_prover, committed_multilins)
}

criterion_main!(poly_commit);
criterion_group!(poly_commit, bench_poly_commit);
