// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::{
	merkle_tree::BinaryMerkleTreeProver,
	piop,
	piop::CommitMeta,
	protocols::{fri, fri::FRIParams},
};
use binius_field::{
	AESTowerField32b, AESTowerField128b, BinaryField, BinaryField32b, BinaryField128b,
	ByteSlicedAES16x128b, ByteSlicedAES32x128b, ByteSlicedAES64x128b, ExtensionField,
	PackedBinaryField1x128b, PackedExtension, PackedField, TowerField, packed::set_packed_slice,
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::{MLEDirectAdapter, MultilinearExtension, MultilinearPoly};
use binius_ntt::SingleThreadedNTT;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use itertools::Itertools;

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

fn bench_poly_commit<F, P, FEncode>(c: &mut Criterion, name: &str)
where
	F: TowerField + ExtensionField<FEncode>,
	P: PackedField<Scalar = F> + PackedExtension<FEncode>,
	FEncode: BinaryField,
{
	let (fri_params, merkle_prover, committed_multilins) =
		create_poly_commit::<F, P, FEncode>(LOG_SIZE);
	let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len())
		.unwrap()
		.precompute_twiddles()
		.multithreaded();
	let mut group = c.benchmark_group("Polynomial Commitment");
	group.throughput(Throughput::Bytes(
		((1 << LOG_SIZE) * committed_multilins.len() * std::mem::size_of::<F>()) as u64,
	));
	group.bench_function(BenchmarkId::new(format!("{name}/log_size"), LOG_SIZE), |b| {
		b.iter(|| {
			fri::commit_interleaved_with(&fri_params, &ntt, &merkle_prover, |message_buffer| {
				merge_multilins(&committed_multilins, message_buffer)
			})
			.unwrap();
		});
	});
	group.finish();
}

#[allow(clippy::type_complexity)]
fn create_poly_commit<F, P, FEncode>(
	log_size: usize,
) -> (
	FRIParams<F, FEncode>,
	BinaryMerkleTreeProver<F, Groestl256, Groestl256ByteCompression>,
	Vec<impl MultilinearPoly<P>>,
)
where
	F: TowerField + ExtensionField<FEncode>,
	P: PackedField<Scalar = F>,
	FEncode: BinaryField,
{
	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 100;

	let mut rng = rand::rng();
	let poly_data = repeat_with(|| P::random(&mut rng))
		.take(1 << LOG_SIZE.saturating_sub(P::LOG_WIDTH))
		.collect::<Vec<_>>();

	let committed_multilins = vec![MLEDirectAdapter::from(
		MultilinearExtension::new(log_size, poly_data).unwrap(),
	)];

	let merkle_prover = BinaryMerkleTreeProver::new(Groestl256ByteCompression);
	let merkle_scheme = merkle_prover.scheme();

	let commit_meta = CommitMeta::with_vars([LOG_SIZE]);

	let fri_params = piop::make_commit_params_with_optimal_arity::<_, _, _>(
		&commit_meta,
		merkle_scheme,
		SECURITY_BITS,
		LOG_INV_RATE,
	)
	.unwrap();

	(fri_params, merkle_prover, committed_multilins)
}

fn bench_poly_commit_binary_field(c: &mut Criterion) {
	bench_poly_commit::<BinaryField128b, PackedBinaryField1x128b, BinaryField32b>(
		c,
		"binary_field",
	);
}

fn bench_poly_commit_byte_sliced(c: &mut Criterion) {
	bench_poly_commit::<AESTowerField128b, ByteSlicedAES16x128b, AESTowerField32b>(
		c,
		"byte_sliced/ByteSlicedAES16x128b",
	);

	bench_poly_commit::<AESTowerField128b, ByteSlicedAES32x128b, AESTowerField32b>(
		c,
		"byte_sliced/ByteSlicedAES32x128b",
	);

	bench_poly_commit::<AESTowerField128b, ByteSlicedAES64x128b, AESTowerField32b>(
		c,
		"byte_sliced/ByteSlicedAES64x128b",
	);
}

criterion_main!(poly_commit);
criterion_group!(poly_commit, bench_poly_commit_binary_field, bench_poly_commit_byte_sliced);
