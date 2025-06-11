// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_core::{constraint_system, fiat_shamir::HasherChallenger};
use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::builder::{
	B32, B128, ConstraintSystem, Statement, WitnessIndex, test_utils::ClosureFiller,
};
use binius_utils::rayon::adjust_thread_pool;
use bumpalo::Bump;
use bytemuck::zeroed_vec;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of operations to do.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(512..))]
	n_ops: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

fn main() -> Result<()> {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} number of BinaryField32b multiplication", args.n_ops);

	let mut rng = StdRng::seed_from_u64(0);
	let test_vector: Vec<(u32, u32)> = (0..args.n_ops)
		.map(|_| (rng.r#gen(), rng.r#gen()))
		.collect();

	let mut cs = ConstraintSystem::new();
	let mut table = cs.add_table("b32_mul");

	let in_a = table.add_committed::<B32, 1>("in_a");
	let in_b = table.add_committed::<B32, 1>("in_b");
	let out = table.add_committed::<B32, 1>("out");

	table.assert_zero("b32_mul", in_a * in_b - out);

	let table_id = table.id();
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![test_vector.len()],
	};

	let trace_gen_scope = tracing::info_span!("Generating trace", n_ops = args.n_ops).entered();
	let allocator = Bump::new();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

	witness
		.fill_table_parallel(
			&ClosureFiller::new(table_id, |events, index| {
				let mut in_a_vals = index.get_mut_as::<B32, _, 1>(in_a).unwrap();
				let mut in_b_vals = index.get_mut_as::<B32, _, 1>(in_b).unwrap();
				let mut out_vals = index.get_mut_as::<B32, _, 1>(out).unwrap();

				for (i, (a, b)) in events.iter().enumerate() {
					let a_field = B32::new(*a);
					let b_field = B32::new(*b);
					let result = a_field * b_field;

					in_a_vals[i] = a_field;
					in_b_vals[i] = b_field;
					out_vals[i] = result;
				}

				Ok(())
			}),
			&test_vector,
		)
		.unwrap();
	drop(trace_gen_scope);

	let ccs = cs.compile(&statement).unwrap();
	let cs_digest = ccs.digest::<Groestl256>();
	let witness = witness.into_multilinear_extension_index();

	let hal_span = tracing::info_span!("HAL Setup", perfetto_category = "phase.main").entered();

	let hal = FastCpuLayer::<CanonicalTowerFamily, PackedType<OptimalUnderlier, B128>>::default();

	let mut host_mem = zeroed_vec(1 << 20);
	let mut dev_mem_owned = zeroed_vec(1 << (28 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));

	let dev_mem = PackedMemorySliceMut::new_slice(&mut dev_mem_owned);

	drop(hal_span);

	let proof = constraint_system::prove::<
		_,
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256Parallel,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&hal,
		&mut host_mem,
		dev_mem,
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&cs_digest,
		&statement.boundaries,
		&statement.table_sizes,
		witness,
		&make_portable_backend(),
	)?;

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	constraint_system::verify::<
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&cs_digest,
		&statement.boundaries,
		proof,
	)?;

	Ok(())
}
