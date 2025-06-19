// Copyright 2024-2025 Irreducible Inc.
use anyhow::Result;
use binius_compute::{ComputeHolder, cpu::alloc::CpuComputeAllocator};
use binius_core::{
	constraint_system, fiat_shamir::HasherChallenger, protocols::fri::FRISoundnessParams,
};
use binius_fast_compute::layer::FastCpuLayerHolder;
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::{
	builder::{B1, B128, WitnessIndex, test_utils::ClosureFiller},
	gadgets::add::{U32Add, U32AddFlags},
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of additions to do.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(512..))]
	n_additions: u32,
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

	println!("Verifying {} u32 additions", args.n_additions);

	let mut rng = StdRng::seed_from_u64(0);
	let test_vector: Vec<(u32, u32)> = (0..args.n_additions)
		.map(|_| (rng.random(), rng.random()))
		.collect();

	let mut cs = binius_m3::builder::ConstraintSystem::new();
	let mut table = cs.add_table("u32_add");

	let flags = U32AddFlags {
		carry_in_bit: None,
		expose_final_carry: false,
		commit_zout: true,
	};

	let xin = table.add_committed::<B1, 32>("xin");
	let yin = table.add_committed::<B1, 32>("yin");
	let adder = U32Add::new(&mut table, xin, yin, flags);

	let table_id = table.id();
	let boundaries = vec![];
	let table_sizes = vec![test_vector.len()];
	let trace_gen_scope =
		tracing::info_span!("Generating trace", n_adds = args.n_additions).entered();
	let mut allocator = CpuComputeAllocator::new(
		1 << (log2_ceil_usize(args.n_additions as _) + 1
			- PackedType::<OptimalUnderlier, B128>::LOG_WIDTH),
	);
	let allocator = allocator.into_bump_allocator();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);
	witness
		.fill_table_parallel(
			&ClosureFiller::new(table_id, |events, index| {
				let mut xin_bits = index.get_mut_as::<u32, _, 32>(adder.xin)?;
				let mut yin_bits = index.get_mut_as::<u32, _, 32>(adder.yin)?;
				for (i, (x, y)) in events.iter().enumerate() {
					xin_bits[i] = *x;
					yin_bits[i] = *y;
				}
				drop((xin_bits, yin_bits));
				adder.populate(index)?;
				Ok(())
			}),
			&test_vector,
		)
		.unwrap();
	drop(trace_gen_scope);

	let ccs = cs.compile().unwrap();
	let cs_digest = ccs.digest::<Groestl256>();
	let witness = witness.into_multilinear_extension_index();

	let hal_span = tracing::info_span!("HAL Setup", perfetto_category = "phase.main").entered();

	let mut compute_holder = FastCpuLayerHolder::<
		CanonicalTowerFamily,
		PackedType<OptimalUnderlier, B128>,
	>::new(1 << 20, 1 << 28);

	drop(hal_span);

	let fri_soundness_params = FRISoundnessParams::new(SECURITY_BITS, args.log_inv_rate as usize);
	let proof = constraint_system::prove::<
		_,
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256Parallel,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
		_,
		_,
	>(
		&mut compute_holder.to_data(),
		&ccs,
		&fri_soundness_params,
		&cs_digest,
		&boundaries,
		&table_sizes,
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
	>(&ccs, &fri_soundness_params, &cs_digest, &boundaries, proof)?;

	Ok(())
}
