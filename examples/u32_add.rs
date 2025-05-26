// Copyright 2024-2025 Irreducible Inc.

// Uses binius_circuits which is being phased out.
#![allow(deprecated)]

use anyhow::Result;
use binius_circuits::{
	arithmetic::Flags,
	builder::{ConstraintSystemBuilder, types::U},
};
use binius_compute::{
	ComputeLayer, FSliceMut,
	alloc::{BumpAllocator, HostBumpAllocator},
	cpu::CpuLayer,
};
use binius_core::{
	constraint_system, fiat_shamir::HasherChallenger, witness::HalMultilinearExtensionIndex,
};
use binius_field::{BinaryField1b, BinaryField128b, tower::CanonicalTowerFamily};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytemuck::zeroed_vec;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
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

	let log_n_additions = log2_ceil_usize(args.n_additions as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let in_a = binius_circuits::unconstrained::unconstrained::<BinaryField1b>(
		&mut builder,
		"in_a",
		log_n_additions + 5,
	)?;
	let in_b = binius_circuits::unconstrained::unconstrained::<BinaryField1b>(
		&mut builder,
		"in_b",
		log_n_additions + 5,
	)?;
	let _sum =
		binius_circuits::arithmetic::u32::add(&mut builder, "sum", in_a, in_b, Flags::Unchecked)?;
	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");
	let constraint_system = builder.build()?;

	let backend = make_portable_backend();

	let hal = <CpuLayer<CanonicalTowerFamily>>::default();
	let mut dev_mem = zeroed_vec(1 << 20);
	let mut host_mem = hal.host_alloc(1 << 20);
	let host_alloc = HostBumpAllocator::new(host_mem.as_mut());
	let dev_alloc = BumpAllocator::new(
		(&mut dev_mem) as FSliceMut<BinaryField128b, CpuLayer<CanonicalTowerFamily>>,
	);
	let hal_witness = HalMultilinearExtensionIndex::new(&dev_alloc, &hal);

	let proof = constraint_system::prove::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
		_,
	>(
		&constraint_system,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&[],
		witness,
		&hal_witness,
		&backend,
		&host_alloc,
		&dev_alloc,
	)?;

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, &[], proof)?;

	Ok(())
}
