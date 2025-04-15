// Copyright 2024-2025 Irreducible Inc.

use std::array;

use anyhow::Result;
use binius_circuits::{
	blake3::Blake3CompressState,
	builder::{types::U, ConstraintSystemBuilder},
};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_utils::rayon::adjust_thread_pool;
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use rand::{rngs::OsRng, Rng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of compressions to verify.
	#[arg(short, long, default_value_t = 32, value_parser = value_parser!(u32).range(1 << 3..))]
	n_compressions: u32,
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

	println!("Verifying {} Blake3 compressions", args.n_compressions);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();

	let mut rng = OsRng;
	let input_witness = (0..args.n_compressions as usize)
		.into_iter()
		.map(|_| {
			let cv: [u32; 8] = array::from_fn(|_| rng.gen::<u32>());
			let block: [u32; 16] = array::from_fn(|_| rng.gen::<u32>());
			let counter = rng.gen::<u64>();
			let counter_low = counter as u32;
			let counter_high = (counter >> 32) as u32;
			let block_len = rng.gen::<u32>();
			let flags = rng.gen::<u32>();

			Blake3CompressState {
				cv,
				block,
				counter_low,
				counter_high,
				block_len,
				flags,
			}
		})
		.collect::<Vec<Blake3CompressState>>();

	let _state_out = binius_circuits::blake3::blake3_compress(
		&mut builder,
		&Some(input_witness),
		args.n_compressions as usize,
	)?;
	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");

	let constraint_system = builder.build()?;

	let backend = make_portable_backend();

	let proof =
		constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, &[], witness, &backend)?;

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
