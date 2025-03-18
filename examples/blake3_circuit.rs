// Copyright 2024-2025 Irreducible Inc.

#![feature(array_try_from_fn)]

use std::array;

use anyhow::Result;
use binius_circuits::{
	blake3::{BLAKE3_STATE_LEN, CHAINING_VALUE_LEN},
	builder::{types::U, ConstraintSystemBuilder},
	unconstrained::unconstrained,
};
use binius_core::{
	constraint_system, fiat_shamir::HasherChallenger, oracle::OracleId, tower::CanonicalTowerFamily,
};
use binius_field::BinaryField1b;
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_math::DefaultEvaluationDomainFactory;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use rand::Rng;
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

const COMPRESSION_LOG_LEN: usize = 5;

fn main() -> Result<()> {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} Blake3 compressions", args.n_compressions);

	let log_n_compressions = log2_ceil_usize(args.n_compressions as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let input: [OracleId; BLAKE3_STATE_LEN] = array::try_from_fn(|i| {
		unconstrained::<BinaryField1b>(&mut builder, i, log_n_compressions + COMPRESSION_LOG_LEN)
	})?;

	let chaining_value: [OracleId; CHAINING_VALUE_LEN] = array::try_from_fn(|i| {
		unconstrained::<BinaryField1b>(&mut builder, i, log_n_compressions + COMPRESSION_LOG_LEN)
	})?;

	let mut rng = rand::thread_rng();
	let _state_out = binius_circuits::blake3::compress(
		&mut builder,
		"blake3-compression",
		&chaining_value,
		&input,
		rng.gen(),
		rng.gen(),
		rng.gen(),
		log_n_compressions + COMPRESSION_LOG_LEN,
	)?;
	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");

	let constraint_system = builder.build()?;

	let domain_factory = DefaultEvaluationDomainFactory::default();
	let backend = make_portable_backend();

	let proof = constraint_system::prove::<
		U,
		CanonicalTowerFamily,
		_,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&constraint_system,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&[],
		witness,
		&domain_factory,
		&backend,
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
