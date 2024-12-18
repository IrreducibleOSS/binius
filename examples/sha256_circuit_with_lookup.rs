// Copyright 2024 Irreducible Inc.

#![feature(array_try_from_fn)]

use anyhow::Result;
use binius_circuits::{builder::ConstraintSystemBuilder, unconstrained::unconstrained};
use binius_core::{
	constraint_system, fiat_shamir::HasherChallenger, oracle::OracleId, tower::CanonicalTowerFamily,
};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, BinaryField128b, BinaryField1b,
	BinaryField8b,
};
use binius_hal::make_portable_backend;
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use binius_math::DefaultEvaluationDomainFactory;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use clap::{value_parser, Parser};
use groestl_crypto::Groestl256;
use std::array;
use tracing_profile::init_tracing;

// P:LOG_WIDTH + BinaryField8b::TOWER_LEVEL - COMPRESSION_LOG_LEN
const MIN_N_COMPRESSIONS: usize =
	1 << (PackedType::<OptimalUnderlier, BinaryField1b>::LOG_WIDTH - 2);

#[derive(Debug, Parser)]
struct Args {
	/// The number of compressions to verify.
	#[arg(short, long, default_value_t = MIN_N_COMPRESSIONS as u32, value_parser = value_parser!(u32).range((MIN_N_COMPRESSIONS) as i64 ..))]
	n_compressions: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

const COMPRESSION_LOG_LEN: usize = 5;

fn main() -> Result<()> {
	type U = OptimalUnderlier;
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} sha256 compressions with lookups", args.n_compressions);

	let log_n_compressions = log2_ceil_usize(args.n_compressions as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, BinaryField128b>::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating witness").entered();
	let input: [OracleId; 16] = array::try_from_fn(|i| {
		unconstrained::<_, _, BinaryField1b>(
			&mut builder,
			i,
			log_n_compressions + COMPRESSION_LOG_LEN,
		)
	})?;

	let _state_out = binius_circuits::lasso::sha256(
		&mut builder,
		input,
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
		BinaryField8b,
		_,
		_,
		GroestlHasher<BinaryField128b>,
		GroestlDigestCompression<BinaryField8b>,
		HasherChallenger<Groestl256>,
		_,
	>(
		&constraint_system,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		witness,
		&domain_factory,
		&backend,
	)?;

	constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		_,
		GroestlHasher<BinaryField128b>,
		GroestlDigestCompression<BinaryField8b>,
		HasherChallenger<Groestl256>,
	>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, vec![], proof)?;

	Ok(())
}