// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_circuits::{
	builder::{types::U, ConstraintSystemBuilder},
	collatz::{Advice, Collatz},
};
use binius_core::{
	constraint_system::{self, Proof},
	fiat_shamir::HasherChallenger,
	tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::compress::Groestl256ByteCompression;
use binius_math::DefaultEvaluationDomainFactory;
use binius_utils::rayon::adjust_thread_pool;
use clap::{value_parser, Parser};
use groestl_crypto::Groestl256;
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The starting value for the collatz orbit
	#[arg(long, default_value_t = 837_799, value_parser = value_parser!(u32).range(2..))]
	starting_value: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

const SECURITY_BITS: usize = 100;

fn main() -> Result<()> {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();
	let _guard = init_tracing().expect("failed to initialize tracing");

	let x0 = args.starting_value; //9999999;
	println!("Verifying collatz orbit over u32 with starting value {}", x0);

	let log_inv_rate = args.log_inv_rate as usize;

	let (advice, proof) = prove(x0, log_inv_rate)?;

	verify(x0, advice, proof, log_inv_rate)?;

	Ok(())
}

fn prove(x0: u32, log_inv_rate: usize) -> Result<(Advice, Proof), anyhow::Error> {
	let mut collatz = Collatz::new(x0);
	let advice = collatz.init_prover();

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let boundaries = collatz.build(&mut builder, advice)?;

	let witness = builder
		.take_witness()
		.expect("builder created with witness");
	let constraint_system = builder.build()?;

	constraint_system::validate::validate_witness(&constraint_system, &boundaries, &witness)?;

	let domain_factory = DefaultEvaluationDomainFactory::default();
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
		log_inv_rate,
		SECURITY_BITS,
		&boundaries,
		witness,
		&domain_factory,
		&make_portable_backend(),
	)?;

	Ok((advice, proof))
}

fn verify(x0: u32, advice: Advice, proof: Proof, log_inv_rate: usize) -> Result<(), anyhow::Error> {
	let collatz = Collatz::new(x0);

	let mut builder = ConstraintSystemBuilder::new();

	let boundaries = collatz.build(&mut builder, advice)?;

	let constraint_system = builder.build()?;

	constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&constraint_system, log_inv_rate, SECURITY_BITS, &boundaries, proof)?;

	Ok(())
}
