// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
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
use tracing_profile::init_tracing;

const MIN_N_ADDITIONS: usize = 1 << (PackedType::<OptimalUnderlier, BinaryField1b>::LOG_WIDTH);

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = MIN_N_ADDITIONS as u32, value_parser = value_parser!(u32).range(MIN_N_ADDITIONS as i64..))]
	n_additions: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

fn main() -> Result<()> {
	type U = OptimalUnderlier;
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} u32 additions", args.n_additions);

	let log_n_additions = log2_ceil_usize(args.n_additions as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, BinaryField128b>::new_with_witness(&allocator);
	let in_a = binius_circuits::unconstrained::unconstrained::<_, _, BinaryField8b>(
		&mut builder,
		"in_a",
		log_n_additions + 2,
	)?;
	let in_b = binius_circuits::unconstrained::unconstrained::<_, _, BinaryField8b>(
		&mut builder,
		"in_b",
		log_n_additions + 2,
	)?;
	let _product = binius_circuits::lasso::u32add::<_, _, BinaryField8b, BinaryField8b>(
		&mut builder,
		"out_c",
		in_a,
		in_b,
	)?;

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
		_,
		GroestlHasher<BinaryField128b>,
		GroestlDigestCompression<BinaryField8b>,
		HasherChallenger<Groestl256>,
	>(
		&constraint_system,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&domain_factory,
		vec![],
		proof,
	)?;

	Ok(())
}
