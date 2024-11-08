// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_field::{arch::OptimalUnderlier128b, BinaryField128b, BinaryField8b};
use binius_hal::make_portable_backend;
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use binius_math::DefaultEvaluationDomainFactory;
use binius_utils::{
	checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool, tracing::init_tracing,
};
use clap::{value_parser, Parser};
use groestl_crypto::Groestl256;

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 1024, value_parser = value_parser!(u32).range(1 << 10..))]
	n_multiplications: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

fn main() -> Result<()> {
	type U = OptimalUnderlier128b;
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} u8 multiplications", args.n_multiplications);

	let log_n_multiplications = log2_ceil_usize(args.n_multiplications as usize);

	let mut builder = ConstraintSystemBuilder::<U, BinaryField128b>::new_with_witness();
	let in_a = binius_circuits::unconstrained::unconstrained::<_, _, BinaryField8b>(
		&mut builder,
		"in_a",
		log_n_multiplications,
	)?;
	let in_b = binius_circuits::unconstrained::unconstrained::<_, _, BinaryField8b>(
		&mut builder,
		"in_b",
		log_n_multiplications,
	)?;
	let _product =
		binius_circuits::lasso::u8mul(&mut builder, "out_c", in_a, in_b, log_n_multiplications)?;

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

	constraint_system::verify::<U, CanonicalTowerFamily, _, _, _, _, HasherChallenger<Groestl256>>(
		&constraint_system,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&domain_factory,
		proof,
	)?;

	Ok(())
}
