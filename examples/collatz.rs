// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{
	constraint_system::{
		self,
		channel::{Boundary, FlushDirection},
	},
	fiat_shamir::HasherChallenger,
	tower::CanonicalTowerFamily,
};
use binius_field::{arch::OptimalUnderlier128b, BinaryField128b, BinaryField32b};
use binius_hal::make_portable_backend;
use binius_hash::{GroestlDigestCompression, GroestlHasher};
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

fn main() -> Result<()> {
	type U = OptimalUnderlier128b;
	type F = BinaryField128b;
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();
	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying collatz orbit over u32 with starting value {}", args.starting_value);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);

	let channel_id = binius_circuits::collatz::collatz(&mut builder, args.starting_value)?;
	let boundaries = vec![
		Boundary {
			channel_id,
			direction: FlushDirection::Pull,
			values: vec![BinaryField32b::new(1).into()],
			multiplicity: 1,
		},
		Boundary {
			channel_id,
			direction: FlushDirection::Push,
			values: vec![BinaryField32b::new(args.starting_value).into()],
			multiplicity: 1,
		},
	];

	let witness = builder
		.take_witness()
		.expect("builder created with witness");
	let constraint_system = builder.build()?;

	constraint_system::validate::validate_witness(&constraint_system, &boundaries, &witness)?;

	let domain_factory = DefaultEvaluationDomainFactory::default();
	let proof = constraint_system::prove::<
		U,
		CanonicalTowerFamily,
		BinaryField32b,
		_,
		_,
		GroestlHasher<_>,
		GroestlDigestCompression<_>,
		HasherChallenger<Groestl256>,
		_,
	>(
		&constraint_system,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		witness,
		&domain_factory,
		&make_portable_backend(),
	)?;

	constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		_,
		_,
		GroestlHasher<_>,
		GroestlDigestCompression<_>,
		HasherChallenger<Groestl256>,
	>(
		&constraint_system.no_base_constraints(),
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&domain_factory,
		boundaries,
		proof,
	)?;

	Ok(())
}
