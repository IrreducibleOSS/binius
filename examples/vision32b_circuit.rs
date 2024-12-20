// Copyright 2024 Irreducible Inc.

//! Example of a Binius SNARK that proves execution of [Vision Mark-32] permutations.
//!
//! The arithmetization uses committed columns of 32-bit binary tower field elements. Every row of
//! the trace attests to the validity of 2 Vision rounds. Each permutation consists of 16 rounds.
//!
//! [Vision Mark-32]: https://eprint.iacr.org/2024/633

use std::array;

use anyhow::Result;
use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{
	constraint_system, fiat_shamir::HasherChallenger, oracle::OracleId, tower::CanonicalTowerFamily,
};
use binius_field::{
	arch::OptimalUnderlier, BinaryField128b, BinaryField32b, BinaryField64b, BinaryField8b,
};
use binius_hal::make_portable_backend;
use binius_hash::{GroestlDigestCompression, GroestlHasher};
use binius_math::IsomorphicEvaluationDomainFactory;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use clap::{value_parser, Parser};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 256, value_parser = value_parser!(u32).range(1 << 8..))]
	n_permutations: u32,
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

	println!("Verifying {} Vision-32b permutations", args.n_permutations);

	let log_n_permutations = log2_ceil_usize(args.n_permutations as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, BinaryField128b>::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let state_in: [OracleId; 24] = array::from_fn(|i| {
		binius_circuits::unconstrained::unconstrained::<_, _, BinaryField32b>(
			&mut builder,
			format!("p_in_{i}"),
			log_n_permutations,
		)
		.unwrap()
	});
	let _state_out =
		binius_circuits::vision::vision_permutation(&mut builder, log_n_permutations, state_in)?;
	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");
	let constraint_system = builder.build()?;

	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
	let backend = make_portable_backend();

	let proof = constraint_system::prove::<
		U,
		CanonicalTowerFamily,
		BinaryField64b,
		_,
		_,
		GroestlHasher<BinaryField128b>,
		GroestlDigestCompression<BinaryField8b>,
		HasherChallenger<groestl_crypto::Groestl256>,
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
		HasherChallenger<groestl_crypto::Groestl256>,
	>(
		&constraint_system.no_base_constraints(),
		args.log_inv_rate as usize,
		SECURITY_BITS,
		vec![],
		proof,
	)?;

	Ok(())
}
