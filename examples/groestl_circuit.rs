// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_circuits::builder::ConstraintSystemBuilder;
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::AESTowerFamily};
use binius_field::{
	arch::OptimalUnderlier, AESTowerField128b, AESTowerField16b, AESTowerField8b, BinaryField8b,
};
use binius_hal::make_portable_backend;
use binius_hash::{Groestl256, GroestlDigestCompression};
use binius_math::IsomorphicEvaluationDomainFactory;
use binius_utils::{
	checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool, tracing::init_tracing,
};
use clap::{value_parser, Parser};

const LOG_ROWS_PER_PERMUTATION: usize = 0;

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(1 << 9..))]
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

	println!("Verifying {} Grøstl-256 P permutations", args.n_permutations);

	let log_n_permutations = log2_ceil_usize(args.n_permutations as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder =
		ConstraintSystemBuilder::<U, AESTowerField128b, AESTowerField16b>::new_with_witness(
			&allocator,
		);
	let _state_out = binius_circuits::groestl::groestl_p_permutation(
		&mut builder,
		log_n_permutations + LOG_ROWS_PER_PERMUTATION,
	);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");
	let constraint_system = builder.build()?;

	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
	let backend = make_portable_backend();

	let proof = constraint_system::prove::<
		U,
		AESTowerFamily,
		_,
		_,
		_,
		Groestl256<AESTowerField128b, _>,
		GroestlDigestCompression<AESTowerField8b>,
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
		AESTowerFamily,
		_,
		_,
		_,
		_,
		HasherChallenger<groestl_crypto::Groestl256>,
	>(
		&constraint_system.no_base_constraints(),
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&domain_factory,
		proof,
	)?;

	Ok(())
}
