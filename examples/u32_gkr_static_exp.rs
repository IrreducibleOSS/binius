// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_circuits::builder::{types::U, ConstraintSystemBuilder};
use binius_core::{
	constraint_system::{self, exp::ExpBase},
	fiat_shamir::HasherChallenger,
	tower::CanonicalTowerFamily,
};
use binius_field::{BinaryField, BinaryField1b, BinaryField64b, TowerField};
use binius_hal::make_portable_backend;
use binius_hash::compress::Groestl256ByteCompression;
use binius_math::DefaultEvaluationDomainFactory;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use groestl_crypto::Groestl256;
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of multiplication to do.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(512..))]
	n_exp: u32,
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

	println!("Verifying {} u32 static exponentiation", args.n_exp);

	let log_n_exp = log2_ceil_usize(args.n_exp as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let in_a = (0..32)
		.map(|i| {
			binius_circuits::unconstrained::unconstrained::<BinaryField1b>(
				&mut builder,
				format!("in_a_{}", i),
				log_n_exp,
			)
			.unwrap()
		})
		.collect::<Vec<_>>();

	let a_exp_res = builder.add_committed("a_exp_res", log_n_exp, BinaryField64b::TOWER_LEVEL);

	builder.add_exp(
		in_a,
		a_exp_res,
		ExpBase::Static(BinaryField64b::MULTIPLICATIVE_GENERATOR.into()),
		BinaryField64b::TOWER_LEVEL,
	);

	builder.assert_not_zero(a_exp_res);

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
