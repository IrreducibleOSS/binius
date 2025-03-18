// Copyright 2024-2025 Irreducible Inc.

use std::array;

use anyhow::Result;
use binius_circuits::{
	arithmetic::static_exp::u16_static_exp_lookups,
	builder::{types::U, ConstraintSystemBuilder},
};
use binius_core::{
	constraint_system, fiat_shamir::HasherChallenger, oracle::OracleId, tower::CanonicalTowerFamily,
};
use binius_field::{BinaryField, BinaryField16b, BinaryField64b, Field, TowerField};
use binius_hal::make_portable_backend;
use binius_hash::compress::Groestl256ByteCompression;
use binius_macros::arith_expr;
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

	let log_n_muls = log2_ceil_usize(args.n_exp as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let in_a: [OracleId; 2] = array::from_fn(|_| {
		binius_circuits::unconstrained::unconstrained::<BinaryField16b>(
			&mut builder,
			"in_a",
			log_n_muls,
		)
		.unwrap()
	});

	let (a_low_exp_res_id, _) = u16_static_exp_lookups::<32>(
		&mut builder,
		"g^a",
		in_a[0],
		BinaryField64b::MULTIPLICATIVE_GENERATOR,
		None,
	)
	.unwrap();

	let (a_high_exp_res_id, _) = u16_static_exp_lookups::<32>(
		&mut builder,
		"g^a",
		in_a[0],
		BinaryField64b::MULTIPLICATIVE_GENERATOR.pow([1 << 16]),
		None,
	)
	.unwrap();

	let a_exp_res_id =
		builder.add_committed("a_exp_result", log_n_muls, BinaryField64b::TOWER_LEVEL);

	builder.assert_zero(
		"a_exp_res_id zerocheck",
		[a_low_exp_res_id, a_high_exp_res_id, a_exp_res_id],
		arith_expr!(
			[a_low_exp_res, a_high_exp_res, a_exp_result_id] =
				a_low_exp_res * a_high_exp_res - a_exp_result_id
		)
		.convert_field(),
	);

	if let Some(witness) = builder.witness() {
		let a_low_exp_res = witness
			.get::<BinaryField64b>(a_low_exp_res_id)?
			.as_slice::<BinaryField64b>();

		let a_high_exp_res = witness
			.get::<BinaryField64b>(a_high_exp_res_id)?
			.as_slice::<BinaryField64b>();

		let mut a_exp_res = witness.new_column::<BinaryField64b>(a_exp_res_id);
		let a_exp_res = a_exp_res.as_mut_slice::<BinaryField64b>();
		a_exp_res.iter_mut().enumerate().for_each(|(i, a_exp_res)| {
			*a_exp_res = a_low_exp_res[i] * a_high_exp_res[i];
		});
	}

	builder.assert_not_zero(a_exp_res_id);

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
