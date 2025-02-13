// Copyright 2024-2025 Irreducible Inc.

use std::{fmt::Display, str::FromStr};

use anyhow::Result;
use binius_circuits::builder::{types::U, ConstraintSystemBuilder};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_field::{BinaryField1b, BinaryField32b, TowerField};
use binius_hal::make_portable_backend;
use binius_hash::compress::Groestl256ByteCompression;
use binius_macros::arith_expr;
use binius_math::DefaultEvaluationDomainFactory;
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use groestl_crypto::Groestl256;
use tracing_profile::init_tracing;

#[derive(Debug, Clone, Copy)]
enum BitwiseOp {
	And,
	Xor,
	Or,
}

impl FromStr for BitwiseOp {
	type Err = anyhow::Error;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s.to_ascii_lowercase().as_str() {
			"and" => Ok(Self::And),
			"xor" => Ok(Self::Xor),
			"or" => Ok(Self::Or),
			_ => Err(anyhow::anyhow!("Unknown bitwise op")),
		}
	}
}

impl Display for BitwiseOp {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let str = match self {
			BitwiseOp::And => String::from("And"),
			BitwiseOp::Xor => String::from("Xor"),
			BitwiseOp::Or => String::from("Or"),
		};
		write!(f, "{}", str)
	}
}

#[derive(Debug, Parser)]
struct Args {
	/// The operation to perform.
	#[arg(long, default_value_t = BitwiseOp::And)]
	op: BitwiseOp,
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 32, value_parser = value_parser!(u32).range(32..))]
	n_u32_ops: u32,
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

	println!("Verifying {} bitwise u32 {}'s", args.n_u32_ops, args.op);

	let log_n_1b_operations =
		log2_ceil_usize(args.n_u32_ops as usize) + BinaryField32b::TOWER_LEVEL;

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	// Assuming our 32bit values have been committed as bits
	let in_a = binius_circuits::unconstrained::unconstrained::<BinaryField1b>(
		&mut builder,
		"in_a",
		log_n_1b_operations,
	)?;
	let in_b = binius_circuits::unconstrained::unconstrained::<BinaryField1b>(
		&mut builder,
		"in_b",
		log_n_1b_operations,
	)?;
	let _result = match args.op {
		BitwiseOp::And => binius_circuits::bitwise::and(&mut builder, "a_and_b", in_a, in_b),
		BitwiseOp::Xor => {
			let out = binius_circuits::bitwise::xor(&mut builder, "a_xor_b", in_a, in_b)?;
			// TODO: Assert equality so that something is constrained.
			builder.assert_zero("zero", [in_a], arith_expr!([x] = x - x).convert_field());
			Ok(out)
		}
		BitwiseOp::Or => binius_circuits::bitwise::or(&mut builder, "a_or_b", in_a, in_b),
	};
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
