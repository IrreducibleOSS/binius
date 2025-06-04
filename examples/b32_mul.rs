// Copyright 2024-2025 Irreducible Inc.

// Uses binius_circuits which is being phased out.
#![allow(deprecated)]

use anyhow::Result;
use binius_circuits::builder::{ConstraintSystemBuilder, types::U};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger};
use binius_field::{BinaryField32b, TowerField, tower::CanonicalTowerFamily};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_macros::arith_expr;

use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use itertools::izip;
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of operations to do.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(512..))]
	n_ops: u32,
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

	println!("Verifying {} number of BinaryField32b multiplication", args.n_ops);

	let log_n_muls = log2_ceil_usize(args.n_ops as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();

	let in_a = binius_circuits::unconstrained::unconstrained::<BinaryField32b>(
		&mut builder,
		"in_a",
		log_n_muls,
	)
	.unwrap();

	let in_b = binius_circuits::unconstrained::unconstrained::<BinaryField32b>(
		&mut builder,
		"in_b",
		log_n_muls,
	)
	.unwrap();
	let out = builder.add_committed("out", log_n_muls, BinaryField32b::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		let in_a_witness = witness
			.get::<BinaryField32b>(in_a)?
			.as_slice::<BinaryField32b>();
		let in_b_witness = witness
			.get::<BinaryField32b>(in_b)?
			.as_slice::<BinaryField32b>();
		let mut out_witness = witness.new_column::<BinaryField32b>(out);

		let out_scalars = out_witness.as_mut_slice::<BinaryField32b>();

		for (&a, &b, out) in izip!(in_a_witness, in_b_witness, out_scalars) {
			*out = a * b;
		}
	}

	builder.assert_zero(
		"b32_mul",
		[in_a, in_b, out],
		binius_math::ArithCircuit::from(
			binius_math::ArithExpr::<binius_field::BinaryField1b>::Var(0usize)
				* binius_math::ArithExpr::<binius_field::BinaryField1b>::Var(1usize)
				- binius_math::ArithExpr::<binius_field::BinaryField1b>::Var(2usize),
		)
		.convert_field(),
	);

	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");
	let constraint_system = builder.build()?;

	let backend = make_portable_backend();

	let proof =
		constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256Parallel,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, &[], witness, &backend)?;

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
