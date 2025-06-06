// Copyright 2024-2025 Irreducible Inc.

// Uses binius_circuits which is being phased out.
#![allow(deprecated)]

use std::array;

use alloy_primitives::U512;
use anyhow::Result;
use binius_circuits::{
	builder::{ConstraintSystemBuilder, types::U},
	lasso::big_integer_ops::{byte_sliced_modular_mul, byte_sliced_test_utils::random_u512},
	transparent,
};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger};
use binius_field::{
	BinaryField1b, BinaryField8b, Field, TowerField,
	tower::CanonicalTowerFamily,
	tower_levels::{TowerLevel4, TowerLevel8},
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::thread_rng;
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of operations to verify.
	#[arg(short, long, default_value_t = 4096, value_parser = value_parser!(u32).range(1 << 5..))]
	n_multiplications: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

fn main() -> Result<()> {
	type B8 = BinaryField8b;
	const SECURITY_BITS: usize = 100;
	const WIDTH: usize = 4;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} u32 modular multiplications", args.n_multiplications);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
	let log_size = log2_ceil_usize(args.n_multiplications as usize);

	let mut rng = thread_rng();

	let mult_a = builder.add_committed_multiple::<WIDTH>("a", log_size, B8::TOWER_LEVEL);
	let mult_b = builder.add_committed_multiple::<WIDTH>("b", log_size, B8::TOWER_LEVEL);

	let input_bitmask = (U512::from(1u8) << (8 * WIDTH)) - U512::from(1u8);

	let modulus = (random_u512(&mut rng) % input_bitmask) + U512::from(1u8);

	if let Some(witness) = builder.witness() {
		let mut mult_a: [_; WIDTH] =
			array::from_fn(|byte_idx| witness.new_column::<BinaryField8b>(mult_a[byte_idx]));

		let mult_a_u8 = mult_a.each_mut().map(|col| col.as_mut_slice::<u8>());

		let mut mult_b: [_; WIDTH] =
			array::from_fn(|byte_idx| witness.new_column::<BinaryField8b>(mult_b[byte_idx]));

		let mult_b_u8 = mult_b.each_mut().map(|col| col.as_mut_slice::<u8>());

		for row_idx in 0..1 << log_size {
			let mut a = random_u512(&mut rng);
			let mut b = random_u512(&mut rng);

			a %= modulus;
			b %= modulus;

			for byte_idx in 0..WIDTH {
				mult_a_u8[byte_idx][row_idx] = a.byte(byte_idx);
				mult_b_u8[byte_idx][row_idx] = b.byte(byte_idx);
			}
		}
	}

	let modulus_input: [_; WIDTH] = array::from_fn(|byte_idx| modulus.byte(byte_idx));

	let zero_oracle_byte =
		transparent::constant(&mut builder, "zero byte", log_size, BinaryField8b::ZERO).unwrap();

	let zero_oracle_carry =
		transparent::constant(&mut builder, "zero carry", log_size, BinaryField1b::ZERO).unwrap();

	let _modded_product = byte_sliced_modular_mul::<TowerLevel4, TowerLevel8>(
		&mut builder,
		"lasso_bytesliced_mul",
		&mult_a,
		&mult_b,
		&modulus_input,
		log_size,
		zero_oracle_byte,
		zero_oracle_carry,
	)
	.unwrap();

	let witness = builder.take_witness().unwrap();
	let constraint_system = builder.build().unwrap();
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
