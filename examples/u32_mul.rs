// Copyright 2024 Irreducible Inc.

use std::array;

use anyhow::Result;
use binius_circuits::{
	builder::ConstraintSystemBuilder,
	lasso::{
		batch::LookupBatch,
		big_integer_ops::byte_sliced_mul,
		lookups::u8_arithmetic::{add_lookup, dci_lookup, mul_lookup},
	},
	transparent,
};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_field::{
	arch::OptimalUnderlier,
	tower_levels::{TowerLevel4, TowerLevel8},
	BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b, Field,
};
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
	n_muls: u32,
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

	println!("Verifying {} u32 multiplication", args.n_muls);

	let log_n_muls = log2_ceil_usize(args.n_muls as usize);

	let allocator = bumpalo::Bump::new();
	let mut builder = ConstraintSystemBuilder::<U, BinaryField128b>::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	// Assuming our input data is already transposed, i.e a length 4 array of B8's
	let in_a = array::from_fn(|i| {
		binius_circuits::unconstrained::unconstrained::<_, _, BinaryField8b>(
			&mut builder,
			format!("in_a_{}", i),
			log_n_muls,
		)
		.unwrap()
	});
	let in_b = array::from_fn(|i| {
		binius_circuits::unconstrained::unconstrained::<_, _, BinaryField8b>(
			&mut builder,
			format!("in_b_{}", i),
			log_n_muls,
		)
		.unwrap()
	});
	let zero_oracle_carry =
		transparent::constant(&mut builder, "zero carry", log_n_muls, BinaryField1b::ZERO).unwrap();

	let lookup_t_mul = mul_lookup(&mut builder, "mul lookup")?;
	let lookup_t_add = add_lookup(&mut builder, "add lookup")?;
	let lookup_t_dci = dci_lookup(&mut builder, "dci lookup")?;

	let mut lookup_batch_mul = LookupBatch::new(lookup_t_mul);
	let mut lookup_batch_add = LookupBatch::new(lookup_t_add);
	let mut lookup_batch_dci = LookupBatch::new(lookup_t_dci);
	let _mul_and_cout = byte_sliced_mul::<_, _, TowerLevel4, TowerLevel8>(
		&mut builder,
		"lasso_bytesliced_mul",
		&in_a,
		&in_b,
		log_n_muls,
		zero_oracle_carry,
		&mut lookup_batch_mul,
		&mut lookup_batch_add,
		&mut lookup_batch_dci,
	)?;
	lookup_batch_mul.execute::<U, BinaryField128b, BinaryField16b, BinaryField32b>(&mut builder)?;
	lookup_batch_add.execute::<U, BinaryField128b, BinaryField16b, BinaryField32b>(&mut builder)?;
	lookup_batch_dci.execute::<U, BinaryField128b, BinaryField16b, BinaryField32b>(&mut builder)?;

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
		BinaryField8b,
		_,
		Groestl256,
		Groestl256ByteCompression,
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

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, vec![], proof)?;

	Ok(())
}
