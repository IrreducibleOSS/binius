// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::{fiat_shamir::HasherChallenger, tower::CanonicalTowerFamily};
use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_m3::{
	builder::{ConstraintSystem, Statement, WitnessIndex, B128, B32, B64},
	gadgets::mul::{MulUU32, MulUU64},
};
use bumpalo::Bump;
use rand::{prelude::StdRng, Rng, SeedableRng};

#[test]
fn test_muluu64() {
	let mut cs = ConstraintSystem::new();
	let mut table = cs.add_table("Test");

	let muluu = MulUU64::new(&mut table, "64bit");

	let table_id = table.id();

	let allocator = Bump::new();

	const TABLE_SIZE: usize = 1 << 8;

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![TABLE_SIZE],
	};
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

	let table_witness = witness.init_table(table_id, TABLE_SIZE).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let x_vals = repeat_with(|| B64::new(rng.gen::<u64>()))
		.take(1 << 8)
		.collect::<Vec<_>>();
	let y_vals = repeat_with(|| B64::new(rng.gen::<u64>()))
		.take(1 << 8)
		.collect::<Vec<_>>();

	let mut segment = table_witness.full_segment();
	muluu
		.populate_with_inputs(&mut segment, x_vals.clone(), y_vals.clone())
		.unwrap();

	let ccs = cs.compile(&statement).unwrap();
	let witness = witness.into_multilinear_extension_index();

	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 100;

	let proof = binius_core::constraint_system::prove::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&ccs,
		LOG_INV_RATE,
		SECURITY_BITS,
		&statement.boundaries,
		witness,
		&binius_hal::make_portable_backend(),
	)
	.unwrap();

	// TODO: ADD Test here to query witness data for correct high and low outputs.

	binius_core::constraint_system::verify::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&ccs, LOG_INV_RATE, SECURITY_BITS, &statement.boundaries, proof)
	.unwrap();
}

#[test]
fn test_muluu32() {
	let mut cs = ConstraintSystem::new();
	let mut table = cs.add_table("Test");

	let muluu = MulUU32::new(&mut table, "32bit");

	let table_id = table.id();

	let allocator = Bump::new();

	const TABLE_SIZE: usize = 1 << 8;

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![TABLE_SIZE],
	};
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

	let table_witness = witness.init_table(table_id, TABLE_SIZE).unwrap();

	let mut rng = StdRng::seed_from_u64(0);
	let x_vals = repeat_with(|| B32::new(rng.gen::<u32>()))
		.take(1 << 8)
		.collect::<Vec<_>>();
	let y_vals = repeat_with(|| B32::new(rng.gen::<u32>()))
		.take(1 << 8)
		.collect::<Vec<_>>();

	let mut segment = table_witness.full_segment();
	muluu
		.populate_with_inputs(&mut segment, x_vals.clone(), y_vals.clone())
		.unwrap();

	let ccs = cs.compile(&statement).unwrap();
	let witness = witness.into_multilinear_extension_index();

	const LOG_INV_RATE: usize = 1;
	const SECURITY_BITS: usize = 100;

	let proof = binius_core::constraint_system::prove::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&ccs,
		LOG_INV_RATE,
		SECURITY_BITS,
		&statement.boundaries,
		witness,
		&binius_hal::make_portable_backend(),
	)
	.unwrap();

	// TODO: ADD Test here to query witness data for correct high and low outputs.

	binius_core::constraint_system::verify::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&ccs, LOG_INV_RATE, SECURITY_BITS, &statement.boundaries, proof)
	.unwrap();
}
