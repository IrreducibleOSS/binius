// Copyright 2025 Irreducible Inc.

// Uses binius_circuits which is being phased out.
#![allow(deprecated)]

use std::iter::repeat_with;

use anyhow::Result;
use binius_circuits::builder::types::U;
use binius_core::fiat_shamir::HasherChallenger;
use binius_field::{
	PackedExtension, PackedFieldIndexable, PackedSubfield, arch::OptimalUnderlier,
	as_packed_field::PackedType, linear_transformation::PackedTransformationFactory,
	tower::CanonicalTowerFamily,
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_m3::{
	builder::{
		B1, B8, B64, B128, ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment,
		WitnessIndex,
	},
	gadgets::hash::keccak::{StateMatrix, stacked::Keccakf},
};
use binius_utils::rayon::adjust_thread_pool;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{RngCore, thread_rng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of leaves to verify.
	#[arg(short, long, default_value_t = 1<<20, value_parser = value_parser!(u32).range(1 << 9..))]
	n_leaves: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

fn main() -> Result<()> {
	let merkle_tree_cs = MerkleTreeCS::new(&mut cs);

	let mut rng = StdRng::seed_from_u64(0);
	// Create a Merkle tree with 8 leaves
	let index = rng.gen_range(0..1 << 10);
	let leaves = (0..3)
		.map(|_| {
			(0..1 << 10)
				.map(|_| rng.r#gen::<[u8; 32]>())
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();

	let trees = (0..3)
		.map(|i| MerkleTree::new(&leaves[i]))
		.collect::<Vec<_>>();
	let roots = (0..3).map(|i| trees[i].root()).collect::<Vec<_>>();
	let paths = trees
		.iter()
		.enumerate()
		.map(|(i, tree)| MerklePath {
			root_id: i as u8,
			index,
			leaf: leaves[i][index],
			nodes: tree.merkle_path(index),
		})
		.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let trace = MerkleTreeTrace::generate(roots, &paths);

	// Allocate memory for the witness
	let allocator = Bump::new();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);

	// Fill the tables with the trace
	merkle_tree_cs.fill_tables(&trace, &cs, &mut witness);

	// Create boundary values based on the trace's boundaries
	let boundaries = merkle_tree_cs.make_boundaries(&trace);
	drop(trace_gen_scope);

	let statement = Statement {
		boundaries,
		table_sizes: vec![n_permutations],
	};
	let ccs = cs.compile(&statement).unwrap();
	let witness = witness.into_multilinear_extension_index();

	let proof = binius_core::constraint_system::prove::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&statement.boundaries,
		witness,
		&binius_hal::make_portable_backend(),
	)
	.unwrap();

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	binius_core::constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&ccs, args.log_inv_rate as usize, SECURITY_BITS, &statement.boundaries, proof)
	.unwrap();

	Ok(())
}
