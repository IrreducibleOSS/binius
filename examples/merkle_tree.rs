// Copyright 2025 Irreducible Inc.

use anyhow::Result;
use binius_compute::{ComputeHolder, cpu::alloc::CpuComputeAllocator};
use binius_core::{constraint_system, fiat_shamir::HasherChallenger};
use binius_fast_compute::layer::FastCpuLayerHolder;
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::{
	builder::{B128, ConstraintSystem, Statement, WitnessIndex},
	gadgets::merkle_tree::{
		MerkleTreeCS,
		trace::{MerklePath, MerkleTree, MerkleTreeTrace},
	},
};
use binius_utils::rayon::adjust_thread_pool;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of leaves in the merkle tree.
	#[arg(long, default_value_t = 20, value_parser = value_parser!(u32).range(1..))]
	log_leaves: u32,

	/// The number of Merkle paths to verify.
	#[arg(short,long, default_value_t = 10, value_parser = value_parser!(u32).range(1..))]
	log_paths: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 2, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

fn main() -> Result<()> {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	let mut cs = ConstraintSystem::new();
	let merkle_tree_cs = MerkleTreeCS::new(&mut cs);

	let mut rng = StdRng::seed_from_u64(0);
	// Create a Merkle tree with 8 leaves
	let leaves = (0..1 << args.log_leaves)
		.map(|_| rng.r#gen::<[u8; 32]>())
		.collect::<Vec<_>>();

	let tree = MerkleTree::new(&leaves);

	let roots = tree.root();
	let paths = (0..1 << args.log_paths)
		.map(|_| {
			let index = rng.gen_range(0..1 << args.log_leaves);
			MerklePath {
				root_id: 0,
				index,
				leaf: leaves[index],
				nodes: tree.merkle_path(index),
			}
		})
		.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!(
		"Generating trace",
		log_leaves = args.log_leaves,
		log_paths = args.log_paths
	)
	.entered();
	let trace = MerkleTreeTrace::generate(vec![roots], &paths);

	// Allocate memory for the witness
	let mut allocator =
		CpuComputeAllocator::new(1 << (22 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));
	let allocator = allocator.into_bump_allocator();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

	// Fill the tables with the trace
	merkle_tree_cs.fill_tables(&trace, &cs, &mut witness)?;

	// Create boundary values based on the trace's boundaries
	let boundaries = merkle_tree_cs.make_boundaries(&trace);
	drop(trace_gen_scope);

	let table_sizes = witness.table_sizes();

	let statement = Statement::<B128> {
		boundaries,
		table_sizes,
	};

	let ccs = cs.compile(&statement).unwrap();
	let cs_digest = ccs.digest::<Groestl256>();
	let witness = witness.into_multilinear_extension_index();

	let hal_span = tracing::info_span!("HAL Setup", perfetto_category = "phase.main").entered();

	let mut compute_holder = FastCpuLayerHolder::<
		CanonicalTowerFamily,
		PackedType<OptimalUnderlier, B128>,
	>::new(1 << 20, 1 << 28);

	drop(hal_span);

	let proof = constraint_system::prove::<
		_,
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256Parallel,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
		_,
		_,
	>(
		&mut compute_holder.to_data(),
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&cs_digest,
		&statement.boundaries,
		&statement.table_sizes,
		witness,
		&make_portable_backend(),
	)
	.unwrap();

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	binius_core::constraint_system::verify::<
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&cs_digest,
		&statement.boundaries,
		proof,
	)?;

	Ok(())
}
