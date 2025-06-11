// Copyright 2025 Irreducible Inc.

use std::{cmp::Reverse, iter::repeat_with};

use anyhow::Result;
use binius_compute::{ComputeHolder, cpu::alloc::CpuComputeAllocator};
use binius_core::{
	constraint_system::{self, channel::ChannelId},
	fiat_shamir::HasherChallenger,
};
use binius_fast_compute::layer::FastCpuLayerHolder;
use binius_field::{
	PackedExtension, PackedFieldIndexable, PackedSubfield, arch::OptimalUnderlier,
	as_packed_field::PackedType, linear_transformation::PackedTransformationFactory,
	tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::{
	builder::{
		B1, B8, B32, B64, B128, ConstraintSystem, TableFiller, TableId, TableWitnessSegment,
		WitnessIndex, tally,
	},
	gadgets::{
		hash::keccak::{StateMatrix, lookedup::KeccakfLookedup},
		indexed_lookup::and::{BitAndIndexedLookup, BitAndLookup},
	},
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use itertools::Itertools;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(1 << 9..))]
	n_permutations: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

pub struct PermutationTable {
	table_id: TableId,
	keccakf: KeccakfLookedup,
}

impl PermutationTable {
	pub fn new(cs: &mut ConstraintSystem, lookup_channel: ChannelId) -> Self {
		let mut table = cs.add_table("Keccak permutation with lookup");

		let keccakf = KeccakfLookedup::new(&mut table, lookup_channel);

		Self {
			table_id: table.id(),
			keccakf,
		}
	}
}

impl<P> TableFiller<P> for PermutationTable
where
	P: PackedFieldIndexable<Scalar = B128>
		+ PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B32>
		+ PackedExtension<B64>,
	PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
	PackedSubfield<P, B32>: PackedFieldIndexable<Scalar = B32>,
{
	type Event = StateMatrix<u64>;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &mut TableWitnessSegment<P>,
	) -> Result<()> {
		self.keccakf.populate_state_in(witness, rows)?;
		self.keccakf.populate(witness)?;
		Ok(())
	}
}

fn main() -> Result<()> {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	let n_permutations = args.n_permutations as usize;
	println!("Verifying {n_permutations} Keccakf permutations");

	let mut allocator = CpuComputeAllocator::new(
		1 << (12 + log2_ceil_usize(n_permutations)
			- PackedType::<OptimalUnderlier, B128>::LOG_WIDTH),
	);
	let allocator = allocator.into_bump_allocator();
	let mut cs = ConstraintSystem::new();
	let lookup_chan = cs.add_channel("lookup");
	let permutation_chan = cs.add_channel("permutation");
	let mut lookup = cs.add_table("lookup");
	let bitand_lookup = BitAndLookup::new(&mut lookup, lookup_chan, permutation_chan, 10);

	let table = PermutationTable::new(&mut cs, lookup_chan);

	let mut rng = StdRng::seed_from_u64(0);
	let events = repeat_with(|| StateMatrix::from_fn(|_| rng.next_u64()))
		.take(n_permutations)
		.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!("Generating trace", n_permutations).entered();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);
	witness.fill_table_parallel(&table, &events)?;
	drop(trace_gen_scope);

	let counts = tally(&cs, &mut witness, &[], lookup_chan, &BitAndIndexedLookup).unwrap();
	// Fill the lookup table with the sorted counts
	let sorted_counts = counts
		.into_iter()
		.enumerate()
		.sorted_by_key(|(_, count)| Reverse(*count))
		.collect::<Vec<_>>();

	witness
		.fill_table_parallel(&bitand_lookup, &sorted_counts)
		.unwrap();

	let boundaries = vec![];
	let table_sizes = witness.table_sizes();

	let ccs = cs.compile().unwrap();
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
		&boundaries,
		&table_sizes,
		witness,
		&make_portable_backend(),
	)?;

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	binius_core::constraint_system::verify::<
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&ccs, args.log_inv_rate as usize, SECURITY_BITS, &cs_digest, &boundaries, proof)?;

	Ok(())
}
