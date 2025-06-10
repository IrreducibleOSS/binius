// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use anyhow::Result;
use binius_core::{constraint_system, fiat_shamir::HasherChallenger};
use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
use binius_field::{
	PackedExtension, PackedFieldIndexable, PackedSubfield, arch::OptimalUnderlier,
	as_packed_field::PackedType, linear_transformation::PackedTransformationFactory,
	tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::{
	builder::{
		B1, B8, B64, B128, ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment,
		WitnessIndex,
	},
	gadgets::hash::keccak::{StateMatrix, stacked::Keccakf},
};
use binius_utils::rayon::adjust_thread_pool;
use bytemuck::zeroed_vec;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{RngCore, thread_rng};
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
	keccakf: Keccakf,
}

impl PermutationTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("Keccak permutation");

		let keccakf = Keccakf::new(&mut table);

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
		+ PackedExtension<B64>,
	PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
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

	let allocator = bumpalo::Bump::new();
	let mut cs = ConstraintSystem::new();
	let table = PermutationTable::new(&mut cs);

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![n_permutations],
	};

	let mut rng = thread_rng();
	let events = repeat_with(|| StateMatrix::from_fn(|_| rng.next_u64()))
		.take(n_permutations)
		.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!("Generating trace", n_permutations).entered();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);
	witness.fill_table_parallel(&table, &events)?;
	drop(trace_gen_scope);

	let ccs = cs.compile(&statement).unwrap();
	let cs_digest = ccs.digest::<Groestl256>();
	let witness = witness.into_multilinear_extension_index();

	let hal = FastCpuLayer::<CanonicalTowerFamily, PackedType<OptimalUnderlier, B128>>::default();

	let mut host_mem = zeroed_vec(1 << 20);
	let mut dev_mem_owned = zeroed_vec(1 << (28 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));

	let dev_mem = PackedMemorySliceMut::new_slice(&mut dev_mem_owned);

	let proof = constraint_system::prove::<
		_,
		OptimalUnderlier,
		CanonicalTowerFamily,
		Groestl256Parallel,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&hal,
		&mut host_mem,
		dev_mem,
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&cs_digest,
		&statement.boundaries,
		&statement.table_sizes,
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
