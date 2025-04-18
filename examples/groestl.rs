// Copyright 2025 Irreducible Inc.

use std::{array, fs::File, iter::repeat_with};

use anyhow::Result;
use binius_core::{
	fiat_shamir::HasherChallenger,
	tower::{
		AESOptimalPackedTowerFamily, AESTowerFamily, CanonicalOptimalPackedTowerFamily,
		CanonicalTowerFamily, PackedTowerFamily,
	},
};
use binius_field::{
	arch::{OptimalUnderlier, OptimalUnderlier128b, OptimalUnderlierByteSliced},
	as_packed_field::PackedType,
	linear_transformation::PackedTransformationFactory,
	AESTowerField128b, ByteSliced16x512x1b, Field, PackedExtension, PackedField,
	PackedFieldIndexable, PackedSubfield,
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_m3::{
	builder::{
		ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment, WitnessIndex, B1,
		B128, B8,
	},
	gadgets::hash::groestl,
};
use binius_utils::rayon::adjust_thread_pool;
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use rand::{rngs::StdRng, thread_rng, SeedableRng};
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

#[derive(Debug)]
pub struct PermutationTable {
	table_id: TableId,
	permutation: groestl::Permutation,
}

impl PermutationTable {
	pub fn new(cs: &mut ConstraintSystem, pq: groestl::PermutationVariant) -> Self {
		let mut table = cs.add_table(format!("Grøstl {pq} permutation"));

		let state_in_bytes = table.add_committed_multiple::<B8, 8, 8>("state_in_bytes");
		let permutation = groestl::Permutation::new(&mut table, pq, state_in_bytes);

		Self {
			table_id: table.id(),
			permutation,
		}
	}
}

impl<P> TableFiller<P> for PermutationTable
where
	P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B8>,
	PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
{
	type Event = [B8; 64];

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &mut TableWitnessSegment<P>,
	) -> Result<()> {
		self.permutation.populate_state_in(witness, rows)?;
		self.permutation.populate(witness)?;
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
	println!("Verifying {} Grøstl-256 P permutations", n_permutations);

	let allocator = bumpalo::Bump::new();
	let mut cs = ConstraintSystem::new();
	let table = PermutationTable::new(&mut cs, groestl::PermutationVariant::P);

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![n_permutations],
	};

	let mut rng = StdRng::seed_from_u64(0);
	let events = repeat_with(|| array::from_fn::<_, 64, _>(|_| <B8 as Field>::random(&mut rng)))
		.take(n_permutations)
		.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);
	witness.fill_table_sequential(&table, &events)?;
	drop(trace_gen_scope);

	// let ccs = cs
	// 	.convert_to_tower::<CanonicalTowerFamily, AESTowerFamily>()
	// 	.compile::<AESOptimalPackedTowerFamily>(&statement)
	// 	.unwrap();

	let prover_cs = cs.convert_to_tower::<CanonicalTowerFamily, AESTowerFamily>();

	let witness = witness
		.repack::<CanonicalOptimalPackedTowerFamily, AESOptimalPackedTowerFamily>(&prover_cs)
		.unwrap();

	let ccs = prover_cs
		.compile::<AESOptimalPackedTowerFamily>(&statement)
		.unwrap();

	let proof = binius_core::constraint_system::prove::<
		OptimalUnderlier,
		AESTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
		_,
	>(
		&ccs,
		args.log_inv_rate as usize,
		SECURITY_BITS,
		&statement.boundaries,
		witness.into_multilinear_extension_index::<AESOptimalPackedTowerFamily>(),
		&binius_hal::make_portable_backend(),
	)
	.unwrap();

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	let statement = statement.convert_field();

	let ccs = cs
		.compile::<CanonicalOptimalPackedTowerFamily>(&statement)
		.unwrap();

	binius_core::constraint_system::verify::<
		OptimalUnderlier128b,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&ccs, args.log_inv_rate as usize, SECURITY_BITS, &statement.boundaries, proof)
	.unwrap();

	Ok(())
}
