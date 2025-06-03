// Copyright 2025 Irreducible Inc.

// Uses binius_circuits which is being phased out.
#![allow(deprecated)]

use std::iter::repeat_with;

use anyhow::Result;
use binius_circuits::builder::types::U;
use binius_core::{constraint_system::channel::ChannelId, fiat_shamir::HasherChallenger};
use binius_field::{
	PackedExtension, PackedFieldIndexable, PackedSubfield, arch::OptimalUnderlier,
	as_packed_field::PackedType, linear_transformation::PackedTransformationFactory,
	tower::CanonicalTowerFamily,
};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::{
	builder::{
		B1, B8, B64, B128, Boundary, ConstraintSystem, FlushDirection, Statement, TableFiller,
		TableId, TableWitnessSegment, WitnessIndex,
	},
	gadgets::hash::keccak::{
		StateMatrix,
		stacked::Keccakf,
		trace::{self, PermutationTrace, RoundTrace},
	},
};
use binius_utils::rayon::adjust_thread_pool;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{RngCore, thread_rng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(2..))]
	n_permutations: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

pub struct PermutationTable {
	table_id: TableId,
	channel_id: ChannelId,
	keccakf: Keccakf,
}

impl PermutationTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let channel_id = cs.add_channel("channel");
		let mut table = cs.add_table("Keccak permutation");

		let keccakf = Keccakf::new(&mut table, channel_id);

		Self {
			table_id: table.id(),
			channel_id,
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
	type Event = PermutationTrace;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &mut TableWitnessSegment<P>,
	) -> Result<()> {
		let rows = rows.cloned().collect::<Vec<_>>();
		self.keccakf.populate(&rows, witness)?;
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

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

	let mut permutations = Vec::with_capacity(n_permutations);
	let mut state = StateMatrix::default();
	for _ in 0..n_permutations {
		let permutation_trace = trace::keccakf_trace(state);
		state = permutation_trace.output().clone();
		permutations.push(permutation_trace);
	}

	let statement = Statement {
		boundaries: vec![
			Boundary {
				values: permutations[0]
					.input()
					.as_inner()
					.iter()
					.map(|v| B128::from(*v as u128))
					.collect(),
				channel_id: table.channel_id,
				direction: FlushDirection::Push,
				multiplicity: 1,
			},
			Boundary {
				values: permutations[n_permutations - 1][11]
					.state_out
					.as_inner()
					.iter()
					.map(|v| B128::from(*v as u128))
					.collect(),
				channel_id: table.channel_id,
				direction: FlushDirection::Pull,
				multiplicity: 1,
			},
		],
		table_sizes: vec![n_permutations],
	};

	witness.fill_table_parallel(&table, &permutations)?;
	drop(trace_gen_scope);

	let ccs = cs.compile(&statement).unwrap();
	let witness = witness.into_multilinear_extension_index();

	let proof = binius_core::constraint_system::prove::<
		U,
		CanonicalTowerFamily,
		Groestl256Parallel,
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
