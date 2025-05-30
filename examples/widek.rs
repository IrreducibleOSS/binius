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
	gadgets::hash::keccak::{RoundTrace, StateMatrix, wide},
};
use binius_utils::rayon::adjust_thread_pool;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{RngCore, thread_rng};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
struct Args {
	/// The number of permutations to verify.
	#[arg(short, long, default_value_t = 8192, value_parser = value_parser!(u32).range(1 << 9..))]
	n_permutations: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

pub struct WideRoundTable {
	table_id: TableId,
	wide_round: wide::WideRound,
}

impl WideRoundTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("Keccak round");

		Self {
			table_id: table.id(),
			wide_round: wide::WideRound::new(&mut table),
		}
	}
}

impl<P> TableFiller<P> for WideRoundTable
where
	P: PackedFieldIndexable<Scalar = B128>
		+ PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B64>,
	PackedSubfield<P, B8>: PackedTransformationFactory<PackedSubfield<P, B8>>,
{
	type Event = RoundTrace;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &mut TableWitnessSegment<P>,
	) -> Result<()> {
		self.wide_round.populate(witness, rows)?;
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
	println!("Verifying {n_permutations} Keccakf rounds!");

	let allocator = bumpalo::Bump::new();
	let mut cs = ConstraintSystem::new();
	let table = WideRoundTable::new(&mut cs);

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![n_permutations],
	};

	let mut rng = thread_rng();
	let mut round_no = 0;
	let events = repeat_with(|| {
		RoundTrace::gather(StateMatrix::from_fn(|_| rng.next_u64()), {
			let round = round_no;
			round_no += 1;
			round % 24
		})
	})
	.take(n_permutations)
	.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);
	witness.fill_table_parallel(&table, &events)?;
	drop(trace_gen_scope);

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
