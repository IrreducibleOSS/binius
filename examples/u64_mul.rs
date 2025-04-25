// Copyright 2024-2025 Irreducible Inc.

use std::iter::repeat_with;

use anyhow::Result;
use binius_circuits::builder::types::U;
use binius_core::{
	constraint_system,
	fiat_shamir::HasherChallenger,
	tower::{CanonicalOptimalPackedTowerFamily, CanonicalTowerFamily},
};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, Field, PackedExtension,
	PackedFieldIndexable,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_m3::{
	builder::{
		ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment, WitnessIndex, B1,
		B128, B64,
	},
	gadgets::mul::MulUU64,
};
use binius_utils::{checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool};
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use rand::thread_rng;
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

#[derive(Debug)]
pub struct MulTable {
	table_id: TableId,
	mul_table: MulUU64,
}

impl MulTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("Mulu64 Example");
		let mul_table = MulUU64::new(&mut table);

		Self {
			table_id: table.id(),
			mul_table,
		}
	}
}

impl<P> TableFiller<P> for MulTable
where
	P: PackedFieldIndexable<Scalar = B128> + PackedExtension<B1> + PackedExtension<B64>,
{
	type Event = (B64, B64);

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment<P>,
	) -> Result<()> {
		let (x_vals, y_vals): (Vec<_>, Vec<_>) = rows.cloned().unzip();
		self.mul_table.populate_with_inputs(witness, x_vals, y_vals)
	}
}

fn main() -> Result<()> {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} u64 multiplication", args.n_muls);

	let log_n_muls = log2_ceil_usize(args.n_muls as usize);

	let allocator = bumpalo::Bump::new();
	let mut cs = ConstraintSystem::new();
	let table = MulTable::new(&mut cs);

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![1 << log_n_muls],
	};

	let mut rng = thread_rng();
	let events = repeat_with(|| (B64::random(&mut rng), B64::random(&mut rng)))
		.take(1 << log_n_muls)
		.collect::<Vec<_>>();

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);
	witness.fill_table_sequential(&table, &events)?;

	drop(trace_gen_scope);

	let constraint_system = cs.compile::<CanonicalOptimalPackedTowerFamily>(&statement)?;
	let witness = witness.into_multilinear_extension_index::<CanonicalOptimalPackedTowerFamily>();

	let backend = make_portable_backend();

	let proof =
		constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256,
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
