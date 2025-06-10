// Copyright 2024-2025 Irreducible Inc.

use std::{fmt::Display, str::FromStr};

use anyhow::Result;
use binius_core::{constraint_system, fiat_shamir::HasherChallenger};
use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression, Groestl256Parallel};
use binius_m3::builder::{
	B1, B128, Col, Statement, TableBuilder, TableWitnessSegment, WitnessIndex,
	test_utils::ClosureFiller,
};
use binius_utils::rayon::adjust_thread_pool;
use bumpalo::Bump;
use bytemuck::zeroed_vec;
use bytesize::ByteSize;
use clap::{Parser, value_parser};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use tracing_profile::init_tracing;

#[derive(Debug, Clone, Copy)]
enum BitwiseOp {
	And,
	Xor,
	Or,
}

impl FromStr for BitwiseOp {
	type Err = anyhow::Error;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s.to_ascii_lowercase().as_str() {
			"and" => Ok(Self::And),
			"xor" => Ok(Self::Xor),
			"or" => Ok(Self::Or),
			_ => Err(anyhow::anyhow!("Unknown bitwise op")),
		}
	}
}

impl Display for BitwiseOp {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"{}",
			match self {
				BitwiseOp::And => "and",
				BitwiseOp::Xor => "xor",
				BitwiseOp::Or => "or",
			}
		)
	}
}

#[derive(Debug, Parser)]
struct Args {
	/// The operation to perform.
	#[arg(long, default_value_t = BitwiseOp::And)]
	op: BitwiseOp,
	/// The number of operations to verify.
	#[arg(short, long, default_value_t = 512, value_parser = value_parser!(u32).range(512..))]
	n_u32_ops: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

pub struct BitwiseAnd {
	pub xin: Col<B1, 32>,
	pub yin: Col<B1, 32>,
	pub zout: Col<B1, 32>,
}

impl BitwiseAnd {
	pub fn new(table: &mut TableBuilder, xin: Col<B1, 32>, yin: Col<B1, 32>) -> Self {
		let zout = table.add_committed::<B1, 32>("zout");
		table.assert_zero("and", zout - xin * yin);
		Self { xin, yin, zout }
	}

	pub fn populate<P>(&self, segment: &mut TableWitnessSegment<P>) -> Result<()>
	where
		P: binius_field::PackedFieldIndexable<Scalar = B128> + binius_field::PackedExtension<B1>,
	{
		let xin_values = segment.get_as::<u32, _, 32>(self.xin)?;
		let yin_values = segment.get_as::<u32, _, 32>(self.yin)?;
		let mut zout_values = segment.get_mut_as::<u32, _, 32>(self.zout)?;

		for i in 0..xin_values.len() {
			zout_values[i] = xin_values[i] & yin_values[i];
		}
		Ok(())
	}
}

pub struct BitwiseXor {
	pub xin: Col<B1, 32>,
	pub yin: Col<B1, 32>,
	pub zout: Col<B1, 32>,
}

impl BitwiseXor {
	pub fn new(table: &mut TableBuilder, xin: Col<B1, 32>, yin: Col<B1, 32>) -> Self {
		let zout = table.add_committed::<B1, 32>("zout");
		table.assert_zero("xor", zout - xin - yin);
		Self { xin, yin, zout }
	}

	pub fn populate<P>(&self, segment: &mut TableWitnessSegment<P>) -> Result<()>
	where
		P: binius_field::PackedFieldIndexable<Scalar = B128> + binius_field::PackedExtension<B1>,
	{
		let xin_values = segment.get_as::<u32, _, 32>(self.xin)?;
		let yin_values = segment.get_as::<u32, _, 32>(self.yin)?;
		let mut zout_values = segment.get_mut_as::<u32, _, 32>(self.zout)?;

		for i in 0..xin_values.len() {
			zout_values[i] = xin_values[i] ^ yin_values[i];
		}
		Ok(())
	}
}

pub struct BitwiseOr {
	pub xin: Col<B1, 32>,
	pub yin: Col<B1, 32>,
	pub zout: Col<B1, 32>,
}

impl BitwiseOr {
	pub fn new(table: &mut TableBuilder, xin: Col<B1, 32>, yin: Col<B1, 32>) -> Self {
		let zout = table.add_committed::<B1, 32>("zout");
		// z = x + y - x * y in binary field
		table.assert_zero("or", zout - xin - yin + xin * yin);
		Self { xin, yin, zout }
	}

	pub fn populate<P>(&self, segment: &mut TableWitnessSegment<P>) -> Result<()>
	where
		P: binius_field::PackedFieldIndexable<Scalar = B128> + binius_field::PackedExtension<B1>,
	{
		let xin_values = segment.get_as::<u32, _, 32>(self.xin)?;
		let yin_values = segment.get_as::<u32, _, 32>(self.yin)?;
		let mut zout_values = segment.get_mut_as::<u32, _, 32>(self.zout)?;

		for i in 0..xin_values.len() {
			zout_values[i] = xin_values[i] | yin_values[i];
		}
		Ok(())
	}
}

enum BitwiseGadget {
	And(BitwiseAnd),
	Xor(BitwiseXor),
	Or(BitwiseOr),
}

impl BitwiseGadget {
	fn populate<P>(&self, segment: &mut TableWitnessSegment<P>) -> Result<()>
	where
		P: binius_field::PackedFieldIndexable<Scalar = B128> + binius_field::PackedExtension<B1>,
	{
		match self {
			BitwiseGadget::And(gadget) => gadget.populate(segment),
			BitwiseGadget::Xor(gadget) => gadget.populate(segment),
			BitwiseGadget::Or(gadget) => gadget.populate(segment),
		}
	}
}

fn main() -> Result<()> {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args::parse();

	let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} bitwise u32 {}'s", args.n_u32_ops, args.op);

	let mut rng = StdRng::seed_from_u64(0);
	let test_vector: Vec<(u32, u32)> = (0..args.n_u32_ops)
		.map(|_| (rng.r#gen(), rng.r#gen()))
		.collect();

	let mut cs = binius_m3::builder::ConstraintSystem::new();
	let mut table = cs.add_table("bitwise_ops");

	let xin = table.add_committed::<B1, 32>("xin");
	let yin = table.add_committed::<B1, 32>("yin");

	let bitwise_gadget = match args.op {
		BitwiseOp::And => BitwiseGadget::And(BitwiseAnd::new(&mut table, xin, yin)),
		BitwiseOp::Xor => BitwiseGadget::Xor(BitwiseXor::new(&mut table, xin, yin)),
		BitwiseOp::Or => BitwiseGadget::Or(BitwiseOr::new(&mut table, xin, yin)),
	};

	let table_id = table.id();
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![test_vector.len()],
	};

	let trace_gen_scope =
		tracing::info_span!("Generating trace", op = args.op.to_string(), n_ops = args.n_u32_ops)
			.entered();
	let allocator = Bump::new();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

	witness
		.fill_table_parallel(
			&ClosureFiller::new(table_id, |events, index| {
				let mut xin_bits = index.get_mut_as::<u32, _, 32>(xin).unwrap();
				let mut yin_bits = index.get_mut_as::<u32, _, 32>(yin).unwrap();
				for (i, (x, y)) in events.iter().enumerate() {
					xin_bits[i] = *x;
					yin_bits[i] = *y;
				}
				drop((xin_bits, yin_bits));
				bitwise_gadget.populate(index)?;
				Ok(())
			}),
			&test_vector,
		)
		.unwrap();

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

	constraint_system::verify::<
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
