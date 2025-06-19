// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_compute::{alloc::HostBumpAllocator, cpu::alloc::CpuComputeAllocator};
use binius_field::{
	arch::{OptimalUnderlier, OptimalUnderlier128b},
	as_packed_field::PackedType,
	packed::get_packed_slice,
};
use binius_m3::{
	builder::{
		B32, B64, B128, ConstraintSystem, TableFiller, TableId, TableWitnessSegment, WitnessIndex,
		test_utils::{ClosureFiller, validate_system_witness},
	},
	gadgets::{
		div::{DivSS32, DivUU32},
		mul::{MulSS32, MulSU32, MulUU32, MulUU64},
	},
};
use bytemuck::Contiguous;
use itertools::chain;
use rand::{Rng, SeedableRng, prelude::StdRng};

// This needs to create witness data as well as later query for checking outputs.
trait MulDivTestSuiteHelper
where
	Self: TableFiller,
{
	fn generate_inputs(&self, table_size: usize) -> Vec<Self::Event>;

	fn check_outputs(&self, inputs: &[Self::Event], table_witness: &TableWitnessSegment);
}

struct MulDivTestSuite;

impl MulDivTestSuite {
	fn execute<'a, Test>(
		&self,
		cs: ConstraintSystem,
		allocator: &'a HostBumpAllocator<'a, PackedType<OptimalUnderlier, B128>>,
		test_table: Test,
		test_size: usize,
	) -> anyhow::Result<()>
	where
		Test: MulDivTestSuiteHelper + TableFiller,
	{
		let inputs = test_table.generate_inputs(test_size);

		let mut witness = WitnessIndex::new(&cs, allocator);

		witness.fill_table_sequential(&test_table, &inputs)?;
		let table_index = witness.get_table(test_table.id()).unwrap();
		test_table.check_outputs(&inputs, &table_index.full_segment());

		let boundaries = vec![];
		validate_system_witness::<OptimalUnderlier>(&cs, witness, boundaries);
		Ok(())
	}
}

struct MulUU64TestTable {
	table_id: TableId,
	muluu: MulUU64,
}

impl MulUU64TestTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("MulUU64Table");
		let table_id = table.id();
		let muluu = MulUU64::new(&mut table);
		Self { table_id, muluu }
	}
}

impl TableFiller for MulUU64TestTable {
	type Event = (B64, B64);

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill(&self, rows: &[Self::Event], witness: &mut TableWitnessSegment) -> anyhow::Result<()> {
		let x_vals = rows.iter().map(|(x, _)| *x);
		let y_vals = rows.iter().map(|(_, y)| *y);
		self.muluu.populate_with_inputs(witness, x_vals, y_vals)
	}
}

impl MulDivTestSuiteHelper for MulUU64TestTable {
	fn generate_inputs(&self, table_size: usize) -> Vec<(B64, B64)> {
		let mut rng = StdRng::seed_from_u64(0);
		repeat_with(|| (B64::new(rng.random::<u64>()), B64::new(rng.random::<u64>())))
			.take(table_size)
			.collect::<Vec<_>>()
	}

	fn check_outputs(&self, inputs: &[(B64, B64)], table_witness: &TableWitnessSegment) {
		let out_low = table_witness.get(self.muluu.out_low).unwrap();
		let out_high = table_witness.get(self.muluu.out_high).unwrap();
		for (i, (x, y)) in inputs.iter().enumerate() {
			let prod = x.val() as u128 * y.val() as u128;
			let low = get_packed_slice(&out_low, i);
			let high = get_packed_slice(&out_high, i);
			let mut got = high.val() as u128;
			got = (got << 64) | low.val() as u128;
			assert_eq!(prod, got);
		}
	}
}

#[test]
fn test_muluu64() {
	let mut cs = ConstraintSystem::new();
	let mut allocator = CpuComputeAllocator::new(1 << 12);
	let allocator = allocator.into_bump_allocator();
	let muluu = MulUU64TestTable::new(&mut cs);
	MulDivTestSuite
		.execute(cs, &allocator, muluu, 1 << 9)
		.unwrap();
}

enum MulDivType {
	MulUU32,
	MulSU32,
	MulSS32,
	DivUU32,
	DivSS32,
}

#[allow(clippy::large_enum_variant)]
enum MulDivEnum {
	MulUU32(MulUU32),
	MulSU32(MulSU32),
	MulSS32(MulSS32),
	DivUU32(DivUU32),
	DivSS32(DivSS32),
}

struct MulDiv32TestTable {
	table_id: TableId,
	mul_div: MulDivEnum,
}

impl MulDiv32TestTable {
	pub fn new(cs: &mut ConstraintSystem, mul_div_type: MulDivType) -> Self {
		let mut table = cs.add_table("MulUU64Table");
		let table_id = table.id();
		let mul_div = match mul_div_type {
			MulDivType::MulUU32 => MulDivEnum::MulUU32(MulUU32::new(&mut table)),
			MulDivType::MulSU32 => MulDivEnum::MulSU32(MulSU32::new(&mut table)),
			MulDivType::MulSS32 => MulDivEnum::MulSS32(MulSS32::new(&mut table)),
			MulDivType::DivUU32 => MulDivEnum::DivUU32(DivUU32::new(&mut table)),
			MulDivType::DivSS32 => MulDivEnum::DivSS32(DivSS32::new(&mut table)),
		};
		Self { table_id, mul_div }
	}
}

impl TableFiller for MulDiv32TestTable {
	type Event = (B32, B32);

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill(&self, rows: &[Self::Event], witness: &mut TableWitnessSegment) -> anyhow::Result<()> {
		let x_vals = rows.iter().map(|(x, _)| *x);
		let y_vals = rows.iter().map(|(_, y)| *y);
		match &self.mul_div {
			MulDivEnum::MulUU32(muluu) => muluu.populate_with_inputs(witness, x_vals, y_vals)?,
			MulDivEnum::MulSU32(mulsu) => mulsu.populate_with_inputs(witness, x_vals, y_vals)?,
			MulDivEnum::MulSS32(mulss) => mulss.populate_with_inputs(witness, x_vals, y_vals)?,
			MulDivEnum::DivUU32(divuu) => divuu.populate_with_inputs(witness, x_vals, y_vals)?,
			MulDivEnum::DivSS32(divss) => divss.populate_with_inputs(witness, x_vals, y_vals)?,
		};
		Ok(())
	}
}

impl MulDivTestSuiteHelper for MulDiv32TestTable {
	fn generate_inputs(&self, table_size: usize) -> Vec<(B32, B32)> {
		// Just to be sure we are not reusing seed for the same inputs.
		let seed = match &self.mul_div {
			MulDivEnum::MulUU32(_) => 0xdeadbeef,
			MulDivEnum::MulSU32(_) => 0xc0ffee,
			MulDivEnum::MulSS32(_) => 0xbadcafe,
			MulDivEnum::DivUU32(_) => 0xdeadbeef,
			MulDivEnum::DivSS32(_) => 0xc0ffee,
		};
		let mut rng = StdRng::seed_from_u64(seed);
		match self.mul_div {
			MulDivEnum::MulUU32(_) => {
				repeat_with(|| (B32::new(rng.random::<u32>()), B32::new(rng.random::<u32>())))
					.take(table_size)
					.collect()
			}
			MulDivEnum::MulSU32(_) => {
				const EXPLICIT_TESTS: [(B32, B32); 4] = [
					(B32::new(i32::MIN_VALUE as u32), B32::new(u32::MAX_VALUE)),
					(B32::new(u32::MAX_VALUE), B32::new(u32::MAX_VALUE)),
					(B32::new(i32::MAX_VALUE as u32), B32::new(u32::MAX_VALUE)),
					(B32::new(0), B32::new(0)),
				];

				chain!(
					EXPLICIT_TESTS.into_iter(),
					repeat_with(|| (
						B32::new(rng.random::<i32>() as u32),
						B32::new(rng.random::<u32>())
					))
				)
				.take(table_size)
				.collect()
			}
			MulDivEnum::MulSS32(_) => repeat_with(|| {
				(B32::new(rng.random::<i32>() as u32), B32::new(rng.random::<i32>() as u32))
			})
			.take(table_size)
			.collect(),
			MulDivEnum::DivUU32(_) => {
				repeat_with(|| (B32::new(rng.random::<u32>()), B32::new(rng.random::<u32>())))
					.filter(|(_, y)| y.val() != 0)
					.take(table_size)
					.collect()
			}
			MulDivEnum::DivSS32(_) => {
				repeat_with(|| (B32::new(rng.random::<u32>()), B32::new(rng.random::<u32>())))
					.filter(|(_, y)| y.val() != 0)
					.take(table_size)
					.collect()
			}
		}
	}

	fn check_outputs(&self, inputs: &[(B32, B32)], table_witness: &TableWitnessSegment) {
		match &self.mul_div {
			MulDivEnum::MulUU32(muluu) => {
				let out_low = table_witness.get(muluu.out_low).unwrap();
				let out_high = table_witness.get(muluu.out_high).unwrap();
				for (i, (x, y)) in inputs.iter().enumerate() {
					let prod = x.val() as u64 * y.val() as u64;
					let low = get_packed_slice(&out_low, i);
					let high = get_packed_slice(&out_high, i);
					assert!((prod >> 32) as u32 == high.val() && prod as u32 == low.val());
				}
			}
			MulDivEnum::MulSU32(mulsu) => {
				let out_low = table_witness.get(mulsu.out_low).unwrap();
				let out_high = table_witness.get(mulsu.out_high).unwrap();
				for (i, (x, y)) in inputs.iter().enumerate() {
					let prod = (x.val() as i32 as i64) * y.val() as i64;
					let low = get_packed_slice(&out_low, i);
					let high = get_packed_slice(&out_high, i);
					assert!((prod >> 32) as u32 == high.val() && prod as u32 == low.val());
				}
			}
			MulDivEnum::MulSS32(mulss) => {
				let out_low = table_witness.get(mulss.out_low).unwrap();
				let out_high = table_witness.get(mulss.out_high).unwrap();
				for (i, (x, y)) in inputs.iter().enumerate() {
					let prod = (x.val() as i32 as i64) * (y.val() as i32 as i64);
					let low = get_packed_slice(&out_low, i);
					let high = get_packed_slice(&out_high, i);
					assert!((prod >> 32) as u32 == high.val() && prod as u32 == low.val());
				}
			}
			MulDivEnum::DivUU32(divuu) => {
				let out_div = table_witness.get(divuu.out_div).unwrap();
				let out_rem = table_witness.get(divuu.out_rem).unwrap();
				for (i, (p, q)) in inputs.iter().enumerate() {
					let exp_div = p.val() / q.val();
					let exp_rem = p.val() % q.val();
					let got_div = get_packed_slice(&out_div, i).val();
					let got_rem = get_packed_slice(&out_rem, i).val();
					assert!(exp_div == got_div && exp_rem == got_rem);
				}
			}
			MulDivEnum::DivSS32(divss) => {
				let out_div = table_witness.get(divss.out_div).unwrap();
				let out_rem = table_witness.get(divss.out_rem).unwrap();
				for (i, (p, q)) in inputs.iter().enumerate() {
					let p_i32 = p.val() as i32;
					let q_i32 = q.val() as i32;
					let exp_div = p_i32 / q_i32;
					let exp_rem = p_i32 % q_i32;
					let got_div = get_packed_slice(&out_div, i).val() as i32;
					let got_rem = get_packed_slice(&out_rem, i).val() as i32;
					assert!(exp_div == got_div && exp_rem == got_rem);
				}
			}
		};
	}
}

#[test]
fn test_muluu32() {
	let mut cs = ConstraintSystem::new();
	let mut allocator =
		CpuComputeAllocator::new(1 << (13 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));
	let allocator = allocator.into_bump_allocator();
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::MulUU32);
	MulDivTestSuite
		.execute(cs, &allocator, mul_div_32, 1 << 9)
		.unwrap();
}

#[test]
fn test_mulsu32() {
	let mut cs = ConstraintSystem::new();
	let mut allocator =
		CpuComputeAllocator::new(1 << (13 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));
	let allocator = allocator.into_bump_allocator();
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::MulSU32);
	MulDivTestSuite
		.execute(cs, &allocator, mul_div_32, 1 << 9)
		.unwrap();
}

#[test]
fn test_mulss32() {
	let mut cs = ConstraintSystem::new();
	let mut allocator =
		CpuComputeAllocator::new(1 << (13 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));
	let allocator = allocator.into_bump_allocator();
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::MulSS32);
	MulDivTestSuite
		.execute(cs, &allocator, mul_div_32, 1 << 9)
		.unwrap();
}

#[test]
fn test_divuu32() {
	let mut cs = ConstraintSystem::new();
	let mut allocator =
		CpuComputeAllocator::new(1 << (13 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));
	let allocator = allocator.into_bump_allocator();
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::DivUU32);
	MulDivTestSuite
		.execute(cs, &allocator, mul_div_32, 1 << 9)
		.unwrap();
}

#[test]
fn test_divss32() {
	let mut cs = ConstraintSystem::new();
	let mut allocator =
		CpuComputeAllocator::new(1 << (13 - PackedType::<OptimalUnderlier, B128>::LOG_WIDTH));
	let allocator = allocator.into_bump_allocator();
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::DivSS32);
	MulDivTestSuite
		.execute(cs, &allocator, mul_div_32, 1 << 9)
		.unwrap();
}

// This test exercises the case when a multiplication gadget is embedded in the same table as a
// column with a different stacking factor. In this case, table column indices and partition column
// indices don't align.
#[test]
fn test_mul_next_to_stacked_col() {
	let mut cs = ConstraintSystem::new();
	let mut table = cs.add_table("test");
	let table_id = table.id();
	let _stacked_col = table.add_committed::<B32, 2>("dummy");
	let mul = MulUU32::new(&mut table.with_namespace("mul1"));

	let mut rng = StdRng::seed_from_u64(0);
	let test_inputs = repeat_with(|| {
		let a = rng.random::<u32>();
		let b = rng.random::<u32>();
		(a, b)
	})
	.take(17)
	.collect::<Vec<_>>();

	let mut allocator = CpuComputeAllocator::new(1 << 12);
	let allocator = allocator.into_bump_allocator();
	let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
	witness
		.fill_table_sequential(
			&ClosureFiller::new(table_id, |events, witness| {
				mul.populate_with_inputs(
					witness,
					events.iter().map(|(a, _)| B32::new(*a)),
					events.iter().map(|(_, b)| B32::new(*b)),
				)?;
				Ok(())
			}),
			&test_inputs,
		)
		.unwrap();

	validate_system_witness::<OptimalUnderlier128b>(&cs, witness, vec![]);
}
