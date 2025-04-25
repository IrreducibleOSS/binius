// Copyright 2025 Irreducible Inc.

use std::iter::repeat_with;

use binius_core::fiat_shamir::HasherChallenger;
use binius_field::{
	arch::OptimalUnderlier, tower::CanonicalTowerFamily, packed::{get_packed_slice, set_packed_slice}, };
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_m3::{
	builder::{
		Col, ConstraintSystem, Statement, TableFiller, TableId, TableWitnessSegment, WitnessIndex,
		B1, B32, B64,
	},
	gadgets::mul::{MulSS32, MulSU32, MulUU32, MulUU64, SignConverter, UnsignedMulPrimitives},
};
use bumpalo::Bump;
use rand::{prelude::StdRng, Rng, SeedableRng};

// This needs to create witness data as well as later query for checking outputs.
trait MulDivTestSuiteHelper
where
	Self: TableFiller,
{
	fn generate_inputs(&self, table_size: usize) -> Vec<Self::Event>;

	fn check_outputs(&self, inputs: &[Self::Event], table_witness: &TableWitnessSegment);
}

struct MulDivTestSuite {
	pub prove_verify: bool,
}

impl MulDivTestSuite {
	fn execute<Test>(
		&self,
		cs: ConstraintSystem,
		allocator: Bump,
		statement: Statement,
		test_table: Test,
	) -> anyhow::Result<()>
	where
		Test: MulDivTestSuiteHelper + TableFiller,
	{
		let inputs = test_table.generate_inputs(*statement.table_sizes.first().unwrap());

		let mut witness = WitnessIndex::new(&cs, &allocator);

		witness.fill_table_sequential(&test_table, &inputs)?;
		let table_index = witness.get_table(test_table.id()).unwrap();
		test_table.check_outputs(&inputs, &table_index.full_segment());

		const LOG_INV_RATE: usize = 1;

		// Lower security bits for testing only!!
		const SECURITY_BITS: usize = 70;
		let ccs = cs.compile(&statement)?;
		let witness = witness.into_multilinear_extension_index();

		binius_core::constraint_system::validate::validate_witness(
			&ccs,
			&statement.boundaries,
			&witness,
		)?;

		if self.prove_verify {
			let proof = binius_core::constraint_system::prove::<
				OptimalUnderlier,
				CanonicalTowerFamily,
				Groestl256,
				Groestl256ByteCompression,
				HasherChallenger<Groestl256>,
				_,
			>(
				&ccs,
				LOG_INV_RATE,
				SECURITY_BITS,
				&statement.boundaries,
				witness,
				&binius_hal::make_portable_backend(),
			)?;

			binius_core::constraint_system::verify::<
				OptimalUnderlier,
				CanonicalTowerFamily,
				Groestl256,
				Groestl256ByteCompression,
				HasherChallenger<Groestl256>,
			>(&ccs, LOG_INV_RATE, SECURITY_BITS, &statement.boundaries, proof)?;
		}
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

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment,
	) -> anyhow::Result<()> {
		let x_vals = rows.clone().map(|(x, _)| *x);
		let y_vals = rows.map(|(_, y)| *y);
		self.muluu.populate_with_inputs(witness, x_vals, y_vals)
	}
}

impl MulDivTestSuiteHelper for MulUU64TestTable {
	fn generate_inputs(&self, table_size: usize) -> Vec<(B64, B64)> {
		let mut rng = StdRng::seed_from_u64(0);
		repeat_with(|| (B64::new(rng.gen::<u64>()), B64::new(rng.gen::<u64>())))
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
	let allocator = Bump::new();
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![1 << 8],
	};
	let muluu = MulUU64TestTable::new(&mut cs);
	let test_suite = MulDivTestSuite { prove_verify: true };
	test_suite.execute(cs, allocator, statement, muluu).unwrap();
}

enum MulDivType {
	MulUU32,
	MulSU32,
	MulSS32,
}

#[allow(clippy::large_enum_variant)]
enum MulDivEnum {
	MulUU32(MulUU32),
	MulSU32(MulSU32),
	MulSS32(MulSS32),
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
		};
		Self { table_id, mul_div }
	}
}

impl TableFiller for MulDiv32TestTable {
	type Event = (B32, B32);

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment,
	) -> anyhow::Result<()> {
		let x_vals = rows.clone().map(|(x, _)| *x);
		let y_vals = rows.map(|(_, y)| *y);
		match &self.mul_div {
			MulDivEnum::MulUU32(muluu) => muluu.populate_with_inputs(witness, x_vals, y_vals)?,
			MulDivEnum::MulSU32(mulsu) => mulsu.populate_with_inputs(witness, x_vals, y_vals)?,
			MulDivEnum::MulSS32(mulss) => mulss.populate_with_inputs(witness, x_vals, y_vals)?,
		};
		Ok(())
	}
}

impl MulDivTestSuiteHelper for MulDiv32TestTable {
	fn generate_inputs(&self, table_size: usize) -> Vec<(B32, B32)> {
		// Just to be sure we are not resuing seed for the same inputs.
		let seed = match &self.mul_div {
			MulDivEnum::MulUU32(_) => 0xdeadbeef,
			MulDivEnum::MulSU32(_) => 0xc0ffee,
			MulDivEnum::MulSS32(_) => 0xbadcafe,
		};
		let mut rng = StdRng::seed_from_u64(seed);
		match self.mul_div {
			MulDivEnum::MulUU32(_) => {
				repeat_with(|| (B32::new(rng.gen::<u32>()), B32::new(rng.gen::<u32>())))
					.take(table_size)
					.collect()
			}
			MulDivEnum::MulSU32(_) => {
				repeat_with(|| (B32::new(rng.gen::<i32>() as u32), B32::new(rng.gen::<u32>())))
					.take(table_size)
					.collect()
			}
			MulDivEnum::MulSS32(_) => repeat_with(|| {
				(B32::new(rng.gen::<i32>() as u32), B32::new(rng.gen::<i32>() as u32))
			})
			.take(table_size)
			.collect(),
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
		};
	}
}

#[test]
fn test_muluu32() {
	let mut cs = ConstraintSystem::new();
	let allocator = Bump::new();
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![1 << 8],
	};
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::MulUU32);
	let test_suite = MulDivTestSuite { prove_verify: true };
	test_suite
		.execute(cs, allocator, statement, mul_div_32)
		.unwrap();
}

#[test]
fn test_mulsu32() {
	let mut cs = ConstraintSystem::new();
	let allocator = Bump::new();
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![1 << 8],
	};
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::MulSU32);
	let test_suite = MulDivTestSuite { prove_verify: true };
	test_suite
		.execute(cs, allocator, statement, mul_div_32)
		.unwrap();
}

#[test]
fn test_mulss32() {
	let mut cs = ConstraintSystem::new();
	let allocator = Bump::new();
	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![1 << 8],
	};
	let mul_div_32 = MulDiv32TestTable::new(&mut cs, MulDivType::MulSS32);
	let test_suite = MulDivTestSuite { prove_verify: true };
	test_suite
		.execute(cs, allocator, statement, mul_div_32)
		.unwrap();
}

#[derive(Debug)]
pub struct AbsoluteValueTable {
	table_id: TableId,
	input: [Col<B1>; 32],
	abs_value_bits: [Col<B1>; 32],
	signed_input: SignConverter<u32, 32>,
}

impl AbsoluteValueTable {
	pub fn new(cs: &mut ConstraintSystem) -> Self {
		let mut table = cs.add_table("TwosComplementTestTable");
		let table_id = table.id();
		let input = table.add_committed_multiple("input");
		let signed_input =
			SignConverter::new(&mut table, "abs_value_bits", input, input[31].into());
		let abs_value_bits = signed_input.converted_bits;
		Self {
			table_id,
			input,
			signed_input,
			abs_value_bits,
		}
	}
}

impl TableFiller for AbsoluteValueTable {
	type Event = B32;

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment,
	) -> anyhow::Result<()> {
		{
			let mut input_witness = array_util::try_map(self.input, |col| witness.get_mut(col))?;
			let mut abs_input_witness =
				array_util::try_map(self.abs_value_bits, |col| witness.get_mut(col))?;

			for (i, val) in rows.enumerate() {
				let val_i32 = val.val() as i32;
				let abs_val = if val_i32 >= 0 {
					val_i32 as u32
				} else {
					(-val_i32) as u32
				};
				for bit in 0..32 {
					set_packed_slice(
						&mut input_witness[bit],
						i,
						B1::from(u32::is_bit_set_at(*val, bit)),
					);
					set_packed_slice(
						&mut abs_input_witness[bit],
						i,
						B1::from(u32::is_bit_set_at(B32::new(abs_val), bit)),
					);
				}
			}
		}
		self.signed_input.populate(witness)?;
		Ok(())
	}
}

impl MulDivTestSuiteHelper for AbsoluteValueTable {
	fn generate_inputs(&self, table_size: usize) -> Vec<B32> {
		let mut rng = StdRng::seed_from_u64(0);
		repeat_with(|| B32::new(rng.gen::<i32>() as u32))
			.take(table_size)
			.collect()
	}

	fn check_outputs(&self, _inputs: &[B32], _table_witness: &TableWitnessSegment) {
		// It's redundant to check here
	}
}

#[test]
fn test_twos_complement() {
	let mut cs = ConstraintSystem::new();

	let allocator = Bump::new();

	const TABLE_SIZE: usize = 1 << 8;

	let statement = Statement {
		boundaries: vec![],
		table_sizes: vec![TABLE_SIZE],
	};

	let table = AbsoluteValueTable::new(&mut cs);
	let test_suite = MulDivTestSuite { prove_verify: true };
	test_suite.execute(cs, allocator, statement, table).unwrap()
}
