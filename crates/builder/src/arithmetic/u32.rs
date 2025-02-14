// Copyright 2024-2025 Irreducible Inc.

use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackScalar, packed::set_packed_slice, BinaryField1b, BinaryField32b,
	ExtensionField, Field, TowerField,
};
use binius_macros::arith_expr;
use binius_maybe_rayon::prelude::*;
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};

use crate::{
	constraint_system::{ProjectionVariant, ShiftVariant},
	derived_fillers as fillers,
	witness::table::OriginalFiller,
	ConstraintSystemBuilder, Filler, TableBuilder, U,
};

pub fn packed(
	builder: &mut TableBuilder,
	name: impl ToString,
	input: OracleId,
) -> Result<OracleId, anyhow::Error> {
	let packed = builder.add_packed(name, input, 5)?;
	// if let Some(witness) = builder.witness() {
	// 	witness.set(
	// 		packed,
	// 		witness
	// 			.get::<BinaryField1b>(input)?
	// 			.repacked::<BinaryField32b>(),
	// 	)?;
	// }
	Ok(packed)
}

pub fn mul_const(
	builder: &mut TableBuilder,
	name: impl ToString,
	input: OracleId,
	value: u32,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
	if value == 0 {
		// let log_rows = builder.log_rows([input])?;
		// return binius_circuits::transparent::constant(
		// 	builder,
		// 	name,
		// 	log_rows,
		// 	BinaryField1b::ZERO,
		// );
	}

	if value == 1 {
		return Ok(input);
	}

	builder.push_namespace(name);
	let mut tmp = value;
	let mut offset = 0;
	let mut result = input;
	let mut first = true;
	while tmp != 0 {
		if tmp & 1 == 1 {
			let shifted = shl(builder, format!("input_shl{offset}"), input, offset)?;
			if first {
				result = shifted;
				first = false;
			} else {
				result = add(builder, format!("add_shl{offset}"), result, shifted, flags)?;
			}
		}
		tmp >>= 1;
		if tmp != 0 {
			offset += 1;
		}
	}

	if matches!(flags, super::Flags::Checked) {
		// Shift overflow checking
		for i in 32 - offset..32 {
			let x = select_bit(builder, format!("bit{i}"), input, i)?;
			builder.assert_zero("overflow", [x], arith_expr!([x] = x).convert_field());
		}
	}

	builder.pop_namespace();
	Ok(result)
}

// this is tricky, be
/// what should the interface be?
/// you would for every table be able to
///
/// how do things compose well?
/// he was thinkingwe'd fill a table all at once. we'd have to receive all the slices at once, and we'd have to then think about which order they were declared in, which ones should get sent to which helpers.
/// but now i
///
///
///
/// what's the point of a table?
/// it's the unit in which we think of stuff.
/// wha is a table?
/// when you add stuff to another function, is there any reason it should go in the same table?
/// actually name scoping should then happen at the table level
///
/// well it depends, maybe both.
/// we might ask another thing for help,
/// oh, and it woul be derived base on ours.
///
/// wait,
///
/// // I'm confused here.
// 1. how do we connect the table name?
// 2. how do we
struct AdderPair {
	x: u32,
	y: u32,
}
impl OriginalFiller for AdderPair {
	const ROWS_PER_INPUT: usize = 32;
	fn name() -> impl ToString {
		"adder pair"
	}
	fn inputs_per_batch() -> usize {
		1
	}
	fn populate<'a>(inputs: &[Self], underliers: &mut [&'a mut [U]]) {
		let cout = must_cast_slice_mut::<U, u32>(underliers[0]);
		for (i, AdderPair { x, y }) in inputs.iter().enumerate() {
			let (zout, carry) = x.overflowing_add(*y);
			let cin = x ^ y ^ zout;
			cout[i] = ((carry as u32) << 31) | (cin >> 1);
		}
	}
}

pub fn add(
	builder: &mut ConstraintSystemBuilder,
	table: &mut TableBuilder,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
	let name = name.to_string();
	table.push_namespace(name.clone());

	let cout = table.add_derived(
		"cout",
		BinaryField1b::TOWER_LEVEL,
		[xin, yin],
		arith_expr!([xin, yin, cin] = (xin + cin) * (yin + cin) + cin).convert_field(),
		fillers::arithmetic::u32_add_carry_out(xin, yin),
	);

	let cin = table.add_shifted("cin", cout, 1, Some(5), ShiftVariant::LogicalLeft)?;
	let cin = table.add_shifted_with_filler(
		"cin",
		cout,
		1,
		Some(5),
		ShiftVariant::LogicalLeft,
		fillers::shifted(0, 1, Some(5), ShiftVariant::LogicalLeft),
	)?;

	let zout = table.add_derived(
		"zout",
		BinaryField1b::TOWER_LEVEL,
		[xin, yin, cin],
		arith_expr!([xin, yin, cin] = xin + yin + cin).convert_field(),
		fillers::arithmetic::u32_sum_with_cin(xin, yin, cin),
	);
	let zout = {
		let expr = arith_expr!([xin, yin, cin] = xin + yin + cin);
		table.add_derived(
			"zout",
			BinaryField1b::TOWER_LEVEL,
			[xin, yin, cin],
			expr.convert_field(),
			fillers::arithmetic_filler::<BinaryField1b>(expr),
		)
	};

	// Overflow checking
	if matches!(flags, super::Flags::Checked) {
		let mut overflow_table =
			builder.new_table_builder(format!("{}: overflow check", name.to_string()));
		let last_cout = select_bit(&mut overflow_table, "last_cout", cout, 31)?;
		overflow_table.assert_zero(
			"overflow",
			[last_cout],
			arith_expr!([last_cout] = last_cout).convert_field(),
		);
	}

	table.pop_namespace();
	Ok(zout)
}

pub fn sub(
	builder: &mut ConstraintSystemBuilder,
	table: &mut TableBuilder,
	name: impl ToString,
	zin: OracleId,
	yin: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
	let name = name.to_string();
	table.push_namespace(name.clone());

	let cout = table.add_derived(
		"cout",
		BinaryField1b::TOWER_LEVEL,
		[zin, yin],
		arith_expr!([xout, yin, cin] = (xout + cin) * (yin + cin) + cin).convert_field(),
		fillers::arithmetic::u32_sub_carry_out(zin, yin),
	);

	let cin = table.add_shifted("cin", cout, 1, Some(5), ShiftVariant::LogicalLeft)?;

	let xout = table.add_derived(
		"xout",
		BinaryField1b::TOWER_LEVEL,
		[yin, zin, cin],
		arith_expr!([yin, zin, cin] = yin + zin + cin).convert_field(),
		fillers::arithmetic::u32_sum_with_cin(yin, zin, cin),
	);

	// Underflow checking
	if matches!(flags, super::Flags::Checked) {
		let mut underflow_table =
			builder.new_table_builder(format!("{}: underflow check", name.to_string()));
		let last_cout = select_bit(&mut underflow_table, "last_cout", cout, 31)?;
		underflow_table.assert_zero(
			"underflow",
			[last_cout],
			arith_expr!([last_cout] = last_cout).convert_field(),
		);
	}

	table.pop_namespace();
	Ok(xout)
}

pub fn half(
	table: &mut TableBuilder,
	name: impl ToString,
	input: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
	if matches!(flags, super::Flags::Checked) {
		// Assert that the number is even
		let lsb = select_bit(table, "lsb", input, 0)?;
		table.assert_zero("is_even", [lsb], arith_expr!([lsb] = lsb).convert_field());
	}
	shr(table, name, input, 1)
}

pub fn shl(
	table: &mut TableBuilder,
	name: impl ToString,
	input: OracleId,
	offset: usize,
) -> Result<OracleId, anyhow::Error> {
	if offset == 0 {
		return Ok(input);
	}

	let shifted = table.add_shifted(name, input, offset, Some(5), ShiftVariant::LogicalLeft)?;

	Ok(shifted)
}

pub fn shr(
	table: &mut TableBuilder,
	name: impl ToString,
	input: OracleId,
	offset: usize,
) -> Result<OracleId, anyhow::Error> {
	if offset == 0 {
		return Ok(input);
	}

	let shifted = table.add_shifted(name, input, offset, Some(5), ShiftVariant::LogicalRight)?;

	Ok(shifted)
}

pub fn select_bit(
	table: &mut TableBuilder,
	name: impl ToString,
	input: OracleId,
	index: usize,
) -> Result<OracleId, anyhow::Error> {
	// let log_rows = builder.log_rows([input])?;
	// anyhow::ensure!(log_rows >= 5, "Polynomial must have n_vars >= 5. Got {log_rows}");
	anyhow::ensure!(index < 32, "Only index values between 0 and 32 are allowed. Got {index}");

	let query = binius_core::polynomial::test_utils::decompose_index_to_hypercube_point(5, index);
	let bits = table.add_projected_with_filler(
		name,
		input,
		query.clone(),
		ProjectionVariant::FirstVars,
		// default filler plugged in by table builder
		fillers::projected(BinaryField1b::TOWER_LEVEL, query, ProjectionVariant::FirstVars),
	)?;

	Ok(bits)
}

pub fn constant(
	builder: &mut ConstraintSystemBuilder,
	table: &mut TableBuilder,
	name: impl ToString,
	log_count: usize,
	value: u32,
) -> Result<OracleId, anyhow::Error> {
	let name = name.to_string();
	table.push_namespace(name.clone());
	// This would not need to be committed if we had `builder.add_unpacked(..)`
	let unpacked_table = builder.new_table_builder(format!("{}: unpacked", name));
	let output = unpacked_table.add_derived(
		"output",
		BinaryField1b::TOWER_LEVEL,
		[],
		arith_expr!([x] = 0).convert_field(),
		fillers::arithmetic::constant(value),
	);

	let output_packed = table.add_packed("output_packed", output, 5)?;

	let transparent = table.add_transparent(
		"transparent",
		binius_core::transparent::constant::Constant::new(log_count, BinaryField32b::new(value)),
	)?;

	if let Some(witness) = table.witness() {
		let packed = witness
			.get::<BinaryField1b>(output)?
			.repacked::<BinaryField32b>();
		witness.set(output_packed, packed)?;
		witness.set(transparent, packed)?;
	}
	table.assert_zero(
		"unpack",
		[output_packed, transparent],
		arith_expr!([x, y] = x - y).convert_field(),
	);
	table.pop_namespace();
	Ok(output)
}
#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{arch::OptimalUnderlier, BinaryField128b, BinaryField1b, TowerField};

	use crate::{arithmetic, builder::ConstraintSystemBuilder, unconstrained::unconstrained};

	type U = OptimalUnderlier;
	type F = BinaryField128b;

	#[test]
	fn test_mul_const() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);

		let a = builder.add_committed("a", 5, BinaryField1b::TOWER_LEVEL);
		if let Some(witness) = builder.witness() {
			witness
				.new_column::<BinaryField1b>(a)
				.as_mut_slice::<u32>()
				.iter_mut()
				.for_each(|v| *v = 0b01000000_00000000_00000000_00000000u32);
		}

		let _c = arithmetic::u32::mul_const(&mut builder, "mul3", a, 3, arithmetic::Flags::Checked)
			.unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}

	#[test]
	fn test_sub() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::<U, F>::new_with_witness(&allocator);

		let a = unconstrained::<U, F, BinaryField1b>(&mut builder, "a", 7).unwrap();
		let b = unconstrained::<U, F, BinaryField1b>(&mut builder, "a", 7).unwrap();
		let _c =
			arithmetic::u32::sub(&mut builder, "c", a, b, arithmetic::Flags::Unchecked).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}
}
