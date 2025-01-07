// Copyright 2024-2025 Irreducible Inc.

use binius_core::oracle::{OracleId, ProjectionVariant, ShiftVariant};
use binius_field::{
	as_packed_field::PackScalar, packed::set_packed_slice, BinaryField1b, BinaryField32b,
	ExtensionField, Field, TowerField,
};
use binius_macros::arith_expr;
use bytemuck::Pod;
use rayon::prelude::*;

use crate::{builder::ConstraintSystemBuilder, transparent};

pub fn packed<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	input: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
	F: TowerField + ExtensionField<BinaryField32b>,
{
	let packed = builder.add_packed(name, input, 5)?;
	if let Some(witness) = builder.witness() {
		witness.set(
			packed,
			witness
				.get::<BinaryField1b>(input)?
				.repacked::<BinaryField32b>(),
		)?;
	}
	Ok(packed)
}

pub fn mul_const<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	input: OracleId,
	value: u32,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	if value == 0 {
		let log_rows = builder.log_rows([input])?;
		return transparent::constant(builder, name, log_rows, BinaryField1b::ZERO);
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
			builder.assert_zero([x], arith_expr!([x] = x).convert_field());
		}
	}

	builder.pop_namespace();
	Ok(result)
}

pub fn add<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	builder.push_namespace(name);
	let log_rows = builder.log_rows([xin, yin])?;
	let cout = builder.add_committed("cout", log_rows, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted("cin", cout, 1, 5, ShiftVariant::LogicalLeft)?;
	let zout = builder.add_committed("zout", log_rows, BinaryField1b::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		(
			witness.get::<BinaryField1b>(xin)?.as_slice::<u32>(),
			witness.get::<BinaryField1b>(yin)?.as_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(zout)
				.as_mut_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(cout)
				.as_mut_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(cin)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout, cout, cin)| {
				let carry;
				(*zout, carry) = (*xin).overflowing_add(*yin);
				*cin = (*xin) ^ (*yin) ^ (*zout);
				*cout = ((carry as u32) << 31) | (*cin >> 1);
			});
	}

	builder.assert_zero(
		[xin, yin, cin, zout],
		arith_expr!([xin, yin, cin, zout] = xin + yin + cin - zout).convert_field(),
	);

	builder.assert_zero(
		[xin, yin, cin, cout],
		arith_expr!([xin, yin, cin, cout] = (xin + cin) * (yin + cin) + cin - cout).convert_field(),
	);

	// Overflow checking
	if matches!(flags, super::Flags::Checked) {
		let last_cout = select_bit(builder, "last_cout", cout, 31)?;
		builder.assert_zero([last_cout], arith_expr!([last_cout] = last_cout).convert_field());
	}

	builder.pop_namespace();
	Ok(zout)
}

pub fn half<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	input: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	if matches!(flags, super::Flags::Checked) {
		// Assert that the number is even
		let lsb = select_bit(builder, "lsb", input, 0)?;
		builder.assert_zero([lsb], arith_expr!([lsb] = lsb).convert_field());
	}
	shr(builder, name, input, 1)
}

pub fn shl<F, U>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	input: OracleId,
	offset: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	if offset == 0 {
		return Ok(input);
	}

	let shifted = builder.add_shifted(name, input, offset, 5, ShiftVariant::LogicalLeft)?;
	if let Some(witness) = builder.witness() {
		(witness.new_column(shifted).as_mut_slice::<u32>(), witness.get(input)?.as_slice::<u32>())
			.into_par_iter()
			.for_each(|(shifted, input)| *shifted = *input << offset);
	}

	Ok(shifted)
}

pub fn shr<F, U>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	input: OracleId,
	offset: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	if offset == 0 {
		return Ok(input);
	}

	let shifted = builder.add_shifted(name, input, offset, 5, ShiftVariant::LogicalRight)?;
	if let Some(witness) = builder.witness() {
		(witness.new_column(shifted).as_mut_slice::<u32>(), witness.get(input)?.as_slice::<u32>())
			.into_par_iter()
			.for_each(|(shifted, input)| *shifted = *input >> offset);
	}

	Ok(shifted)
}

pub fn select_bit<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	input: OracleId,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	let log_rows = builder.log_rows([input])?;
	anyhow::ensure!(log_rows >= 5, "Polynomial must have n_vars >= 5. Got {log_rows}");
	anyhow::ensure!(index < 32, "Only index values between 0 and 32 are allowed. Got {index}");

	let query = binius_core::polynomial::test_utils::decompose_index_to_hypercube_point(5, index);
	let bits = builder.add_projected(name, input, query, ProjectionVariant::FirstVars)?;

	if let Some(witness) = builder.witness() {
		let mut bits = witness.new_column::<BinaryField1b>(bits);
		let bits = bits.packed();
		let input = witness.get(input)?.as_slice::<u32>();
		input.iter().enumerate().for_each(|(i, &val)| {
			let value = match (val >> index) & 1 {
				0 => BinaryField1b::ZERO,
				_ => BinaryField1b::ONE,
			};
			set_packed_slice(bits, i, value);
		});
	}

	Ok(bits)
}

pub fn constant<F, U>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_count: usize,
	value: u32,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b> + Pod,
	F: TowerField + ExtensionField<BinaryField32b>,
{
	builder.push_namespace(name);
	// This would not need to be committed if we had `builder.add_unpacked(..)`
	let output = builder.add_committed("output", log_count + 5, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		witness
			.new_column::<BinaryField1b>(output)
			.as_mut_slice()
			.fill(value);
	}

	let output_packed = builder.add_packed("output_packed", output, 5)?;
	let transparent = builder.add_transparent(
		"transparent",
		binius_core::transparent::constant::Constant::new(log_count, BinaryField32b::new(value)),
	)?;
	if let Some(witness) = builder.witness() {
		let packed = witness
			.get::<BinaryField1b>(output)?
			.repacked::<BinaryField32b>();
		witness.set(output_packed, packed)?;
		witness.set(transparent, packed)?;
	}
	builder.assert_zero([output_packed, transparent], arith_expr!([x, y] = x - y).convert_field());
	builder.pop_namespace();
	Ok(output)
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::{arch::OptimalUnderlier, BinaryField128b, BinaryField1b, TowerField};

	use crate::{arithmetic, builder::ConstraintSystemBuilder};

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
}
