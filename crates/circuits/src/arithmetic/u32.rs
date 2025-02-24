// Copyright 2024-2025 Irreducible Inc.

use binius_core::oracle::{OracleId, ProjectionVariant, ShiftVariant};
use binius_field::{packed::set_packed_slice, BinaryField1b, BinaryField32b, Field, TowerField};
use binius_macros::arith_expr;
use binius_maybe_rayon::prelude::*;

use crate::{builder::ConstraintSystemBuilder, transparent};

pub fn packed(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	input: OracleId,
) -> Result<OracleId, anyhow::Error> {
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

pub fn mul_const(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	input: OracleId,
	value: u32,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
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
			builder.assert_zero("overflow", [x], arith_expr!([x] = x).convert_field());
		}
	}

	builder.pop_namespace();
	Ok(result)
}

pub fn add(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
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
		"sum",
		[xin, yin, cin, zout],
		arith_expr!([xin, yin, cin, zout] = xin + yin + cin - zout).convert_field(),
	);

	builder.assert_zero(
		"carry",
		[xin, yin, cin, cout],
		arith_expr!([xin, yin, cin, cout] = (xin + cin) * (yin + cin) + cin - cout).convert_field(),
	);

	// Overflow checking
	if matches!(flags, super::Flags::Checked) {
		let last_cout = select_bit(builder, "last_cout", cout, 31)?;
		builder.assert_zero(
			"overflow",
			[last_cout],
			arith_expr!([last_cout] = last_cout).convert_field(),
		);
	}

	builder.pop_namespace();
	Ok(zout)
}

pub fn sub(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	zin: OracleId,
	yin: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);
	let log_rows = builder.log_rows([zin, yin])?;
	let cout = builder.add_committed("cout", log_rows, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted("cin", cout, 1, 5, ShiftVariant::LogicalLeft)?;
	let xout = builder.add_committed("xin", log_rows, BinaryField1b::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		(
			witness.get::<BinaryField1b>(zin)?.as_slice::<u32>(),
			witness.get::<BinaryField1b>(yin)?.as_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(xout)
				.as_mut_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(cout)
				.as_mut_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(cin)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(zout, yin, xin, cout, cin)| {
				let carry;
				(*xin, carry) = (*zout).overflowing_sub(*yin);
				*cin = (*xin) ^ (*yin) ^ (*zout);
				*cout = ((carry as u32) << 31) | (*cin >> 1);
			});
	}

	builder.assert_zero(
		"sum",
		[xout, yin, cin, zin],
		arith_expr!([xout, yin, cin, zin] = xout + yin + cin - zin).convert_field(),
	);

	builder.assert_zero(
		"carry",
		[xout, yin, cin, cout],
		arith_expr!([xout, yin, cin, cout] = (xout + cin) * (yin + cin) + cin - cout)
			.convert_field(),
	);

	// Underflow checking
	if matches!(flags, super::Flags::Checked) {
		let last_cout = select_bit(builder, "last_cout", cout, 31)?;
		builder.assert_zero(
			"underflow",
			[last_cout],
			arith_expr!([last_cout] = last_cout).convert_field(),
		);
	}

	builder.pop_namespace();
	Ok(xout)
}

pub fn half(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	input: OracleId,
	flags: super::Flags,
) -> Result<OracleId, anyhow::Error> {
	if matches!(flags, super::Flags::Checked) {
		// Assert that the number is even
		let lsb = select_bit(builder, "lsb", input, 0)?;
		builder.assert_zero("is_even", [lsb], arith_expr!([lsb] = lsb).convert_field());
	}
	shr(builder, name, input, 1)
}

pub fn shl(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	input: OracleId,
	offset: usize,
) -> Result<OracleId, anyhow::Error> {
	if offset == 0 {
		return Ok(input);
	}

	let shifted = builder.add_shifted(name, input, offset, 5, ShiftVariant::LogicalLeft)?;
	if let Some(witness) = builder.witness() {
		(
			witness
				.new_column::<BinaryField1b>(shifted)
				.as_mut_slice::<u32>(),
			witness.get::<BinaryField1b>(input)?.as_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(shifted, input)| *shifted = *input << offset);
	}

	Ok(shifted)
}

pub fn shr(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	input: OracleId,
	offset: usize,
) -> Result<OracleId, anyhow::Error> {
	if offset == 0 {
		return Ok(input);
	}

	let shifted = builder.add_shifted(name, input, offset, 5, ShiftVariant::LogicalRight)?;
	if let Some(witness) = builder.witness() {
		(
			witness
				.new_column::<BinaryField1b>(shifted)
				.as_mut_slice::<u32>(),
			witness.get::<BinaryField1b>(input)?.as_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(shifted, input)| *shifted = *input >> offset);
	}

	Ok(shifted)
}

pub fn select_bit(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	input: OracleId,
	index: usize,
) -> Result<OracleId, anyhow::Error> {
	let log_rows = builder.log_rows([input])?;
	anyhow::ensure!(log_rows >= 5, "Polynomial must have n_vars >= 5. Got {log_rows}");
	anyhow::ensure!(index < 32, "Only index values between 0 and 32 are allowed. Got {index}");

	let query = binius_core::polynomial::test_utils::decompose_index_to_hypercube_point(5, index);
	let bits = builder.add_projected(name, input, query, ProjectionVariant::FirstVars)?;

	if let Some(witness) = builder.witness() {
		let mut bits = witness.new_column::<BinaryField1b>(bits);
		let bits = bits.packed();
		let input = witness.get::<BinaryField1b>(input)?.as_slice::<u32>();
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

pub fn constant(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_count: usize,
	value: u32,
) -> Result<OracleId, anyhow::Error> {
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
	builder.assert_zero(
		"unpack",
		[output_packed, transparent],
		arith_expr!([x, y] = x - y).convert_field(),
	);
	builder.pop_namespace();
	Ok(output)
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField1b, TowerField};

	use crate::{arithmetic, builder::test_utils::test_circuit, unconstrained::unconstrained};

	#[test]
	fn test_mul_const() {
		test_circuit(|builder| {
			let a = builder.add_committed("a", 5, BinaryField1b::TOWER_LEVEL);
			if let Some(witness) = builder.witness() {
				witness
					.new_column::<BinaryField1b>(a)
					.as_mut_slice::<u32>()
					.iter_mut()
					.for_each(|v| *v = 0b01000000_00000000_00000000_00000000u32);
			}
			let _c = arithmetic::u32::mul_const(builder, "mul3", a, 3, arithmetic::Flags::Checked)?;
			Ok(vec![])
		})
		.unwrap();
	}

	#[test]
	fn test_add() {
		test_circuit(|builder| {
			let log_size = 14;
			let a = unconstrained::<BinaryField1b>(builder, "a", log_size)?;
			let b = unconstrained::<BinaryField1b>(builder, "b", log_size)?;
			let _c = arithmetic::u32::add(builder, "u32add", a, b, arithmetic::Flags::Unchecked)?;
			Ok(vec![])
		})
		.unwrap();
	}

	#[test]
	fn test_sub() {
		test_circuit(|builder| {
			let a = unconstrained::<BinaryField1b>(builder, "a", 7).unwrap();
			let b = unconstrained::<BinaryField1b>(builder, "a", 7).unwrap();
			let _c = arithmetic::u32::sub(builder, "c", a, b, arithmetic::Flags::Unchecked)?;
			Ok(vec![])
		})
		.unwrap();
	}
}
