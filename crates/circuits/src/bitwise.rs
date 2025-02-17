// Copyright 2024-2025 Irreducible Inc.

use binius_core::oracle::OracleId;
use binius_field::{BinaryField1b, Field, TowerField};
use binius_macros::arith_expr;
use binius_maybe_rayon::prelude::*;

use crate::builder::ConstraintSystemBuilder;

pub fn and(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);
	let log_rows = builder.log_rows([xin, yin])?;
	let zout = builder.add_committed("zout", log_rows, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		(
			witness.get::<BinaryField1b>(xin)?.as_slice::<u32>(),
			witness.get::<BinaryField1b>(yin)?.as_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(zout)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout)| {
				*zout = (*xin) & (*yin);
			});
	}
	builder.assert_zero(
		"bitwise_and",
		[xin, yin, zout],
		arith_expr!([x, y, z] = x * y - z).convert_field(),
	);
	builder.pop_namespace();
	Ok(zout)
}

pub fn xor(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);
	let log_rows = builder.log_rows([xin, yin])?;
	let zout =
		builder.add_linear_combination("zout", log_rows, [(xin, Field::ONE), (yin, Field::ONE)])?;
	if let Some(witness) = builder.witness() {
		(
			witness.get::<BinaryField1b>(xin)?.as_slice::<u32>(),
			witness.get::<BinaryField1b>(yin)?.as_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(zout)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout)| {
				*zout = (*xin) ^ (*yin);
			});
	}
	builder.pop_namespace();
	Ok(zout)
}

pub fn or(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error> {
	builder.push_namespace(name);
	let log_rows = builder.log_rows([xin, yin])?;
	let zout = builder.add_committed("zout", log_rows, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		(
			witness.get::<BinaryField1b>(xin)?.as_slice::<u32>(),
			witness.get::<BinaryField1b>(yin)?.as_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(zout)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout)| {
				*zout = (*xin) | (*yin);
			});
	}
	builder.assert_zero(
		"bitwise_or",
		[xin, yin, zout],
		arith_expr!([x, y, z] = (x + y) + (x * y) - z).convert_field(),
	);
	builder.pop_namespace();
	Ok(zout)
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::validate::validate_witness;
	use binius_field::BinaryField1b;

	use crate::{builder::ConstraintSystemBuilder, unconstrained::unconstrained};

	#[test]
	fn test_bitwise() {
		let allocator = bumpalo::Bump::new();
		let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);
		let log_size = 6;
		let a = unconstrained::<BinaryField1b>(&mut builder, "a", log_size).unwrap();
		let b = unconstrained::<BinaryField1b>(&mut builder, "b", log_size).unwrap();
		let _and = super::and(&mut builder, "and", a, b).unwrap();
		let _xor = super::xor(&mut builder, "xor", a, b).unwrap();
		let _or = super::or(&mut builder, "or", a, b).unwrap();

		let witness = builder.take_witness().unwrap();
		let constraint_system = builder.build().unwrap();
		let boundaries = vec![];
		validate_witness(&constraint_system, &boundaries, &witness).unwrap();
	}
}
