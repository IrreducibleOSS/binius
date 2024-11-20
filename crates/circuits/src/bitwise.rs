// Copyright 2024 Irreducible Inc.

use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField1b, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{must_cast_slice, Pod};
use rayon::prelude::*;

use crate::builder::ConstraintSystemBuilder;

pub fn and<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);
	let zout = builder.add_committed("out", log_size, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		(
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(xin)?),
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(yin)?),
			witness
				.new_column::<BinaryField1b>(zout, log_size)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout)| {
				*zout = (*xin) & (*yin);
			});
	}
	builder.assert_zero([xin, yin, zout], composition_poly!([x, y, z] = x * y - z));
	builder.pop_namespace();
	Ok(zout)
}

pub fn xor<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);
	let zout = builder.add_linear_combination("out", log_size, [(xin, F::ONE), (yin, F::ONE)])?;
	if let Some(witness) = builder.witness() {
		(
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(xin)?),
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(yin)?),
			witness
				.new_column::<BinaryField1b>(zout, log_size)
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

pub fn or<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);
	let zout = builder.add_committed("out", log_size, BinaryField1b::TOWER_LEVEL);
	if let Some(witness) = builder.witness() {
		(
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(xin)?),
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(yin)?),
			witness
				.new_column::<BinaryField1b>(zout, log_size)
				.as_mut_slice::<u32>(),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout)| {
				*zout = (*xin) | (*yin);
			});
	}
	builder.assert_zero([xin, yin, zout], composition_poly!([x, y, z] = (x + y) + (x * y) - z));
	builder.pop_namespace();
	Ok(zout)
}
