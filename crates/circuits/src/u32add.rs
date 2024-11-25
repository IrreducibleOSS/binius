// Copyright 2024 Irreducible Inc.

use crate::builder::ConstraintSystemBuilder;
use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField1b, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::Pod;
use rayon::prelude::*;

fn u32add_common<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	xin: OracleId,
	yin: OracleId,
	zout: OracleId,
	cin: OracleId,
	cout: OracleId,
) -> Result<(), anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
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
		[xin, yin, cin, cout],
		composition_poly!([xin, yin, cin, cout] = (xin + cin) * (yin + cin) + cin - cout),
	);
	Ok(())
}

pub fn u32add<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);
	let log_rows = builder.log_rows([xin, yin])?;
	let cout = builder.add_committed("cout", log_rows, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted("cin", cout, 1, 5, ShiftVariant::LogicalLeft)?;

	let zout = builder.add_linear_combination(
		"zout",
		log_rows,
		[(xin, F::ONE), (yin, F::ONE), (cin, F::ONE)].into_iter(),
	)?;

	u32add_common(builder, xin, yin, zout, cin, cout)?;

	builder.pop_namespace();
	Ok(zout)
}

pub fn u32add_committed<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);
	let log_rows = builder.log_rows([xin, yin])?;
	let cout = builder.add_committed("cout", log_rows, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted("cin", cout, 1, 5, ShiftVariant::LogicalLeft)?;
	let zout = builder.add_committed("zout", log_rows, BinaryField1b::TOWER_LEVEL);

	u32add_common(builder, xin, yin, zout, cin, cout)?;

	builder.assert_zero(
		[xin, yin, cin, zout],
		composition_poly!([xin, yin, cin, zout] = xin + yin + cin - zout),
	);

	builder.pop_namespace();
	Ok(zout)
}
