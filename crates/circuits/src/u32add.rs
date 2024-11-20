// Copyright 2024 Irreducible Inc.

use crate::builder::ConstraintSystemBuilder;
use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField1b, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{must_cast_slice, Pod};
use rayon::prelude::*;

fn u32add_common<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	log_size: usize,
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
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(xin)?),
			must_cast_slice::<_, u32>(witness.get::<BinaryField1b>(yin)?),
			witness
				.new_column::<BinaryField1b>(zout, log_size)
				.as_mut_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(cout, log_size)
				.as_mut_slice::<u32>(),
			witness
				.new_column::<BinaryField1b>(cin, log_size)
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
	log_size: usize,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	builder.push_namespace(name);
	let cout = builder.add_committed("cout", log_size, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted("cin", cout, 1, 5, ShiftVariant::LogicalLeft)?;

	let zout = builder.add_linear_combination(
		"zout",
		log_size,
		[(xin, F::ONE), (yin, F::ONE), (cin, F::ONE)].into_iter(),
	)?;

	u32add_common(builder, log_size, xin, yin, zout, cin, cout)?;

	builder.pop_namespace();
	Ok(zout)
}

pub fn u32add_committed<U, F>(
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
	let cout = builder.add_committed("cout", log_size, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted("cin", cout, 1, 5, ShiftVariant::LogicalLeft)?;
	let zout = builder.add_committed("zout", log_size, BinaryField1b::TOWER_LEVEL);

	u32add_common(builder, log_size, xin, yin, zout, cin, cout)?;

	builder.assert_zero(
		[xin, yin, cin, zout],
		composition_poly!([xin, yin, cin, zout] = xin + yin + cin - zout),
	);

	builder.pop_namespace();
	Ok(zout)
}
