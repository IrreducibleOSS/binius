// Copyright 2024 Ulvetanna Inc.

use crate::builder::ConstraintSystemBuilder;
use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, PackedField, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};
use rayon::prelude::*;

pub fn u32add<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	log_size: usize,
	xin: OracleId,
	yin: OracleId,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	let cout = builder.add_committed(log_size, BinaryField1b::TOWER_LEVEL);
	let cin = builder.add_shifted(cout, 1, 5, ShiftVariant::LogicalLeft)?;
	let zout = builder.add_linear_combination(
		log_size,
		[(xin, F::ONE), (yin, F::ONE), (cin, F::ONE)].into_iter(),
	)?;

	if let Some(witness) = builder.witness() {
		let len = 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH);
		let mut zout_witness = vec![U::default(); len].into_boxed_slice();
		let mut cout_witness = vec![U::default(); len].into_boxed_slice();
		let mut cin_witness = vec![U::default(); len].into_boxed_slice();
		(
			must_cast_slice::<_, u32>(WithUnderlier::to_underliers_ref(
				witness.get::<BinaryField1b>(xin)?.evals(),
			)),
			must_cast_slice::<_, u32>(WithUnderlier::to_underliers_ref(
				witness.get::<BinaryField1b>(yin)?.evals(),
			)),
			must_cast_slice_mut::<_, u32>(&mut zout_witness),
			must_cast_slice_mut::<_, u32>(&mut cout_witness),
			must_cast_slice_mut::<_, u32>(&mut cin_witness),
		)
			.into_par_iter()
			.for_each(|(xin, yin, zout, cout, cin)| {
				let carry;
				(*zout, carry) = (*xin).overflowing_add(*yin);
				*cin = (*xin) ^ (*yin) ^ (*zout);
				*cout = ((carry as u32) << 31) | (*cin >> 1);
			});
		*witness = std::mem::take(witness).update_owned::<BinaryField1b, Box<[U]>>(
			[
				(zout, zout_witness),
				(cout, cout_witness),
				(cin, cin_witness),
			]
			.into_iter(),
		)?;
	}

	builder.assert_zero(
		[xin, yin, cin, cout],
		composition_poly!([xin, yin, cin, cout] = (xin + cin) * (yin + cin) + cin - cout),
	);

	Ok(zout)
}
