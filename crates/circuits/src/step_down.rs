// Copyright 2024 Irreducible Inc.

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField1b, PackedField, TowerField,
};
use bytemuck::{must_cast_slice_mut, Pod};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::cmp::Ordering;

use crate::builder::ConstraintSystemBuilder;

pub fn step_down<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	let id = builder.add_transparent(transparent::step_down::StepDown::new(log_size, index)?)?;

	if let Some(witness) = builder.witness() {
		let len = 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH);
		let mut stepdown = vec![U::default(); len].into_boxed_slice();
		{
			let stepdown = must_cast_slice_mut::<_, u8>(&mut stepdown);
			let byte_index = index >> 3;
			stepdown
				.into_par_iter()
				.enumerate()
				.for_each(|(i, stepdown)| {
					*stepdown = match i.cmp(&byte_index) {
						Ordering::Less => 0b11111111,
						Ordering::Equal => 0b11111111 >> (8 - (index & 0b111)),
						Ordering::Greater => 0b00000000,
					}
				});
		}
		*witness = std::mem::take(witness)
			.update_owned::<BinaryField1b, Box<[U]>>([(id, stepdown)].into_iter())?;
	}

	Ok(id)
}
