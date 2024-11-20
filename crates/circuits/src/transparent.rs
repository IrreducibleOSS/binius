// Copyright 2024 Irreducible Inc.

use std::cmp::Ordering;

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, ExtensionField, PackedField, TowerField,
};
use bytemuck::{must_cast_slice_mut, Pod};
use rayon::prelude::*;

use crate::{builder::ConstraintSystemBuilder, helpers::make_underliers};

pub fn step_down<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase> + PackScalar<BinaryField1b> + Pod,
	F: TowerField + ExtensionField<FBase>,
	FBase: TowerField,
{
	let step_down = transparent::step_down::StepDown::new(log_size, index)?;
	let id = builder.add_transparent(name, step_down.clone())?;
	if let Some(witness) = builder.witness() {
		let mut data = make_underliers::<U, BinaryField1b>(log_size);
		let byte_index = index >> 3;
		must_cast_slice_mut::<_, u8>(&mut data)
			.into_par_iter()
			.enumerate()
			.for_each(|(i, stepdown)| {
				*stepdown = match i.cmp(&byte_index) {
					Ordering::Less => 0b11111111,
					Ordering::Equal => (0b11111111u16 >> (8 - (index & 0b111))) as u8,
					Ordering::Greater => 0b00000000,
				}
			});
		witness.set_owned::<BinaryField1b, _>([(id, data)])?;
	}
	Ok(id)
}

pub fn constant<U, F, FS, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	log_size: usize,
	value: FS,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType
		+ PackScalar<F>
		+ PackScalar<FS>
		+ PackScalar<FBase>
		+ PackScalar<BinaryField1b>,
	F: TowerField + ExtensionField<FBase> + ExtensionField<FS>,
	FS: TowerField,
	FBase: TowerField,
{
	let poly = transparent::constant::Constant {
		n_vars: log_size,
		value: value.into(),
	};
	let id = builder.add_transparent(name, poly)?;
	if let Some(witness) = builder.witness() {
		let mut data = make_underliers::<U, FS>(log_size);
		data.fill(WithUnderlier::to_underlier(<PackedType<U, FS>>::broadcast(value)));
		witness.set_owned::<FS, _>([(id, data)])?;
	}
	Ok(id)
}
