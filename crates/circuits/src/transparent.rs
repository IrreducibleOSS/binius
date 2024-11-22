// Copyright 2024 Irreducible Inc.

use std::cmp::Ordering;

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField1b, ExtensionField, PackedField, TowerField,
};
use bytemuck::Pod;
use rayon::prelude::*;

use crate::builder::ConstraintSystemBuilder;

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
	let id = builder.add_transparent(name, step_down)?;
	if let Some(witness) = builder.witness() {
		let byte_index = index >> 3;
		witness
			.new_column::<BinaryField1b>(id)
			.as_mut_slice::<u8>()
			.into_par_iter()
			.enumerate()
			.for_each(|(i, stepdown)| {
				*stepdown = match i.cmp(&byte_index) {
					Ordering::Less => 0b11111111,
					Ordering::Equal => (0b11111111u16 >> (8 - (index & 0b111))) as u8,
					Ordering::Greater => 0b00000000,
				}
			});
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
	U: UnderlierType + PackScalar<F> + PackScalar<FS> + PackScalar<FBase>,
	F: TowerField + ExtensionField<FBase> + ExtensionField<FS>,
	FS: TowerField,
	FBase: TowerField,
{
	let poly = transparent::constant::Constant::new(log_size, value);
	let id = builder.add_transparent(name, poly)?;
	if let Some(witness) = builder.witness() {
		witness
			.new_column::<FS>(id)
			.packed()
			.fill(<PackedType<U, FS>>::broadcast(value));
	}
	Ok(id)
}
