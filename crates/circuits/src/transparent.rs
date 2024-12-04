// Copyright 2024 Irreducible Inc.

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField1b, ExtensionField, PackedField, TowerField,
};
use bytemuck::Pod;

use crate::builder::ConstraintSystemBuilder;

pub fn step_down<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	let step_down = transparent::step_down::StepDown::new(log_size, index)?;
	let id = builder.add_transparent(name, step_down.clone())?;
	if let Some(witness) = builder.witness() {
		step_down.populate(witness.new_column::<BinaryField1b>(id).packed());
	}
	Ok(id)
}

pub fn step_up<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<BinaryField1b> + Pod,
	F: TowerField,
{
	let step_up = transparent::step_up::StepUp::new(log_size, index)?;
	let id = builder.add_transparent(name, step_up.clone())?;
	if let Some(witness) = builder.witness() {
		step_up.populate(witness.new_column::<BinaryField1b>(id).packed());
	}
	Ok(id)
}

pub fn constant<U, F, FS>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	value: FS,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FS>,
	F: TowerField + ExtensionField<FS>,
	FS: TowerField,
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
