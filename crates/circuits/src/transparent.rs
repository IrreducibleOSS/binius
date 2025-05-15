// Copyright 2024-2025 Irreducible Inc.

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	BinaryField1b, ExtensionField, PackedField, TowerField,
	as_packed_field::{PackScalar, PackedType},
};

use crate::builder::{
	ConstraintSystemBuilder,
	types::{F, U},
};

pub fn step_down(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error> {
	let step_down = transparent::step_down::StepDown::new(log_size, index)?;
	let id = builder.add_transparent(name, step_down.clone())?;
	if let Some(witness) = builder.witness() {
		step_down.populate(witness.new_column::<BinaryField1b>(id).packed());
	}
	Ok(id)
}

pub fn step_up(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error> {
	let step_up = transparent::step_up::StepUp::new(log_size, index)?;
	let id = builder.add_transparent(name, step_up.clone())?;
	if let Some(witness) = builder.witness() {
		step_up.populate(witness.new_column::<BinaryField1b>(id).packed());
	}
	Ok(id)
}

pub fn constant<FS>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_size: usize,
	value: FS,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<FS>,
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

pub fn make_transparent<FS>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	values: &[FS],
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<FS>,
	F: TowerField + ExtensionField<FS>,
	FS: TowerField,
{
	let packed_length = values.len().div_ceil(PackedType::<U, FS>::WIDTH);
	let mut packed_values = vec![PackedType::<U, FS>::default(); packed_length];
	for (i, value) in values.iter().enumerate() {
		binius_field::packed::set_packed_slice(&mut packed_values, i, *value);
	}

	use binius_core::transparent::multilinear_extension::MultilinearExtensionTransparent;
	let mle = MultilinearExtensionTransparent::<_, PackedType<U, F>, _>::from_values(
		packed_values.clone(),
	)?;

	let oracle = builder.add_transparent(name, mle)?;

	if let Some(witness) = builder.witness() {
		let mut entry_builder = witness.new_column::<FS>(oracle);
		entry_builder.packed().copy_from_slice(&packed_values);
	}

	Ok(oracle)
}
