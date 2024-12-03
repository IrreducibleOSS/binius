// Copyright 2024 Irreducible Inc.

use crate::builder::ConstraintSystemBuilder;
use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, ExtensionField, TowerField,
};

pub fn pack<U, F, FInput, FOutput>(
	oracle_id: OracleId,
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
) -> Result<OracleId>
where
	F: TowerField + ExtensionField<FInput> + ExtensionField<FOutput>,
	FInput: TowerField,
	FOutput: TowerField,
	FOutput: ExtensionField<FInput>,
	U: UnderlierType + PackScalar<F> + PackScalar<FInput> + PackScalar<FOutput>,
{
	if FInput::TOWER_LEVEL == FOutput::TOWER_LEVEL {
		return Ok(oracle_id);
	}

	let packed_id =
		builder.add_packed(name, oracle_id, FOutput::TOWER_LEVEL - FInput::TOWER_LEVEL)?;

	if let Some(witness) = builder.witness() {
		let values_witness = witness.get::<FInput>(oracle_id)?;

		witness.set(packed_id, values_witness.repacked::<FOutput>())?;
	}

	Ok(packed_id)
}
