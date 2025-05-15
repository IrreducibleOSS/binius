// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{ExtensionField, TowerField, as_packed_field::PackScalar};

use crate::builder::{
	ConstraintSystemBuilder,
	types::{F, U},
};

pub fn pack<FInput, FOutput>(
	oracle_id: OracleId,
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
) -> Result<OracleId>
where
	F: ExtensionField<FInput> + ExtensionField<FOutput>,
	FInput: TowerField,
	FOutput: TowerField + ExtensionField<FInput>,
	U: PackScalar<FInput> + PackScalar<FOutput>,
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
