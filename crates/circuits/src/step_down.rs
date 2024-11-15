// Copyright 2024 Irreducible Inc.

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField1b, ExtensionField, TowerField,
};

use crate::builder::ConstraintSystemBuilder;

pub fn step_down<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase> + PackScalar<BinaryField1b>,
	F: TowerField + ExtensionField<FBase>,
	FBase: TowerField,
{
	let step_down = transparent::step_down::StepDown::new(log_size, index)?;
	let id = builder.add_transparent(name, step_down.clone())?;
	if let Some(witness) = builder.witness() {
		witness.update_multilin_poly([(
			id,
			step_down
				.multilinear_extension::<PackedType<U, BinaryField1b>>()?
				.specialize_arc_dyn(),
		)])?;
	}
	Ok(id)
}
