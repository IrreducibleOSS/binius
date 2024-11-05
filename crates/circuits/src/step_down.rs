// Copyright 2024 Irreducible Inc.

use binius_core::{oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField1b, TowerField,
};

use crate::builder::ConstraintSystemBuilder;

pub fn step_down<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
	index: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + PackScalar<F> + PackScalar<BinaryField1b>,
	F: TowerField,
{
	let step_down = transparent::step_down::StepDown::new(log_size, index)?;
	let id = builder.add_transparent(name, step_down.clone())?;
	if let Some(witness) = builder.witness() {
		witness.update_multilin_poly([(
			id,
			step_down.multilinear_extension()?.specialize_arc_dyn(),
		)])?;
	}
	Ok(id)
}
