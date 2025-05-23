// Copyright 2024-2025 Irreducible Inc.
use binius_core::oracle::OracleId;
use binius_field::{ExtensionField, TowerField, as_packed_field::PackScalar};
use binius_maybe_rayon::prelude::*;
use bytemuck::Pod;
use rand::{Rng, thread_rng};

use crate::builder::{
	ConstraintSystemBuilder,
	types::{F, U},
};

pub fn unconstrained<FS>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_size: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<FS> + Pod,
	F: TowerField + ExtensionField<FS>,
	FS: TowerField,
{
	let rng = builder.add_committed(name, log_size, FS::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		witness
			.new_column::<FS>(rng)
			.as_mut_slice::<u8>()
			.into_par_iter()
			.for_each_init(thread_rng, |rng, data| {
				*data = rng.r#gen();
			});
	}

	Ok(rng)
}

// Same as 'unconstrained' but uses some pre-defined values instead of a random ones
pub fn fixed_u32<FS>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	log_size: usize,
	values: Vec<u32>,
) -> Result<OracleId, anyhow::Error>
where
	U: PackScalar<FS> + Pod,
	F: TowerField + ExtensionField<FS>,
	FS: TowerField,
{
	let fixed = builder.add_committed(name, log_size, FS::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		witness
			.new_column::<FS>(fixed)
			.as_mut_slice::<u32>()
			.into_par_iter()
			.zip(values.into_par_iter())
			.for_each(|(data, value)| {
				*data = value;
			});
	}

	Ok(fixed)
}
