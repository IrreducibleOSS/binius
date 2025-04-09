// Copyright 2024-2025 Irreducible Inc.
use binius_core::oracle::OracleId;
use binius_field::{as_packed_field::PackScalar, ExtensionField, TowerField};
use binius_maybe_rayon::prelude::*;
use bytemuck::Pod;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::builder::{
	types::{F, U},
	ConstraintSystemBuilder,
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
		let mut rand_rng = StdRng::from_seed([42u8; 32]);
		witness
			.new_column::<FS>(rng)
			.as_mut_slice::<u8>()
			.into_iter()
			.for_each(|data| {
				*data = rand_rng.gen::<u8>();
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
