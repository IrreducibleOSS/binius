// Copyright 2024-2025 Irreducible Inc.
use binius_core::oracle::OracleId;
use binius_field::{as_packed_field::PackScalar, ExtensionField, TowerField};
use binius_maybe_rayon::prelude::*;
use bytemuck::Pod;
use rand::{thread_rng, Rng};

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
		witness
			.new_column::<FS>(rng)
			.as_mut_slice::<u8>()
			.into_par_iter()
			.for_each_init(thread_rng, |rng, data| {
				*data = rng.gen();
			});
	}

	Ok(rng)
}
