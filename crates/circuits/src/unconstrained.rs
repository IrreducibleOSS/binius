// Copyright 2024 Irreducible Inc.
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, ExtensionField, TowerField,
};
use bytemuck::Pod;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::builder::ConstraintSystemBuilder;

pub fn unconstrained<U, F, FS>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<FS>,
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
