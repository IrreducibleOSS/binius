// Copyright 2024 Irreducible Inc.
use crate::builder::ConstraintSystemBuilder;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, ExtensionField, TowerField,
};
use bytemuck::Pod;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

pub fn unconstrained<U, F, FBase, FS>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	log_size: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<FBase> + PackScalar<FS>,
	F: TowerField + ExtensionField<FBase> + ExtensionField<FS>,
	FBase: TowerField + ExtensionField<FS>,
	FS: TowerField,
{
	let rng = builder.add_committed(name, log_size, FS::TOWER_LEVEL);

	if let Some(witness) = builder.witness() {
		witness
			.new_column::<FS>(rng, log_size)
			.as_mut_slice::<u8>()
			.into_par_iter()
			.for_each_init(thread_rng, |rng, data| {
				*data = rng.gen();
			});
	}

	Ok(rng)
}
