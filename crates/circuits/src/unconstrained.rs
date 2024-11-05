// Copyright 2024 Irreducible Inc.
use crate::builder::ConstraintSystemBuilder;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, PackedField, TowerField,
};
use bytemuck::{must_cast_slice_mut, Pod};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

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
		let len = 1 << (log_size - <PackedType<U, FS>>::LOG_WIDTH);
		let mut data = vec![U::default(); len].into_boxed_slice();
		must_cast_slice_mut::<_, u8>(&mut data)
			.into_par_iter()
			.for_each_init(thread_rng, |rng, data| {
				*data = rng.gen();
			});
		witness.set_owned::<FS, _>([(rng, data)])?;
	}

	Ok(rng)
}
