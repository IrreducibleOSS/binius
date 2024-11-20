// Copyright 2024 Irreducible Inc.

use binius_core::oracle::{OracleId, ShiftVariant};
use binius_field::{
	as_packed_field::PackScalar, underlier::UnderlierType, BinaryField1b, BinaryField32b,
	ExtensionField, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::Pod;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::{builder::ConstraintSystemBuilder, transparent::step_down, u32add::u32add};

pub fn u32fib<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	log_size: usize,
) -> Result<OracleId, anyhow::Error>
where
	U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b> + PackScalar<BinaryField32b>,
	F: TowerField + ExtensionField<BinaryField32b>,
{
	builder.push_namespace(name);
	let current = builder.add_committed("current", log_size, BinaryField1b::TOWER_LEVEL);
	let next = builder.add_shifted("next", current, 32, log_size, ShiftVariant::LogicalRight)?;
	let next_next =
		builder.add_shifted("next_next", current, 64, log_size, ShiftVariant::LogicalRight)?;

	if let Some(witness) = builder.witness() {
		let mut current = witness.new_column::<BinaryField1b>(current, log_size);
		let mut next = witness.new_column::<BinaryField1b>(next, log_size);
		let mut next_next = witness.new_column::<BinaryField1b>(next_next, log_size);

		let mut rng = thread_rng();
		let current = current.as_mut_slice::<u32>();
		current[0] = rng.gen();
		current[1] = rng.gen();
		for i in 2..current.len() {
			current[i] = rng.gen();
			(current[i], _) = current[i - 1].overflowing_add(current[i - 2]);
		}
		(next.as_mut_slice::<u32>(), &current[1..])
			.into_par_iter()
			.for_each(|(next, current)| {
				*next = *current;
			});
		(next_next.as_mut_slice::<u32>(), &current[2..])
			.into_par_iter()
			.for_each(|(next_next, current)| {
				*next_next = *current;
			});
	}

	let packed_log_size = log_size - 5;
	let enabled = step_down(builder, "enabled", packed_log_size, (1 << packed_log_size) - 2)?;
	let sum = u32add(builder, "sum", log_size, current, next)?;
	let sum_packed = builder.add_packed("sum_packed", sum, 5)?;
	let next_next_packed = builder.add_packed("next_next_packed", next_next, 5)?;

	if let Some(witness) = builder.witness() {
		let next_next_packed_witness = witness.get::<BinaryField1b>(next_next)?;
		witness.set_data::<BinaryField32b>(
			next_next_packed,
			packed_log_size,
			next_next_packed_witness,
		)?;
		let sum_packed_witness = witness.get::<BinaryField1b>(sum)?;
		witness.set_data::<BinaryField32b>(sum_packed, packed_log_size, sum_packed_witness)?;
	}

	builder.assert_zero(
		[sum_packed, next_next_packed, enabled],
		composition_poly!([a, b, enabled] = (a - b) * enabled),
	);

	builder.pop_namespace();
	Ok(current)
}
