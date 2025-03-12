use anyhow::Ok;
use binius_core::oracle::OracleId;
use binius_field::{
	underlier::WithUnderlier, BinaryField128b, BinaryField16b, BinaryField64b, PackedField,
	TowerField,
};
use binius_maybe_rayon::iter::{
	IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tracing::instrument;

use crate::{
	builder::{types::F, ConstraintSystemBuilder},
	plain_lookup, transparent,
};

pub fn build_exp_table(
	g: BinaryField64b,
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
) -> Result<OracleId, anyhow::Error> {
	let chunk_size = 1024;

	let table_values = (0..1 << 16)
		.into_par_iter()
		.chunks(chunk_size)
		.enumerate()
		.flat_map(|(i, chunk)| {
			let start = i * chunk_size;
			let mut current_g = g.pow(start as u128);
			chunk
				.iter()
				.map(|&i| {
					let val = ((i as u128) << 64) | (current_g.to_underlier() as u128);
					current_g *= g;
					val
				})
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();

	transparent::make_transparent(
		builder,
		name,
		bytemuck::cast_slice::<_, BinaryField128b>(&table_values),
	)
}

pub fn static_exp_lookups<const LOG_MAX_MULTIPLICITY: usize>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	xin: OracleId,
	g: BinaryField64b,
) -> Result<OracleId, anyhow::Error> {
	let log_rows = builder.log_rows([xin])?;

	let name = name.to_string();

	let exp_result = builder.add_committed("exp_result", log_rows, BinaryField64b::TOWER_LEVEL);
	println!("{}: log_rows: {} TOWER_LEVEL:{}", name, log_rows, BinaryField64b::TOWER_LEVEL);

	let g_lookup_table = build_exp_table(g, builder, "g")?;

	let lookup_values = builder.add_linear_combination(
		"lookup_values",
		log_rows,
		[
			(xin, <F as TowerField>::basis(4, 4)?),
			(exp_result, <F as TowerField>::basis(6, 0)?),
		],
	)?;

	if let Some(witness) = builder.witness() {
		let xin = witness.get::<BinaryField16b>(xin)?.as_slice::<u16>();

		let mut exp_result = witness.new_column::<BinaryField64b>(exp_result);

		let exp_result = exp_result.as_mut_slice::<u64>();

		exp_result.par_iter_mut().enumerate().for_each(|(i, exp)| {
			*exp = g.pow(xin[i] as u128).to_underlier();
		});

		let mut lookup_values = witness.new_column::<BinaryField128b>(lookup_values);

		let lookup_values = lookup_values.as_mut_slice::<u128>();

		lookup_values
			.iter_mut()
			.enumerate()
			.for_each(|(i, look_val)| {
				*look_val = ((xin[i] as u128) << 64) | exp_result[i] as u128;
			});
	}

	plain_lookup::plain_lookup::<BinaryField128b, LOG_MAX_MULTIPLICITY>(
		builder,
		g_lookup_table,
		lookup_values,
		1 << log_rows,
	)?;

	Ok(exp_result)
}
