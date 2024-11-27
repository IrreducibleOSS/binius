// Copyright 2024 Irreducible Inc.

use anyhow::{ensure, Result};
use binius_core::{constraint_system::channel::ChannelId, oracle::OracleId};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, PackedFieldIndexable, TowerField,
};
use itertools::izip;

use crate::{builder::ConstraintSystemBuilder, transparent};

pub fn lasso<U, F, FBase, FS, FC>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	n_lookups: usize,
	u_to_t_mapping: Option<Vec<usize>>,
	lookup_u: OracleId,
	lookup_t: OracleId,
	channel: ChannelId,
) -> Result<()>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase> + PackScalar<FC>,
	F: TowerField + ExtensionField<FBase> + ExtensionField<FC> + From<FC>,
	PackedType<U, FC>: PackedFieldIndexable,
	FBase: TowerField,
	FC: TowerField,
	FS: TowerField,
{
	builder.push_namespace(name);

	let u_log_rows = builder.log_rows([lookup_u])?;
	let t_log_rows = builder.log_rows([lookup_t])?;
	ensure!(n_lookups <= 1 << u_log_rows);

	let lookup_o = transparent::constant(builder, "lookup_o", t_log_rows, F::ONE)?;
	let lookup_f = builder.add_committed("lookup_f", t_log_rows, FC::TOWER_LEVEL);
	let lookup_r = builder.add_committed("lookup_r", u_log_rows, FC::TOWER_LEVEL);
	let lookup_w = builder.add_linear_combination(
		"lookup_w",
		u_log_rows,
		[(lookup_r, FC::MULTIPLICATIVE_GENERATOR.into())],
	)?;

	if let Some(witness) = builder.witness() {
		let mut lookup_r_witness = witness.new_column::<FC>(lookup_r);
		let mut lookup_w_witness = witness.new_column::<FC>(lookup_w);
		let mut lookup_f_witness = witness.new_column::<FC>(lookup_f);

		let lookup_r_scalars = PackedType::<U, FC>::unpack_scalars_mut(lookup_r_witness.packed());
		let lookup_w_scalars = PackedType::<U, FC>::unpack_scalars_mut(lookup_w_witness.packed());
		let lookup_f_scalars = PackedType::<U, FC>::unpack_scalars_mut(lookup_f_witness.packed());

		let alpha = FC::MULTIPLICATIVE_GENERATOR;

		lookup_r_scalars.fill(FC::ONE);
		lookup_w_scalars.fill(alpha);
		lookup_f_scalars.fill(FC::ONE);

		let u_to_t_mapping = u_to_t_mapping
			.ok_or(anyhow::Error::msg("the u_to_t_mapping must be provided for the witness"))?;

		for (j, (r, w)) in izip!(lookup_r_scalars.iter_mut(), lookup_w_scalars.iter_mut())
			.enumerate()
			.take(n_lookups)
		{
			let index = u_to_t_mapping[j];
			let ts = lookup_f_scalars[index];
			*r = ts;
			*w = ts * alpha;
			lookup_f_scalars[index] *= alpha;
		}
	}

	builder.assert_not_zero(lookup_r);

	// populate table using initial timestamps
	builder.send(channel, 1 << t_log_rows, [lookup_t, lookup_o]);

	// for every value looked up, pull using current timestamp and push with incremented timestamp
	builder.receive(channel, n_lookups, [lookup_u, lookup_r]);
	builder.send(channel, n_lookups, [lookup_u, lookup_w]);

	// depopulate table using final timestamps
	builder.receive(channel, 1 << t_log_rows, [lookup_t, lookup_f]);

	builder.pop_namespace();
	Ok(())
}
