// Copyright 2024 Irreducible Inc.

use anyhow::Result;
use binius_core::{constraint_system::channel::ChannelId, oracle::OracleId, transparent};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, PackedFieldIndexable, TowerField,
};
use itertools::izip;

use crate::builder::ConstraintSystemBuilder;

use crate::helpers::{make_underliers, underliers_unpack_scalars_mut};

pub fn lasso<U, F, FBase, FS, FC, const T_LOG_SIZE: usize>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	name: impl ToString,
	n_vars: usize,
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

	let lookup_f = builder.add_committed("lookup_f", T_LOG_SIZE, FC::TOWER_LEVEL);
	let lookup_o = builder.add_transparent(
		"lookup_o",
		transparent::constant::Constant {
			n_vars: T_LOG_SIZE,
			value: F::ONE,
		},
	)?;
	let lookup_r = builder.add_committed("lookup_r", n_vars, FC::TOWER_LEVEL);
	let lookup_w = builder.add_linear_combination(
		"lookup_w",
		n_vars,
		[(lookup_r, FC::MULTIPLICATIVE_GENERATOR.into())],
	)?;

	if let Some(witness) = builder.witness() {
		let mut lookup_r_witness = make_underliers::<_, FC>(n_vars);
		let mut lookup_w_witness = make_underliers::<_, FC>(n_vars);
		let mut lookup_f_witness = make_underliers::<_, FC>(T_LOG_SIZE);
		let mut lookup_o_witness = make_underliers::<_, FC>(T_LOG_SIZE);

		let lookup_r_scalars = underliers_unpack_scalars_mut::<_, FC>(&mut lookup_r_witness);
		let lookup_w_scalars = underliers_unpack_scalars_mut::<_, FC>(&mut lookup_w_witness);
		let lookup_f_scalars = underliers_unpack_scalars_mut::<_, FC>(&mut lookup_f_witness);
		let lookup_o_scalars = underliers_unpack_scalars_mut::<_, FC>(&mut lookup_o_witness);

		lookup_f_scalars.fill(FC::ONE);
		lookup_o_scalars.fill(FC::ONE);

		let alpha = FC::MULTIPLICATIVE_GENERATOR;

		let u_to_t_mapping = u_to_t_mapping
			.ok_or(anyhow::Error::msg("the u_to_t_mapping must be provided for the witness"))?;

		for (j, (r, w)) in
			izip!(lookup_r_scalars.iter_mut(), lookup_w_scalars.iter_mut()).enumerate()
		{
			let index = u_to_t_mapping[j];
			let ts = lookup_f_scalars[index];
			*r = ts;
			*w = ts * alpha;
			lookup_f_scalars[index] *= alpha;
		}

		witness.set_owned::<FC, _>([
			(lookup_r, lookup_r_witness),
			(lookup_w, lookup_w_witness),
			(lookup_f, lookup_f_witness),
			(lookup_o, lookup_o_witness),
		])?;
	}

	builder.assert_not_zero(lookup_r);

	// populate table using initial timestamps
	builder.send(channel, [lookup_t, lookup_o]);

	// for every value looked up, pull using current timestamp and push with incremented timestamp
	builder.receive(channel, [lookup_u, lookup_r]);
	builder.send(channel, [lookup_u, lookup_w]);

	// depopulate table using final timestamps
	builder.receive(channel, [lookup_t, lookup_f]);

	Ok(())
}
