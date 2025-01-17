// Copyright 2024-2025 Irreducible Inc.

use anyhow::{ensure, Error, Result};
use binius_core::{constraint_system::channel::ChannelId, oracle::OracleId};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, PackedFieldIndexable, TowerField,
};
use itertools::{izip, Itertools};

use crate::{builder::ConstraintSystemBuilder, transparent};

pub fn lasso<U, F, FC>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	n_lookups: &[usize],
	u_to_t_mappings: &[impl AsRef<[usize]>],
	lookups_u: &[impl AsRef<[OracleId]>],
	lookup_t: impl AsRef<[OracleId]>,
	channel: ChannelId,
) -> Result<()>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FC>,
	F: TowerField + ExtensionField<FC> + From<FC>,
	PackedType<U, FC>: PackedFieldIndexable,
	FC: TowerField,
{
	if n_lookups.len() != lookups_u.len() {
		Err(anyhow::Error::msg("n_vars and lookups_u must be of the same length"))?;
	}

	if n_lookups.iter().sum::<usize>() >= 1 << FC::N_BITS {
		Err(anyhow::Error::msg("FC too small"))?;
	}

	if lookups_u
		.iter()
		.any(|oracles| oracles.as_ref().len() != lookup_t.as_ref().len())
	{
		Err(anyhow::Error::msg(
			"looked up and lookup tables must have the same number of oracles",
		))?;
	}

	builder.push_namespace(name);

	let u_log_rows = lookups_u
		.iter()
		.map(|row_oracles| builder.log_rows(row_oracles.as_ref().iter().copied()))
		.collect::<Result<Vec<_>, Error>>()?;

	for (n_lookups, u_log_rows) in n_lookups.iter().zip(&u_log_rows) {
		ensure!(*n_lookups <= 1 << *u_log_rows);
	}

	let t_log_rows = builder.log_rows(lookup_t.as_ref().iter().copied())?;
	let lookup_o = transparent::constant(builder, "lookup_o", t_log_rows, F::ONE)?;
	let lookup_f = builder.add_committed("lookup_f", t_log_rows, FC::TOWER_LEVEL);
	let lookups_r = u_log_rows
		.iter()
		.map(|&u_log_rows| builder.add_committed("lookup_r", u_log_rows, FC::TOWER_LEVEL))
		.collect_vec();

	let lookups_w = u_log_rows
		.iter()
		.zip(&lookups_r)
		.map(|(u_log_rows, lookup_r)| {
			Ok(builder.add_linear_combination(
				"lookup_w",
				*u_log_rows,
				[(*lookup_r, FC::MULTIPLICATIVE_GENERATOR.into())],
			)?)
		})
		.collect::<Result<Vec<_>, anyhow::Error>>()?;

	if let Some(witness) = builder.witness() {
		if u_log_rows.len() != u_to_t_mappings.len() {
			Err(anyhow::Error::msg("u_log_rows and u_to_t_mappings must be of the same length"))?;
		}

		let mut lookup_f_witness = witness.new_column::<FC>(lookup_f);

		let lookup_f_scalars = PackedType::<U, FC>::unpack_scalars_mut(lookup_f_witness.packed());

		let alpha = FC::MULTIPLICATIVE_GENERATOR;

		lookup_f_scalars.fill(FC::ONE);

		for (u_to_t_mapping, &n_lookups, &lookup_r, &lookup_w) in
			izip!(u_to_t_mappings, n_lookups, &lookups_r, &lookups_w)
		{
			let mut lookup_r_witness = witness.new_column::<FC>(lookup_r);
			let mut lookup_w_witness = witness.new_column::<FC>(lookup_w);

			let lookup_r_scalars =
				PackedType::<U, FC>::unpack_scalars_mut(lookup_r_witness.packed());
			let lookup_w_scalars =
				PackedType::<U, FC>::unpack_scalars_mut(lookup_w_witness.packed());

			lookup_r_scalars.fill(FC::ONE);
			lookup_w_scalars.fill(alpha);

			for (&index, r, w) in
				izip!(u_to_t_mapping.as_ref(), lookup_r_scalars, lookup_w_scalars).take(n_lookups)
			{
				let ts = lookup_f_scalars[index];
				*r = ts;
				*w = ts * alpha;
				lookup_f_scalars[index] *= alpha;
			}
		}
	}

	lookups_r
		.iter()
		.for_each(|lookup_r| builder.assert_not_zero(*lookup_r));

	let oracles_prefix_t = lookup_t.as_ref().iter().copied();

	// populate table using initial timestamps
	builder.send(channel, 1 << t_log_rows, oracles_prefix_t.clone().chain([lookup_o]));

	// for every value looked up, pull using current timestamp and push with incremented timestamp
	izip!(lookups_u, lookups_r, lookups_w, n_lookups).for_each(
		|(lookup_u, lookup_r, lookup_w, &n_lookup)| {
			let oracle_prefix_u = lookup_u.as_ref().iter().copied();
			builder.receive(channel, n_lookup, oracle_prefix_u.clone().chain([lookup_r]));
			builder.send(channel, n_lookup, oracle_prefix_u.chain([lookup_w]));
		},
	);

	// depopulate table using final timestamps
	builder.receive(channel, 1 << t_log_rows, oracles_prefix_t.chain([lookup_f]));

	Ok(())
}
