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

pub fn lasso<U, F, FS, FC>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	n_lookups: &[usize],
	u_to_t_mappings: &[Vec<usize>],
	lookups_u: &[OracleId],
	lookup_t: OracleId,
	channel: ChannelId,
) -> Result<()>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FC>,
	F: TowerField + ExtensionField<FC> + From<FC>,
	PackedType<U, FC>: PackedFieldIndexable,
	FC: TowerField,
	FS: TowerField,
{
	if n_lookups.len() != lookups_u.len() {
		Err(anyhow::Error::msg("n_vars and lookups_u must be of the same length"))?;
	}

	if n_lookups.iter().sum::<usize>() >= 1 << FC::N_BITS {
		Err(anyhow::Error::msg("FC too small"))?;
	}

	builder.push_namespace(name);

	let u_log_rows = lookups_u
		.iter()
		.map(|id| builder.log_rows([*id]))
		.collect::<Result<Vec<_>, Error>>()?;

	for (n_lookups, u_log_rows) in n_lookups.iter().zip(&u_log_rows) {
		ensure!(*n_lookups <= 1 << *u_log_rows);
	}

	let t_log_rows = builder.log_rows([lookup_t])?;
	let lookup_o = transparent::constant(builder, "lookup_o", t_log_rows, F::ONE)?;
	let lookup_f = builder.add_committed("lookup_f", t_log_rows, FC::TOWER_LEVEL);
	let lookups_r = u_log_rows
		.iter()
		.map(|u_log_rows| builder.add_committed("lookup_r", *u_log_rows, FC::TOWER_LEVEL))
		.collect_vec();

	let lookups_w = u_log_rows
		.iter()
		.zip(&lookups_r)
		.map(|(u_log_rows, lookup_r)| {
			builder
				.add_linear_combination(
					"lookup_w",
					*u_log_rows,
					[(*lookup_r, FC::MULTIPLICATIVE_GENERATOR.into())],
				)
				.map_err(|e| e.into())
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

		for (i, (u_to_t_mapping, n_lookups)) in u_to_t_mappings.iter().zip(n_lookups).enumerate() {
			let mut lookup_r_witness = witness.new_column::<FC>(lookups_r[i]);
			let mut lookup_w_witness = witness.new_column::<FC>(lookups_w[i]);

			let lookup_r_scalars =
				PackedType::<U, FC>::unpack_scalars_mut(lookup_r_witness.packed());
			let lookup_w_scalars =
				PackedType::<U, FC>::unpack_scalars_mut(lookup_w_witness.packed());

			lookup_r_scalars.fill(FC::ONE);
			lookup_w_scalars.fill(alpha);

			for (j, (r, w)) in izip!(lookup_r_scalars.iter_mut(), lookup_w_scalars.iter_mut())
				.enumerate()
				.take(*n_lookups)
			{
				let index = u_to_t_mapping[j];
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

	// populate table using initial timestamps
	builder.send(channel, 1 << t_log_rows, [lookup_t, lookup_o]);

	// for every value looked up, pull using current timestamp and push with incremented timestamp
	izip!(lookups_u, &lookups_r, &lookups_w, n_lookups).for_each(
		|(lookup_u, lookup_r, lookup_w, n_lookup)| {
			builder.receive(channel, *n_lookup, [*lookup_u, *lookup_r]);
			builder.send(channel, *n_lookup, [*lookup_u, *lookup_w]);
		},
	);

	// depopulate table using final timestamps
	builder.receive(channel, 1 << t_log_rows, [lookup_t, lookup_f]);

	Ok(())
}
