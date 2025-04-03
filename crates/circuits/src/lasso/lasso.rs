// Copyright 2024-2025 Irreducible Inc.

use anyhow::{ensure, Error, Result};
use binius_core::{
	constraint_system::channel::{ChannelId, OracleOrConst},
	oracle::OracleId,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::{get_packed_slice, set_packed_slice},
	ExtensionField, Field, PackedField, TowerField,
};
use itertools::{izip, Itertools};

use crate::{
	builder::{
		types::{F, U},
		ConstraintSystemBuilder,
	},
	transparent,
};

pub fn lasso<FC>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	n_lookups: &[usize],
	u_to_t_mappings: &[impl AsRef<[usize]>],
	lookups_u: &[impl AsRef<[OracleId]>],
	lookup_t: impl AsRef<[OracleId]>,
	channel: ChannelId,
) -> Result<()>
where
	FC: TowerField,
	U: PackScalar<FC>,
	F: ExtensionField<FC> + From<FC>,
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
	let lookup_o = transparent::constant(builder, "lookup_o", t_log_rows, Field::ONE)?;
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

		let lookup_f = lookup_f_witness.packed();

		let alpha = FC::MULTIPLICATIVE_GENERATOR;
		let alpha_packed = PackedType::<U, FC>::broadcast(alpha);
		let alpha_inverted_packed = alpha_packed.invert_or_zero();

		lookup_f.fill(PackedField::one());

		for (u_to_t_mapping, &n_lookups, &lookup_r, &lookup_w) in
			izip!(u_to_t_mappings, n_lookups, &lookups_r, &lookups_w)
		{
			let mut lookup_r_witness = witness.new_column::<FC>(lookup_r);
			let mut lookup_w_witness = witness.new_column::<FC>(lookup_w);

			let lookup_r = lookup_r_witness.packed();
			let lookup_w = lookup_w_witness.packed();

			lookup_r.fill(PackedField::one());
			lookup_w.fill(alpha_packed);

			// Process the number of lookups that fits into full packed field items.
			let packed_lookups = n_lookups >> PackedType::<U, FC>::LOG_WIDTH;
			let mut scalars = vec![FC::ZERO; PackedType::<U, FC>::WIDTH];
			for i in 0..packed_lookups {
				let offset = i << PackedType::<U, FC>::LOG_WIDTH;

				// Since indicees may repeat withing the same packed field item, we need to
				// process them one by one and use the result only after that.
				for (j, scalar) in scalars.iter_mut().enumerate() {
					let index = u_to_t_mapping.as_ref()[offset + j];
					*scalar = get_packed_slice(lookup_f, index) * alpha;
					set_packed_slice(lookup_f, index, *scalar);
				}

				let ts_by_alpha = PackedField::from_scalars(scalars.iter().copied());

				lookup_r[i] = ts_by_alpha * alpha_inverted_packed;
				lookup_w[i] = ts_by_alpha;
			}

			// Process the remainder
			let offset = packed_lookups << PackedType::<U, FC>::LOG_WIDTH;
			for i in offset..n_lookups {
				let ts = get_packed_slice(lookup_f, u_to_t_mapping.as_ref()[i]);
				set_packed_slice(lookup_r, i, ts);
				set_packed_slice(lookup_w, i, ts * alpha);
				set_packed_slice(lookup_f, u_to_t_mapping.as_ref()[i], ts * alpha);
			}
		}
	}

	lookups_r
		.iter()
		.for_each(|lookup_r| builder.assert_not_zero(*lookup_r));

	let oracles_prefix_t = lookup_t.as_ref().iter().copied();

	// populate table using initial timestamps
	builder.send(
		channel,
		1 << t_log_rows,
		oracles_prefix_t
			.clone()
			.chain([lookup_o])
			.map(OracleOrConst::Oracle),
	)?;

	// for every value looked up, pull using current timestamp and push with incremented timestamp
	izip!(lookups_u, lookups_r, lookups_w, n_lookups).try_for_each(
		|(lookup_u, lookup_r, lookup_w, &n_lookup)| -> Result<()> {
			let oracle_prefix_u = lookup_u.as_ref().iter().copied();
			builder.receive(
				channel,
				n_lookup,
				oracle_prefix_u
					.clone()
					.chain([lookup_r])
					.map(OracleOrConst::Oracle),
			)?;
			builder.send(
				channel,
				n_lookup,
				oracle_prefix_u.chain([lookup_w]).map(OracleOrConst::Oracle),
			)?;
			Ok(())
		},
	)?;

	// depopulate table using final timestamps
	builder.receive(
		channel,
		1 << t_log_rows,
		oracles_prefix_t
			.chain([lookup_f])
			.map(OracleOrConst::Oracle),
	)?;

	Ok(())
}
