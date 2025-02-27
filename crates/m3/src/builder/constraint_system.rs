// Copyright 2025 Irreducible Inc.

// Tables

// Lookups
// Columns <- Associated with tables

// Statement
// - Channel boundaries
// - Table sizes
pub use binius_core::constraint_system::channel::{
	Boundary, Flush as CompiledFlush, FlushDirection,
};
use binius_core::{
	constraint_system::{channel::ChannelId, ConstraintSystem as CompiledConstraintSystem},
	oracle::{Constraint, ConstraintPredicate, ConstraintSet, MultilinearOracleSet, OracleId},
	transparent::step_down::StepDown,
};
use binius_field::{underlier::UnderlierType, TowerField};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bumpalo::Bump;

use super::{
	channel::{Channel, Flush},
	column::{Column, ColumnInfo},
	error::Error,
	statement::Statement,
	table::{Table, TablePartition},
	types::B128,
	witness::{TableWitnessIndex, WitnessIndex},
};
use crate::builder::expr::ArithExprNamedVars;

#[derive(Debug, Default)]
pub struct ConstraintSystem<F: TowerField = B128> {
	pub tables: Vec<Table<F>>,
	pub channels: Vec<Channel>,
	/// All valid channel IDs are strictly less than this bound.
	pub channel_id_bound: ChannelId,
}

impl std::fmt::Display for ConstraintSystem {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "ConstraintSystem {{")?;

		for channel in self.channels.iter() {
			writeln!(f, "    CHANNEL {}", channel.name)?;
		}

		let mut oracle_id = 0;

		for table in self.tables.iter() {
			writeln!(f, "    TABLE {} {{", table.name)?;

			for partition in table.partitions.iter() {
				let names = partition
					.column_info
					.iter()
					.map(|c| c.name.clone())
					.collect::<Vec<_>>();
				for constraint in partition.zero_constraints.iter() {
					let name = constraint.name.clone();
					let expr = ArithExprNamedVars(&constraint.expr, &names);
					writeln!(f, "        ZERO {name}: {expr}")?;
				}

				for flush in partition.flushes.iter() {
					let channel = self.channels[flush.channel_id].name.clone();
					let columns = flush
						.column_indices
						.iter()
						.map(|i| partition.column_info[*i].name.clone())
						.collect::<Vec<_>>()
						.join(", ");
					match flush.direction {
						FlushDirection::Push => {
							writeln!(f, "        PUSH ({columns}) to {channel}")?
						}
						FlushDirection::Pull => {
							writeln!(f, "        PULL ({columns}) from {channel}")?
						}
					};
				}

				for col in partition.column_info.iter() {
					let name = col.name.clone();
					let pack_factor = 1 << col.shape.pack_factor;
					let field = match col.shape.tower_height {
						0 => "B1",
						1 => "B2",
						2 => "B4",
						3 => "B8",
						4 => "B16",
						5 => "B32",
						6 => "B64",
						_ => "B128",
					};

					let type_str = if pack_factor > 1 {
						format!("{field}x{pack_factor}")
					} else {
						field.to_string()
					};
					writeln!(f, "        {oracle_id:04} {type_str} {name}")?;
					oracle_id += 1;
				}

				let selector_type_str = if partition.pack_factor > 1 {
					format!("B1x{}", 1 << partition.pack_factor)
				} else {
					"B1".to_string()
				};
				writeln!(f, "        {oracle_id:04} {selector_type_str} (ROW_SELECTOR)")?;
				oracle_id += 1; // step_down selector for the table

				if partition.partition_index < table.partitions.len() - 1 {
					writeln!(f)?;
				}
			}

			writeln!(f, "    }}")?;
		}
		writeln!(f, "}}")
	}
}

impl<F: TowerField> ConstraintSystem<F> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn add_table(&mut self, name: impl ToString) -> &mut Table<F> {
		let id = self.tables.len();
		self.tables.push(Table::new(id, name));
		self.tables.last_mut().expect("table was just pushed")
	}

	pub fn add_channel(&mut self, name: impl ToString) -> ChannelId {
		let id = self.channels.len();
		self.channels.push(Channel {
			name: name.to_string(),
		});
		id
	}

	pub fn build_witness<'a, U: UnderlierType>(
		&self,
		allocator: &'a Bump,
		statement: &Statement,
	) -> Result<WitnessIndex<'a, U>, Error> {
		Ok(WitnessIndex::<U> {
			tables: self
				.tables
				.iter()
				.map(|table| {
					TableWitnessIndex::new(allocator, table, statement.table_sizes[table.id])
				})
				.collect(),
		})
	}

	/// Compiles a [`CompiledConstraintSystem`] for a particular statement.
	///
	/// The most important transformation that takes place in this step is creating multilinear
	/// oracles for all columns. The main difference between column definitions and oracle
	/// definitions is that multilinear oracle definitions have a number of variables, whereas the
	/// column definitions contained in a [`ConstraintSystem`] do not have size information.
	pub fn compile(&self, statement: &Statement<F>) -> Result<CompiledConstraintSystem<F>, Error> {
		if statement.table_sizes.len() != self.tables.len() {
			return Err(Error::StatementMissingTableSize {
				expected: self.tables.len(),
				actual: statement.table_sizes.len(),
			});
		}

		// TODO: new -> with_capacity
		let mut oracles = MultilinearOracleSet::new();
		let mut table_constraints = Vec::new();
		let mut compiled_flushes = Vec::new();
		let mut non_zero_oracle_ids = Vec::new();

		for (table, &count) in std::iter::zip(&self.tables, &statement.table_sizes) {
			let mut oracle_lookup = Vec::new();

			for partition in table.partitions.iter() {
				oracle_lookup.push(Vec::new());

				let TablePartition {
					column_info,
					flushes,
					zero_constraints,
					pack_factor,
					..
				} = partition;

				let n_vars = log2_ceil_usize(count) + pack_factor;
				let first_oracle_id_in_table = oracles.size();

				// Add multilinear oracles for all table columns.
				let partition_oracle_ids = column_info
					.iter()
					.map(|column_info| {
						let oracle_id = add_oracle_for_column(
							&mut oracles,
							&oracle_lookup,
							column_info,
							n_vars,
						)
						.unwrap();

						oracle_lookup[column_info.id.partition].push(oracle_id);

						if column_info.is_nonzero {
							non_zero_oracle_ids.push(oracle_id);
						}
						Ok::<_, Error>(oracle_id)
					})
					.collect::<Result<Vec<_>, _>>()?;
				// StepDown witness data is populated in WitnessIndex::into_multilinear_extension_index
				let selector =
					oracles.add_transparent(StepDown::new(n_vars, count << pack_factor)?)?;

				// Translate flushes for the compiled constraint system.
				for Flush {
					column_indices,
					channel_id,
					direction,
				} in flushes
				{
					let flush_oracles = column_indices
						.iter()
						.map(|&column_index| first_oracle_id_in_table + column_index)
						.collect::<Vec<_>>();
					compiled_flushes.push(CompiledFlush {
						oracles: flush_oracles,
						channel_id: *channel_id,
						direction: *direction,
						selector,
						multiplicity: 1,
					});
				}

				if !zero_constraints.is_empty() {
					// Translate zero constraints for the compiled constraint system.
					let compiled_constraints = zero_constraints
						.iter()
						.map(|zero_constraint| Constraint {
							name: zero_constraint.name.clone(),
							composition: zero_constraint.expr.clone(),
							predicate: ConstraintPredicate::Zero,
						})
						.collect::<Vec<_>>();

					table_constraints.push(ConstraintSet {
						n_vars,
						oracle_ids: partition_oracle_ids,
						constraints: compiled_constraints,
					});
				}
			}
		}

		Ok(CompiledConstraintSystem {
			oracles,
			table_constraints,
			flushes: compiled_flushes,
			non_zero_oracle_ids,
			max_channel_id: self.channel_id_bound.saturating_sub(1),
		})
	}
}

fn add_oracle_for_column<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	oracle_lookup: &[Vec<OracleId>],
	column_info: &ColumnInfo<F>,
	n_vars: usize,
) -> Result<OracleId, Error> {
	let ColumnInfo { id, col, name, .. } = column_info;
	let addition = oracles.add_named(name.clone());
	let oracle_id = match col {
		Column::Committed { tower_level } => addition.committed(n_vars, *tower_level),
		Column::LinearCombination(lincom) => {
			let inner_oracles = lincom
				.var_coeffs
				.iter()
				.enumerate()
				.map(|(index, &coeff)| (oracle_lookup[id.partition][index], coeff))
				.collect::<Vec<_>>();
			addition.linear_combination_with_offset(n_vars, lincom.constant, inner_oracles)?
		}
		Column::Shifted {
			col,
			offset,
			log_block_size,
			variant,
		} => {
			assert_eq!(col.partition, id.partition);
			addition.shifted(
				oracle_lookup[col.partition][col.index],
				*offset,
				*log_block_size,
				*variant,
			)?
		}
		Column::Packed { col, log_degree } => {
			addition.packed(oracle_lookup[col.partition][col.index], *log_degree)?
		}
	};
	Ok(oracle_id)
}
// table[partition][column] -> oracle_id
