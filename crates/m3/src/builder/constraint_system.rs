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
	table::Table,
	types::B128,
	witness::WitnessIndex,
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

			let names = table
				.column_info
				.iter()
				.map(|c| c.name.clone())
				.collect::<Vec<_>>();
			for constraint in table.zero_constraints.iter() {
				let name = constraint.name.clone();
				let expr = ArithExprNamedVars(&constraint.expr, &names);
				writeln!(f, "        ZERO {name}: {expr}")?;
			}

			if !table.zero_constraints.is_empty() && !table.flushes.is_empty() {
				writeln!(f, "")?;
			}

			for flush in table.flushes.iter() {
				let channel = self.channels[flush.channel_id].name.clone();
				let columns = flush
					.column_indices
					.iter()
					.map(|i| table.column_info[*i].name.clone())
					.collect::<Vec<_>>()
					.join(", ");
				match flush.direction {
					FlushDirection::Push => writeln!(f, "        PUSH ({columns}) to {channel}")?,
					FlushDirection::Pull => writeln!(f, "        PULL ({columns}) from {channel}")?,
				};
			}

			if !table.column_info.is_empty() && !table.flushes.is_empty() {
				writeln!(f, "")?;
			}

			for col in table.column_info.iter() {
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
					format!("{field}")
				};
				writeln!(f, "        {oracle_id:04} {type_str} {name}")?;
				oracle_id += 1;
			}

			if !table.column_info.is_empty() && !table.flushes.is_empty() {
				writeln!(f, "")?;
			}

			writeln!(f, "        {oracle_id:04} B1 (ROW_SELECTOR)")?;
			oracle_id += 1; // step_down selector for the table

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
		let mut witness = WitnessIndex::<U>::default();
		witness.tables = self
			.tables
			.iter()
			.enumerate()
			.map(|(i, table)| {
				let log_capacity = log2_ceil_usize(statement.table_sizes[i]);
				let column_shapes = table
					.column_info
					.iter()
					.map(|col| col.shape)
					.collect::<Vec<_>>();
				super::witness::TableWitnessIndex::new(
					allocator,
					table.id,
					&column_shapes,
					log_capacity,
				)
			})
			.collect();
		Ok(witness)
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
		let mut table_constraints = Vec::with_capacity(self.tables.len());
		let mut compiled_flushes = Vec::new();
		let mut non_zero_oracle_ids = Vec::new();
		for (table, &count) in std::iter::zip(&self.tables, &statement.table_sizes) {
			let Table {
				column_info,
				flushes,
				zero_constraints,
				is_fixed: _,
				..
			} = table;
			let n_vars = log2_ceil_usize(count);
			let first_oracle_id_in_table = oracles.size();

			// Add multilinear oracles for all table columns.
			let table_oracle_ids = column_info
				.iter()
				.map(|column_info| {
					let oracle_id = add_oracle_for_column(
						&mut oracles,
						first_oracle_id_in_table,
						column_info,
						n_vars,
					)?;
					if column_info.is_nonzero {
						non_zero_oracle_ids.push(oracle_id);
					}
					Ok::<_, Error>(oracle_id)
				})
				.collect::<Result<Vec<_>, _>>()?;

			// StepDown witness data is populated in WitnessIndex::into_multilinear_extension_index
			let selector = oracles.add_transparent(StepDown::new(n_vars, count)?)?;

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
					multiplicity: count as u64,
				});
			}

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
				oracle_ids: table_oracle_ids,
				constraints: compiled_constraints,
			});
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
	first_oracle_id_in_table: OracleId,
	column_info: &ColumnInfo<F>,
	n_vars: usize,
) -> Result<OracleId, Error> {
	let ColumnInfo { col, name, .. } = column_info;
	let addition = oracles.add_named(name.clone());
	let oracle_id = match col {
		Column::Committed { tower_level } => addition.committed(n_vars, *tower_level),
		Column::LinearCombination(lincom) => {
			let inner_oracles = lincom
				.var_coeffs
				.iter()
				.enumerate()
				.map(|(index, &coeff)| (first_oracle_id_in_table + index, coeff))
				.collect::<Vec<_>>();
			addition.linear_combination_with_offset(n_vars, lincom.constant, inner_oracles)?
		}
		Column::Shifted {
			col_index: col,
			offset,
			log_block_size,
			variant,
		} => {
			addition.shifted(first_oracle_id_in_table + *col, *offset, *log_block_size, *variant)?
		}
		Column::Packed {
			col_index,
			log_degree,
		} => addition.packed(first_oracle_id_in_table + *col_index, *log_degree)?,
	};
	Ok(oracle_id)
}
