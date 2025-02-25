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
	statement::Instance,
	table::Table,
	types::B128,
	witness::WitnessIndex,
};

#[derive(Debug, Default)]
pub struct ConstraintSystem<F: TowerField = B128> {
	pub tables: Vec<Table<F>>,
	pub channels: Vec<Channel>,
	/// All valid channel IDs are strictly less than this bound.
	pub channel_id_bound: ChannelId,
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

	pub fn build_witness<U: UnderlierType>(
		&self,
		allocator: &Bump,
		instance: &Instance,
	) -> Result<WitnessIndex<U>, Error> {
		todo!()
	}

	/// Compiles a [`CompiledConstraintSystem`] for a particular statement.
	///
	/// The most important transformation that takes place in this step is creating multilinear
	/// oracles for all columns. The main difference between column definitions and oracle
	/// definitions is that multilinear oracle definitions have a number of variables, whereas the
	/// column definitions contained in a [`ConstraintSystem`] do not have size information.
	pub fn compile(&self, statement: &Instance<F>) -> Result<CompiledConstraintSystem<F>, Error> {
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
				id,
				name,
				column_info,
				flushes,
				zero_constraints,
				is_fixed: _,
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

			// TODO: How do we add StepDown data to the witness index?
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
				.map(|zero_constraint| {
					// The table zero constraint is an expression with column indices in the table.
					// We need to remap these column indices to oracle IDs.
					let composition = zero_constraint.expr.clone().remap_vars(&table_oracle_ids)?;
					Ok::<_, Error>(Constraint {
						name: zero_constraint.name.clone(),
						composition,
						predicate: ConstraintPredicate::Zero,
					})
				})
				.collect::<Result<Vec<_>, _>>()?;
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
	//table_id: TableId,
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
	};
	Ok(oracle_id)
}
