// Copyright 2025 Irreducible Inc.

// Tables

// Lookups
// Columns <- Associated with tables

use std::iter;

// Statement
// - Channel boundaries
// - Table sizes
pub use binius_core::constraint_system::channel::{
	Boundary, Flush as CompiledFlush, FlushDirection,
};
use binius_core::{
	constraint_system::ConstraintSystem as CompiledConstraintSystem,
	oracle::{Constraint, ConstraintPredicate, ConstraintSet, MultilinearOracleSet, OracleId},
};
use binius_field::TowerField;
use binius_math::ArithExpr;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bumpalo::Bump;

use super::error::Error;
use crate::{types::B128, witness::TableWitnessIndex};

pub type TableId = usize;
pub type ChannelId = usize;
pub type ColumnIndex = usize; // REVIEW: Could make these opaque without a constructor, to protect
							  // access

#[derive(Debug, Clone, Copy)]
pub struct ColumnShape {
	pub tower_height: usize,
	pub pack_factor: usize,
}

#[derive(Debug)]
pub struct ColumnId {
	pub table_id: TableId,
	pub index: ColumnIndex,
}

// TODO: Impl Add/Sub/Mul for Col, returning Expr

// feature: TableBuilder needs namespacing
pub enum Column {
	Committed { tower_level: usize },
}

pub struct ColumnInfo {
	pub col: Column,
	pub name: String,
	/// This represents how many
	pub pack_factor: usize,
	pub is_nonzero: bool,
}

pub struct Table<F: TowerField = B128> {
	pub id: TableId,
	pub name: String,
	pub column_info: Vec<ColumnInfo>,
	pub flushes: Vec<Flush>,
	pub zero_constraints: Vec<ArithExpr<F>>,
	/// This indicates whether a table is fixed for constraint system or part of the dynamic trace.
	///
	/// Fixed tables are either entirely transparent or committed during a preprocessing step that
	/// occurs before any statements are proven.
	pub is_fixed: bool,
}

pub struct Flush {
	pub column_indices: Vec<usize>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
}

pub struct Instance<F: TowerField = B128> {
	pub boundaries: Vec<Boundary<F>>,
	/// Direct index mapping table IDs to the count of rows per table.
	///
	/// The table sizes seem like advice values that don't affect the semantic meaning of the
	/// statement, but we include them in the statement directly. This makes sense because
	///
	/// 1. These values affect the control flow of the verification routine.
	/// 2. These values are necessarily made public.
	/// 3. For some constraint systems, the verifier does care about the values. For example, the
	///    statement could be that a VM execution state is reachable within a certain number of
	///    cycles.
	pub table_sizes: Vec<usize>,
}

pub struct Channel<F: TowerField = B128> {
	pub name: String,
}

pub struct ConstraintSystem<F: TowerField = B128> {
	tables: Vec<Table<F>>,
	/// All valid channel IDs are strictly less than this bound.
	channel_id_bound: ChannelId,
}

impl<F: TowerField> ConstraintSystem<F> {
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
		for (table, &count) in iter::zip(&self.tables, &statement.table_sizes) {
			let Table {
				id,
				name,
				column_info,
				flushes,
				zero_constraints,
				is_fixed: _,
			} = table;
			let n_vars = log2_ceil_usize(count);

			// Add multilinear oracles for all table columns.
			let table_oracle_ids = column_info
				.iter()
				.map(|column_info| {
					let oracle_id = add_oracle_for_column(&mut oracles, column_info, n_vars);
					if column_info.is_nonzero {
						non_zero_oracle_ids.push(oracle_id);
					}
					oracle_id
				})
				.collect::<Vec<_>>();

			// Translate flushes for the compiled constraint system.
			for Flush {
				column_indices,
				channel_id,
				direction,
			} in flushes
			{
				let flush_oracles = column_indices
					.iter()
					.map(|&column_index| table_oracle_ids[column_index])
					.collect::<Vec<_>>();
				compiled_flushes.push(CompiledFlush {
					oracles: flush_oracles,
					channel_id: *channel_id,
					direction: *direction,
					count,
				});
			}

			// Translate zero constraints for the compiled constraint system.
			let compiled_constraints = zero_constraints
				.iter()
				.map(|zero_constraint| {
					let composition = zero_constraint.clone().remap_vars(&table_oracle_ids)?;
					Ok(Constraint {
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

	pub fn build_witness(&self, allocator: &Bump, instance: &Instance) -> TableWitnessIndex<F> {
		todo!()
	}
}

fn add_oracle_for_column<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	column_info: &ColumnInfo,
	n_vars: usize,
) -> OracleId {
	let ColumnInfo { col, name, .. } = column_info;
	let addition = oracles.add_named(Some(name.clone()));
	match col {
		Column::Committed { tower_level } => addition.add_committed(n_vars, *tower_level),
	}
}
