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
	transparent::step_down::StepDown,
};
use binius_field::{underlier::UnderlierType, Field, TowerField};
use binius_math::{ArithExpr, LinearNormalForm};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bumpalo::Bump;

use super::error::Error;
use crate::{types::B128, witness::WitnessIndex};

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
#[derive(Debug)]
pub enum Column<F: TowerField = B128> {
	Committed { tower_level: usize },
	LinearCombination(LinearNormalForm<F>),
}

#[derive(Debug)]
pub struct ColumnInfo<F: TowerField = B128> {
	pub col: Column<F>,
	pub name: String,
	/// This represents how many
	pub pack_factor: usize,
	pub is_nonzero: bool,
}

#[derive(Debug)]
pub struct ZeroConstraint<F: Field> {
	pub name: String,
	pub expr: ArithExpr<F>,
}

#[derive(Debug)]
pub struct Table<F: TowerField = B128> {
	pub id: TableId,
	pub name: String,
	pub column_info: Vec<ColumnInfo<F>>,
	pub flushes: Vec<Flush>,
	pub zero_constraints: Vec<ZeroConstraint<F>>,
	/// This indicates whether a table is fixed for constraint system or part of the dynamic trace.
	///
	/// Fixed tables are either entirely transparent or committed during a preprocessing step that
	/// occurs before any statements are proven.
	pub is_fixed: bool,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct Channel {
	pub name: String,
}

pub struct ConstraintSystem<F: TowerField = B128> {
	pub tables: Vec<Table<F>>,
	/// All valid channel IDs are strictly less than this bound.
	pub channel_id_bound: ChannelId,
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

	pub fn build_witness<U: UnderlierType>(
		&self,
		allocator: &Bump,
		instance: &Instance,
	) -> Result<WitnessIndex<U>, Error> {
		todo!()
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
