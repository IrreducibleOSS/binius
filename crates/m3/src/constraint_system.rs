// Copyright 2025 Irreducible Inc.

// Tables

// Lookups
// Columns <- Associated with tables

use std::marker::PhantomData;

// Statement
// - Channel boundaries
// - Table sizes
pub use binius_core::constraint_system::channel::{
	Boundary, Flush as CompiledFlush, FlushDirection,
};
use binius_core::{
	constraint_system::ConstraintSystem as CompiledConstraintSystem,
	oracle::{
		Constraint, ConstraintPredicate, ConstraintSet, MultilinearOracleSet, OracleId,
		ShiftVariant,
	},
};
use binius_field::{BinaryField128b, ExtensionField, Field, TowerField};
use binius_math::ArithExpr;
use binius_utils::checked_arithmetics::log2_ceil_usize;

use super::error::Error;
use crate::types::B128;

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

/// A type representing a column in a table.
///
/// The column has entries that are elements of `F`. In practice, the fields used will always be
/// from the canonical tower (B1, B8, B16, B32, B64, B128). The second constant represents how many
/// elements are packed vertically into a single logical row. For example, a column of type
/// `Col<B1, 5>` will have 2^5 = 32 elements of `B1` packed into a single row.
#[derive(Debug, Clone, Copy)]
pub struct Col<F: TowerField, const LOG_VALS_PER_ROW: usize = 0> {
	// REVIEW: Maybe this should have denormalized name for debugging.
	pub table_id: TableId,
	pub index: ColumnIndex,
	_marker: PhantomData<F>,
}

impl<F: TowerField, const V: usize> Col<F, V> {
	pub fn new(id: ColumnId) -> Self {
		Self {
			table_id: id.table_id,
			index: id.index,
			_marker: PhantomData,
		}
	}

	pub fn shape(&self) -> ColumnShape {
		ColumnShape {
			tower_height: F::TOWER_LEVEL,
			pack_factor: V,
		}
	}
}

pub fn upcast_col<F, FSub, const V: usize>(col: Col<FSub, V>) -> Col<F, V>
where
	FSub: TowerField,
	F: TowerField + ExtensionField<FSub>,
{
	Col {
		table_id: col.table_id,
		index: col.index,
		_marker: PhantomData,
	}
}

/// A type representing an arithmetic expression composed over some table columns.
///
/// If the expression degree is 1, then it is a linear expression.
pub struct Expr<F: TowerField, const LOG_VALS_PER_ROW: usize> {
	expr: ArithExpr<F>,
	cols: Vec<Col<F, LOG_VALS_PER_ROW>>,
}

// feature: TableBuilder needs namespacing

/*
pub enum Column {
	Committed { tower_level: usize },
}

pub struct ColumnInfo {
	col: Column,
	name: String,
	/// This represents how many
	pack_factor: usize,
	is_nonzero: bool,
}

/// A table in an M3 constraint system.
///
/// ## Invariants
///
/// * All expressions in `zero_constraints` have a number of variables less than or equal to the
///   number of table columns (the length of `column_info`).
/// * All flushes in `flushes` contain column indices less than the number of table columns (the
///   length of `column_info`).
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

impl<F: TowerField> Table<F> {
	pub fn new(id: TableId, name: impl ToString) -> Self {
		Self {
			id,
			name: name.to_string(),
			column_info: Vec::new(),
			flushes: Vec::new(),
			zero_constraints: Vec::new(),
			is_fixed: false,
		}
	}

	pub fn new_fixed(id: TableId, name: impl ToString) -> Self {
		Self {
			is_fixed: true,
			..Self::new(id, name)
		}
	}

	pub fn add_committed<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
	) -> Col<FSub, LOG_VALS_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let index = self.column_info.len();
		self.column_info.push(ColumnInfo {
			col: Column::Committed {
				tower_level: FSub::TOWER_LEVEL,
			},
			name: name.to_string(),
			pack_factor: LOG_VALS_PER_ROW,
			is_nonzero: false,
		});
		Col {
			table_id: self.id,
			index,
			_marker: PhantomData,
		}
	}

	pub fn add_shifted<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		original: Col<FSub, LOG_VALS_PER_ROW>,
		shift_bits: usize,
		shift: usize,
		shift_mode: ShiftVariant,
	) -> Col<FSub, LOG_VALS_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		todo!()
	}

	// Selected is a special form of projected with hypercube indices
	pub fn add_selected<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		original: Col<FSub, LOG_VALS_PER_ROW>,
		index: usize,
	) -> Col<FSub, 1>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		todo!()
	}

	pub fn add_linear_combination<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		cols: impl IntoIterator<Item = (Col<FSub, V>, FSub)>,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.add_linear_combination_with_offset(name, cols, FSub::ZERO)
	}

	pub fn add_linear_combination_with_offset<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		cols: impl IntoIterator<Item = Col<FSub, V>>,
		offset: FSub,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		todo!()
	}

	pub fn add_committed_multiple<FSub, const LOG_VALS_PER_ROW: usize, const N: usize>(
		&mut self,
		name: impl ToString,
		tower_level: usize,
	) -> [Col<FSub, LOG_VALS_PER_ROW>; N]
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		array::from_fn(|i| self.add_committed(format!("{}[{}]", name.to_string(), i)))
	}
}

pub struct Flush {
	pub column_indices: Vec<usize>,
	pub channel_id: ChannelId,
	pub direction: FlushDirection,
}

pub struct ConstraintSystem<F: TowerField = B128> {
	tables: Vec<Table<F>>,
	/// All valid channel IDs are strictly less than this bound.
	channel_id_bound: ChannelId,
}

pub struct Instance<F: TowerField = B128> {
	boundaries: Vec<Boundary<F>>,
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
	table_sizes: Vec<usize>,
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
*/
