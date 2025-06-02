// Copyright 2025 Irreducible Inc.

use std::fmt;

use binius_field::TowerField;
use binius_math::EvalCost;
use binius_utils::{checked_arithmetics::log2_strict_usize, sparse_index::SparseIndex};

use super::{Table, TablePartition};

struct Constraint {
	name: String,
	degree: usize,
	eval_cost: EvalCost,
}

#[derive(Default)]
struct PerPartition {
	constraints: Vec<Constraint>,
}

#[derive(Default)]
struct PerTowerLevel {
	/// Index from the log2 V (`values_per_row`) to the struct holding a set of constraints.
	per_v: SparseIndex<PerPartition>,
}

/// Table statistics.
pub struct TableStat {
	name: String,
	/// Index from the log2 tower level to the struct holding a set of partitions.
	per_tower_level: SparseIndex<PerTowerLevel>,
	bits_per_row_committed: usize,
	bits_per_row_virtual: usize,
}

impl TableStat {
	pub(super) fn new<F: TowerField>(table: &Table<F>) -> Self {
		let mut bits_per_row_committed = 0;
		let mut bits_per_row_virtual = 0;
		for column in &table.columns {
			let bits_per_column = 1 << column.shape.log_cell_size();
			if matches!(column.col, super::ColumnDef::Committed { .. }) {
				bits_per_row_committed += bits_per_column;
			} else {
				bits_per_row_virtual += bits_per_column;
			}
		}

		let mut me = Self {
			name: table.name.clone(),
			per_tower_level: SparseIndex::new(),
			bits_per_row_committed,
			bits_per_row_virtual,
		};

		for (_, partition) in table.partitions.iter() {
			let &TablePartition {
				values_per_row,
				zero_constraints,
				..
			} = &partition;
			let v_log2 = log2_strict_usize(*values_per_row);
			for zero_constraint in zero_constraints {
				let tower_level = zero_constraint.tower_level;
				let name = zero_constraint.name.clone();
				let degree = zero_constraint.expr.degree();
				let eval_cost = zero_constraint.expr.eval_cost();
				me.add_constraint(
					tower_level,
					v_log2,
					Constraint {
						name,
						degree,
						eval_cost,
					},
				);
			}
		}

		me
	}

	fn add_constraint(&mut self, tower_level: usize, v_log2: usize, c: Constraint) {
		let per_tower_level = self
			.per_tower_level
			.entry(tower_level)
			.or_insert_with(PerTowerLevel::default);
		let per_partition = per_tower_level
			.per_v
			.entry(v_log2)
			.or_insert_with(PerPartition::default);
		per_partition.constraints.push(c);
	}

	/// Returns the table name.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Returns the approximate cost of [`assert_zero`][zero_constraint] constraints.
	///
	/// The dominating factor of the proving cost of zero constraints is a product of:
	///
	/// - The degree of the constraint expression.
	/// - The number of multiplications in the expression.
	/// - The tower level over which the constraint expression is defined. Note that sub-byte towers
	///   are counted the same way as full 8-bit fields.
	/// - The `values_per_row` number.
	///
	/// This is an approximate number and there are potentially other factors that can influence
	/// the cost.
	///
	/// [zero_constraint]: super::TableBuilder::assert_zero
	pub fn assert_zero_cost_approx(&self) -> usize {
		let mut cost = 0;
		for (tower_level, per_tower_level) in self.per_tower_level.iter() {
			for (v_log2, per_partition) in per_tower_level.per_v.iter() {
				for constraint in &per_partition.constraints {
					let values_per_row = 1 << v_log2;
					// Scale the constraint cost by the number of bits in the constrained elements.
					// Treat the minimum number of bits as 8 because the B8 field is the smallest
					// that we can evaluate zerocheck constraints over, using the univariate skip
					// strategy.
					let bits = (1 << tower_level).max(8) * values_per_row;
					let degree = constraint.degree;
					let mult_cost_approx = constraint.eval_cost.mult_cost_approx();
					cost += bits * degree * mult_cost_approx;
				}
			}
		}
		cost
	}

	/// Returns the number of committed bits per row of this table. Committed bits are contributed
	/// by the [`add_committed`][add_committed] columns.
	///
	/// Committed bits increase the proof size and generally lead to faster verification time
	/// relative to comparable virtual bits.
	///
	/// [add_committed]: super::TableBuilder::add_committed
	pub fn bits_per_row_committed(&self) -> usize {
		self.bits_per_row_committed
	}

	/// Returns the number of virtual bits per row of this table. Virtual bits are pretty much any
	/// non-[`add_committed`][`add_committed`] columns. E.g. [`add_shifted`][`add_shifted`],
	/// [`add_computed`][`add_computed`], etc.
	///
	/// Virtual bits do not affect the proof size but generally lead to slower verification time
	/// relative to committed bits.
	///
	/// [`add_committed`]: super::TableBuilder::add_committed
	/// [`add_shifted`]: super::TableBuilder::add_shifted
	/// [`add_computed`]: super::TableBuilder::add_computed
	pub fn bits_per_row_virtual(&self) -> usize {
		self.bits_per_row_virtual
	}
}

impl fmt::Display for TableStat {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		writeln!(f, "table '{}':", self.name)?;
		writeln!(f, "* bits per row: {}", self.bits_per_row_committed + self.bits_per_row_virtual)?;
		writeln!(f, "  committed: {}", self.bits_per_row_committed)?;
		writeln!(f, "  virtual: {}", self.bits_per_row_virtual)?;
		writeln!(f, "* zero checks:")?;
		for (tower_level, per_tower_level) in self.per_tower_level.iter() {
			let bits = 1 << tower_level;
			writeln!(f, "  B{bits}:")?;
			for (v_log2, per_partition) in per_tower_level.per_v.iter() {
				let values_per_row = 1 << v_log2;
				writeln!(f, "    values_per_row={values_per_row}:")?;
				for (i, constraint) in per_partition.constraints.iter().enumerate() {
					let ordinal = i + 1;
					let Constraint {
						name,
						degree,
						eval_cost,
					} = constraint;
					let n_adds = eval_cost.n_adds;
					let n_muls = eval_cost.n_muls;
					let n_squares = eval_cost.n_squares;
					writeln!(
						f,
						"      {ordinal}. {name}: deg={degree},  #+={n_adds}, #×={n_muls}, #²={n_squares}"
					)?;
				}
			}
		}

		let cost = self.assert_zero_cost_approx();
		writeln!(f, "Total approximate assert_zero costs: {cost}")?;
		Ok(())
	}
}
