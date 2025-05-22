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
	total_flush_multiplicity: usize,
	effective_log_capacity: Option<usize>,
}

impl TableStat {
	pub(super) fn new<F: TowerField>(table: &Table<F>) -> Self {
		let mut bits_per_row_committed = 0;
		let mut bits_per_row_virtual = 0;
		for column in &table.columns {
			let bits_per_column = 1 << column.shape.log_cell_size();
			if matches!(column.col, super::ColumnDef::Committed { .. }) {
				bits_per_row_virtual += bits_per_column;
			} else {
				bits_per_row_committed += bits_per_column;
			}
		}

		let mut me = Self {
			name: table.name.clone(),
			per_tower_level: SparseIndex::new(),
			bits_per_row_committed,
			bits_per_row_virtual,
			total_flush_multiplicity: 0,
			effective_log_capacity: None,
		};

		let mut calculated_total_flush_multiplicity = 0;
		for (_, partition) in table.partitions.iter() {
			for flush in partition.flushes.iter() {
				calculated_total_flush_multiplicity += flush.multiplicity as usize;
			}
		}
		me.total_flush_multiplicity = calculated_total_flush_multiplicity;

		match table.size_spec() {
			super::table::TableSizeSpec::Fixed { log_size } => {
				me.effective_log_capacity = Some(log_size);
			}
			_ => {
				// Keep None for PowerOfTwo and Arbitrary
			}
		}

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

	/// Returns the total flush multiplicity of this table.
	pub fn total_flush_multiplicity(&self) -> usize {
		self.total_flush_multiplicity
	}

	/// Returns the effective log capacity of this table, if fixed.
	pub fn effective_log_capacity(&self) -> Option<usize> {
		self.effective_log_capacity
	}

	/// Returns the approximate cost of flush operations for this table.
	///
	/// This is calculated as `total_flush_multiplicity * table_size`.
	/// Returns `None` if the table size is not fixed.
	pub fn flush_cost_approx(&self) -> Option<usize> {
		self.effective_log_capacity
			.map(|log_cap| self.total_flush_multiplicity * (1 << log_cap))
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

		writeln!(f, "* Total flush multiplicity: {}", self.total_flush_multiplicity())?;
		match self.flush_cost_approx() {
			Some(cost) => {
				writeln!(f, "* Estimated total flush cost: {}", cost)?;
			}
			None => {
				writeln!(
					f,
					"* Estimated total flush cost: N/A (table size not fixed; multiply total flush multiplicity by 2^log_capacity to estimate)"
				)?;
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::builder::table::{Table, TableBuilder, TableSizeSpec};
	use crate::builder::channel::FlushOpts;
	use binius_core::constraint_system::channel::ChannelId;
	use binius_field::B128; // Assuming B128 is a suitable field type

	#[test]
	fn test_no_flushes_fixed_size() {
		let mut table = Table::<B128>::new(0, "test_no_flushes_fixed");
		let mut tb = TableBuilder::new(&mut table);
		tb.require_fixed_size(10);
		// No flushes added

		let stat = table.stat();

		assert_eq!(stat.total_flush_multiplicity(), 0);
		assert_eq!(stat.effective_log_capacity(), Some(10));
		assert_eq!(stat.flush_cost_approx(), Some(0));

		let display_output = format!("{}", stat);
		assert!(display_output.contains("Total flush multiplicity: 0"));
		assert!(display_output.contains("Estimated total flush cost: 0"));
	}

	#[test]
	fn test_no_flushes_power_of_two_size() {
		let mut table = Table::<B128>::new(0, "test_no_flushes_pot");
		let mut tb = TableBuilder::new(&mut table);
		tb.require_power_of_two_size();
		// No flushes added

		let stat = table.stat();

		assert_eq!(stat.total_flush_multiplicity(), 0);
		assert_eq!(stat.effective_log_capacity(), None);
		assert_eq!(stat.flush_cost_approx(), None);

		let display_output = format!("{}", stat);
		assert!(display_output.contains("Total flush multiplicity: 0"));
		assert!(display_output.contains("Estimated total flush cost: N/A (table size not fixed; multiply total flush multiplicity by 2^log_capacity to estimate)"));
	}

	#[test]
	fn test_with_flushes_fixed_size() {
		let mut table = Table::<B128>::new(0, "test_with_flushes_fixed");
		let mut tb = TableBuilder::new(&mut table);
		tb.require_fixed_size(8);

		let col1 = tb.add_committed::<B128, 1>("col1");
		let col2 = tb.add_committed::<B128, 1>("col2");
        let col3 = tb.add_committed::<B128, 1>("col3");


		// Flush 1: channel 0, one column, multiplicity = 2
		tb.push_with_opts(
			ChannelId::new(0),
			vec![col1.into()],
			FlushOpts { multiplicity: 2, ..Default::default() },
		);

		// Flush 2: channel 1, two columns, multiplicity = 1 (default)
        tb.push_with_opts(
            ChannelId::new(1),
            vec![col2.into(), col3.into()],
            Default::default(),
        );


		let stat = table.stat();
		let expected_total_flush_multiplicity = 2 + 1; // col1 (mult 2) + col2,col3 (mult 1)
		let expected_flush_cost_approx = expected_total_flush_multiplicity * (1 << 8); // 3 * 256 = 768

		assert_eq!(stat.total_flush_multiplicity(), expected_total_flush_multiplicity);
		assert_eq!(stat.effective_log_capacity(), Some(8));
		assert_eq!(stat.flush_cost_approx(), Some(expected_flush_cost_approx));

		let display_output = format!("{}", stat);
		assert!(display_output.contains(&format!("Total flush multiplicity: {}", expected_total_flush_multiplicity)));
		assert!(display_output.contains(&format!("Estimated total flush cost: {}", expected_flush_cost_approx)));
	}

	#[test]
	fn test_with_flushes_arbitrary_size() {
		let mut table = Table::<B128>::new(0, "test_with_flushes_arbitrary");
		let mut tb = TableBuilder::new(&mut table);
		// Default size spec is Arbitrary

		let col1 = tb.add_committed::<B128, 1>("col1");
		let col2 = tb.add_committed::<B128, 1>("col2");
        let col3 = tb.add_committed::<B128, 1>("col3");
        let col4 = tb.add_committed::<B128, 1>("col4");


		// Flush 1: channel 0, two columns, multiplicity = 3
        tb.push_with_opts(
            ChannelId::new(0),
            vec![col1.into(), col2.into()],
            FlushOpts { multiplicity: 3, ..Default::default() },
        );

		// Flush 2: channel 1, two columns, multiplicity = 1 (default)
        tb.push_with_opts(
            ChannelId::new(1),
            vec![col3.into(), col4.into()],
            Default::default(),
        );

		let stat = table.stat();
        let expected_total_flush_multiplicity = 3 + 1; // (col1,col2) mult 3 + (col3,col4) mult 1

		assert_eq!(stat.total_flush_multiplicity(), expected_total_flush_multiplicity);
		assert_eq!(stat.effective_log_capacity(), None);
		assert_eq!(stat.flush_cost_approx(), None);

		let display_output = format!("{}", stat);
		assert!(display_output.contains(&format!("Total flush multiplicity: {}", expected_total_flush_multiplicity)));
		assert!(display_output.contains("Estimated total flush cost: N/A (table size not fixed; multiply total flush multiplicity by 2^log_capacity to estimate)"));
	}
}
