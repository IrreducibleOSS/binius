// Copyright 2025 Irreducible Inc.

pub use binius_core::constraint_system::channel::{
	Boundary, Flush as CompiledFlush, FlushDirection,
};
use binius_core::{
	constraint_system::{channel::ChannelId, ConstraintSystem as CompiledConstraintSystem},
	oracle::{
		Constraint, ConstraintPredicate, ConstraintSet, MultilinearOracleSet, OracleId,
		ProjectionVariant,
	},
	transparent::step_down::StepDown,
};
use binius_field::{underlier::UnderlierType, TowerField};
use binius_math::LinearNormalForm;
use binius_utils::checked_arithmetics::log2_strict_usize;
use bumpalo::Bump;

use super::{
	channel::{Channel, Flush},
	column::{ColumnDef, ColumnInfo},
	error::Error,
	statement::Statement,
	table::TablePartition,
	types::B128,
	witness::{TableWitnessIndex, WitnessIndex},
	Table, TableBuilder,
};
use crate::builder::expr::ArithExprNamedVars;

/// An M3 constraint system, independent of the table sizes.
#[derive(Debug, Default)]
pub struct ConstraintSystem<F: TowerField = B128> {
	pub tables: Vec<Table<F>>,
	pub channels: Vec<Channel>,
}

impl<F: TowerField> std::fmt::Display for ConstraintSystem<F> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "ConstraintSystem {{")?;

		for channel in self.channels.iter() {
			writeln!(f, "    CHANNEL {}", channel.name)?;
		}

		let mut oracle_id = 0;

		for table in self.tables.iter() {
			writeln!(f, "    TABLE {} {{", table.name)?;

			for partition in table.partitions.values() {
				for flush in partition.flushes.iter() {
					let channel = self.channels[flush.channel_id].name.clone();
					let columns = flush
						.column_indices
						.iter()
						.map(|i| table.columns[partition.columns[*i]].name.clone())
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

				let names = partition
					.columns
					.iter()
					.map(|&index| table.columns[index].name.clone())
					.collect::<Vec<_>>();

				for constraint in partition.zero_constraints.iter() {
					let name = constraint.name.clone();
					let expr = ArithExprNamedVars(&constraint.expr, &names);
					writeln!(f, "        ZERO {name}: {expr}")?;
				}
			}

			for col in table.columns.iter() {
				if matches!(col.col, ColumnDef::Constant { .. }) {
					oracle_id += 1;
				}
			}

			for col in table.columns.iter() {
				let name = col.name.clone();
				let log_values_per_row = col.shape.log_values_per_row;
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
				let type_str = if log_values_per_row > 0 {
					let values_per_row = 1 << log_values_per_row;
					format!("{field}x{values_per_row}")
				} else {
					field.to_string()
				};
				writeln!(f, "        {oracle_id:04} {type_str} {name}")?;
				oracle_id += 1;
			}

			// step_down selectors for the table
			for log_values_per_row in table.partitions.keys() {
				let values_per_row = 1 << log_values_per_row;
				let selector_type_str = if values_per_row > 1 {
					format!("B1x{}", values_per_row)
				} else {
					"B1".to_string()
				};
				writeln!(f, "        {oracle_id:04} {selector_type_str} (ROW_SELECTOR)")?;
				oracle_id += 1;
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

	pub fn add_table(&mut self, name: impl ToString) -> TableBuilder<'_, F> {
		let id = self.tables.len();
		self.tables.push(Table::new(id, name.to_string()));
		TableBuilder::new(self.tables.last_mut().expect("table was just pushed"))
	}

	pub fn add_channel(&mut self, name: impl ToString) -> ChannelId {
		let id = self.channels.len();
		self.channels.push(Channel {
			name: name.to_string(),
		});
		id
	}

	/// Creates and allocates the witness index for a statement.
	///
	/// The statement includes information about the tables sizes, which this requires in order to
	/// allocate the column data correctly. The created witness index needs to be populated before
	/// proving.
	pub fn build_witness<'cs, 'alloc, U: UnderlierType>(
		&'cs self,
		allocator: &'alloc Bump,
		statement: &Statement,
	) -> Result<WitnessIndex<'cs, 'alloc, U, F>, Error> {
		Ok(WitnessIndex {
			tables: self
				.tables
				.iter()
				.zip(&statement.table_sizes)
				.map(|(table, &table_size)| {
					let witness = if table_size > 0 {
						Some(TableWitnessIndex::new(allocator, table, table_size))
					} else {
						None
					};
					witness.transpose()
				})
				.collect::<Result<_, _>>()?,
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
			if count == 0 {
				continue;
			}
			let mut oracle_lookup = Vec::new();

			let mut transparent_single = vec![None; table.columns.len()];
			for (table_index, info) in table.columns.iter().enumerate() {
				if let ColumnDef::Constant { poly } = &info.col {
					let oracle_id = oracles
						.add_named(format!("{}_single", info.name))
						.transparent(poly.clone())?;
					transparent_single[table_index] = Some(oracle_id);
				}
			}

			// Add multilinear oracles for all table columns.
			let log_capacity = table.log_capacity(count);
			for column_info in table.columns.iter() {
				let n_vars = log_capacity + column_info.shape.log_values_per_row;
				let oracle_id = add_oracle_for_column(
					&mut oracles,
					&oracle_lookup,
					&transparent_single,
					column_info,
					n_vars,
				)?;
				oracle_lookup.push(oracle_id);
				if column_info.is_nonzero {
					non_zero_oracle_ids.push(oracle_id);
				}
			}

			for partition in table.partitions.values() {
				let TablePartition {
					columns,
					flushes,
					zero_constraints,
					values_per_row,
					..
				} = partition;

				let n_vars = log_capacity + log2_strict_usize(*values_per_row);

				let partition_oracle_ids = columns
					.iter()
					.map(|&index| oracle_lookup[index])
					.collect::<Vec<_>>();

				// StepDown witness data is populated in WitnessIndex::into_multilinear_extension_index
				let step_down =
					oracles.add_transparent(StepDown::new(n_vars, count * values_per_row)?)?;

				// Translate flushes for the compiled constraint system.
				for Flush {
					column_indices,
					channel_id,
					direction,
					multiplicity,
					selector,
				} in flushes
				{
					let flush_oracles = column_indices
						.iter()
						.map(|&column_index| oracle_lookup[column_index])
						.collect::<Vec<_>>();
					compiled_flushes.push(CompiledFlush {
						oracles: flush_oracles,
						channel_id: *channel_id,
						direction: *direction,
						selector: selector.unwrap_or(step_down),
						multiplicity: *multiplicity as u64,
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
			max_channel_id: self.channels.len().saturating_sub(1),
			exponents: Vec::new(),
		})
	}
}

/// Add a table column to the multilinear oracle set with a specified number of variables.
///
/// ## Arguments
///
/// * `oracles` - The set of multilinear oracles to add to.
/// * `oracle_lookup` - mapping of column indices in the table to oracle IDs in the oracle set
/// * `column_info` - information about the column to be added
/// * `n_vars` - number of variables of the multilinear oracle
fn add_oracle_for_column<F: TowerField>(
	oracles: &mut MultilinearOracleSet<F>,
	oracle_lookup: &[OracleId],
	transparent_single: &[Option<OracleId>],
	column_info: &ColumnInfo<F>,
	n_vars: usize,
) -> Result<OracleId, Error> {
	let ColumnInfo {
		id,
		col,
		name,
		shape,
		..
	} = column_info;
	let addition = oracles.add_named(name);
	let oracle_id = match col {
		ColumnDef::Committed { tower_level } => addition.committed(n_vars, *tower_level),
		ColumnDef::Selected {
			col,
			index,
			index_bits,
		} => {
			let index_values = (0..*index_bits)
				.map(|i| {
					if (index >> i) & 1 == 0 {
						F::ZERO
					} else {
						F::ONE
					}
				})
				.collect();
			addition.projected(
				oracle_lookup[col.table_index],
				index_values,
				ProjectionVariant::FirstVars,
			)?
		}
		ColumnDef::Shifted {
			col,
			offset,
			log_block_size,
			variant,
		} => {
			// TODO: debug assert column at col.table_index has the same values_per_row as col.id
			addition.shifted(oracle_lookup[col.table_index], *offset, *log_block_size, *variant)?
		}
		ColumnDef::Packed { col, log_degree } => {
			// TODO: debug assert column at col.table_index has the same values_per_row as col.id
			addition.packed(oracle_lookup[col.table_index], *log_degree)?
		}
		ColumnDef::Computed { cols, expr } => {
			if let Ok(LinearNormalForm {
				constant: offset,
				var_coeffs,
			}) = expr.linear_normal_form()
			{
				let col_scalars = cols
					.iter()
					.zip(var_coeffs)
					.map(|(&col_index, coeff)| (oracle_lookup[col_index], coeff))
					.collect::<Vec<_>>();
				addition.linear_combination_with_offset(n_vars, offset, col_scalars)?
			} else {
				let inner_oracles = cols
					.iter()
					.map(|&col_index| oracle_lookup[col_index])
					.collect::<Vec<_>>();
				addition.composite_mle(n_vars, inner_oracles, expr.clone())?
			}
		}
		ColumnDef::Constant { .. } => addition.repeating(
			transparent_single[id.table_index].unwrap(),
			n_vars - shape.log_values_per_row,
		)?,
	};
	Ok(oracle_id)
}
