// Copyright 2025 Irreducible Inc.

pub use binius_core::constraint_system::channel::{
	Boundary, Flush as CompiledFlush, FlushDirection,
};
use binius_core::{
	constraint_system::{
		channel::{ChannelId, OracleOrConst},
		exp::Exp,
		ConstraintSystem as CompiledConstraintSystem,
	},
	oracle::{Constraint, ConstraintPredicate, ConstraintSet, MultilinearOracleSet, OracleId},
	transparent::step_down::StepDown,
};
use binius_field::{PackedField, TowerField};
use binius_math::{ArithCircuit, LinearNormalForm};
use binius_utils::checked_arithmetics::log2_strict_usize;
use bumpalo::Bump;
use itertools::chain;

use super::{
	channel::{Channel, Flush},
	column::{ColumnDef, ColumnInfo},
	error::Error,
	statement::Statement,
	table::TablePartition,
	types::B128,
	witness::WitnessIndex,
	Table, TableBuilder, ZeroConstraint,
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
						.map(|i| table.columns[*i].name.clone())
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
					format!("B1x{values_per_row}")
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

	/// Creates and allocates the witness index.
	///
	/// **Deprecated**: This is a thin wrapper over [`WitnessIndex::new`] now, which is preferred.
	pub fn build_witness<'cs, 'alloc, P: PackedField<Scalar = F>>(
		&'cs self,
		allocator: &'alloc Bump,
	) -> WitnessIndex<'cs, 'alloc, P> {
		WitnessIndex::new(self, allocator)
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

		let mut oracles = MultilinearOracleSet::new();
		let mut table_constraints = Vec::new();
		let mut compiled_flushes = Vec::new();
		let mut non_zero_oracle_ids = Vec::new();
		let mut exponents = Vec::new();

		for (table, &count) in std::iter::zip(&self.tables, &statement.table_sizes) {
			if count == 0 {
				continue;
			}
			if table.is_power_of_two_sized() {
				if !count.is_power_of_two() {
					return Err(Error::TableSizePowerOfTwoRequired {
						table_id: table.id,
						size: count,
					});
				}
				if count != 1 << table.log_capacity(count) {
					panic!(
						"Tables with required power-of-two size currently cannot have capacity \
						exceeding their count. This is because the flushes do not have automatic \
						selectors applied, and so the table would flush invalid events"
					);
				}
			}

			let mut oracle_lookup = Vec::new();

			let mut transparent_single = vec![None; table.columns.len()];
			for (table_index, info) in table.columns.iter().enumerate() {
				if let ColumnDef::Constant { poly, .. } = &info.col {
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

				// Add Exponents with the same pack factor for the compiled constraint system.
				columns.iter().for_each(|&index| {
					let col_info = &table.columns[index].col;
					if let ColumnDef::StaticExp {
						bit_cols,
						base,
						base_tower_level,
					} = col_info
					{
						let bits_ids = bit_cols
							.iter()
							.map(|&col_idx| oracle_lookup[col_idx])
							.collect();
						exponents.push(Exp {
							base: OracleOrConst::Const {
								base: *base,
								tower_level: *base_tower_level,
							},
							bits_ids,
							exp_result_id: oracle_lookup[index],
						});
					}
					if let ColumnDef::DynamicExp { bit_cols, base, .. } = col_info {
						let bits_ids = bit_cols
							.iter()
							.map(|&col_idx| oracle_lookup[col_idx])
							.collect();
						exponents.push(Exp {
							base: OracleOrConst::Oracle(oracle_lookup[*base]),
							bits_ids,
							exp_result_id: oracle_lookup[index],
						})
					}
				});

				// StepDown witness data is populated in WitnessIndex::into_multilinear_extension_index
				let step_down = (!table.is_power_of_two_sized())
					.then(|| {
						let step_down_poly = StepDown::new(n_vars, count * values_per_row)?;
						oracles.add_transparent(step_down_poly)
					})
					.transpose()?;

				// Translate flushes for the compiled constraint system.
				for Flush {
					column_indices,
					channel_id,
					direction,
					multiplicity,
					selectors,
				} in flushes
				{
					let flush_oracles = column_indices
						.iter()
						.map(|&column_index| OracleOrConst::Oracle(oracle_lookup[column_index]))
						.collect::<Vec<_>>();
					let selectors = chain!(
						selectors
							.iter()
							.map(|column_idx| oracle_lookup[*column_idx]),
						step_down
					)
					.collect::<Vec<_>>();

					compiled_flushes.push(CompiledFlush {
						oracles: flush_oracles,
						channel_id: *channel_id,
						direction: *direction,
						selectors,
						multiplicity: *multiplicity as u64,
					});
				}

				if !zero_constraints.is_empty() {
					let constraint_set =
						translate_constraint_set(n_vars, zero_constraints, partition_oracle_ids);
					table_constraints.push(constraint_set);
				}
			}
		}

		Ok(CompiledConstraintSystem {
			oracles,
			table_constraints,
			flushes: compiled_flushes,
			non_zero_oracle_ids,
			max_channel_id: self.channels.len().saturating_sub(1),
			exponents,
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
			addition.projected(oracle_lookup[col.table_index], index_values, 0)?
		}
		ColumnDef::Projected {
			col,
			start_index,
			query_size,
			query_bits,
		} => {
			let query_values = (0..*query_size)
				.map(|i| -> F {
					if (query_bits >> i) & 1 == 0 {
						F::ZERO
					} else {
						F::ONE
					}
				})
				.collect();
			addition.projected(oracle_lookup[col.table_index], query_values, *start_index)?
		}
		ColumnDef::ZeroPadded {
			col,
			n_pad_vars,
			start_index,
			nonzero_index,
		} => addition.zero_padded(
			oracle_lookup[col.table_index],
			*n_pad_vars,
			*nonzero_index,
			*start_index,
		)?,
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
		ColumnDef::StructuredDynSize(structured) => {
			let expr = structured.expr(n_vars)?;
			addition.transparent(ArithCircuit::from(&expr))?
		}
		ColumnDef::StructuredFixedSize { expr } => addition.transparent(expr.clone())?,
		ColumnDef::StaticExp {
			base_tower_level, ..
		} => addition.committed(n_vars, *base_tower_level),
		ColumnDef::DynamicExp {
			base_tower_level, ..
		} => addition.committed(n_vars, *base_tower_level),
	};
	Ok(oracle_id)
}

/// Translates a set of zero constraints from a particular table partition into a constraint set.
///
/// The resulting constraint set will only contain oracles that were actually referenced from any
/// of the constraint expressions.
fn translate_constraint_set<F: TowerField>(
	n_vars: usize,
	zero_constraints: &[ZeroConstraint<F>],
	partition_oracle_ids: Vec<usize>,
) -> ConstraintSet<F> {
	// We need to figure out which oracle ids from the entire set of the partition oracles is
	// actually referenced in every zero constraint expressions.
	let mut oracle_appears_in_expr = vec![false; partition_oracle_ids.len()];
	let mut n_used_oracles = 0usize;
	for zero_contraint in zero_constraints {
		let vars_usage = zero_contraint.expr.vars_usage();
		for (oracle_index, used) in vars_usage.iter().enumerate() {
			if *used && !oracle_appears_in_expr[oracle_index] {
				oracle_appears_in_expr[oracle_index] = true;
				n_used_oracles += 1;
			}
		}
	}

	// Now that we've got the set of oracle ids that appear in the expr we are going to create
	// a new list of oracle ids each of which is used. Along the way we create a new mapping table
	// that maps the original oracle index to the new index in the dense list.
	const INVALID_SENTINEL: usize = usize::MAX;
	let mut remap_indices_table = vec![INVALID_SENTINEL; partition_oracle_ids.len()];
	let mut dense_oracle_ids = Vec::with_capacity(n_used_oracles);
	for (i, &used) in oracle_appears_in_expr.iter().enumerate() {
		if !used {
			continue;
		}
		let dense_index = dense_oracle_ids.len();
		dense_oracle_ids.push(partition_oracle_ids[i]);
		remap_indices_table[i] = dense_index;
	}

	// Translate zero constraints for the compiled constraint system.
	let compiled_constraints = zero_constraints
		.iter()
		.map(|zero_constraint| {
			let expr = zero_constraint
				.expr
				.clone()
				.remap_vars(&remap_indices_table)
				.expect(
					"the expr must have the same length as partition_oracle_ids which is the\
				 same length of remap_indices_table",
				);
			Constraint {
				name: zero_constraint.name.clone(),
				composition: expr,
				predicate: ConstraintPredicate::Zero,
			}
		})
		.collect::<Vec<_>>();

	ConstraintSet {
		n_vars,
		oracle_ids: dense_oracle_ids,
		constraints: compiled_constraints,
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;

	use super::*;

	#[test]
	fn test_unsatisfied_po2_requirement() {
		let mut cs = ConstraintSystem::<B128>::new();
		let mut table_builder = cs.add_table("fibonacci");
		table_builder.require_power_of_two_size();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![15],
		};
		assert_matches!(cs.compile(&statement), Err(Error::TableSizePowerOfTwoRequired { .. }));
	}
}
