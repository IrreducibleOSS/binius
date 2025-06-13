// Copyright 2025 Irreducible Inc.

use std::{cell, collections::BTreeMap, ops::Index};

use binius_compute::alloc::HostBumpAllocator;
pub use binius_core::constraint_system::channel::{
	Boundary, Flush as CompiledFlush, FlushDirection,
};
use binius_core::{
	constraint_system::{
		ConstraintSystem as CompiledConstraintSystem,
		channel::{ChannelId, OracleOrConst},
		exp::Exp,
	},
	oracle::{
		Constraint, ConstraintPredicate, ConstraintSet, OracleId, SymbolicMultilinearOracleSet,
	},
};
use binius_field::{PackedField, TowerField};
use binius_math::{ArithCircuit, LinearNormalForm};
use binius_utils::checked_arithmetics::log2_strict_usize;

use super::{
	ColumnId, Table, TableBuilder, TableId, ZeroConstraint,
	channel::{Channel, Flush},
	column::{ColumnDef, ColumnInfo},
	error::Error,
	statement::Statement,
	table::TablePartition,
	types::B128,
	witness::WitnessIndex,
};
use crate::builder::expr::ArithExprNamedVars;

/// An M3 constraint system, independent of the table sizes.
#[derive(Debug, Default)]
pub struct ConstraintSystem<F: TowerField = B128> {
	pub tables: Vec<Table<F>>,
	pub channels: Vec<Channel>,

	// This is assigned as part of `ConstraintSystem::compile`.
	oracle_lookup: cell::RefCell<Option<OracleLookup>>,
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
						.columns
						.iter()
						.map(|i| table[*i].name.clone())
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
					.map(|&index| table[index].name.clone())
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
	#[deprecated]
	pub fn build_witness<'cs, 'alloc, P: PackedField<Scalar = F>>(
		&'cs self,
		allocator: &'alloc HostBumpAllocator<'alloc, P>,
	) -> WitnessIndex<'cs, 'alloc, P> {
		WitnessIndex::new(self, allocator)
	}

	/// Returns the oracle lookup for this constraint system.
	///
	/// Note that this function returns the struct as of the last call to
	/// [`ConstraintSystem::lookup`].
	#[track_caller]
	pub(crate) fn oracle_lookup<'a>(&'a self) -> cell::Ref<'a, OracleLookup> {
		const MESSAGE: &str = "oracle_lookup was requested but constraint system was not compiled";
		cell::Ref::map(self.oracle_lookup.borrow(), |o| o.as_ref().expect(MESSAGE))
	}

	/// Compiles a [`CompiledConstraintSystem`] for a particular statement.
	///
	/// The most important transformation that takes place in this step is creating multilinear
	/// oracles for all columns. The main difference between column definitions and oracle
	/// definitions is that multilinear oracle definitions have a number of variables, whereas the
	/// column definitions contained in a [`ConstraintSystem`] do not have size information.
	pub fn compile(&self, statement: &Statement<F>) -> Result<CompiledConstraintSystem<F>, Error> {
		// TODO: This is not used anymore and should be removed from `compile` in the follow up PRs.
		let _ = statement;

		let mut oracles = SymbolicMultilinearOracleSet::new();
		let mut table_constraints = Vec::new();
		let mut compiled_flushes = Vec::new();
		let mut non_zero_oracle_ids = Vec::new();
		let mut exponents = Vec::new();
		let mut table_size_specs = Vec::new();

		let mut oracle_lookup = OracleLookup::new();

		for table in &self.tables {
			table_size_specs.push(table.size_spec());

			// Add multilinear oracles for all table columns.
			add_oracles_for_columns(
				&mut oracle_lookup,
				&mut oracles,
				table,
				&mut non_zero_oracle_ids,
			)?;

			for partition in table.partitions.values() {
				let TablePartition {
					columns,
					flushes,
					zero_constraints,
					values_per_row,
					..
				} = partition;

				let partition_oracle_ids = columns
					.iter()
					.map(|&index| oracle_lookup[index])
					.collect::<Vec<_>>();

				// Add Exponents with the same pack factor for the compiled constraint system.
				columns.iter().for_each(|index| {
					let col = &table[*index];
					let col_info = &table[*index].col;
					match col_info {
						ColumnDef::StaticExp {
							bit_cols,
							base,
							base_tower_level,
						} => {
							let bits_ids = bit_cols
								.iter()
								.map(|&column_id| oracle_lookup[column_id])
								.collect();
							exponents.push(Exp {
								base: OracleOrConst::Const {
									base: *base,
									tower_level: *base_tower_level,
								},
								bits_ids,
								exp_result_id: oracle_lookup[col.id],
							});
						}
						ColumnDef::DynamicExp { bit_cols, base, .. } => {
							let bits_ids = bit_cols
								.iter()
								.map(|&col_idx| oracle_lookup[col_idx])
								.collect();
							exponents.push(Exp {
								base: OracleOrConst::Oracle(oracle_lookup[*base]),
								bits_ids,
								exp_result_id: oracle_lookup[col.id],
							})
						}
						_ => (),
					}
				});

				// Translate flushes for the compiled constraint system.
				for Flush {
					columns: flush_columns,
					channel_id,
					direction,
					multiplicity,
					selectors,
				} in flushes
				{
					let flush_oracles = flush_columns
						.iter()
						.map(|&column_id| OracleOrConst::Oracle(oracle_lookup[column_id]))
						.collect::<Vec<_>>();
					let selectors = selectors
						.iter()
						.map(|column_idx| oracle_lookup[*column_idx])
						.collect::<Vec<_>>();

					compiled_flushes.push(CompiledFlush {
						table_id: table.id(),
						log_values_per_row: log2_strict_usize(*values_per_row),
						oracles: flush_oracles,
						channel_id: *channel_id,
						direction: *direction,
						selectors,
						multiplicity: *multiplicity as u64,
					});
				}

				if !zero_constraints.is_empty() {
					let constraint_set = translate_constraint_set(
						table.id(),
						log2_strict_usize(*values_per_row),
						zero_constraints,
						partition_oracle_ids,
					);
					table_constraints.push(constraint_set);
				}
			}
		}

		*self.oracle_lookup.borrow_mut() = Some(oracle_lookup);

		Ok(CompiledConstraintSystem {
			oracles,
			table_constraints,
			flushes: compiled_flushes,
			non_zero_oracle_ids,
			channel_count: self.channels.len(),
			exponents,
			table_size_specs,
		})
	}
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum OracleMapping {
	Regular(OracleId),
	/// This is used for constant columns.
	///
	/// A constant columns are backed by a transparent oracle. That oracle is a single row and
	/// is not repeating which is not really expected. So in order to reduce the factor of surprise
	/// for the user, the original transparent is wrapped into a repeating virtual oracle.
	TransparentCompound {
		original: OracleId,
		repeating: OracleId,
	},
}

/// This structure holds metadata about every oracle ID in a constraint system.
///
/// This structure maintains mapping between the [`OracleId`] and the related [`ColumnId`].
#[derive(Debug, Default)]
pub(crate) struct OracleLookup {
	column_to_oracle: BTreeMap<ColumnId, OracleMapping>,
}

impl OracleLookup {
	/// Creates a new empty Oracle Registry.
	pub(crate) fn new() -> Self {
		Self {
			column_to_oracle: BTreeMap::new(),
		}
	}

	/// Looks up the [`OracleMapping`] for a given column ID.
	///
	/// # Preconditions
	///
	/// The column ID must exist in the registry, otherwise this function will panic.
	pub fn lookup(&self, column_id: ColumnId) -> &OracleMapping {
		&self.column_to_oracle[&column_id]
	}

	/// Adds a mapping from a column ID to an oracle mapping.
	///
	/// # Preconditions
	///
	/// The column ID must not already be registered in the registry, otherwise this function
	/// will panic.
	fn register_regular(&mut self, column_id: ColumnId, oracle_id: OracleId) {
		let prev = self
			.column_to_oracle
			.insert(column_id, OracleMapping::Regular(oracle_id));
		assert!(prev.is_none());
	}

	/// Registers a transparent oracle mapping for a column.
	///
	/// This creates a compound mapping from a column to both an original and a repeating oracle.
	/// This is specifically used for constant columns that need repeating behavior.
	///
	/// # Preconditions
	///
	/// The column ID must not already be registered in the registry, otherwise this function
	/// will panic.
	fn register_transparent(
		&mut self,
		column_id: ColumnId,
		original: OracleId,
		repeating: OracleId,
	) {
		let prev = self.column_to_oracle.insert(
			column_id,
			OracleMapping::TransparentCompound {
				original,
				repeating,
			},
		);
		assert!(prev.is_none());
	}
}

/// Indexing for [`OracleLookup`]. For transparents this returns the repeating column.
impl Index<ColumnId> for OracleLookup {
	type Output = OracleId;

	fn index(&self, id: ColumnId) -> &Self::Output {
		match &self.column_to_oracle[&id] {
			OracleMapping::Regular(oracle_id) => oracle_id,
			OracleMapping::TransparentCompound { repeating, .. } => repeating,
		}
	}
}

/// Add all columns within the given table into the given `oracle_lookup`. Also, fills out
/// the `non_zero_oracle_ids`.
fn add_oracles_for_columns<F: TowerField>(
	oracle_lookup: &mut OracleLookup,
	oracle_set: &mut SymbolicMultilinearOracleSet<F>,
	table: &Table<F>,
	non_zero_oracle_ids: &mut Vec<OracleId>,
) -> Result<(), Error> {
	for column_info in table.columns.iter() {
		add_oracle_for_column(oracle_set, oracle_lookup, column_info, table.id())?;
		if column_info.is_nonzero {
			non_zero_oracle_ids.push(oracle_lookup[column_info.id]);
		}
	}
	Ok(())
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
	oracles: &mut SymbolicMultilinearOracleSet<F>,
	oracle_lookup: &mut OracleLookup,
	column_info: &ColumnInfo<F>,
	table_id: TableId,
) -> Result<(), Error> {
	let ColumnInfo {
		id: column_id,
		col,
		name,
		shape,
		..
	} = column_info;
	match col {
		ColumnDef::Committed { tower_level } => {
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.committed(*tower_level);
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
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
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.projected(oracle_lookup[*col], index_values, 0)?;
			oracle_lookup.register_regular(*column_id, oracle_id);
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
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.projected(oracle_lookup[*col], query_values, *start_index)?;
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
		ColumnDef::ZeroPadded {
			col,
			n_pad_vars,
			start_index,
			nonzero_index,
		} => {
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.zero_padded(oracle_lookup[*col], *n_pad_vars, *nonzero_index, *start_index)?;
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
		ColumnDef::Shifted {
			col,
			offset,
			log_block_size,
			variant,
		} => {
			// TODO: debug assert column at col.table_index has the same values_per_row as col.id
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.shifted(oracle_lookup[*col], *offset, *log_block_size, *variant)?;
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
		ColumnDef::Packed { col, log_degree } => {
			// TODO: debug assert column at col.table_index has the same values_per_row as col.id
			let source = oracle_lookup[*col];
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.packed(source, *log_degree)?;
			oracle_lookup.register_regular(*column_id, oracle_id);
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
					.map(|(&col_id, coeff)| (oracle_lookup[col_id], coeff))
					.collect::<Vec<_>>();
				let oracle_id = oracles
					.add_oracle(table_id, shape.log_values_per_row, name)
					.linear_combination_with_offset(offset, col_scalars)?;
				oracle_lookup.register_regular(*column_id, oracle_id);
			} else {
				let inner_oracles = cols
					.iter()
					.map(|&col_index| oracle_lookup[col_index])
					.collect::<Vec<_>>();
				let oracle_id = oracles
					.add_oracle(table_id, shape.log_values_per_row, name)
					.composite_mle(inner_oracles, expr.clone())?;
				oracle_lookup.register_regular(*column_id, oracle_id);
			};
		}
		ColumnDef::Constant { poly, .. } => {
			let oracle_id_original = oracles
				.add_oracle(table_id, shape.log_values_per_row, format!("{name}_single"))
				.transparent(poly.clone())?;
			let oracle_id_repeating = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.repeating(oracle_id_original)?;
			oracle_lookup.register_transparent(*column_id, oracle_id_original, oracle_id_repeating);
		}
		ColumnDef::StructuredDynSize(structured) => {
			let expr = structured.expr()?;
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.structured(ArithCircuit::from(&expr))?;
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
		ColumnDef::StructuredFixedSize { expr } => {
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.transparent(expr.clone())?;
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
		ColumnDef::StaticExp {
			base_tower_level, ..
		} => {
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.committed(*base_tower_level);
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
		ColumnDef::DynamicExp {
			base_tower_level, ..
		} => {
			let oracle_id = oracles
				.add_oracle(table_id, shape.log_values_per_row, name)
				.committed(*base_tower_level);
			oracle_lookup.register_regular(*column_id, oracle_id);
		}
	};
	Ok(())
}

/// Translates a set of zero constraints from a particular table partition into a constraint set.
///
/// The resulting constraint set will only contain oracles that were actually referenced from any
/// of the constraint expressions.
fn translate_constraint_set<F: TowerField>(
	table_id: TableId,
	log_values_per_row: usize,
	zero_constraints: &[ZeroConstraint<F>],
	partition_oracle_ids: Vec<OracleId>,
) -> ConstraintSet<F> {
	// We need to figure out which oracle ids from the entire set of the partition oracles is
	// actually referenced in every zero constraint expressions.
	let mut oracle_appears_in_expr = vec![false; partition_oracle_ids.len()];
	let mut n_used_oracles = 0usize;
	for zero_constraint in zero_constraints {
		let vars_usage = zero_constraint.expr.vars_usage();
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
		table_id,
		log_values_per_row,
		oracle_ids: dense_oracle_ids,
		constraints: compiled_constraints,
	}
}
