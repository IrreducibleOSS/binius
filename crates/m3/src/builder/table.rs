// Copyright 2025 Irreducible Inc.

use std::sync::Arc;

use binius_core::{
	constraint_system::channel::{ChannelId, FlushDirection},
	oracle::ShiftVariant,
	transparent::MultilinearExtensionTransparent,
};
use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	packed::pack_slice,
	ExtensionField, TowerField,
};
use binius_math::LinearNormalForm;
use binius_utils::{
	checked_arithmetics::{checked_log_2, log2_strict_usize},
	sparse_index::SparseIndex,
};

use super::{
	channel::Flush,
	column::{upcast_col, Col, ColumnDef, ColumnInfo, ColumnShape},
	expr::{Expr, ZeroConstraint},
	types::B128,
	ColumnIndex,
};
use crate::builder::column::ColumnId;

pub type TableId = usize;

#[derive(Debug)]
pub struct TableBuilder<'a, F: TowerField = B128> {
	namespace: Option<String>,
	table: &'a mut Table<F>,
}

impl<'a, F: TowerField> TableBuilder<'a, F> {
	pub fn new(table: &'a mut Table<F>) -> Self {
		Self {
			namespace: None,
			table,
		}
	}

	pub fn with_namespace(&mut self, namespace: impl ToString) -> TableBuilder<'_, F> {
		TableBuilder {
			namespace: Some(self.namespaced_name(namespace)),
			table: self.table,
		}
	}

	pub fn id(&self) -> TableId {
		self.table.id()
	}

	pub fn add_committed<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Committed {
				tower_level: FSub::TOWER_LEVEL,
			},
		)
	}

	pub fn add_committed_multiple<FSub, const VALUES_PER_ROW: usize, const N: usize>(
		&mut self,
		name: impl ToString,
	) -> [Col<FSub, VALUES_PER_ROW>; N]
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		std::array::from_fn(|i| self.add_committed(format!("{}[{}]", name.to_string(), i)))
	}

	pub fn add_shifted<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, VALUES_PER_ROW>,
		log_block_size: usize,
		offset: usize,
		variant: ShiftVariant,
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(log_block_size <= log2_strict_usize(VALUES_PER_ROW));
		assert!(offset <= 1 << log_block_size);
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Shifted {
				col: col.id(),
				offset,
				log_block_size,
				variant,
			},
		)
	}

	pub fn add_linear_combination<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		expr: Expr<FSub, V>,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let LinearNormalForm {
			constant: offset,
			var_coeffs,
		} = expr
			.expr()
			.convert_field::<F>()
			.linear_normal_form()
			.expect("pre-condition: expression must be linear");

		let col_scalars = var_coeffs
			.into_iter()
			.enumerate()
			.filter_map(|(partition_index, coeff)| {
				if coeff != F::ZERO {
					let partition = &self.table.partitions[partition_id::<V>()];
					Some((partition.columns[partition_index], coeff))
				} else {
					None
				}
			})
			.collect();

		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::LinearCombination {
				offset,
				col_scalars,
			},
		)
	}

	pub fn add_packed<FSubSub, const VALUES_PER_ROW_SUB: usize, FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSubSub, VALUES_PER_ROW_SUB>,
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField + ExtensionField<FSubSub>,
		FSubSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(FSubSub::TOWER_LEVEL < FSub::TOWER_LEVEL);
		assert!(VALUES_PER_ROW_SUB > VALUES_PER_ROW);
		assert_eq!(
			FSub::TOWER_LEVEL + log2_strict_usize(VALUES_PER_ROW),
			FSubSub::TOWER_LEVEL + log2_strict_usize(VALUES_PER_ROW_SUB)
		);
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Packed {
				col: col.id(),
				log_degree: FSub::TOWER_LEVEL - FSubSub::TOWER_LEVEL,
			},
		)
	}

	pub fn add_computed<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		expr: Expr<FSub, V>,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let partition_indexes = expr
			.expr()
			.vars_usage()
			.iter()
			.enumerate()
			.filter(|(_, &used)| used)
			.map(|(i, _)| i)
			.collect::<Vec<_>>();
		let cols = partition_indexes
			.iter()
			.map(|&partition_index| {
				let partition = &self.table.partitions[partition_id::<V>()];
				partition.columns[partition_index]
			})
			.collect::<Vec<_>>();

		let mut var_remapping = vec![0; expr.expr().n_vars()];
		for (new_index, &old_index) in partition_indexes.iter().enumerate() {
			var_remapping[old_index] = new_index;
		}
		let remapped_expr = expr
			.expr()
			.convert_field()
			.remap_vars(&var_remapping)
			.expect("var_remapping should be large enought");

		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Computed {
				cols,
				expr: remapped_expr,
			},
		)
	}

	pub fn add_selected<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, VALUES_PER_ROW>,
		index: usize,
	) -> Col<FSub, 1>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert!(index < VALUES_PER_ROW);
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Selected {
				col: col.id(),
				index,
				index_bits: log2_strict_usize(VALUES_PER_ROW),
			},
		)
	}

	pub fn add_repeating_constants<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		constants: [FSub; VALUES_PER_ROW],
	) -> Col<FSub, VALUES_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
		OptimalUnderlier: PackScalar<FSub> + PackScalar<F>,
	{
		let namespaced_name = self.namespaced_name(name);
		let n_vars = log2_strict_usize(VALUES_PER_ROW);
		let packed_values: Vec<PackedType<OptimalUnderlier, FSub>> = pack_slice(&constants);
		let mle = MultilinearExtensionTransparent::<
			PackedType<OptimalUnderlier, FSub>,
			PackedType<OptimalUnderlier, F>,
			_,
		>::from_values_and_mu(packed_values, n_vars)
		.unwrap();
		self.table.new_column(
			namespaced_name,
			ColumnDef::Constant {
				poly: Arc::new(mle),
			},
		)
	}

	pub fn assert_zero<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		expr: Expr<FSub, VALUES_PER_ROW>,
	) where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table
			.partition_mut(VALUES_PER_ROW)
			.assert_zero(name, expr)
	}

	pub fn pull_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(1).pull_one(channel, col)
	}

	pub fn push_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(1).push_one(channel, col)
	}

	pub fn pull(&mut self, channel: ChannelId, cols: impl IntoIterator<Item = Col<F>>) {
		self.table.partition_mut(1).pull(channel, cols);
	}

	pub fn push(&mut self, channel: ChannelId, cols: impl IntoIterator<Item = Col<F>>) {
		self.table.partition_mut(1).push(channel, cols);
	}

	fn namespaced_name(&self, name: impl ToString) -> String {
		let name = name.to_string();
		match &self.namespace {
			Some(namespace) => format!("{namespace}::{name}"),
			None => name.to_string(),
		}
	}
}

/// A table in an M3 constraint system.
///
/// ## Invariants
///
/// * All expressions in `zero_constraints` have a number of variables less than or equal to the
///   number of table columns (the length of `column_info`).
/// * All flushes in `flushes` contain column indices less than the number of table columns (the
///   length of `column_info`).
#[derive(Debug)]
pub struct Table<F: TowerField = B128> {
	pub id: TableId,
	pub name: String,
	pub columns: Vec<ColumnInfo<F>>,
	pub(super) partitions: SparseIndex<TablePartition<F>>,
}

/// A table partition describes a part of a table where everything has the same pack factor (as well as height)
/// Tower level does not need to be the same.
///
/// Zerocheck constraints can only be defined within table partitions.
#[derive(Debug)]
pub(super) struct TablePartition<F: TowerField = B128> {
	pub table_id: TableId,
	pub values_per_row: usize,
	pub flushes: Vec<Flush>,
	pub columns: Vec<ColumnIndex>,
	pub zero_constraints: Vec<ZeroConstraint<F>>,
}

impl<F: TowerField> TablePartition<F> {
	pub fn new(table_id: TableId, values_per_row: usize) -> Self {
		Self {
			table_id,
			values_per_row,
			flushes: Vec::new(),
			columns: Vec::new(),
			zero_constraints: Vec::new(),
		}
	}

	pub fn assert_zero<FSub, const VALUES_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		expr: Expr<FSub, VALUES_PER_ROW>,
	) where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		// TODO: Should we dynamically keep track of FSub::TOWER_LEVEL?
		// On the other hand, ArithExpr does introspect that already
		self.zero_constraints.push(ZeroConstraint {
			name: name.to_string(),
			expr: expr.expr().convert_field(),
		});
	}

	pub fn pull_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.pull(channel, std::iter::once(upcast_col(col)))
	}

	pub fn push_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.push(channel, std::iter::once(upcast_col(col)))
	}

	pub fn pull(&mut self, channel: ChannelId, cols: impl IntoIterator<Item = Col<F>>) {
		self.flush(channel, FlushDirection::Pull, cols)
	}

	pub fn push(&mut self, channel: ChannelId, cols: impl IntoIterator<Item = Col<F>>) {
		self.flush(channel, FlushDirection::Push, cols)
	}

	fn flush(
		&mut self,
		channel_id: ChannelId,
		direction: FlushDirection,
		cols: impl IntoIterator<Item = Col<F>>,
	) {
		let column_indices = cols
			.into_iter()
			.map(|col| {
				assert_eq!(col.table_id, self.table_id);
				col.table_index
			})
			.collect();
		self.flushes.push(Flush {
			column_indices,
			channel_id,
			direction,
		});
	}
}

impl<F: TowerField> Table<F> {
	pub fn new(id: TableId, name: impl ToString) -> Self {
		Self {
			id,
			name: name.to_string(),
			columns: Vec::new(),
			partitions: SparseIndex::new(),
		}
	}

	pub fn id(&self) -> TableId {
		self.id
	}

	fn new_column<FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		col: ColumnDef<F>,
	) -> Col<FSub, V>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let table_id = self.id;
		let table_index = self.columns.len();
		let partition = self.partition_mut(V);
		let id = ColumnId {
			table_id,
			table_index,
		};
		let info = ColumnInfo {
			id,
			col,
			name: name.to_string(),
			shape: ColumnShape {
				tower_height: FSub::TOWER_LEVEL,
				log_values_per_row: log2_strict_usize(V),
			},
			is_nonzero: false,
		};

		let partition_index = partition.columns.len();
		partition.columns.push(table_index);
		self.columns.push(info);
		Col::new(id, partition_index)
	}

	fn partition_mut(&mut self, values_per_row: usize) -> &mut TablePartition<F> {
		self.partitions
			.entry(log2_strict_usize(values_per_row))
			.or_insert_with(|| TablePartition::new(self.id, values_per_row))
	}
}

const fn partition_id<const V: usize>() -> usize {
	checked_log_2(V)
}
