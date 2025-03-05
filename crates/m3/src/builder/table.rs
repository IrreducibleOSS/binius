// Copyright 2025 Irreducible Inc.

use binius_core::{
	constraint_system::channel::{ChannelId, FlushDirection},
	oracle::ShiftVariant,
};
use binius_field::{ExtensionField, TowerField};
use binius_utils::sparse_index::SparseIndex;

use super::{
	channel::Flush,
	column::{upcast_col, Col, ColumnDef, ColumnInfo, ColumnShape},
	expr::{Expr, ZeroConstraint},
	types::B128,
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

	pub fn add_committed<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
	) -> Col<FSub, LOG_VALS_PER_ROW>
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

	pub fn add_committed_multiple<FSub, const V: usize, const N: usize>(
		&mut self,
		name: impl ToString,
	) -> [Col<FSub, V>; N]
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		std::array::from_fn(|i| self.add_committed(format!("{}[{}]", name.to_string(), i)))
	}

	pub fn add_shifted<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSub, LOG_VALS_PER_ROW>,
		log_block_size: usize,
		offset: usize,
		variant: ShiftVariant,
	) -> Col<FSub, LOG_VALS_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
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
		let lincom = expr
			.expr()
			.convert_field::<F>()
			.linear_normal_form()
			.expect("pre-condition: expression must be linear");

		self.table
			.new_column(self.namespaced_name(name), ColumnDef::LinearCombination(lincom))
	}

	pub fn add_packed<FSubSub, const VSUB: usize, FSub, const V: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSubSub, VSUB>,
	) -> Col<FSub, V>
	where
		FSub: TowerField + ExtensionField<FSubSub>,
		FSubSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.new_column(
			self.namespaced_name(name),
			ColumnDef::Packed {
				col: col.id(),
				log_degree: FSub::TOWER_LEVEL - FSubSub::TOWER_LEVEL,
			},
		)
	}

	pub fn add_selected<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		_name: impl ToString,
		_original: Col<FSub, LOG_VALS_PER_ROW>,
		_index: usize,
	) -> Col<FSub, 0>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		todo!()
	}

	pub fn assert_zero<FSub, const V: usize>(&mut self, name: impl ToString, expr: Expr<FSub, V>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(V).assert_zero(name, expr)
	}

	pub fn pull_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(0).pull_one(channel, col)
	}

	pub fn push_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.table.partition_mut(0).push_one(channel, col)
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
	pub partitions: SparseIndex<TablePartition<F>>,
	/// This indicates whether a table is fixed for constraint system or part of the dynamic trace.
	///
	/// Fixed tables are either entirely transparent or committed during a preprocessing step that
	/// occurs before any statements are proven.
	pub is_fixed: bool,
}

/// A table partition describes a part of a table where everything has the same pack factor (as well as height)
/// Tower level does not need to be the same.
///
/// Zerocheck constraints can only be defined within table partitions.
#[derive(Debug)]
pub struct TablePartition<F: TowerField = B128> {
	pub table_id: TableId,
	pub pack_factor: usize,
	pub flushes: Vec<Flush>,
	pub columns: Vec<ColumnId>,
	pub zero_constraints: Vec<ZeroConstraint<F>>,
}

impl<F: TowerField> TablePartition<F> {
	pub fn new(table_id: TableId, pack_factor: usize) -> Self {
		Self {
			table_id,
			pack_factor,
			flushes: Vec::new(),
			columns: Vec::new(),
			zero_constraints: Vec::new(),
		}
	}

	pub fn assert_zero<FSub, const V: usize>(&mut self, name: impl ToString, expr: Expr<FSub, V>)
	where
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
		cols: impl IntoIterator<Item = Col<F, 0>>,
	) {
		let column_indices = cols
			.into_iter()
			.map(|col| {
				assert_eq!(col.id.table_id, self.table_id);
				assert_eq!(col.id.partition_id, self.pack_factor);
				col.id.partition_index
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
			is_fixed: false,
		}
	}

	pub fn new_fixed(id: TableId, name: impl ToString) -> Self {
		let mut builder = Self::new(id, name);
		builder.is_fixed = true;
		builder
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
			partition_id: partition.pack_factor,
			partition_index: partition.columns.len(),
		};
		let info = ColumnInfo {
			id,
			col,
			name: name.to_string(),
			shape: ColumnShape {
				pack_factor: V,
				tower_height: FSub::TOWER_LEVEL,
			},
			is_nonzero: false,
		};
		partition.columns.push(id);
		self.columns.push(info);
		Col::new(id)
	}

	fn partition_mut(&mut self, pack_factor: usize) -> &mut TablePartition<F> {
		self.partitions
			.entry(pack_factor)
			.or_insert_with(|| TablePartition::new(self.id, pack_factor))
	}
}
