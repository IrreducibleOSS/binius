// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_core::{
	constraint_system::channel::{ChannelId, FlushDirection},
	oracle::ShiftVariant,
};
use binius_field::{ExtensionField, TowerField};

use super::{
	channel::Flush,
	column::{upcast_col, Col, Column, ColumnIndex, ColumnInfo, ColumnShape},
	expr::{Expr, ZeroConstraint},
	types::B128,
};
use crate::builder::column::ColumnId;

pub type TableId = usize;

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
	pub partitions: Vec<TablePartition<F>>,
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
	pub partition_index: usize,
	pub pack_factor: usize,
	pub flushes: Vec<Flush>,
	pub column_info: Vec<ColumnInfo<F>>,
	pub zero_constraints: Vec<ZeroConstraint<F>>,
}

impl<F: TowerField> TablePartition<F> {
	pub fn new(table_id: TableId, partition_index: usize, pack_factor: usize) -> Self {
		Self {
			table_id,
			partition_index,
			pack_factor,
			flushes: Vec::new(),
			column_info: Vec::new(),
			zero_constraints: Vec::new(),
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
			id: self.column_id(index),
			col: Column::Committed {
				tower_level: FSub::TOWER_LEVEL,
			},
			name: name.to_string(),
			shape: ColumnShape {
				pack_factor: LOG_VALS_PER_ROW,
				tower_height: FSub::TOWER_LEVEL,
			},
			is_nonzero: false,
		});
		Col {
			table_id: self.table_id,
			partition: self.partition_index,
			index,
			_marker: PhantomData,
		}
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
		assert!(log_block_size <= LOG_VALS_PER_ROW);
		assert!(offset <= 1 << log_block_size);
		let index = self.column_info.len();
		self.column_info.push(ColumnInfo {
			id: self.column_id(index),
			col: Column::Shifted {
				col: col.id(),
				offset,
				log_block_size,
				variant,
			},
			name: name.to_string(),
			shape: ColumnShape {
				pack_factor: LOG_VALS_PER_ROW,
				tower_height: FSub::TOWER_LEVEL,
			},
			is_nonzero: false,
		});
		Col {
			table_id: self.table_id,
			partition: self.partition_index,
			index,
			_marker: PhantomData,
		}
	}

	// Selected is a special form of projected with hypercube indices
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

		let index = self.column_info.len();
		self.column_info.push(ColumnInfo {
			id: self.column_id(index),
			col: Column::LinearCombination(lincom),
			name: name.to_string(),
			shape: ColumnShape {
				pack_factor: V,
				tower_height: FSub::TOWER_LEVEL,
			},
			is_nonzero: false,
		});
		Col {
			table_id: self.table_id,
			partition: self.partition_index,
			index,
			_marker: PhantomData,
		}
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
		assert!(FSubSub::TOWER_LEVEL < FSub::TOWER_LEVEL);
		assert!(VSUB > V);
		assert_eq!(FSub::TOWER_LEVEL + V, FSubSub::TOWER_LEVEL + VSUB);
		let index = self.column_info.len();
		self.column_info.push(ColumnInfo {
			id: self.column_id(index),
			col: Column::Packed {
				col: col.id(),
				log_degree: FSub::TOWER_LEVEL - FSubSub::TOWER_LEVEL,
			},
			name: name.to_string(),
			shape: ColumnShape {
				pack_factor: V,
				tower_height: FSub::TOWER_LEVEL,
			},
			is_nonzero: false,
		});
		Col {
			table_id: self.table_id,
			partition: self.partition_index,
			index,
			_marker: PhantomData,
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
				assert_eq!(col.table_id, self.table_id);
				assert_eq!(col.partition, self.partition_index);
				col.index
			})
			.collect();
		self.flushes.push(Flush {
			column_indices,
			channel_id,
			direction,
		});
	}

	fn column_id(&self, index: ColumnIndex) -> ColumnId {
		ColumnId {
			table_id: self.table_id,
			partition: self.partition_index,
			index,
		}
	}
}

impl<F: TowerField> Table<F> {
	pub fn new(id: TableId, name: impl ToString) -> Self {
		Self {
			id,
			name: name.to_string(),
			partitions: Vec::new(),
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

	fn partition_mut(&mut self, pack_factor: usize) -> &mut TablePartition<F> {
		let index = self
			.partitions
			.iter()
			.enumerate()
			.find(|(_, p)| p.pack_factor == pack_factor)
			.map(|(i, _)| i);
		if let Some(index) = index {
			self.partitions.get_mut(index).unwrap()
		} else {
			let index = self.partitions.len();
			self.partitions
				.push(TablePartition::new(self.id, index, pack_factor));
			self.partitions.last_mut().unwrap()
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
		self.partition_mut(LOG_VALS_PER_ROW).add_committed(name)
	}

	pub fn add_committed_multiple<FSub, const V: usize, const N: usize>(
		&mut self,
		name: impl ToString,
	) -> [Col<FSub, V>; N]
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.partition_mut(V).add_committed_multiple(name)
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
		self.partition_mut(LOG_VALS_PER_ROW)
			.add_shifted(name, col, log_block_size, offset, variant)
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
		self.partition_mut(V).add_linear_combination(name, expr)
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
		self.partition_mut(V).add_packed(name, col)
	}

	pub fn add_selected<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
		original: Col<FSub, LOG_VALS_PER_ROW>,
		index: usize,
	) -> Col<FSub, 0>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.partition_mut(LOG_VALS_PER_ROW)
			.add_selected(name, original, index)
	}

	pub fn assert_zero<FSub, const V: usize>(&mut self, name: impl ToString, expr: Expr<FSub, V>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.partition_mut(V).assert_zero(name, expr)
	}

	pub fn pull_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.partition_mut(0).pull_one(channel, col)
	}

	pub fn push_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.partition_mut(0).push_one(channel, col)
	}
}
