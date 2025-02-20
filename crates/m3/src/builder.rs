// Copyright 2025 Irreducible Inc.

use std::{array, marker::PhantomData};

use binius_core::oracle::ShiftVariant;
use binius_field::{ExtensionField, TowerField};
use binius_math::ArithExpr;

use crate::{
	constraint_system::{
		Channel, ChannelId, Column, ColumnId, ColumnIndex, ColumnInfo, ColumnShape,
		ConstraintSystem, Table, TableId,
	},
	types::B128,
};

/// A type representing a column in a table.
///
/// The column has entries that are elements of `F`. In practice, the fields used will always be
/// from the canonical tower (B1, B8, B16, B32, B64, B128). The second constant represents how many
/// elements are packed vertically into a single logical row. For example, a column of type
/// `Col<B1, 5>` will have 2^5 = 32 elements of `B1` packed into a single row.
#[derive(Debug, Clone, Copy)]
pub struct Col<F: TowerField, const V: usize = 0> {
	// TODO: Maybe V should be powers of 2 instead of logarithmic

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

	pub fn id(&self) -> ColumnId {
		ColumnId {
			table_id: self.table_id,
			index: self.index,
		}
	}
}

/// Upcast a columns from a subfield to an extension field..
pub fn upcast_col<F, FSub, const V: usize>(col: Col<FSub, V>) -> Col<F, V>
where
	FSub: TowerField,
	F: TowerField + ExtensionField<FSub>,
{
	// REVIEW: Maybe this should retain the info of the smallest tower level
	Col {
		table_id: col.table_id,
		index: col.index,
		_marker: PhantomData,
	}
}

/// A type representing an arithmetic expression composed over some table columns.
///
/// If the expression degree is 1, then it is a linear expression.
#[derive(Debug)]
pub struct Expr<F: TowerField, const V: usize> {
	expr: ArithExpr<F>,
	cols: Vec<Col<F, V>>,
}

/// A table in an M3 constraint system.
///
/// ## Invariants
///
/// * All expressions in `zero_constraints` have a number of variables less than or equal to the
///   number of table columns (the length of `column_info`).
/// * All flushes in `flushes` contain column indices less than the number of table columns (the
///   length of `column_info`).
pub struct TableBuilder<F: TowerField = B128> {
	table: Table,
}

impl<F: TowerField> TableBuilder<F> {
	pub fn new(id: TableId, name: impl ToString) -> Self {
		Self {
			table: Table {
				id,
				name: name.to_string(),
				column_info: Vec::new(),
				flushes: Vec::new(),
				zero_constraints: Vec::new(),
				is_fixed: false,
			},
		}
	}

	pub fn new_fixed(id: TableId, name: impl ToString) -> Self {
		let mut builder = Self::new(id, name);
		builder.table.is_fixed = true;
		builder
	}

	pub fn id(&self) -> TableId {
		self.table.id
	}

	pub fn add_committed<FSub, const LOG_VALS_PER_ROW: usize>(
		&mut self,
		name: impl ToString,
	) -> Col<FSub, LOG_VALS_PER_ROW>
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		let index = self.table.column_info.len();
		self.table.column_info.push(ColumnInfo {
			col: Column::Committed {
				tower_level: FSub::TOWER_LEVEL,
			},
			name: name.to_string(),
			pack_factor: LOG_VALS_PER_ROW,
			is_nonzero: false,
		});
		Col {
			table_id: self.table.id,
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

	pub fn add_committed_multiple<FSub, const V: usize, const N: usize>(
		&mut self,
		name: impl ToString,
		tower_level: usize,
	) -> [Col<FSub, V>; N]
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		array::from_fn(|i| self.add_committed(format!("{}[{}]", name.to_string(), i)))
	}

	pub fn assert_zero<const V: usize>(&mut self, name: impl ToString, expr: Expr<F, V>) {
		self.table.zero_constraints.push(expr.expr);
	}

	pub fn build(self) -> Table<F> {
		todo!()
	}
}

#[derive(Debug, Default)]
pub struct ConstraintSystemBuilder<F: TowerField = B128> {
	tables: Vec<TableBuilder<F>>,
	channels: Vec<Channel<F>>,
	/// All valid channel IDs are strictly less than this bound.
	channel_id_bound: ChannelId,
}

impl<F: TowerField> ConstraintSystemBuilder<F> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn add_table(&mut self, name: impl ToString) -> &mut TableBuilder<F> {
		let id = self.tables.len();
		self.tables.push(TableBuilder::new(id, name));
		self.tables.last_mut().expect("table was just pushed")
	}

	pub fn add_channel(&mut self, name: impl ToString) -> ChannelId {
		let id = self.channels.len();
		self.channels.push(Channel {
			name: name.to_string(),
		});
		id
	}

	pub fn build(self) -> ConstraintSystem<F> {
		todo!()
	}
}
