// Copyright 2025 Irreducible Inc.

use std::{
	array, iter,
	marker::PhantomData,
	ops::{Add, Mul, Sub},
};

use binius_core::{constraint_system::channel::FlushDirection, oracle::ShiftVariant};
use binius_field::{ExtensionField, TowerField};
use binius_math::ArithExpr;
use getset::{CopyGetters, Getters};
use static_assertions::const_assert_eq;

use crate::{
	constraint_system::{
		Channel, ChannelId, Column, ColumnId, ColumnIndex, ColumnInfo, ColumnShape,
		ConstraintSystem, Flush, Table, TableId, ZeroConstraint,
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

impl<F: TowerField, const V: usize> Add<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::Var(self.index);
		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr + rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> Add<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Col<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: self.expr + rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> Add<Expr<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn add(self, rhs: Expr<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		Expr {
			table_id: self.table_id,
			expr: self.expr + rhs.expr,
		}
	}
}

impl<F: TowerField, const V: usize> Sub<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::Var(self.index);
		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr - rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> Sub<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Col<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: self.expr - rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> Sub<Expr<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn sub(self, rhs: Expr<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		Expr {
			table_id: self.table_id,
			expr: self.expr - rhs.expr,
		}
	}
}

impl<F: TowerField, const V: usize> Mul<Self> for Col<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Self) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let lhs_expr = ArithExpr::Var(self.index);
		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: lhs_expr * rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> Mul<Col<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Col<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		let rhs_expr = ArithExpr::Var(rhs.index);

		Expr {
			table_id: self.table_id,
			expr: self.expr * rhs_expr,
		}
	}
}

impl<F: TowerField, const V: usize> Mul<Expr<F, V>> for Expr<F, V> {
	type Output = Expr<F, V>;

	fn mul(self, rhs: Expr<F, V>) -> Self::Output {
		assert_eq!(self.table_id, rhs.table_id);

		Expr {
			table_id: self.table_id,
			expr: self.expr * rhs.expr,
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

// TODO: Do we need upcast_expr too?

/// A type representing an arithmetic expression composed over some table columns.
///
/// If the expression degree is 1, then it is a linear expression.
#[derive(Debug, Getters, CopyGetters)]
pub struct Expr<F: TowerField, const V: usize> {
	#[get_copy = "pub"]
	table_id: TableId,
	#[get = "pub"]
	expr: ArithExpr<F>,
}

impl<F: TowerField, const V: usize> Expr<F, V> {
	pub fn degree(&self) -> usize {
		self.expr.degree()
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
pub struct TableBuilder<F: TowerField = B128> {
	table: Table<F>,
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
			.expr
			.convert_field::<F>()
			.linear_normal_form()
			.expect("pre-condition: expression must be linear");

		let index = self.table.column_info.len();
		self.table.column_info.push(ColumnInfo {
			col: Column::LinearCombination(lincom),
			name: name.to_string(),
			pack_factor: V,
			is_nonzero: false,
		});
		Col {
			table_id: self.table.id,
			index,
			_marker: PhantomData,
		}
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

	pub fn add_packed<FSub, const V: usize, FSubSub, const VSub: usize>(
		&mut self,
		name: impl ToString,
		col: Col<FSubSub, VSub>,
	) -> Col<FSub, V>
	where
		FSub: TowerField + ExtensionField<FSubSub>,
		FSubSub: TowerField,
		F: ExtensionField<FSub>,
	{
		assert_eq!(FSub::TOWER_LEVEL + V, FSubSub::TOWER_LEVEL + VSub);
		todo!()
	}

	pub fn assert_zero<FSub, const V: usize>(&mut self, name: impl ToString, expr: Expr<FSub, V>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		// TODO: Should we dynamically keep track of FSub::TOWER_LEVEL?
		// On the other hand, ArithExpr does introspect that already
		self.table.zero_constraints.push(ZeroConstraint {
			name: name.to_string(),
			expr: expr.expr.convert_field(),
		});
	}

	pub fn pull_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.pull(channel, iter::once(upcast_col(col)))
	}

	pub fn push_one<FSub>(&mut self, channel: ChannelId, col: Col<FSub>)
	where
		FSub: TowerField,
		F: ExtensionField<FSub>,
	{
		self.push(channel, iter::once(upcast_col(col)))
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
				assert_eq!(col.table_id, self.table.id);
				col.index
			})
			.collect();
		self.table.flushes.push(Flush {
			column_indices,
			channel_id,
			direction,
		});
	}

	pub fn build(self) -> Table<F> {
		todo!()
	}
}

#[derive(Debug, Default)]
pub struct ConstraintSystemBuilder<F: TowerField = B128> {
	tables: Vec<TableBuilder<F>>,
	channels: Vec<Channel>,
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
