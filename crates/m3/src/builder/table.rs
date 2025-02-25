use std::marker::PhantomData;

use binius_core::{
	constraint_system::channel::{ChannelId, FlushDirection},
	oracle::ShiftVariant,
};
use binius_field::{ExtensionField, TowerField};

use super::{
	channel::Flush,
	column::{upcast_col, Col, Column, ColumnInfo},
	expr::{Expr, ZeroConstraint},
	types::B128,
};

pub type TableId = usize;
pub type ColumnIndex = usize; // REVIEW: Could make these opaque without a constructor, to protect
							  // access

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
	pub column_info: Vec<ColumnInfo<F>>,
	pub flushes: Vec<Flush>,
	pub zero_constraints: Vec<ZeroConstraint<F>>,
	/// This indicates whether a table is fixed for constraint system or part of the dynamic trace.
	///
	/// Fixed tables are either entirely transparent or committed during a preprocessing step that
	/// occurs before any statements are proven.
	pub is_fixed: bool,
}

impl<F: TowerField> Table<F> {
	pub fn new(id: TableId, name: impl ToString) -> Self {
		Self {
			id,
			name: name.to_string(),
			column_info: Vec::new(),
			flushes: Vec::new(),
			zero_constraints: Vec::new(),
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
			col: Column::Committed {
				tower_level: FSub::TOWER_LEVEL,
			},
			name: name.to_string(),
			pack_factor: LOG_VALS_PER_ROW,
			is_nonzero: false,
		});
		Col {
			table_id: self.id,
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
			col: Column::Shifted {
				col_index: col.index,
				offset,
				log_block_size,
				variant,
			},
			name: name.to_string(),
			pack_factor: LOG_VALS_PER_ROW,
			is_nonzero: false,
		});
		Col {
			table_id: self.id,
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
			col: Column::LinearCombination(lincom),
			name: name.to_string(),
			pack_factor: V,
			is_nonzero: false,
		});
		Col {
			table_id: self.id,
			index,
			_marker: PhantomData,
		}
	}

	pub fn add_committed_multiple<FSub, const V: usize, const N: usize>(
		&mut self,
		name: impl ToString,
		_tower_level: usize,
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
			col: Column::Packed {
				col_index: col.index,
				log_degree: FSub::TOWER_LEVEL - FSubSub::TOWER_LEVEL,
			},
			name: name.to_string(),
			pack_factor: V,
			is_nonzero: false,
		});
		Col {
			table_id: self.id,
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
				assert_eq!(col.table_id, self.id);
				col.index
			})
			.collect();
		self.flushes.push(Flush {
			column_indices,
			channel_id,
			direction,
		});
	}
}
