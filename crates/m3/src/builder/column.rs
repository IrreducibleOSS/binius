use std::marker::PhantomData;

use binius_core::oracle::ShiftVariant;
use binius_field::{ExtensionField, TowerField};
use binius_math::LinearNormalForm;

use super::{
	table::{ColumnIndex, TableId},
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
	pub _marker: PhantomData<F>,
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

#[derive(Debug, Clone, Copy)]
pub struct ColumnShape {
	pub tower_height: usize,
	pub pack_factor: usize,
}

#[derive(Debug)]
pub struct ColumnId {
	pub table_id: TableId,
	pub index: ColumnIndex,
}

#[derive(Debug)]
pub struct ColumnInfo<F: TowerField = B128> {
	pub col: Column<F>,
	pub name: String,
	/// This represents how many
	pub pack_factor: usize,
	pub is_nonzero: bool,
}

// TODO: Impl Add/Sub/Mul for Col, returning Expr

// feature: TableBuilder needs namespacing
#[derive(Debug)]
pub enum Column<F: TowerField = B128> {
	Committed {
		tower_level: usize,
	},
	LinearCombination(LinearNormalForm<F>),
	Shifted {
		col_index: ColumnIndex,
		offset: usize,
		log_block_size: usize,
		variant: ShiftVariant,
	},
}
