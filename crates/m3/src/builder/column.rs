// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_core::oracle::ShiftVariant;
use binius_field::{ExtensionField, TowerField};
use binius_math::LinearNormalForm;

use super::{table::TableId, types::B128};

/// An index of a column within a table.
pub type ColumnIndex = usize;

/// A typed identifier for a column in a table.
///
/// The column has entries that are elements of `F`. In practice, the fields used will always be
/// from the canonical tower (B1, B8, B16, B32, B64, B128). The second constant represents how many
/// elements are packed vertically into a single logical row. For example, a column of type
/// `Col<B1, 5>` will have 2^5 = 32 elements of `B1` packed into a single row.
#[derive(Debug, Clone, Copy)]
pub struct Col<F: TowerField, const V: usize = 0> {
	// TODO: Maybe V should be powers of 2 instead of logarithmic
	pub id: ColumnId,
	pub _marker: PhantomData<F>,
}

impl<F: TowerField, const V: usize> Col<F, V> {
	pub fn new(id: ColumnId) -> Self {
		Self {
			id,
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
		self.id
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
		id: col.id,
		_marker: PhantomData,
	}
}

/// Complete description of a column within a table.
#[derive(Debug)]
pub struct ColumnInfo<F: TowerField = B128> {
	pub id: ColumnId,
	pub col: ColumnDef<F>,
	pub name: String,
	pub shape: ColumnShape,
	/// Whether the column is constrained to be non-zero.
	pub is_nonzero: bool,
}

/// The shape of each cell in a column.
#[derive(Debug, Clone, Copy)]
pub struct ColumnShape {
	/// The tower height of the field elements.
	pub tower_height: usize,
	/// The binary logarithm of the number of elements packed vertically per event row.
	pub pack_factor: usize,
}

/// Unique identifier for a column within a constraint system.
///
/// IDs are assigned when columns are added to the constraint system and remain stable when more
/// columns are added.
#[derive(Debug, Clone, Copy)]
pub struct ColumnId {
	pub table_id: TableId,
	pub table_index: ColumnIndex,
	// REVIEW: Does this strictly correspond to the packing factor?
	// Should it be here or on columnInfo?
	pub partition_id: usize,
	pub partition_index: ColumnIndex,
}

// TODO: TableBuilder needs namespacing

/// A definition of a column in a table.
#[derive(Debug)]
pub enum ColumnDef<F: TowerField = B128> {
	Committed {
		tower_level: usize,
	},
	LinearCombination(LinearNormalForm<F>),
	Selected {
		col: ColumnId,
		index: usize,
		index_bits: usize,
	},
	Shifted {
		col: ColumnId,
		offset: usize,
		log_block_size: usize,
		variant: ShiftVariant,
	},
	Packed {
		col: ColumnId,
		log_degree: usize,
	},
}
