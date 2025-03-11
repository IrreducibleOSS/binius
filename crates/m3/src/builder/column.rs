// Copyright 2025 Irreducible Inc.

use std::{marker::PhantomData, sync::Arc};

use binius_core::{oracle::ShiftVariant, polynomial::MultivariatePoly};
use binius_field::{ExtensionField, TowerField};
use binius_math::ArithExpr;

use super::{table::TableId, types::B128};

/// An index of a column within a table.
pub type ColumnIndex = usize;

/// An index of a column within a table.
pub type ColumnPartitionIndex = usize;

/// A typed identifier for a column in a table.
///
/// The column has entries that are elements of `F`. In practice, the fields used will always be
/// from the canonical tower (B1, B8, B16, B32, B64, B128). The second constant represents how many
/// elements are packed vertically into a single logical row. For example, a column of type
/// `Col<B1, 32>` will have 2^5 = 32 elements of `B1` packed into a single row.
#[derive(Debug, Clone, Copy)]
pub struct Col<F: TowerField, const VALUES_PER_ROW: usize = 1> {
	pub table_id: TableId,
	pub table_index: TableId,
	// Denormalized partition index so that we can use it to construct arithmetic expressions over
	// the partition columns.
	pub partition_index: ColumnPartitionIndex,
	_marker: PhantomData<F>,
}

impl<F: TowerField, const VALUES_PER_ROW: usize> Col<F, VALUES_PER_ROW> {
	pub fn new(id: ColumnId, partition_index: ColumnPartitionIndex) -> Self {
		assert!(VALUES_PER_ROW.is_power_of_two());
		Self {
			table_id: id.table_id,
			table_index: id.table_index,
			partition_index,
			_marker: PhantomData,
		}
	}

	pub fn shape(&self) -> ColumnShape {
		ColumnShape {
			tower_height: F::TOWER_LEVEL,
			log_values_per_row: VALUES_PER_ROW.ilog2() as usize,
		}
	}

	pub fn id(&self) -> ColumnId {
		ColumnId {
			table_id: self.table_id,
			table_index: self.table_index,
		}
	}
}

/// Upcast a columns from a subfield to an extension field..
pub fn upcast_col<F, FSub, const VALUES_PER_ROW: usize>(
	col: Col<FSub, VALUES_PER_ROW>,
) -> Col<F, VALUES_PER_ROW>
where
	FSub: TowerField,
	F: TowerField + ExtensionField<FSub>,
{
	let Col {
		table_id,
		table_index,
		partition_index,
		_marker: _,
	} = col;
	// REVIEW: Maybe this should retain the info of the smallest tower level
	Col {
		table_id,
		table_index,
		partition_index,
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
	pub log_values_per_row: usize,
}

/// Unique identifier for a column within a constraint system.
///
/// IDs are assigned when columns are added to the constraint system and remain stable when more
/// columns are added.
#[derive(Debug, Clone, Copy)]
pub struct ColumnId {
	pub table_id: TableId,
	pub table_index: ColumnIndex,
}

/// A definition of a column in a table.
#[derive(Debug)]
pub enum ColumnDef<F: TowerField = B128> {
	Committed {
		tower_level: usize,
	},
	LinearCombination {
		offset: F,
		col_scalars: Vec<(ColumnIndex, F)>,
	},
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
	Computed {
		cols: Vec<ColumnIndex>,
		expr: ArithExpr<F>,
	},
	RepeatingTransparent {
		poly: Arc<dyn MultivariatePoly<F>>,
	},
}
