// Copyright 2025 Irreducible Inc.

use std::{marker::PhantomData, sync::Arc};

use binius_core::{oracle::ShiftVariant, polynomial::MultivariatePoly};
use binius_field::{ExtensionField, TowerField};
use binius_math::ArithExpr;

use super::{structured::StructuredDynSize, table::TableId, types::B128};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Col<F: TowerField, const VALUES_PER_ROW: usize = 1> {
	pub table_id: TableId,
	pub table_index: TableId,
	// Denormalized partition index so that we can use it to construct arithmetic expressions over
	// the partition columns.
	pub partition_index: ColumnPartitionIndex,
	_marker: PhantomData<F>,
}

impl<F: TowerField, const VALUES_PER_ROW: usize> Col<F, VALUES_PER_ROW> {
	/// Creates a new typed column handle.
	///
	/// This has limited visibility to ensure that only the [`TableBuilder`] can create them. This
	/// ensures the type parameters are consistent with the column definition.
	pub(super) fn new(id: ColumnId, partition_index: ColumnPartitionIndex) -> Self {
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

/// Upcast a column from a subfield to an extension field..
pub fn upcast_col<F, FSub, const V: usize>(col: Col<FSub, V>) -> Col<F, V>
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

impl ColumnShape {
	/// Returns the binary logarithm of the number of bits each cell occupies.
	pub fn log_cell_size(&self) -> usize {
		self.tower_height + self.log_values_per_row
	}
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
	Selected {
		col: ColumnId,
		index: usize,
		index_bits: usize,
	},
	Projected {
		col: ColumnId,
		start_index: usize,
		query_size: usize,
		query_bits: usize,
	},
	ZeroPadded {
		col: ColumnId,
		n_pad_vars: usize,
		start_index: usize,
		nonzero_index: usize,
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
	Constant {
		poly: Arc<dyn MultivariatePoly<F>>,
	},
	StructuredDynSize(StructuredDynSize),
	StaticExp {
		bit_cols: Vec<ColumnIndex>,
		base: F,
		base_tower_level: usize,
	},
	DynamicExp {
		bit_cols: Vec<ColumnIndex>,
		base: ColumnIndex,
		base_tower_level: usize,
	},
}
