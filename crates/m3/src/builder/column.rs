// Copyright 2025 Irreducible Inc.

use std::{any::Any, fmt::Debug, marker::PhantomData, sync::Arc};

use binius_core::{
	oracle::ShiftVariant,
	polynomial::MultivariatePoly,
	tower::{PackedTowerConverter, PackedTowerFamily, TowerFamily},
	transparent::MultilinearExtensionTransparent,
};
use binius_field::{packed::pack_slice, ExtensionField, PackedExtension, TowerField};
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

impl<F: TowerField> ColumnInfo<F> {
	pub(super) fn convert_to_tower<SourcePackedTower, TargetPackedTower>(
		&self,
		converter: &impl PackedTowerConverter<SourcePackedTower, TargetPackedTower>,
	) -> ColumnInfo<<TargetPackedTower::Tower as TowerFamily>::B128>
	where
		SourcePackedTower: PackedTowerFamily,
		SourcePackedTower::Tower: TowerFamily<B128 = F>,
		TargetPackedTower: PackedTowerFamily,
		<TargetPackedTower::Tower as TowerFamily>::B128: From<F>,
	{
		ColumnInfo {
			id: self.id.clone(),
			col: self
				.col
				.convert_to_tower::<SourcePackedTower, TargetPackedTower>(converter),
			name: self.name.clone(),
			shape: self.shape,
			is_nonzero: self.is_nonzero,
		}
	}
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
		data: Box<dyn Any + Send + Sync>,
		poly: Arc<dyn MultivariatePoly<F>>,
	},
}

impl<F: TowerField> ColumnDef<F> {
	pub(super) fn convert_to_tower<SourcePackedTower, TargetPackedTower>(
		&self,
		converter: &impl PackedTowerConverter<SourcePackedTower, TargetPackedTower>,
	) -> ColumnDef<<TargetPackedTower::Tower as TowerFamily>::B128>
	where
		SourcePackedTower: PackedTowerFamily,
		SourcePackedTower::Tower: TowerFamily<B128 = F>,
		TargetPackedTower: PackedTowerFamily,
		<TargetPackedTower::Tower as TowerFamily>::B128: From<F>,
	{
		match self {
			&Self::Committed { tower_level } => ColumnDef::Committed { tower_level },
			&Self::Selected {
				col,
				index,
				index_bits,
			} => ColumnDef::Selected {
				col,
				index,
				index_bits,
			},
			&Self::Shifted {
				col,
				offset,
				log_block_size,
				variant,
			} => ColumnDef::Shifted {
				col,
				offset,
				log_block_size,
				variant,
			},
			&Self::Packed { col, log_degree } => ColumnDef::Packed { col, log_degree },
			Self::Computed { cols, expr } => ColumnDef::Computed {
				cols: cols.clone(),
				expr: expr.convert_field::<<TargetPackedTower::Tower as TowerFamily>::B128>(),
			},
			Self::Constant { data, .. } => {
				let (data, poly) = if let Some(data) =
					data.downcast_ref::<Vec<<SourcePackedTower::Tower as TowerFamily>::B1>>()
				{
					convert_column_constant::<_, _, TargetPackedTower::PackedB128>(&data, |x| {
						converter.convert_b1(x)
					})
				} else if let Some(data) =
					data.downcast_ref::<Vec<<SourcePackedTower::Tower as TowerFamily>::B8>>()
				{
					convert_column_constant::<_, _, TargetPackedTower::PackedB128>(&data, |x| {
						converter.convert_b8(x)
					})
				} else if let Some(data) =
					data.downcast_ref::<Vec<<SourcePackedTower::Tower as TowerFamily>::B16>>()
				{
					convert_column_constant::<_, _, TargetPackedTower::PackedB128>(&data, |x| {
						converter.convert_b16(x)
					})
				} else if let Some(data) =
					data.downcast_ref::<Vec<<SourcePackedTower::Tower as TowerFamily>::B32>>()
				{
					convert_column_constant::<_, _, TargetPackedTower::PackedB128>(&data, |x| {
						converter.convert_b32(x)
					})
				} else if let Some(data) =
					data.downcast_ref::<Vec<<SourcePackedTower::Tower as TowerFamily>::B64>>()
				{
					convert_column_constant::<_, _, TargetPackedTower::PackedB128>(&data, |x| {
						converter.convert_b64(x)
					})
				} else if let Some(data) =
					data.downcast_ref::<Vec<<SourcePackedTower::Tower as TowerFamily>::B128>>()
				{
					convert_column_constant::<_, _, TargetPackedTower::PackedB128>(&data, |x| {
						converter.convert_b128(x)
					})
				} else {
					panic!("constant column must be a vector of tower field elements");
				};

				ColumnDef::Constant { data, poly }
			}
		}
	}
}

fn convert_column_constant<F, TargetField, PackedTowerFieldExt>(
	source: &[F],
	convert_fn: impl Fn(&F) -> TargetField,
) -> (Box<dyn Any + Send + Sync>, Arc<dyn MultivariatePoly<PackedTowerFieldExt::Scalar>>)
where
	F: TowerField,
	TargetField: TowerField,
	PackedTowerFieldExt: PackedExtension<TargetField, Scalar: TowerField>,
{
	let data = source.iter().map(convert_fn).collect::<Vec<_>>();

	let packed_data = pack_slice(&data);
	let mle = MultilinearExtensionTransparent::<
		PackedTowerFieldExt::PackedSubfield,
		PackedTowerFieldExt,
		_,
	>::from_values_and_mu(packed_data, source.len())
	.unwrap();

	(Box::new(data), Arc::new(mle))
}
