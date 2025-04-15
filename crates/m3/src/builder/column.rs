// Copyright 2025 Irreducible Inc.

use std::{
	any::Any,
	fmt::Debug,
	marker::PhantomData,
	sync::{Arc, Mutex},
};

use binius_core::{
	oracle::ShiftVariant,
	polynomial::MultivariatePoly,
	tower::{PackedTowerFamily, TowerFamily},
	transparent::MultilinearExtensionTransparent,
};
use binius_field::{packed::pack_slice, ExtensionField, PackedExtension, TowerField};
use binius_math::ArithExpr;
use binius_utils::checked_arithmetics::checked_log_2;

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
	pub(super) fn convert_to_tower<SourceTower, TargetTower>(&self) -> ColumnInfo<TargetTower::B128>
	where
		SourceTower: TowerFamily<B128 = F>,
		TargetTower: TowerFamily<
			B1: From<SourceTower::B1>,
			B8: From<SourceTower::B8>,
			B16: From<SourceTower::B16>,
			B32: From<SourceTower::B32>,
			B64: From<SourceTower::B64>,
			B128: From<SourceTower::B128>,
		>,
	{
		ColumnInfo {
			id: self.id.clone(),
			col: self.col.convert_to_tower::<SourceTower, TargetTower>(),
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

#[derive(Debug)]
pub struct ConstantColumn<F: TowerField = B128> {
	pub data: Box<dyn Any + Send + Sync>,
	pub cached_poly: Mutex<Option<Arc<dyn MultivariatePoly<F>>>>,
}

impl<F: TowerField> ConstantColumn<F> {
	pub fn new_subfield<FSub: TowerField>(data: &[FSub]) -> Self
	where
		F: ExtensionField<FSub>,
	{
		let data = data.to_vec();
		Self {
			data: Box::new(data),
			cached_poly: Mutex::new(None),
		}
	}

	pub fn new_isomorphic<FSubSource, FSubTarget>(data: &[FSubSource]) -> Self
	where
		FSubSource: TowerField,
		FSubTarget: TowerField + From<FSubSource>,
		F: ExtensionField<FSubTarget>,
	{
		let data = data
			.iter()
			.copied()
			.map(FSubTarget::from)
			.collect::<Vec<_>>();
		Self {
			data: Box::new(data),
			cached_poly: Mutex::new(None),
		}
	}

	pub fn get_or_init_poly<PackedTower: PackedTowerFamily<Tower: TowerFamily<B128 = F>>>(
		&self,
	) -> Arc<dyn MultivariatePoly<F>> {
		let mut cached_poly = self.cached_poly.lock().unwrap();
		if let Some(poly) = cached_poly.as_ref() {
			return poly.clone();
		}

		let poly = if let Some(data) = self
			.data
			.downcast_ref::<Vec<<PackedTower::Tower as TowerFamily>::B1>>()
		{
			make_poly::<_, PackedTower::PackedB128>(data)
		} else if let Some(data) = self
			.data
			.downcast_ref::<Vec<<PackedTower::Tower as TowerFamily>::B8>>()
		{
			make_poly::<_, PackedTower::PackedB128>(data)
		} else if let Some(data) = self
			.data
			.downcast_ref::<Vec<<PackedTower::Tower as TowerFamily>::B16>>()
		{
			make_poly::<_, PackedTower::PackedB128>(data)
		} else if let Some(data) = self
			.data
			.downcast_ref::<Vec<<PackedTower::Tower as TowerFamily>::B32>>()
		{
			make_poly::<_, PackedTower::PackedB128>(data)
		} else if let Some(data) = self
			.data
			.downcast_ref::<Vec<<PackedTower::Tower as TowerFamily>::B64>>()
		{
			make_poly::<_, PackedTower::PackedB128>(data)
		} else if let Some(data) = self
			.data
			.downcast_ref::<Vec<<PackedTower::Tower as TowerFamily>::B128>>()
		{
			make_poly::<_, PackedTower::PackedB128>(data)
		} else {
			panic!("constant column must be a vector of tower field elements");
		};

		*cached_poly = Some(poly.clone());
		poly
	}
}

fn make_poly<F, PackedExt>(source: &[F]) -> Arc<dyn MultivariatePoly<PackedExt::Scalar>>
where
	F: TowerField,
	PackedExt: PackedExtension<F, Scalar: TowerField>,
{
	let packed_data = pack_slice(&source);
	let mle = MultilinearExtensionTransparent::<
	PackedExt::PackedSubfield,
		PackedExt,
		_,
	>::from_values_and_mu(packed_data, checked_log_2(source.len()))
	.unwrap();

	Arc::new(mle)
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
	Constant(ConstantColumn<F>),
}

impl<F: TowerField> ColumnDef<F> {
	pub(super) fn convert_to_tower<SourceTower, TargetTower>(&self) -> ColumnDef<TargetTower::B128>
	where
		SourceTower: TowerFamily<B128 = F>,
		TargetTower: TowerFamily<
			B1: From<SourceTower::B1>,
			B8: From<SourceTower::B8>,
			B16: From<SourceTower::B16>,
			B32: From<SourceTower::B32>,
			B64: From<SourceTower::B64>,
			B128: From<SourceTower::B128>,
		>,
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
			&Self::Projected {
				col,
				start_index,
				query_size,
				query_bits,
			} => ColumnDef::Projected {
				col,
				start_index,
				query_size,
				query_bits,
			},
			Self::Computed { cols, expr } => ColumnDef::Computed {
				cols: cols.clone(),
				expr: expr.convert_field::<TargetTower::B128>(),
			},
			Self::Constant(ConstantColumn { data, .. }) => {
				let data = if let Some(data) = data.downcast_ref::<Vec<SourceTower::B1>>() {
					ConstantColumn::new_isomorphic::<SourceTower::B1, TargetTower::B1>(data)
				} else if let Some(data) = data.downcast_ref::<Vec<SourceTower::B8>>() {
					ConstantColumn::new_isomorphic::<SourceTower::B8, TargetTower::B8>(data)
				} else if let Some(data) = data.downcast_ref::<Vec<SourceTower::B16>>() {
					ConstantColumn::new_isomorphic::<SourceTower::B16, TargetTower::B16>(data)
				} else if let Some(data) = data.downcast_ref::<Vec<SourceTower::B32>>() {
					ConstantColumn::new_isomorphic::<SourceTower::B32, TargetTower::B32>(data)
				} else if let Some(data) = data.downcast_ref::<Vec<SourceTower::B64>>() {
					ConstantColumn::new_isomorphic::<SourceTower::B64, TargetTower::B64>(data)
				} else if let Some(data) = data.downcast_ref::<Vec<SourceTower::B128>>() {
					ConstantColumn::new_isomorphic::<SourceTower::B128, TargetTower::B128>(data)
				} else {
					panic!("constant column must be a vector of tower field elements");
				};

				ColumnDef::Constant(data)
			}
		}
	}
}
