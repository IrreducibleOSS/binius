// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter, slice,
	sync::Arc,
};

use binius_core::{
	oracle::OracleId, polynomial::ArithCircuitPoly, transparent::step_down::StepDown,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	ExtensionField, PackedField, PackedFieldIndexable, TowerField,
};
use binius_math::{CompositionPoly, MultilinearExtension, MultilinearPoly, RowsBatchRef};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{must_cast_slice, must_cast_slice_mut, zeroed_vec, Pod};
use getset::CopyGetters;
use itertools::Itertools;

use super::{
	column::{Col, ColumnShape},
	error::Error,
	table::{Table, TableId},
	types::{B1, B128, B16, B32, B64, B8},
	ColumnDef, ColumnId, ColumnIndex, Expr,
};
use crate::builder::multi_iter::MultiIterator;

/// Holds witness column data for all tables in a constraint system, indexed by column ID.
///
/// The struct has two lifetimes: `'cs` is the lifetime of the constraint system, and `'alloc` is
/// the lifetime of the bump allocator. The reason these must be separate is that the witness index
/// gets converted into a multilinear extension index, which maintains references to the data
/// allocated by the allocator, but does not need to maintain a reference to the constraint system,
/// which can then be dropped.
#[derive(Debug, Default, CopyGetters)]
pub struct WitnessIndex<'cs, 'alloc, U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	pub tables: Vec<Option<TableWitnessIndex<'cs, 'alloc, U, F>>>,
}

impl<'cs, 'alloc, U: UnderlierType, F: TowerField> WitnessIndex<'cs, 'alloc, U, F> {
	pub fn get_table(
		&mut self,
		table_id: TableId,
	) -> Option<&mut TableWitnessIndex<'cs, 'alloc, U, F>> {
		self.tables
			.get_mut(table_id)
			.and_then(|inner| inner.as_mut())
	}

	pub fn fill_table_sequential<T: TableFiller<U, F>>(
		&mut self,
		table: &T,
		rows: &[T::Event],
	) -> Result<(), Error> {
		if !rows.is_empty() {
			let table_id = table.id();
			let witness = self
				.get_table(table_id)
				.ok_or(Error::MissingTable { table_id })?;
			witness.fill_sequential(table, rows)?;
		}
		Ok(())
	}

	pub fn fill_table_parallel<T>(&mut self, table: &T, rows: &[T::Event]) -> Result<(), Error>
	where
		T: TableFiller<U, F> + Sync,
		T::Event: Sync,
	{
		let table_id = table.id();
		let witness = self
			.get_table(table_id)
			.ok_or(Error::MissingTable { table_id })?;
		witness.fill_parallel(table, rows)?;
		Ok(())
	}

	pub fn into_multilinear_extension_index(
		self,
	) -> MultilinearExtensionIndex<'alloc, PackedType<U, B128>>
	where
		U: PackScalar<B1>
			+ PackScalar<B8>
			+ PackScalar<B16>
			+ PackScalar<B32>
			+ PackScalar<B64>
			+ PackScalar<B128>,
	{
		let mut index = MultilinearExtensionIndex::new();
		let mut first_oracle_id_in_table = 0;
		for table_witness in self.tables {
			let Some(table_witness) = table_witness else {
				continue;
			};
			let table = table_witness.table();
			let cols = immutable_witness_index_columns(table_witness.cols);

			// Append oracles for constant columns that are repeated.
			let mut count = 0;
			for (oracle_id_offset, col) in cols.into_iter().enumerate() {
				let oracle_id = first_oracle_id_in_table + oracle_id_offset;
				let log_capacity = if col.is_single_row {
					0
				} else {
					table_witness.log_capacity
				};
				let n_vars = log_capacity + col.shape.log_values_per_row;
				let underlier_count =
					1 << (n_vars + col.shape.tower_height).saturating_sub(U::LOG_BITS);
				let witness = multilin_poly_from_underlier_data(
					&col.data[..underlier_count],
					n_vars,
					col.shape.tower_height,
				);
				index.update_multilin_poly([(oracle_id, witness)]).unwrap();
				count += 1;
			}

			// Every table partition has a step_down appended to the end of the table to support
			// non-power of two height tables.
			for log_values_per_row in table.partitions.keys() {
				let oracle_id = first_oracle_id_in_table + count;
				let size = table_witness.size << log_values_per_row;
				let log_size = table_witness.log_capacity + log_values_per_row;
				let witness = StepDown::new(log_size, size)
					.unwrap()
					.multilinear_extension::<PackedType<U, B1>>()
					.unwrap()
					.specialize_arc_dyn();
				index.update_multilin_poly([(oracle_id, witness)]).unwrap();
				count += 1;
			}

			first_oracle_id_in_table += count;
		}
		index
	}
}

fn multilin_poly_from_underlier_data<U>(
	data: &[U],
	n_vars: usize,
	tower_height: usize,
) -> Arc<dyn MultilinearPoly<PackedType<U, B128>> + Send + Sync + '_>
where
	U: PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<B64>
		+ PackScalar<B128>,
{
	match tower_height {
		0 => MultilinearExtension::new(n_vars, PackedType::<U, B1>::from_underliers_ref(data))
			.unwrap()
			.specialize_arc_dyn(),
		3 => MultilinearExtension::new(n_vars, PackedType::<U, B8>::from_underliers_ref(data))
			.unwrap()
			.specialize_arc_dyn(),
		4 => MultilinearExtension::new(n_vars, PackedType::<U, B16>::from_underliers_ref(data))
			.unwrap()
			.specialize_arc_dyn(),
		5 => MultilinearExtension::new(n_vars, PackedType::<U, B32>::from_underliers_ref(data))
			.unwrap()
			.specialize_arc_dyn(),
		6 => MultilinearExtension::new(n_vars, PackedType::<U, B64>::from_underliers_ref(data))
			.unwrap()
			.specialize_arc_dyn(),
		7 => MultilinearExtension::new(n_vars, PackedType::<U, B128>::from_underliers_ref(data))
			.unwrap()
			.specialize_arc_dyn(),
		_ => {
			panic!("Unsupported tower height: {tower_height}");
		}
	}
}

/// Holds witness column data for a table, indexed by column index.
#[derive(Debug, CopyGetters)]
pub struct TableWitnessIndex<'cs, 'alloc, U: UnderlierType = OptimalUnderlier, F: TowerField = B128>
{
	#[get_copy = "pub"]
	table: &'cs Table<F>,
	oracle_offset: usize,
	cols: Vec<WitnessIndexColumn<'alloc, U>>,
	/// The number of table events that the index should contain.
	#[get_copy = "pub"]
	size: usize,
	#[get_copy = "pub"]
	log_capacity: usize,
	/// Binary logarithm of the mininimum segment size.
	///
	/// This is the minimum number of logical rows that can be put into one segment during
	/// iteration. It is the maximum number of logical rows occupied by a single underlier.
	#[get_copy = "pub"]
	min_log_segment_size: usize,
}

#[derive(Debug)]
pub struct WitnessIndexColumn<'a, U: UnderlierType> {
	shape: ColumnShape,
	data: WitnessDataMut<'a, U>,
	is_single_row: bool,
}

#[derive(Debug, Clone)]
enum WitnessColumnInfo<T> {
	Owned(T),
	SameAsOracleIndex(usize),
}

type WitnessDataMut<'a, U> = WitnessColumnInfo<&'a mut [U]>;

impl<'a, U: UnderlierType> WitnessDataMut<'a, U> {
	pub fn new_owned(allocator: &'a bumpalo::Bump, log_underlier_count: usize) -> Self {
		Self::Owned(allocator.alloc_slice_fill_default(1 << log_underlier_count))
	}
}

type RefCellData<'a, U> = WitnessColumnInfo<RefCell<&'a mut [U]>>;

#[derive(Debug)]
pub struct ImmutableWitnessIndexColumn<'a, U: UnderlierType> {
	pub shape: ColumnShape,
	pub data: &'a [U],
	pub is_single_row: bool,
}

/// Converts the vector of witness columns into immutable references to column data that may be
/// shared.
fn immutable_witness_index_columns<U: UnderlierType>(
	cols: Vec<WitnessIndexColumn<U>>,
) -> Vec<ImmutableWitnessIndexColumn<U>> {
	let mut result = Vec::<ImmutableWitnessIndexColumn<_>>::with_capacity(cols.len());
	for col in cols {
		result.push(ImmutableWitnessIndexColumn {
			shape: col.shape,
			data: match col.data {
				WitnessDataMut::Owned(data) => data,
				WitnessDataMut::SameAsOracleIndex(index) => result[index].data,
			},
			is_single_row: col.is_single_row,
		});
	}
	result
}

impl<'cs, 'alloc, U: UnderlierType, F: TowerField> TableWitnessIndex<'cs, 'alloc, U, F> {
	pub fn new(
		allocator: &'alloc bumpalo::Bump,
		table: &'cs Table<F>,
		size: usize,
	) -> Result<Self, Error> {
		if size == 0 {
			return Err(Error::EmptyTable {
				table_id: table.id(),
			});
		}

		let log_capacity = table.log_capacity(size);

		let mut cols = Vec::new();
		let mut oracle_offset = 0;
		let mut transparent_single_backing = vec![None; table.columns.len()];

		for col in &table.columns {
			if matches!(col.col, ColumnDef::Constant { .. }) {
				transparent_single_backing[col.id.table_index] = Some(oracle_offset);
				cols.push(WitnessIndexColumn {
					shape: col.shape,
					data: WitnessDataMut::new_owned(
						allocator,
						(col.shape.log_cell_size() + log_capacity).saturating_sub(U::LOG_BITS),
					),
					is_single_row: true,
				});
				oracle_offset += 1;
			}
		}

		cols.extend(table.columns.iter().map(|col| WitnessIndexColumn {
			shape: col.shape,
			data: match col.col {
				ColumnDef::Packed { col: inner_col, .. } => {
					WitnessDataMut::SameAsOracleIndex(oracle_offset + inner_col.table_index)
				}
				ColumnDef::Constant { .. } => WitnessDataMut::SameAsOracleIndex(
					transparent_single_backing[col.id.table_index].unwrap(),
				),
				_ => WitnessDataMut::new_owned(
					allocator,
					(col.shape.log_cell_size() + log_capacity).saturating_sub(U::LOG_BITS),
				),
			},
			is_single_row: false,
		}));

		// The minimum segment size is chosen such that the segment of each column is at least one
		// underlier in size.
		let min_log_segment_size = U::LOG_BITS
			- table
				.columns
				.iter()
				.map(|col| col.shape.log_cell_size())
				.fold(U::LOG_BITS, |a, b| a.min(b));

		// But, in case the minimum segment size is larger than the capacity, we lower it so the
		// caller can get the full witness index in one segment. This is OK because the extra field
		// elements in the smallest columns are just padding.
		let min_log_segment_size = min_log_segment_size.min(log_capacity);

		Ok(Self {
			table,
			cols,
			size,
			log_capacity,
			min_log_segment_size,
			oracle_offset,
		})
	}

	pub fn table_id(&self) -> TableId {
		self.table.id
	}

	pub fn capacity(&self) -> usize {
		1 << self.log_capacity
	}

	/// Returns a witness index segment covering the entire table.
	pub fn full_segment(&mut self) -> TableWitnessSegment<U, F> {
		let cols = self
			.cols
			.iter_mut()
			.map(|col| match &mut col.data {
				WitnessDataMut::SameAsOracleIndex(index) => RefCellData::SameAsOracleIndex(*index),
				WitnessDataMut::Owned(data) => RefCellData::Owned(RefCell::new(data)),
			})
			.collect();
		TableWitnessSegment {
			table: self.table,
			cols,
			log_size: self.log_capacity,
			oracle_offset: self.oracle_offset,
		}
	}

	/// Fill a full table witness index using the given row data.
	///
	/// This function iterates through witness segments sequentially in a single thread.
	pub fn fill_sequential<T: TableFiller<U, F>>(
		&mut self,
		table: &T,
		rows: &[T::Event],
	) -> Result<(), Error> {
		let log_size = self.optimal_segment_size_heuristic();
		self.fill_sequential_with_segment_size(table, rows, log_size)
	}

	/// Fill a full table witness index using the given row data.
	///
	/// This function iterates through witness segments in parallel in multiple threads.
	pub fn fill_parallel<T>(&mut self, table: &T, rows: &[T::Event]) -> Result<(), Error>
	where
		T: TableFiller<U, F> + Sync,
		T::Event: Sync,
	{
		let log_size = self.optimal_segment_size_heuristic();
		self.fill_parallel_with_segment_size(table, rows, log_size)
	}

	fn optimal_segment_size_heuristic(&self) -> usize {
		// As a heuristic, choose log_size so that the median column segment size is 4 KiB.
		const TARGET_SEGMENT_LOG_BITS: usize = 12 + 3;

		let n_cols = self.table.columns.len();
		let median_col_log_bits = self
			.table
			.columns
			.iter()
			.map(|col| col.shape.log_cell_size())
			.sorted()
			.nth(n_cols / 2)
			.unwrap_or_default();

		TARGET_SEGMENT_LOG_BITS.saturating_sub(median_col_log_bits)
	}

	/// Fill a full table witness index using the given row data.
	///
	/// This function iterates through witness segments sequentially in a single thread.
	pub fn fill_sequential_with_segment_size<T: TableFiller<U, F>>(
		&mut self,
		table: &T,
		rows: &[T::Event],
		log_size: usize,
	) -> Result<(), Error> {
		if rows.len() != self.size {
			return Err(Error::IncorrectNumberOfTableEvents {
				expected: self.size,
				actual: rows.len(),
			});
		}

		let mut segmented_view = TableWitnessSegmentedView::new(self, log_size);

		// Overwrite log_size because it may need to get clamped.
		let log_size = segmented_view.log_segment_size;
		let segment_size = 1 << log_size;

		// rows.len() equals self.size and self.size is check to be non-zero in the constructor
		debug_assert_ne!(rows.len(), 0);
		// number of chunks is rounded up
		let n_chunks = (rows.len() - 1) / segment_size + 1;

		let (full_chunk_segments, mut rest_segments) = segmented_view.split_at(n_chunks - 1);

		// Fill segments of the table with full chunks
		full_chunk_segments
			.into_iter()
			// by taking n_chunks - 1, we guarantee that all row chunks are full
			.zip(rows.chunks(segment_size).take(n_chunks - 1))
			.try_for_each(|(mut witness_segment, row_chunk)| {
				table
					.fill(row_chunk.iter(), &mut witness_segment)
					.map_err(Error::TableFill)
			})?;

		// Fill the last segment. There may not be enough events to match the size of the segment,
		// which is a pre-condition for TableFiller::fill. In that case, we clone the last event to
		// pad the row_chunk. Since it's a clone, the filled witness should satisfy all row-wise
		// constraints as long as all the given events do.
		let row_chunk = &rows[(n_chunks - 1) * segment_size..];
		let mut padded_row_chunk = Vec::new();
		let row_chunk = if row_chunk.len() != segment_size {
			padded_row_chunk.reserve(segment_size);
			padded_row_chunk.extend_from_slice(row_chunk);
			let last_event = row_chunk
				.last()
				.expect("row_chunk must be non-empty because of how n_chunk is calculated")
				.clone();
			padded_row_chunk.resize(segment_size, last_event);
			&padded_row_chunk
		} else {
			row_chunk
		};

		let (partial_chunk_segments, rest_segments) = rest_segments.split_at(1);
		let mut partial_chunk_segment_iter = partial_chunk_segments.into_iter();
		let mut witness_segment = partial_chunk_segment_iter.next().expect(
			"segmented_view.split_at called with 1 must return a view with exactly one segment",
		);
		table
			.fill(row_chunk.iter(), &mut witness_segment)
			.map_err(Error::TableFill)?;
		assert!(partial_chunk_segment_iter.next().is_none());

		// Finally, copy the last filled segment to the remaining segments. This should satisfy all
		// row-wise constraints if the last segment does.
		let last_segment_cols = witness_segment
			.cols
			.iter_mut()
			.map(|col| match col {
				RefCellData::Owned(data) => WitnessColumnInfo::Owned(data.get_mut()),
				RefCellData::SameAsOracleIndex(idx) => WitnessColumnInfo::SameAsOracleIndex(*idx),
			})
			.collect::<Vec<_>>();

		rest_segments.into_iter().for_each(|mut segment| {
			for (dst_col, src_col) in iter::zip(&mut segment.cols, &last_segment_cols) {
				if let (RefCellData::Owned(dst), WitnessColumnInfo::Owned(src)) = (dst_col, src_col)
				{
					dst.get_mut().copy_from_slice(src)
				}
			}
		});

		Ok(())
	}

	/// Fill a full table witness index using the given row data.
	///
	/// This function iterates through witness segments in parallel in multiple threads.
	pub fn fill_parallel_with_segment_size<T>(
		&mut self,
		table: &T,
		rows: &[T::Event],
		log_size: usize,
	) -> Result<(), Error>
	where
		T: TableFiller<U, F> + Sync,
		T::Event: Sync,
	{
		if rows.len() != self.size {
			return Err(Error::IncorrectNumberOfTableEvents {
				expected: self.size,
				actual: rows.len(),
			});
		}

		// This implementation duplicates a lot of code with `fill_sequential_with_segment_size`.
		// We could either refactor to deduplicate or just remove `fill_sequential` once this
		// method is more battle-tested.

		let mut segmented_view = TableWitnessSegmentedView::new(self, log_size);

		// Overwrite log_size because it may need to get clamped.
		let log_size = segmented_view.log_segment_size;
		let segment_size = 1 << log_size;

		// rows.len() equals self.size and self.size is check to be non-zero in the constructor
		debug_assert_ne!(rows.len(), 0);
		// number of chunks is rounded up
		let n_chunks = (rows.len() - 1) / segment_size + 1;

		let (full_chunk_segments, mut rest_segments) = segmented_view.split_at(n_chunks - 1);

		// Fill segments of the table with full chunks
		full_chunk_segments
			.into_par_iter()
			// by taking n_chunks - 1, we guarantee that all row chunks are full
			.zip(rows.par_chunks(segment_size).take(n_chunks - 1))
			.try_for_each(|(mut witness_segment, row_chunk)| {
				table
					.fill(row_chunk.iter(), &mut witness_segment)
					.map_err(Error::TableFill)
			})?;

		// Fill the last segment. There may not be enough events to match the size of the segment,
		// which is a pre-condition for TableFiller::fill. In that case, we clone the last event to
		// pad the row_chunk. Since it's a clone, the filled witness should satisfy all row-wise
		// constraints as long as all the given events do.
		let row_chunk = &rows[(n_chunks - 1) * segment_size..];
		let mut padded_row_chunk = Vec::new();
		let row_chunk = if row_chunk.len() != segment_size {
			padded_row_chunk.reserve(segment_size);
			padded_row_chunk.extend_from_slice(row_chunk);
			let last_event = row_chunk
				.last()
				.expect("row_chunk must be non-empty because of how n_chunk is calculated")
				.clone();
			padded_row_chunk.resize(segment_size, last_event);
			&padded_row_chunk
		} else {
			row_chunk
		};

		let (partial_chunk_segments, rest_segments) = rest_segments.split_at(1);
		let mut partial_chunk_segment_iter = partial_chunk_segments.into_iter();
		let mut witness_segment = partial_chunk_segment_iter.next().expect(
			"segmented_view.split_at called with 1 must return a view with exactly one segment",
		);
		table
			.fill(row_chunk.iter(), &mut witness_segment)
			.map_err(Error::TableFill)?;
		assert!(partial_chunk_segment_iter.next().is_none());

		// Finally, copy the last filled segment to the remaining segments. This should satisfy all
		// row-wise constraints if the last segment does.
		let last_segment_cols = witness_segment
			.cols
			.iter_mut()
			.map(|col| match col {
				RefCellData::Owned(data) => WitnessColumnInfo::Owned(data.get_mut()),
				RefCellData::SameAsOracleIndex(idx) => WitnessColumnInfo::SameAsOracleIndex(*idx),
			})
			.collect::<Vec<_>>();

		rest_segments.into_par_iter().for_each(|mut segment| {
			for (dst_col, src_col) in iter::zip(&mut segment.cols, &last_segment_cols) {
				if let (RefCellData::Owned(dst), WitnessColumnInfo::Owned(src)) = (dst_col, src_col)
				{
					dst.get_mut().copy_from_slice(src)
				}
			}
		});

		Ok(())
	}

	/// Returns an iterator over segments of witness index rows.
	///
	/// This method clamps the segment size, requested as `log_size`, to a minimum of
	/// `self.min_log_segment_size()` and a maximum of `self.log_capacity()`. The actual segment
	/// size can be queried on the items yielded by the iterator.
	pub fn segments(&mut self, log_size: usize) -> impl Iterator<Item = TableWitnessSegment<U, F>> {
		TableWitnessSegmentedView::new(self, log_size).into_iter()
	}

	pub fn par_segments(
		&mut self,
		log_size: usize,
	) -> impl IndexedParallelIterator<Item = TableWitnessSegment<'_, U, F>> {
		TableWitnessSegmentedView::new(self, log_size).into_par_iter()
	}
}

/// A view over a table witness that splits the table into segments.
///
/// The purpose of this struct is to implement the `split_at` method, which safely splits the view
/// of the table witness vertically. This aids in the implementation of `fill_sequential` and
/// `fill_parallel`.
#[derive(Debug)]
struct TableWitnessSegmentedView<'a, U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	table: &'a Table<F>,
	oracle_offset: usize,
	cols: Vec<WitnessColumnInfo<(&'a mut [U], usize)>>,
	log_segment_size: usize,
	n_segments: usize,
}

impl<'a, U: UnderlierType, F: TowerField> TableWitnessSegmentedView<'a, U, F> {
	fn new(witness: &'a mut TableWitnessIndex<U, F>, log_segment_size: usize) -> Self {
		// Clamp the segment size.
		let log_segment_size = log_segment_size
			.min(witness.log_capacity)
			.max(witness.min_log_segment_size);

		let cols = witness
			.cols
			.iter_mut()
			.map(|col| match &mut col.data {
				WitnessColumnInfo::Owned(data) => {
					let chunk_size =
						(log_segment_size + col.shape.log_cell_size()).saturating_sub(U::LOG_BITS);
					WitnessColumnInfo::Owned((&mut **data, 1 << chunk_size))
				}
				WitnessColumnInfo::SameAsOracleIndex(idx) => {
					WitnessColumnInfo::SameAsOracleIndex(*idx)
				}
			})
			.collect::<Vec<_>>();
		Self {
			table: witness.table,
			oracle_offset: witness.oracle_offset,
			cols,
			log_segment_size,
			n_segments: 1 << (witness.log_capacity - log_segment_size),
		}
	}

	fn split_at(
		&mut self,
		index: usize,
	) -> (TableWitnessSegmentedView<U, F>, TableWitnessSegmentedView<U, F>) {
		assert!(index <= self.n_segments);
		let (cols_0, cols_1) = self
			.cols
			.iter_mut()
			.map(|col| match col {
				WitnessColumnInfo::Owned((data, chunk_size)) => {
					let (data_0, data_1) = data.split_at_mut(*chunk_size * index);
					(
						WitnessColumnInfo::Owned((data_0, *chunk_size)),
						WitnessColumnInfo::Owned((data_1, *chunk_size)),
					)
				}
				WitnessColumnInfo::SameAsOracleIndex(idx) => (
					WitnessColumnInfo::SameAsOracleIndex(*idx),
					WitnessColumnInfo::SameAsOracleIndex(*idx),
				),
			})
			.unzip();
		(
			TableWitnessSegmentedView {
				table: self.table,
				oracle_offset: self.oracle_offset,
				cols: cols_0,
				log_segment_size: self.log_segment_size,
				n_segments: index,
			},
			TableWitnessSegmentedView {
				table: self.table,
				oracle_offset: self.oracle_offset,
				cols: cols_1,
				log_segment_size: self.log_segment_size,
				n_segments: self.n_segments - index,
			},
		)
	}

	fn into_iter(self) -> impl Iterator<Item = TableWitnessSegment<'a, U, F>> {
		let TableWitnessSegmentedView {
			table,
			oracle_offset,
			cols,
			log_segment_size,
			n_segments,
		} = self;

		if n_segments == 0 {
			itertools::Either::Left(iter::empty())
		} else {
			let iter = MultiIterator::new(
				cols.into_iter()
					.map(|col| match col {
						WitnessColumnInfo::Owned((data, chunk_size)) => itertools::Either::Left(
							data.chunks_mut(chunk_size)
								.map(|chunk| RefCellData::Owned(RefCell::new(chunk))),
						),
						WitnessColumnInfo::SameAsOracleIndex(index) => itertools::Either::Right(
							iter::repeat_n(index, n_segments).map(RefCellData::SameAsOracleIndex),
						),
					})
					.collect(),
			)
			.map(move |cols| TableWitnessSegment {
				table,
				cols,
				log_size: log_segment_size,
				oracle_offset,
			});
			itertools::Either::Right(iter)
		}
	}

	fn into_par_iter(self) -> impl IndexedParallelIterator<Item = TableWitnessSegment<'a, U, F>> {
		let TableWitnessSegmentedView {
			table,
			oracle_offset,
			cols,
			log_segment_size,
			n_segments: _,
		} = self;

		// This implementation uses unsafe code to iterate the segments of the view. A fully safe
		// implementation is also possible, which would look similar to that of
		// `rayon::slice::ChunksMut`. That just requires more code and doesn't seem justified.

		// TODO: clippy error (clippy::mut_from_ref): mutable borrow from immutable input(s)
		#[allow(clippy::mut_from_ref)]
		unsafe fn cast_slice_ref_to_mut<T>(slice: &[T]) -> &mut [T] {
			slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len())
		}

		// Convert cols with mutable references into cols with const refs so that they can be
		// cloned. Within the loop, we unsafely cast back to mut refs.
		let cols = cols
			.into_iter()
			.map(|col| -> WitnessColumnInfo<(&'a [U], usize)> {
				match col {
					WitnessColumnInfo::Owned((data, chunk_size)) => {
						WitnessColumnInfo::Owned((data, chunk_size))
					}
					WitnessColumnInfo::SameAsOracleIndex(index) => {
						WitnessColumnInfo::SameAsOracleIndex(index)
					}
				}
			})
			.collect::<Vec<_>>();

		(0..self.n_segments).into_par_iter().map(move |i| {
			let col_strides = cols
				.iter()
				.map(|col| match col {
					WitnessColumnInfo::SameAsOracleIndex(index) => {
						RefCellData::SameAsOracleIndex(*index)
					}
					WitnessColumnInfo::Owned((data, chunk_size)) => {
						RefCellData::Owned(RefCell::new(unsafe {
							// Safety: The function borrows self mutably, so we have mutable access
							// to all columns and thus none can be borrowed by anyone else. The
							// loop is constructed to borrow disjoint segments of each column -- if
							// this loop were transposed, we would use `chunks_mut`.
							cast_slice_ref_to_mut(&data[i * chunk_size..(i + 1) * chunk_size])
						}))
					}
				})
				.collect();
			TableWitnessSegment {
				table,
				cols: col_strides,
				log_size: log_segment_size,
				oracle_offset,
			}
		})
	}
}

/// A vertical segment of a table witness index.
///
/// This provides runtime-checked access to slices of the witness columns. This is used separately
/// from [`TableWitnessIndex`] so that witness population can be parallelized over segments.
#[derive(Debug, CopyGetters)]
pub struct TableWitnessSegment<'a, U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	table: &'a Table<F>,
	cols: Vec<RefCellData<'a, U>>,
	#[get_copy = "pub"]
	log_size: usize,
	oracle_offset: usize,
}

impl<'a, U: UnderlierType, F: TowerField> TableWitnessSegment<'a, U, F> {
	pub fn get<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[PackedType<U, FSub>]>, Error>
	where
		U: PackScalar<FSub>,
	{
		if col.table_id != self.table.id() {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table.id(),
			});
		}

		let col = self
			.get_col_data(col.table_index)
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		Ok(Ref::map(col_ref, |x| <PackedType<U, FSub>>::from_underliers_ref(x)))
	}

	pub fn get_mut<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<RefMut<[PackedType<U, FSub>]>, Error>
	where
		U: PackScalar<FSub>,
		F: ExtensionField<FSub>,
	{
		if col.table_id != self.table.id() {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table.id(),
			});
		}

		let col = self
			.get_col_data(col.table_index)
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |x| <PackedType<U, FSub>>::from_underliers_ref_mut(x)))
	}

	pub fn get_scalars<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[FSub]>, Error>
	where
		U: PackScalar<FSub>,
		F: ExtensionField<FSub>,
		PackedType<U, FSub>: PackedFieldIndexable,
	{
		self.get(col)
			.map(|packed| Ref::map(packed, <PackedType<U, FSub>>::unpack_scalars))
	}

	pub fn get_scalars_mut<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<RefMut<[FSub]>, Error>
	where
		U: PackScalar<FSub>,
		F: ExtensionField<FSub>,
		PackedType<U, FSub>: PackedFieldIndexable,
	{
		self.get_mut(col)
			.map(|packed| RefMut::map(packed, <PackedType<U, FSub>>::unpack_scalars_mut))
	}

	pub fn get_as<T: Pod, FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[T]>, Error>
	where
		U: Pod,
		F: ExtensionField<FSub>,
	{
		let col = self
			.get_col_data(col.table_index)
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		Ok(Ref::map(col_ref, |x| must_cast_slice(x)))
	}

	pub fn get_mut_as<T: Pod, FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<RefMut<[T]>, Error>
	where
		U: Pod,
		F: ExtensionField<FSub>,
	{
		if col.table_id != self.table.id() {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table.id(),
			});
		}

		let col = self
			.get_col_data(col.table_index)
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |x| must_cast_slice_mut(x)))
	}

	/// Evaluate an expression over columns that are assumed to be already populated.
	///
	/// This function evaluates an expression over the columns in the segment and returns an
	/// iterator over the packed elements. This borrows the columns segments it reads from, so
	/// they must not be borrowed mutably elsewhere (which is possible due to runtime-checked
	/// column borrowing).
	pub fn eval_expr<FSub: TowerField, const V: usize>(
		&self,
		expr: &Expr<FSub, V>,
	) -> Result<impl Iterator<Item = PackedType<U, FSub>>, Error>
	where
		U: PackScalar<FSub>,
		F: ExtensionField<FSub>,
	{
		let log_vals_per_row = checked_log_2(V);

		let partition =
			self.table
				.partitions
				.get(log_vals_per_row)
				.ok_or_else(|| Error::MissingPartition {
					table_id: self.table.id(),
					log_vals_per_row,
				})?;
		let col_refs = partition
			.columns
			.iter()
			.zip(expr.expr().vars_usage())
			.map(|(col_index, used)| {
				used.then(|| {
					self.get(Col::<FSub, V>::new(
						ColumnId {
							table_id: self.table.id(),
							table_index: partition.columns[*col_index],
						},
						*col_index,
					))
				})
				.transpose()
			})
			.collect::<Result<Vec<_>, _>>()?;

		let log_packed_elems =
			(self.log_size + log_vals_per_row).saturating_sub(<PackedType<U, FSub>>::LOG_WIDTH);

		// Batch evaluate requires value slices even for the indices it will not read.
		let dummy_col = zeroed_vec(1 << log_packed_elems);

		let cols = col_refs
			.iter()
			.map(|col| col.as_ref().map(|col_ref| &**col_ref).unwrap_or(&dummy_col))
			.collect::<Vec<_>>();
		let cols = RowsBatchRef::new(&cols, 1 << log_packed_elems);

		// REVIEW: This could be inefficient with very large segments because batch evaluation
		// allocates more memory, proportional to the size of the segment. Because of how segments
		// get split up in practice, it's not a problem yet. If we see stack overflows, we should
		// split up the evaluation into multiple batches.
		let mut evals = zeroed_vec(1 << log_packed_elems);
		ArithCircuitPoly::new(expr.expr().clone()).batch_evaluate(&cols, &mut evals)?;
		Ok(evals.into_iter())
	}

	pub fn size(&self) -> usize {
		1 << self.log_size
	}

	fn get_col_data(&self, table_index: ColumnIndex) -> Option<&RefCell<&'a mut [U]>> {
		self.get_col_data_by_oracle_offset(self.oracle_offset + table_index)
	}

	fn get_col_data_by_oracle_offset(&self, oracle_id: OracleId) -> Option<&RefCell<&'a mut [U]>> {
		match self.cols.get(oracle_id) {
			Some(RefCellData::Owned(data)) => Some(data),
			Some(RefCellData::SameAsOracleIndex(id)) => self.get_col_data_by_oracle_offset(*id),
			None => None,
		}
	}
}

/// A struct that can populate segments of a table witness using row descriptors.
pub trait TableFiller<U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	/// A struct that specifies the row contents.
	type Event: Clone;

	/// Returns the table ID.
	fn id(&self) -> TableId;

	/// Fill the table witness with data derived from the given rows.
	///
	/// ## Preconditions
	///
	/// * the number of elements in `rows` must equal `witness.size()`
	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event> + Clone,
		witness: &'a mut TableWitnessSegment<U, F>,
	) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use assert_matches::assert_matches;
	use binius_field::{
		arch::{OptimalUnderlier128b, OptimalUnderlier256b},
		packed::{len_packed_slice, set_packed_slice},
	};
	use rand::{rngs::StdRng, Rng, SeedableRng};

	use super::*;
	use crate::builder::{
		types::{B1, B32, B8},
		ConstraintSystem, Statement, TableBuilder,
	};

	#[test]
	fn test_table_witness_borrows() {
		let table_id = 0;
		let mut inner_table = Table::<B128>::new(table_id, "table".to_string());
		let mut table = TableBuilder::new(&mut inner_table);
		let col0 = table.add_committed::<B1, 8>("col0");
		let col1 = table.add_committed::<B1, 32>("col1");
		let col2 = table.add_committed::<B8, 1>("col2");
		let col3 = table.add_committed::<B32, 1>("col3");

		let allocator = bumpalo::Bump::new();
		let table_size = 64;
		let mut index =
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &inner_table, table_size)
				.unwrap();
		let segment = index.full_segment();

		{
			let col0_ref0 = segment.get(col0).unwrap();
			let _col0_ref1 = segment.get(col0).unwrap();
			assert_matches!(segment.get_mut(col0), Err(Error::WitnessBorrowMut(_)));
			drop(col0_ref0);

			let col1_ref = segment.get_mut(col1).unwrap();
			assert_matches!(segment.get(col1), Err(Error::WitnessBorrow(_)));
			drop(col1_ref);
		}

		assert_eq!(len_packed_slice(&segment.get_mut(col0).unwrap()), 1 << 9);
		assert_eq!(len_packed_slice(&segment.get_mut(col1).unwrap()), 1 << 11);
		assert_eq!(len_packed_slice(&segment.get_mut(col2).unwrap()), 1 << 6);
		assert_eq!(len_packed_slice(&segment.get_mut(col3).unwrap()), 1 << 6);
	}

	#[test]
	fn test_table_witness_segments() {
		let table_id = 0;
		let mut inner_table = Table::<B128>::new(table_id, "table".to_string());
		let mut table = TableBuilder::new(&mut inner_table);
		let col0 = table.add_committed::<B1, 8>("col0");
		let col1 = table.add_committed::<B1, 32>("col1");
		let col2 = table.add_committed::<B8, 1>("col2");
		let col3 = table.add_committed::<B32, 1>("col3");

		let allocator = bumpalo::Bump::new();
		let table_size = 64;
		let mut index =
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &inner_table, table_size)
				.unwrap();

		assert_eq!(index.min_log_segment_size(), 4);
		let mut iter = index.segments(5);
		let seg0 = iter.next().unwrap();
		let seg1 = iter.next().unwrap();
		assert!(iter.next().is_none());

		assert_eq!(len_packed_slice(&seg0.get_mut(col0).unwrap()), 1 << 8);
		assert_eq!(len_packed_slice(&seg0.get_mut(col1).unwrap()), 1 << 10);
		assert_eq!(len_packed_slice(&seg0.get_mut(col2).unwrap()), 1 << 5);
		assert_eq!(len_packed_slice(&seg0.get_mut(col3).unwrap()), 1 << 5);

		assert_eq!(len_packed_slice(&seg1.get_mut(col0).unwrap()), 1 << 8);
		assert_eq!(len_packed_slice(&seg1.get_mut(col1).unwrap()), 1 << 10);
		assert_eq!(len_packed_slice(&seg1.get_mut(col2).unwrap()), 1 << 5);
		assert_eq!(len_packed_slice(&seg1.get_mut(col3).unwrap()), 1 << 5);
	}

	#[test]
	fn test_eval_expr() {
		let table_id = 0;
		let mut inner_table = Table::<B128>::new(table_id, "table".to_string());
		let mut table = TableBuilder::new(&mut inner_table);
		let col0 = table.add_committed::<B8, 2>("col0");
		let col1 = table.add_committed::<B8, 2>("col0");
		let col2 = table.add_committed::<B8, 2>("col0");

		let allocator = bumpalo::Bump::new();
		let table_size = 1 << 6;
		let mut index =
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &inner_table, table_size)
				.unwrap();

		let segment = index.full_segment();
		assert_eq!(segment.log_size(), 6);

		// Fill the columns with a deterministic pattern.
		{
			let mut col0 = segment.get_mut(col0).unwrap();
			let mut col1 = segment.get_mut(col1).unwrap();
			let mut col2 = segment.get_mut(col2).unwrap();

			// 3 = 6 (log table size) + 1 (log values per row) - 4 (log packed field width)
			let expected_slice_len = 1 << 3;
			assert_eq!(col0.len(), expected_slice_len);
			assert_eq!(col0.len(), expected_slice_len);
			assert_eq!(col0.len(), expected_slice_len);

			for i in 0..expected_slice_len * <PackedType<OptimalUnderlier128b, B8>>::WIDTH {
				set_packed_slice(&mut col0, i, B8::new(i as u8) + B8::new(0x00));
				set_packed_slice(&mut col1, i, B8::new(i as u8) + B8::new(0x40));
				set_packed_slice(&mut col2, i, B8::new(i as u8) + B8::new(0x80));
			}
		}

		let evals = segment.eval_expr(&(col0 * col1 - col2)).unwrap();
		for (i, eval_i) in evals
			.into_iter()
			.flat_map(PackedField::into_iter)
			.enumerate()
		{
			let col0_val = B8::new(i as u8) + B8::new(0x00);
			let col1_val = B8::new(i as u8) + B8::new(0x40);
			let col2_val = B8::new(i as u8) + B8::new(0x80);
			assert_eq!(eval_i, col0_val * col1_val - col2_val);
		}
	}

	#[test]
	fn test_small_tables() {
		let table_id = 0;
		let mut inner_table = Table::<B128>::new(table_id, "table".to_string());
		let mut table = TableBuilder::new(&mut inner_table);
		let _col0 = table.add_committed::<B1, 8>("col0");
		let _col1 = table.add_committed::<B1, 32>("col1");
		let _col2 = table.add_committed::<B8, 1>("col2");
		let _col3 = table.add_committed::<B32, 1>("col3");

		let allocator = bumpalo::Bump::new();
		let table_size = 7;
		let mut index =
			TableWitnessIndex::<OptimalUnderlier256b>::new(&allocator, &inner_table, table_size)
				.unwrap();

		assert_eq!(index.log_capacity(), 4);
		assert_eq!(index.min_log_segment_size(), 4);

		let mut iter = index.segments(5);
		// Check that the segment size is clamped to the capacity.
		assert_eq!(iter.next().unwrap().log_size(), 4);
		assert!(iter.next().is_none());
		drop(iter);

		let mut iter = index.segments(2);
		// Check that the segment size is clamped to the minimum segment size.
		assert_eq!(iter.next().unwrap().log_size(), 4);
		assert!(iter.next().is_none());
		drop(iter);
	}

	struct TestTable {
		id: TableId,
		col0: Col<B32>,
		col1: Col<B32>,
	}

	impl TestTable {
		fn new(cs: &mut ConstraintSystem) -> Self {
			let mut table = cs.add_table("test");

			let col0 = table.add_committed("col0");
			let col1 = table.add_computed("col1", col0 * col0 + B32::new(0x03));

			Self {
				id: table.id(),
				col0,
				col1,
			}
		}
	}

	impl TableFiller<OptimalUnderlier128b> for TestTable {
		type Event = u32;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event> + Clone,
			witness: &'a mut TableWitnessSegment<OptimalUnderlier128b>,
		) -> anyhow::Result<()> {
			let mut col0 = witness.get_scalars_mut(self.col0)?;
			let mut col1 = witness.get_scalars_mut(self.col1)?;
			for (i, &val) in rows.enumerate() {
				col0[i] = B32::new(val);
				col1[i] = col0[i].pow(2) + B32::new(0x03);
			}
			Ok(())
		}
	}

	#[test]
	fn test_fill_sequential_with_incomplete_events() {
		let mut cs = ConstraintSystem::new();
		let test_table = TestTable::new(&mut cs);

		let allocator = bumpalo::Bump::new();

		let table_size = 11;
		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![table_size],
		};
		let mut index = cs.build_witness(&allocator, &statement).unwrap();
		let table_index = index.get_table(test_table.id()).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let rows = repeat_with(|| rng.gen())
			.take(table_size)
			.collect::<Vec<_>>();

		// 2^4 is the next power of two after 1, the table size.
		assert_eq!(table_index.log_capacity(), 4);

		// 2^2 B32 values fit into a 128-bit underlier.
		assert_eq!(table_index.min_log_segment_size(), 2);

		// Assert that fill_sequential validates the number of events..
		assert_matches!(
			table_index.fill_sequential_with_segment_size(&test_table, &rows[1..], 2),
			Err(Error::IncorrectNumberOfTableEvents { .. })
		);

		table_index
			.fill_sequential_with_segment_size(&test_table, &rows, 2)
			.unwrap();

		let segment = table_index.full_segment();
		let col0 = segment.get_scalars(test_table.col0).unwrap();
		for i in 0..11 {
			assert_eq!(col0[i].val(), rows[i]);
		}
		assert_eq!(col0[11].val(), rows[10]);
		assert_eq!(col0[12].val(), rows[8]);
		assert_eq!(col0[13].val(), rows[9]);
		assert_eq!(col0[14].val(), rows[10]);
		assert_eq!(col0[15].val(), rows[10]);
	}

	#[test]
	fn test_fill_parallel_with_incomplete_events() {
		let mut cs = ConstraintSystem::new();
		let test_table = TestTable::new(&mut cs);

		let allocator = bumpalo::Bump::new();

		let table_size = 11;
		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![table_size],
		};
		let mut index = cs.build_witness(&allocator, &statement).unwrap();
		let table_index = index.get_table(test_table.id()).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let rows = repeat_with(|| rng.gen())
			.take(table_size)
			.collect::<Vec<_>>();

		// 2^4 is the next power of two after 1, the table size.
		assert_eq!(table_index.log_capacity(), 4);

		// 2^2 B32 values fit into a 128-bit underlier.
		assert_eq!(table_index.min_log_segment_size(), 2);

		// Assert that fill_sequential validates the number of events..
		assert_matches!(
			table_index.fill_parallel_with_segment_size(&test_table, &rows[1..], 2),
			Err(Error::IncorrectNumberOfTableEvents { .. })
		);

		table_index
			.fill_parallel_with_segment_size(&test_table, &rows, 2)
			.unwrap();

		let segment = table_index.full_segment();
		let col0 = segment.get_scalars(test_table.col0).unwrap();
		for i in 0..11 {
			assert_eq!(col0[i].val(), rows[i]);
		}
		assert_eq!(col0[11].val(), rows[10]);
		assert_eq!(col0[12].val(), rows[8]);
		assert_eq!(col0[13].val(), rows[9]);
		assert_eq!(col0[14].val(), rows[10]);
		assert_eq!(col0[15].val(), rows[10]);
	}
}
