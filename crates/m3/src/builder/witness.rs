// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter,
	ops::{Deref, DerefMut},
	slice,
	sync::Arc,
};

use binius_compute::alloc::{ComputeAllocator, HostBumpAllocator};
use binius_core::witness::{MultilinearExtensionIndex, MultilinearWitness};
use binius_fast_compute::arith_circuit::ArithCircuitPoly;
use binius_field::{
	ExtensionField, PackedExtension, PackedField, PackedFieldIndexable, PackedSubfield, TowerField,
	arch::OptimalUnderlier,
	as_packed_field::PackedType,
	packed::{get_packed_slice, set_packed_slice},
};
use binius_math::{
	ArithCircuit, CompositionPoly, MultilinearExtension, MultilinearPoly, RowsBatchRef,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{Pod, must_cast_slice, must_cast_slice_mut, zeroed_vec};
use either::Either;
use getset::CopyGetters;
use itertools::Itertools;

use super::{
	ColumnDef, ColumnId, ColumnInfo, ConstraintSystem, Expr,
	column::{Col, ColumnShape},
	constraint_system::OracleMapping,
	error::Error,
	table::{self, Table, TableId},
	types::{B1, B8, B16, B32, B64, B128},
};
use crate::builder::multi_iter::MultiIterator;

/// Holds witness column data for all tables in a constraint system, indexed by column ID.
///
/// The struct has two lifetimes: `'cs` is the lifetime of the constraint system, and `'alloc` is
/// the lifetime of the bump allocator. The reason these must be separate is that the witness index
/// gets converted into a multilinear extension index, which maintains references to the data
/// allocated by the allocator, but does not need to maintain a reference to the constraint system,
/// which can then be dropped.
pub struct WitnessIndex<'cs, 'alloc, P = PackedType<OptimalUnderlier, B128>>
where
	P: PackedField,
	P::Scalar: TowerField,
{
	cs: &'cs ConstraintSystem<P::Scalar>,
	allocator: &'alloc HostBumpAllocator<'alloc, P>,
	/// Each entry is Left if the index hasn't been initialized & filled, and Right if it has.
	tables: Vec<Either<&'cs Table<P::Scalar>, TableWitnessIndex<'cs, 'alloc, P>>>,
}

impl<'cs, 'alloc, F: TowerField, P: PackedField<Scalar = F>> WitnessIndex<'cs, 'alloc, P> {
	/// Creates and allocates the witness index for a constraint system.
	pub fn new(
		cs: &'cs ConstraintSystem<F>,
		allocator: &'alloc HostBumpAllocator<'alloc, P>,
	) -> Self {
		Self {
			cs,
			allocator,
			tables: cs.tables.iter().map(Either::Left).collect(),
		}
	}

	pub fn init_table(
		&mut self,
		table_id: TableId,
		size: usize,
	) -> Result<&mut TableWitnessIndex<'cs, 'alloc, P>, Error> {
		match self.tables.get_mut(table_id) {
			Some(entry) => match entry {
				Either::Left(table) => {
					if size == 0 {
						Err(Error::EmptyTable { table_id })
					} else {
						let table_witness = TableWitnessIndex::new(self.allocator, table, size)?;
						*entry = Either::Right(table_witness);
						let Either::Right(table_witness) = entry else {
							unreachable!("entry is assigned to this pattern on the previous line")
						};
						Ok(table_witness)
					}
				}
				Either::Right(_) => Err(Error::TableIndexAlreadyInitialized { table_id }),
			},
			None => Err(Error::MissingTable { table_id }),
		}
	}

	pub fn get_table(
		&mut self,
		table_id: TableId,
	) -> Option<&mut TableWitnessIndex<'cs, 'alloc, P>> {
		self.tables.get_mut(table_id).and_then(|table| match table {
			Either::Left(_) => None,
			Either::Right(index) => Some(index),
		})
	}

	pub fn fill_table_sequential<T: TableFiller<P>>(
		&mut self,
		filler: &T,
		rows: &[T::Event],
	) -> Result<(), Error> {
		self.init_and_fill_table(
			filler.id(),
			|table_witness, rows| table_witness.fill_sequential(filler, rows),
			rows,
		)
	}

	pub fn fill_table_parallel<T>(&mut self, filler: &T, rows: &[T::Event]) -> Result<(), Error>
	where
		T: TableFiller<P> + Sync,
		T::Event: Sync,
	{
		self.init_and_fill_table(
			filler.id(),
			|table_witness, rows| table_witness.fill_parallel(filler, rows),
			rows,
		)
	}

	fn init_and_fill_table<Event>(
		&mut self,
		table_id: TableId,
		fill: impl FnOnce(&mut TableWitnessIndex<'cs, 'alloc, P>, &[Event]) -> Result<(), Error>,
		rows: &[Event],
	) -> Result<(), Error> {
		match self.tables.get_mut(table_id) {
			Some(entry) => match entry {
				Either::Right(witness) => fill(witness, rows),
				Either::Left(table) => {
					if rows.is_empty() {
						Ok(())
					} else {
						let mut table_witness =
							TableWitnessIndex::new(self.allocator, table, rows.len())?;
						fill(&mut table_witness, rows)?;
						*entry = Either::Right(table_witness);
						Ok(())
					}
				}
			},
			None => Err(Error::MissingTable { table_id }),
		}
	}

	/// Returns the sizes of all tables in the witness, indexed by table ID.
	pub fn table_sizes(&self) -> Vec<usize> {
		self.tables
			.iter()
			.map(|entry| match entry {
				Either::Left(_) => 0,
				Either::Right(index) => index.size(),
			})
			.collect()
	}

	fn mk_column_witness<'a>(
		log_capacity: usize,
		shape: ColumnShape,
		data: &'a [P],
	) -> MultilinearWitness<'a, P>
	where
		P: PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B16>
			+ PackedExtension<B32>
			+ PackedExtension<B64>
			+ PackedExtension<B128>,
	{
		let n_vars = log_capacity + shape.log_values_per_row;
		let underlier_count =
			1 << (n_vars + shape.tower_height).saturating_sub(P::LOG_WIDTH + F::TOWER_LEVEL);
		multilin_poly_from_underlier_data(&data[..underlier_count], n_vars, shape.tower_height)
	}

	/// Converts this witness into binius_core's [`MultilinearExtensionIndex`].
	///
	/// Note that this function must be called only after the [`ConstraintSystem::compile`].
	pub fn into_multilinear_extension_index(self) -> MultilinearExtensionIndex<'alloc, P>
	where
		P: PackedExtension<B1>
			+ PackedExtension<B8>
			+ PackedExtension<B16>
			+ PackedExtension<B32>
			+ PackedExtension<B64>
			+ PackedExtension<B128>,
	{
		let oracle_lookup = self.cs.oracle_lookup();

		let mut index = MultilinearExtensionIndex::new();

		for table_witness in self.tables {
			let Either::Right(table_witness) = table_witness else {
				continue;
			};
			let cols = immutable_witness_index_columns(table_witness.cols);

			// Here our objective is to add a witness for every oracle the table has created.
			//
			// There are some tricky parts that is worth keeping in mind:
			//
			// 1. Some oracles share witnesses, e.g. packed column has the same witness as the
			//    column that it packs. Despite that they cannot share the underlying witness
			//    polynomial because of the difference in n_vars.
			//
			// 2. Similarly, the constant column creates two oracles: the original constant and the
			//    the user-visible one, repeating column. Instead of making the user to fill both
			//    witnesses, we fill the original constant oracle with the truncated version of the
			//    repeating column.
			//

			for col in cols.into_iter() {
				let oracle_mapping = *oracle_lookup.lookup(col.column_id);
				match oracle_mapping {
					OracleMapping::Regular(oracle_id) => index
						.update_multilin_poly([(
							oracle_id,
							Self::mk_column_witness(
								table_witness.log_capacity,
								col.shape,
								col.data,
							),
						)])
						.unwrap(),
					OracleMapping::TransparentCompound {
						original,
						repeating,
					} => {
						// Create a single row poly witness for the original oracle and the
						// repeating version of that for the repeating oracle.
						let original_witness = Self::mk_column_witness(0, col.shape, col.data);
						let repeating_witness = Self::mk_column_witness(
							table_witness.log_capacity,
							col.shape,
							col.data,
						);
						index
							.update_multilin_poly([
								(original, original_witness),
								(repeating, repeating_witness),
							])
							.unwrap();
					}
				}
			}
		}
		index
	}
}

impl<'cs, 'alloc, P> WitnessIndex<'cs, 'alloc, P>
where
	P: PackedField<Scalar: TowerField>
		+ PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B16>
		+ PackedExtension<B32>
		+ PackedExtension<B64>
		+ PackedExtension<B128>,
{
	/// Automatically populate the witness data for all the constant columns in all the tables with
	/// a [`TableWitnessIndex<P>`].
	pub fn fill_constant_cols(&mut self) -> Result<(), Error> {
		for table in self.tables.iter_mut() {
			match table.as_mut() {
				Either::Left(_) => (),
				// If we have witness index data, populate the witness
				Either::Right(table_witness_index) => {
					let table = table_witness_index.table();
					let segment = table_witness_index.full_segment();
					for col in table.columns.iter() {
						if let ColumnDef::Constant { data, .. } = &col.col {
							let mut witness_data = segment.get_dyn_mut(col.id)?;
							let len = witness_data.size();
							for (i, scalar) in data.iter().cycle().take(len).enumerate() {
								witness_data.set(i, *scalar)?
							}
						}
					}
				}
			}
		}
		Ok(())
	}
}

fn multilin_poly_from_underlier_data<P>(
	data: &[P],
	n_vars: usize,
	tower_height: usize,
) -> Arc<dyn MultilinearPoly<P> + Send + Sync + '_>
where
	P: PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B16>
		+ PackedExtension<B32>
		+ PackedExtension<B64>
		+ PackedExtension<B128>,
{
	match tower_height {
		0 => MultilinearExtension::new(n_vars, PackedExtension::<B1>::cast_bases(data))
			.unwrap()
			.specialize_arc_dyn(),
		3 => MultilinearExtension::new(n_vars, PackedExtension::<B8>::cast_bases(data))
			.unwrap()
			.specialize_arc_dyn(),
		4 => MultilinearExtension::new(n_vars, PackedExtension::<B16>::cast_bases(data))
			.unwrap()
			.specialize_arc_dyn(),
		5 => MultilinearExtension::new(n_vars, PackedExtension::<B32>::cast_bases(data))
			.unwrap()
			.specialize_arc_dyn(),
		6 => MultilinearExtension::new(n_vars, PackedExtension::<B64>::cast_bases(data))
			.unwrap()
			.specialize_arc_dyn(),
		7 => MultilinearExtension::new(n_vars, PackedExtension::<B128>::cast_bases(data))
			.unwrap()
			.specialize_arc_dyn(),
		_ => {
			panic!("Unsupported tower height: {tower_height}");
		}
	}
}

/// Holds witness column data for a table, indexed by column index.
#[derive(Debug, CopyGetters)]
pub struct TableWitnessIndex<'cs, 'alloc, P = PackedType<OptimalUnderlier, B128>>
where
	P: PackedField,
	P::Scalar: TowerField,
{
	#[get_copy = "pub"]
	table: &'cs Table<P::Scalar>,
	cols: Vec<WitnessIndexColumn<'alloc, P>>,
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
pub struct WitnessIndexColumn<'a, P: PackedField> {
	shape: ColumnShape,
	data: WitnessDataMut<'a, P>,
	column_id: ColumnId,
}

#[derive(Debug, Clone)]
enum WitnessColumnInfo<T> {
	Owned(T),
	/// This column is same as the column stored in `cols[.0]`.
	SameAsIndex(usize),
}

type WitnessDataMut<'a, P> = WitnessColumnInfo<&'a mut [P]>;

impl<'a, P: PackedField> WitnessDataMut<'a, P> {
	pub fn new_owned(allocator: &'a HostBumpAllocator<'a, P>, log_underlier_count: usize) -> Self {
		let slice = allocator
			.alloc(1 << log_underlier_count)
			.expect("failed to allocate witness data slice");

		Self::Owned(slice)
	}
}

type RefCellData<'a, P> = WitnessColumnInfo<RefCell<&'a mut [P]>>;

#[derive(Debug)]
struct ImmutableWitnessIndexColumn<'a, P: PackedField> {
	shape: ColumnShape,
	data: &'a [P],
	column_id: ColumnId,
}

/// Converts the vector of witness columns into immutable references to column data that may be
/// shared.
fn immutable_witness_index_columns<P: PackedField>(
	cols: Vec<WitnessIndexColumn<P>>,
) -> Vec<ImmutableWitnessIndexColumn<P>> {
	let mut result = Vec::<ImmutableWitnessIndexColumn<_>>::with_capacity(cols.len());
	for col in cols {
		result.push(ImmutableWitnessIndexColumn {
			shape: col.shape,
			data: match col.data {
				WitnessDataMut::Owned(data) => data,
				WitnessDataMut::SameAsIndex(index) => result[index].data,
			},
			column_id: col.column_id,
		});
	}
	result
}

impl<'cs, 'alloc, F: TowerField, P: PackedField<Scalar = F>> TableWitnessIndex<'cs, 'alloc, P> {
	pub(crate) fn new(
		allocator: &'alloc HostBumpAllocator<'alloc, P>,
		table: &'cs Table<F>,
		size: usize,
	) -> Result<Self, Error> {
		if size == 0 {
			return Err(Error::EmptyTable {
				table_id: table.id(),
			});
		}

		let log_capacity = table::log_capacity(size);
		let packed_elem_log_bits = P::LOG_WIDTH + F::TOWER_LEVEL;

		let mut cols = Vec::with_capacity(table.columns.len());
		for ColumnInfo { id, col, shape, .. } in &table.columns {
			let data: WitnessDataMut<P> = if let ColumnDef::Packed { col: source, .. } = col {
				// Packed column reuses the witness of the one it is based on.
				WitnessDataMut::SameAsIndex(source.table_index.0)
			} else {
				// Everything else has it's own column.
				WitnessDataMut::new_owned(
					allocator,
					(shape.log_cell_size() + log_capacity).saturating_sub(packed_elem_log_bits),
				)
			};
			cols.push(WitnessIndexColumn {
				shape: *shape,
				data,
				column_id: *id,
			});
		}

		// The minimum segment size is chosen such that the segment of each column is at least one
		// underlier in size.
		let min_log_segment_size = packed_elem_log_bits
			- table
				.columns
				.iter()
				.map(|col| col.shape.log_cell_size())
				.fold(packed_elem_log_bits, |a, b| a.min(b));

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
		})
	}

	pub fn table_id(&self) -> TableId {
		self.table.id
	}

	pub fn capacity(&self) -> usize {
		1 << self.log_capacity
	}

	/// Returns a witness index segment covering the entire table.
	pub fn full_segment(&mut self) -> TableWitnessSegment<P> {
		let cols = self
			.cols
			.iter_mut()
			.map(|col| match &mut col.data {
				WitnessDataMut::SameAsIndex(id) => RefCellData::SameAsIndex(*id),
				WitnessDataMut::Owned(data) => RefCellData::Owned(RefCell::new(data)),
			})
			.collect();
		TableWitnessSegment {
			table: self.table,
			cols,
			log_size: self.log_capacity,
			index: 0,
		}
	}

	/// Fill a full table witness index using the given row data.
	///
	/// This function iterates through witness segments sequentially in a single thread.
	pub fn fill_sequential<T: TableFiller<P>>(
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
		T: TableFiller<P> + Sync,
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
	pub fn fill_sequential_with_segment_size<T: TableFiller<P>>(
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
				RefCellData::SameAsIndex(idx) => WitnessColumnInfo::SameAsIndex(*idx),
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
		T: TableFiller<P> + Sync,
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
				RefCellData::SameAsIndex(idx) => WitnessColumnInfo::SameAsIndex(*idx),
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
	pub fn segments(&mut self, log_size: usize) -> impl Iterator<Item = TableWitnessSegment<P>> {
		TableWitnessSegmentedView::new(self, log_size).into_iter()
	}

	pub fn par_segments(
		&mut self,
		log_size: usize,
	) -> impl IndexedParallelIterator<Item = TableWitnessSegment<'_, P>> {
		TableWitnessSegmentedView::new(self, log_size).into_par_iter()
	}
}

/// A view over a table witness that splits the table into segments.
///
/// The purpose of this struct is to implement the `split_at` method, which safely splits the view
/// of the table witness vertically. This aids in the implementation of `fill_sequential` and
/// `fill_parallel`.
#[derive(Debug)]
struct TableWitnessSegmentedView<'a, P = PackedType<OptimalUnderlier, B128>>
where
	P: PackedField,
	P::Scalar: TowerField,
{
	table: &'a Table<P::Scalar>,
	cols: Vec<WitnessColumnInfo<(&'a mut [P], usize)>>,
	log_segment_size: usize,
	start_index: usize,
	n_segments: usize,
}

impl<'a, F: TowerField, P: PackedField<Scalar = F>> TableWitnessSegmentedView<'a, P> {
	fn new(witness: &'a mut TableWitnessIndex<P>, log_segment_size: usize) -> Self {
		// Clamp the segment size.
		let log_segment_size = log_segment_size
			.min(witness.log_capacity)
			.max(witness.min_log_segment_size);

		let cols = witness
			.cols
			.iter_mut()
			.map(|col| match &mut col.data {
				WitnessColumnInfo::Owned(data) => {
					let chunk_size = (log_segment_size + col.shape.log_cell_size())
						.saturating_sub(P::LOG_WIDTH + F::TOWER_LEVEL);
					WitnessColumnInfo::Owned((&mut **data, 1 << chunk_size))
				}
				WitnessColumnInfo::SameAsIndex(id) => WitnessColumnInfo::SameAsIndex(*id),
			})
			.collect::<Vec<_>>();
		Self {
			table: witness.table,
			cols,
			log_segment_size,
			start_index: 0,
			n_segments: 1 << (witness.log_capacity - log_segment_size),
		}
	}

	fn split_at(
		&mut self,
		index: usize,
	) -> (TableWitnessSegmentedView<P>, TableWitnessSegmentedView<P>) {
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
				WitnessColumnInfo::SameAsIndex(id) => {
					(WitnessColumnInfo::SameAsIndex(*id), WitnessColumnInfo::SameAsIndex(*id))
				}
			})
			.unzip();
		(
			TableWitnessSegmentedView {
				table: self.table,
				cols: cols_0,
				log_segment_size: self.log_segment_size,
				start_index: self.start_index,
				n_segments: index,
			},
			TableWitnessSegmentedView {
				table: self.table,
				cols: cols_1,
				log_segment_size: self.log_segment_size,
				start_index: self.start_index + index,
				n_segments: self.n_segments - index,
			},
		)
	}

	fn into_iter(self) -> impl Iterator<Item = TableWitnessSegment<'a, P>> {
		let TableWitnessSegmentedView {
			table,
			cols,
			log_segment_size,
			start_index,
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
						WitnessColumnInfo::SameAsIndex(id) => itertools::Either::Right(
							iter::repeat_n(id, n_segments).map(RefCellData::SameAsIndex),
						),
					})
					.collect(),
			)
			.enumerate()
			.map(move |(index, cols)| TableWitnessSegment {
				table,
				cols,
				log_size: log_segment_size,
				index: start_index + index,
			});
			itertools::Either::Right(iter)
		}
	}

	fn into_par_iter(self) -> impl IndexedParallelIterator<Item = TableWitnessSegment<'a, P>> {
		let TableWitnessSegmentedView {
			table,
			cols,
			log_segment_size,
			start_index,
			n_segments,
		} = self;

		// This implementation uses unsafe code to iterate the segments of the view. A fully safe
		// implementation is also possible, which would look similar to that of
		// `rayon::slice::ChunksMut`. That just requires more code and doesn't seem justified.

		// TODO: clippy error (clippy::mut_from_ref): mutable borrow from immutable input(s)
		#[allow(clippy::mut_from_ref)]
		unsafe fn cast_slice_ref_to_mut<T>(slice: &[T]) -> &mut [T] {
			unsafe { slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len()) }
		}

		// Convert cols with mutable references into cols with const refs so that they can be
		// cloned. Within the loop, we unsafely cast back to mut refs.
		let cols = cols
			.into_iter()
			.map(|col| -> WitnessColumnInfo<(&'a [P], usize)> {
				match col {
					WitnessColumnInfo::Owned((data, chunk_size)) => {
						WitnessColumnInfo::Owned((data, chunk_size))
					}
					WitnessColumnInfo::SameAsIndex(id) => WitnessColumnInfo::SameAsIndex(id),
				}
			})
			.collect::<Vec<_>>();

		(0..n_segments).into_par_iter().map(move |i| {
			let col_strides = cols
				.iter()
				.map(|col| match col {
					WitnessColumnInfo::SameAsIndex(id) => RefCellData::SameAsIndex(*id),
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
				index: start_index + i,
			}
		})
	}
}

/// A vertical segment of a table witness index.
///
/// This provides runtime-checked access to slices of the witness columns. This is used separately
/// from [`TableWitnessIndex`] so that witness population can be parallelized over segments.
#[derive(Debug, CopyGetters)]
pub struct TableWitnessSegment<'a, P = PackedType<OptimalUnderlier, B128>>
where
	P: PackedField,
	P::Scalar: TowerField,
{
	table: &'a Table<P::Scalar>,
	/// Stores the actual data for the witness columns.
	///
	/// The order of the columns corresponds to the same order as defined in the table.
	cols: Vec<RefCellData<'a, P>>,
	#[get_copy = "pub"]
	log_size: usize,
	/// The index of the segment in the segmented table witness.
	#[get_copy = "pub"]
	index: usize,
}

impl<'a, F: TowerField, P: PackedField<Scalar = F>> TableWitnessSegment<'a, P> {
	pub fn get<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[PackedSubfield<P, FSub>]>, Error>
	where
		P: PackedExtension<FSub>,
	{
		if col.table_id != self.table.id() {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table.id(),
			});
		}

		let col = self
			.get_col_data(col.id())
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		Ok(Ref::map(col_ref, |packed| PackedExtension::cast_bases(packed)))
	}

	pub fn get_mut<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<RefMut<[PackedSubfield<P, FSub>]>, Error>
	where
		P: PackedExtension<FSub>,
		F: ExtensionField<FSub>,
	{
		if col.table_id != self.table.id() {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table.id(),
			});
		}

		let col = self
			.get_col_data(col.id())
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |packed| PackedExtension::cast_bases_mut(packed)))
	}

	pub fn get_scalars<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[FSub]>, Error>
	where
		P: PackedExtension<FSub>,
		F: ExtensionField<FSub>,
		PackedSubfield<P, FSub>: PackedFieldIndexable,
	{
		self.get(col)
			.map(|packed| Ref::map(packed, <PackedSubfield<P, FSub>>::unpack_scalars))
	}

	pub fn get_scalars_mut<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<RefMut<[FSub]>, Error>
	where
		P: PackedExtension<FSub>,
		F: ExtensionField<FSub>,
		PackedSubfield<P, FSub>: PackedFieldIndexable,
	{
		self.get_mut(col)
			.map(|packed| RefMut::map(packed, <PackedSubfield<P, FSub>>::unpack_scalars_mut))
	}

	pub fn get_as<T: Pod, FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[T]>, Error>
	where
		P: PackedExtension<FSub> + PackedFieldIndexable,
		F: ExtensionField<FSub> + Pod,
	{
		let col = self
			.get_col_data(col.id())
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		Ok(Ref::map(col_ref, |col| must_cast_slice(P::unpack_scalars(col))))
	}

	pub fn get_mut_as<T: Pod, FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<RefMut<[T]>, Error>
	where
		P: PackedExtension<FSub> + PackedFieldIndexable,
		F: ExtensionField<FSub> + Pod,
	{
		if col.table_id != self.table.id() {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table.id(),
			});
		}

		let col = self
			.get_col_data(col.id())
			.ok_or_else(|| Error::MissingColumn(col.id()))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |col| must_cast_slice_mut(P::unpack_scalars_mut(col))))
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
	) -> Result<impl Iterator<Item = PackedSubfield<P, FSub>> + use<FSub, V, F, P>, Error>
	where
		P: PackedExtension<FSub>,
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
		let expr_circuit = ArithCircuit::from(expr.expr());

		let col_refs = partition
			.columns
			.iter()
			.zip(expr_circuit.vars_usage())
			.map(|(col_id, used)| {
				used.then(|| {
					assert_eq!(FSub::TOWER_LEVEL, self.table[*col_id].shape.tower_height);
					let col = self
						.get_col_data(*col_id)
						.ok_or_else(|| Error::MissingColumn(*col_id))?;
					let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
					Ok::<_, Error>(Ref::map(col_ref, |packed| PackedExtension::cast_bases(packed)))
				})
				.transpose()
			})
			.collect::<Result<Vec<_>, _>>()?;

		let log_packed_elems =
			(self.log_size + log_vals_per_row).saturating_sub(<PackedSubfield<P, FSub>>::LOG_WIDTH);

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
		ArithCircuitPoly::new(expr_circuit).batch_evaluate(&cols, &mut evals)?;
		Ok(evals.into_iter())
	}

	pub fn size(&self) -> usize {
		1 << self.log_size
	}

	fn get_col_data(&self, column_id: ColumnId) -> Option<&RefCell<&'a mut [P]>> {
		let column_index = column_id.table_index.0;
		self.get_col_data_by_index(column_index)
	}

	fn get_col_data_by_index(&self, index: usize) -> Option<&RefCell<&'a mut [P]>> {
		match self.cols.get(index) {
			Some(RefCellData::Owned(data)) => Some(data),
			Some(RefCellData::SameAsIndex(index)) => self.get_col_data_by_index(*index),
			None => None,
		}
	}
}

impl<'a, P> TableWitnessSegment<'a, P>
where
	P: PackedField<Scalar: TowerField>
		+ PackedExtension<B1>
		+ PackedExtension<B8>
		+ PackedExtension<B16>
		+ PackedExtension<B32>
		+ PackedExtension<B64>
		+ PackedExtension<B128>,
{
	/// For a given column index within a table, return the immutable upcasted witness data
	pub fn get_dyn(
		&self,
		col_id: ColumnId,
	) -> Result<Box<dyn WitnessColView<P::Scalar> + '_>, Error> {
		let col = self
			.get_col_data(col_id)
			.ok_or_else(|| Error::MissingColumn(col_id))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		let tower_level = self.table[col_id].shape.tower_height;
		let ret: Box<dyn WitnessColView<_>> = match tower_level {
			0 => Box::new(WitnessColViewImpl(Ref::map(col_ref, |packed| {
				PackedExtension::<B1>::cast_bases(packed)
			}))),
			3 => Box::new(WitnessColViewImpl(Ref::map(col_ref, |packed| {
				PackedExtension::<B8>::cast_bases(packed)
			}))),
			4 => Box::new(WitnessColViewImpl(Ref::map(col_ref, |packed| {
				PackedExtension::<B16>::cast_bases(packed)
			}))),
			5 => Box::new(WitnessColViewImpl(Ref::map(col_ref, |packed| {
				PackedExtension::<B32>::cast_bases(packed)
			}))),
			6 => Box::new(WitnessColViewImpl(Ref::map(col_ref, |packed| {
				PackedExtension::<B64>::cast_bases(packed)
			}))),
			7 => Box::new(WitnessColViewImpl(Ref::map(col_ref, |packed| {
				PackedExtension::<B128>::cast_bases(packed)
			}))),
			_ => panic!("tower_level must be in the range [0, 7]"),
		};
		Ok(ret)
	}

	/// For a given column index within a table, return the mutable witness data.
	pub fn get_dyn_mut(
		&self,
		col_id: ColumnId,
	) -> Result<Box<dyn WitnessColViewMut<P::Scalar> + '_>, Error> {
		let col = self
			.get_col_data(col_id)
			.ok_or_else(|| Error::MissingColumn(col_id))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		let tower_level = self.table[col_id].shape.tower_height;
		let ret: Box<dyn WitnessColViewMut<_>> = match tower_level {
			0 => Box::new(WitnessColViewImpl(RefMut::map(col_ref, |packed| {
				PackedExtension::<B1>::cast_bases_mut(packed)
			}))),
			3 => Box::new(WitnessColViewImpl(RefMut::map(col_ref, |packed| {
				PackedExtension::<B8>::cast_bases_mut(packed)
			}))),
			4 => Box::new(WitnessColViewImpl(RefMut::map(col_ref, |packed| {
				PackedExtension::<B16>::cast_bases_mut(packed)
			}))),
			5 => Box::new(WitnessColViewImpl(RefMut::map(col_ref, |packed| {
				PackedExtension::<B32>::cast_bases_mut(packed)
			}))),
			6 => Box::new(WitnessColViewImpl(RefMut::map(col_ref, |packed| {
				PackedExtension::<B64>::cast_bases_mut(packed)
			}))),
			7 => Box::new(WitnessColViewImpl(RefMut::map(col_ref, |packed| {
				PackedExtension::<B128>::cast_bases_mut(packed)
			}))),
			_ => panic!("tower_level must be in the range [0, 7]"),
		};
		Ok(ret)
	}
}

/// Type erased interface for viewing witness columns. Note that `F` will be an extension field of
/// the underlying column's field.
pub trait WitnessColView<F> {
	/// Returns the scalar at a given index
	fn get(&self, index: usize) -> F;

	/// The size of the scalar elements in this column.
	fn size(&self) -> usize;
}

/// Similar to [`WitnessColView`], for mutating witness columns.
pub trait WitnessColViewMut<F>: WitnessColView<F> {
	/// Modifies the upcasted scalar at a given index.
	fn set(&mut self, index: usize, val: F) -> Result<(), Error>;
}

struct WitnessColViewImpl<Data>(Data);

impl<F, P, Data> WitnessColView<F> for WitnessColViewImpl<Data>
where
	F: ExtensionField<P::Scalar>,
	P: PackedField,
	Data: Deref<Target = [P]>,
{
	fn get(&self, index: usize) -> F {
		get_packed_slice(&self.0, index).into()
	}

	fn size(&self) -> usize {
		self.0.len() << P::LOG_WIDTH
	}
}

impl<F, P, Data> WitnessColViewMut<F> for WitnessColViewImpl<Data>
where
	F: ExtensionField<P::Scalar>,
	P: PackedField,
	Data: DerefMut<Target = [P]>,
{
	fn set(&mut self, index: usize, val: F) -> Result<(), Error> {
		let subfield_val = val.try_into().map_err(|_| Error::FieldElementTooBig)?;
		set_packed_slice(&mut self.0, index, subfield_val);
		Ok(())
	}
}

/// A struct that can populate segments of a table witness using row descriptors.
pub trait TableFiller<P = PackedType<OptimalUnderlier, B128>>
where
	P: PackedField,
	P::Scalar: TowerField,
{
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
		witness: &'a mut TableWitnessSegment<P>,
	) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
	use std::{array, iter::repeat_with};

	use assert_matches::assert_matches;
	use binius_compute::cpu::alloc::CpuComputeAllocator;
	use binius_core::oracle::{OracleId, SymbolicMultilinearOracleSet};
	use binius_field::{
		arch::{OptimalUnderlier128b, OptimalUnderlier256b},
		packed::{len_packed_slice, set_packed_slice},
	};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::builder::{
		ConstraintSystem, Statement, TableBuilder,
		types::{B1, B8, B16, B32},
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

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();
		let table_size = 64;
		let mut index = TableWitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(
			&allocator,
			&inner_table,
			table_size,
		)
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

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();
		let table_size = 64;
		let mut index = TableWitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(
			&allocator,
			&inner_table,
			table_size,
		)
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
		let col1 = table.add_committed::<B8, 2>("col1");
		let col2 = table.add_committed::<B8, 2>("col2");

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();
		let table_size = 1 << 6;
		let mut index = TableWitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(
			&allocator,
			&inner_table,
			table_size,
		)
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
	fn test_eval_expr_different_cols() {
		let table_id = 0;
		let mut inner_table = Table::<B128>::new(table_id, "table".to_string());
		let mut table = TableBuilder::new(&mut inner_table);
		let col0 = table.add_committed::<B32, 1>("col1");
		let col1 = table.add_committed::<B8, 8>("col2");
		let col2 = table.add_committed::<B16, 4>("col3");
		let col3 = table.add_committed::<B8, 8>("col4");

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();
		let table_size = 4;
		let mut index = TableWitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(
			&allocator,
			&inner_table,
			table_size,
		)
		.unwrap();

		let segment = index.full_segment();

		// Fill the columns with a deterministic pattern.
		{
			let mut col0: RefMut<'_, [u32]> = segment.get_mut_as(col0).unwrap();
			let mut col1: RefMut<'_, [[u8; 8]]> = segment.get_mut_as(col1).unwrap();
			let mut col2: RefMut<'_, [[u16; 4]]> = segment.get_mut_as(col2).unwrap();
			let mut col3: RefMut<'_, [[u8; 8]]> = segment.get_mut_as(col3).unwrap();

			col0[0] = 0x40;
			col1[0] = [0x45, 0, 0, 0, 0, 0, 0, 0];
			col2[0] = [0, 0, 0, 0x80];
			col3[0] = [0x85, 0, 0, 0, 0, 0, 0, 00];
		}

		let evals = segment.eval_expr(&(col1 + col3)).unwrap();
		for (i, eval_i) in evals
			.into_iter()
			.flat_map(PackedField::into_iter)
			.enumerate()
		{
			if i == 0 {
				assert_eq!(eval_i, B8::new(0x45) + B8::new(0x85));
			} else {
				assert_eq!(eval_i, B8::new(0x00));
			}
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

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();
		let table_size = 7;
		let mut index = TableWitnessIndex::<PackedType<OptimalUnderlier256b, B128>>::new(
			&allocator,
			&inner_table,
			table_size,
		)
		.unwrap();

		assert_eq!(index.log_capacity(), 3);
		assert_eq!(index.min_log_segment_size(), 3);

		let mut iter = index.segments(5);
		// Check that the segment size is clamped to the capacity.
		assert_eq!(iter.next().unwrap().log_size(), 3);
		assert!(iter.next().is_none());
		drop(iter);

		let mut iter = index.segments(2);
		// Check that the segment size is clamped to the minimum segment size.
		assert_eq!(iter.next().unwrap().log_size(), 3);
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

	impl TableFiller<PackedType<OptimalUnderlier128b, B128>> for TestTable {
		type Event = u32;

		fn id(&self) -> TableId {
			self.id
		}

		fn fill<'a>(
			&'a self,
			rows: impl Iterator<Item = &'a Self::Event> + Clone,
			witness: &'a mut TableWitnessSegment<PackedType<OptimalUnderlier128b, B128>>,
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

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();

		let table_size = 11;
		let mut index = WitnessIndex::new(&cs, &allocator);
		let table_index = index.init_table(test_table.id(), table_size).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let rows = repeat_with(|| rng.r#gen())
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

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();

		let table_size = 11;
		let mut index = WitnessIndex::new(&cs, &allocator);
		let table_index = index.init_table(test_table.id(), table_size).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let rows = repeat_with(|| rng.r#gen())
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

	#[test]
	fn test_fill_empty_rows_non_empty_table() {
		let mut cs = ConstraintSystem::new();
		let test_table = TestTable::new(&mut cs);

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();

		let table_size = 11;
		let mut index = WitnessIndex::new(&cs, &allocator);

		index.init_table(test_table.id(), table_size).unwrap();

		assert_matches!(
			index.fill_table_sequential(&test_table, &[]),
			Err(Error::IncorrectNumberOfTableEvents { .. })
		);
	}

	#[test]
	fn test_dyn_witness() {
		let mut cs = ConstraintSystem::new();
		let mut test_table = cs.add_table("test");
		let test_col: Col<B32, 4> = test_table.add_committed("col");
		let table_id = test_table.id();

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();

		let table_size = 11;
		let mut index = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

		let table_index = index.init_table(table_id, table_size).unwrap();
		let segment = table_index.full_segment();
		let mut rng = StdRng::seed_from_u64(0);
		let row = repeat_with(|| B32::random(&mut rng))
			.take(table_size * 4)
			.collect::<Vec<_>>();
		{
			let mut data: Box<dyn WitnessColViewMut<_>> =
				segment.get_dyn_mut(test_col.id()).unwrap();
			row.iter().enumerate().for_each(|(i, val)| {
				data.set(i, (*val).into()).unwrap();
			})
		}
		let data = segment.get_dyn(test_col.id()).unwrap();
		row.iter().enumerate().for_each(|(i, val)| {
			let down_cast: B32 = data.get(i).try_into().unwrap();
			assert_eq!(down_cast, *val)
		})
	}

	fn find_oracle_id_with_name(
		oracles: &SymbolicMultilinearOracleSet<B128>,
		name: &str,
	) -> Option<OracleId> {
		oracles
			.iter()
			.find(|(_, oracle)| oracle.name.as_deref() == Some(name))
			.map(|(id, _)| id)
	}

	#[test]
	fn test_constant_filling() {
		let mut cs = ConstraintSystem::new();

		let mut test_table = cs.add_table("test");
		let mut rng = StdRng::seed_from_u64(0);
		let unpack_value = B16::random(&mut rng);
		let pack_const_arr: [B32; 4] = array::from_fn(|_| B32::random(&mut rng));

		let _ = test_table.add_constant("unpacked_col", [unpack_value]);
		let _ = test_table.add_constant("packed_col", pack_const_arr);
		let table_id = test_table.id();

		let mut allocator = CpuComputeAllocator::new(1 << 12);
		let allocator = allocator.into_bump_allocator();

		let table_size = 123;
		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![table_size],
		};
		let ccs = cs.compile(&statement).unwrap();
		let mut index = WitnessIndex::<PackedType<OptimalUnderlier, B128>>::new(&cs, &allocator);

		{
			let _ = index.init_table(table_id, table_size).unwrap();
			index.fill_constant_cols().unwrap();
		}

		let witness = index.into_multilinear_extension_index();
		let non_packed_col_id = find_oracle_id_with_name(&ccs.oracles, "unpacked_col").unwrap();

		// Query MultilinearExtensionIndex to see if the constants are correct.
		let non_pack_witness = witness.get_multilin_poly(non_packed_col_id).unwrap();
		for index in 0..non_pack_witness.size() {
			let got = non_pack_witness.evaluate_on_hypercube(index).unwrap();
			assert_eq!(got, unpack_value.into());
		}
		let packed_col_id = find_oracle_id_with_name(&ccs.oracles, "packed_col").unwrap();

		let pack_witness = witness.get_multilin_poly(packed_col_id).unwrap();
		for index in 0..pack_witness.size() {
			let got = pack_witness.evaluate_on_hypercube(index).unwrap();

			assert_eq!(got, pack_const_arr[index % 4].into());
		}
	}
}
