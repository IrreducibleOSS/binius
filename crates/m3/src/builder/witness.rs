// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter,
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
use binius_math::{CompositionPoly, MultilinearExtension, MultilinearPoly};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{must_cast_slice, must_cast_slice_mut, zeroed_vec, Pod};
use getset::CopyGetters;
use itertools::Itertools;

use super::{
	column::{Col, ColumnShape},
	error::Error,
	multi_iter::MultiIterator,
	statement::Statement,
	table::{Table, TableId},
	types::{B1, B128, B16, B32, B64, B8},
	ColumnDef, ColumnId, ColumnIndex, Expr,
};

/// Holds witness column data for all tables in a constraint system, indexed by column ID.
///
/// The struct has two lifetimes: `'cs` is the lifetime of the constraint system, and `'alloc` is
/// the lifetime of the bump allocator. The reason these must be separate is that the witness index
/// gets converted into a multilinear extension index, which maintains references to the data
/// allocated by the allocator, but does not need to maintain a reference to the constraint system,
/// which can then be dropped.
///
/// TODO: Change Underlier to P: PackedFieldIndexable
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

	pub fn into_multilinear_extension_index(
		self,
		statement: &Statement<B128>,
	) -> MultilinearExtensionIndex<'alloc, U, B128>
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
				let log_size = table.log_capacity + log_values_per_row;
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

#[derive(Debug)]
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
	pub fn full_segment(&mut self) -> TableWitnessIndexSegment<U, F> {
		let cols = self
			.cols
			.iter_mut()
			.map(|col| match &mut col.data {
				WitnessDataMut::SameAsOracleIndex(index) => RefCellData::SameAsOracleIndex(*index),
				WitnessDataMut::Owned(data) => RefCellData::Owned(RefCell::new(data)),
			})
			.collect();
		TableWitnessIndexSegment {
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

		let log_size = TARGET_SEGMENT_LOG_BITS.saturating_sub(median_col_log_bits);
		self.fill_sequential_with_segment_size(table, rows, log_size)
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

		let segment_size = 1 << log_size;
		let n_full_chunks = rows.len() / segment_size;
		let n_extra_rows = rows.len() % segment_size;
		let max_full_chunk_index = n_full_chunks * segment_size;

		// Fill segments of the table with full chunks
		let mut segments_iter = self.segments(log_size);

		let mut last_segment = None;
		for (row_chunk, mut witness_segment) in
			iter::zip(rows[..max_full_chunk_index].chunks(segment_size), &mut segments_iter)
		{
			table
				.fill(row_chunk.iter(), &mut witness_segment)
				.map_err(Error::TableFill)?;
			last_segment = Some(witness_segment);
		}

		// Fill the segment that is only partially assigned row events.
		if n_extra_rows != 0 {
			let mut witness_segment = segments_iter
				.next()
				.expect("the witness capacity is at least as much as the number of rows");

			let repeating_rows = rows[max_full_chunk_index..]
				.iter()
				.cycle()
				.take(segment_size);
			table
				.fill(repeating_rows, &mut witness_segment)
				.map_err(Error::TableFill)?;
			last_segment = Some(witness_segment);
		}

		// Copy the filled segment to the remaining segments. This should satisfy all row-wise
		// constraints if the last segment does.
		let mut last_segment = last_segment.expect(
			"there must be at least one row chunk; \
			 rows length is equal to self.size; \
			 self.size is checked to be non-zero in the constructor",
		);
		for mut segment in segments_iter {
			println!("Hi");
			for (dst_col, src_col) in iter::zip(&mut segment.cols, &mut last_segment.cols) {
				if let (RefCellData::Owned(dst), RefCellData::Owned(src)) = (dst_col, src_col) {
					dst.get_mut().copy_from_slice(src.get_mut())
				}
			}
		}

		Ok(())
	}

	pub fn segments(
		&mut self,
		log_size: usize,
	) -> impl Iterator<Item = TableWitnessIndexSegment<U, F>> + '_ {
		// Clamp the segment size.
		let log_size = log_size
			.min(self.log_capacity)
			.max(self.min_log_segment_size);

		let log_n = self.log_capacity - log_size;
		let iter = MultiIterator::new(
			self.cols
				.iter_mut()
				.map(|col| match &mut col.data {
					WitnessDataMut::SameAsOracleIndex(index) => itertools::Either::Left(
						iter::repeat_n(*index, log_n).map(RefCellData::SameAsOracleIndex),
					),
					WitnessDataMut::Owned(data) => {
						// The segment size can be smaller than required for the segment to fill
						// one underlier in the case when the whole column only has one
						// partially-filed underlier. In this case it's OK to use a saturating_sub.
						let log_stride =
							(log_size + col.shape.log_cell_size()).saturating_sub(U::LOG_BITS);
						itertools::Either::Right(
							data.chunks_mut(1 << log_stride)
								.map(|data| RefCellData::Owned(RefCell::new(data))),
						)
					}
				})
				.collect(),
		);

		let table = self.table;
		let oracle_offset = self.oracle_offset;
		iter.map(move |cols| TableWitnessIndexSegment {
			table,
			cols,
			log_size,
			oracle_offset,
		})
	}
}

/// A vertical segment of a table witness index.
///
/// This provides runtime-checked access to slices of the witness columns. This is used separately
/// from [`TableWitnessIndex`] so that witness population can be parallelized over segments.
#[derive(Debug, CopyGetters)]
pub struct TableWitnessIndexSegment<'a, U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	table: &'a Table<F>,
	cols: Vec<RefCellData<'a, U>>,
	#[get_copy = "pub"]
	log_size: usize,
	oracle_offset: usize,
}

impl<'a, U: UnderlierType, F: TowerField> TableWitnessIndexSegment<'a, U, F> {
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
	type Event;

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
		witness: &'a mut TableWitnessIndexSegment<U, F>,
	) -> anyhow::Result<()>;
}

// TODO: fill_table_parallel
// TODO: a streaming version that streams in rows and fills in a background thread pool.

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
		ConstraintSystem, TableBuilder,
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
			TableWitnessIndex::<OptimalUnderlier256b>::new(&allocator, &inner_table, table_size).unwrap();

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

	#[test]
	fn test_fill_sequential_with_incomplete_events() {
		type U = OptimalUnderlier128b;

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

		impl TableFiller<U> for TestTable {
			type Event = u32;

			fn id(&self) -> TableId {
				self.id
			}

			fn fill<'a>(
				&'a self,
				rows: impl Iterator<Item = &'a Self::Event> + Clone,
				witness: &'a mut TableWitnessIndexSegment<U>,
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
		assert_eq!(col0[11].val(), rows[8]);
		assert_eq!(col0[12].val(), rows[8]);
		assert_eq!(col0[13].val(), rows[9]);
		assert_eq!(col0[14].val(), rows[10]);
		assert_eq!(col0[15].val(), rows[8]);
	}
}
