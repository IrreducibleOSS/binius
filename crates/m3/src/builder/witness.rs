// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter, slice,
	sync::Arc,
};

use anyhow::ensure;
use binius_core::{
	oracle::OracleId, polynomial::ArithCircuitPoly, transparent::step_down::StepDown,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	ExtensionField, PackedField, TowerField,
};
use binius_math::{CompositionPoly, MultilinearExtension, MultilinearPoly};
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::{checked_log_2, log2_ceil_usize};
use bytemuck::{must_cast_slice, must_cast_slice_mut, zeroed_vec, Pod};
use getset::CopyGetters;

use super::{
	column::{Col, ColumnShape},
	error::Error,
	statement::Statement,
	table::{Table, TableId},
	types::{B1, B128, B16, B32, B64, B8},
	ColumnDef, ColumnId, ColumnIndex, Expr,
};

/// Holds witness column data for all tables in a constraint system, indexed by column ID.
#[derive(Debug, Default, CopyGetters)]
pub struct WitnessIndex<'cs, 'alloc, U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	pub tables: Vec<TableWitnessIndex<'cs, 'alloc, U, F>>,
}

impl<'cs, 'alloc, U: UnderlierType, F: TowerField> WitnessIndex<'cs, 'alloc, U, F> {
	pub fn get_table(
		&mut self,
		table_id: TableId,
	) -> Option<&mut TableWitnessIndex<'cs, 'alloc, U, F>> {
		self.tables.get_mut(table_id)
	}

	pub fn fill_table_sequential<T: TableFiller<U, F>>(
		&mut self,
		table: &T,
		rows: &[T::Event],
	) -> Result<(), Error> {
		let table_id = table.id();
		let witness = self
			.get_table(table_id)
			.ok_or(Error::MissingTable { table_id })?;
		fill_table_sequential(table, rows, witness).map_err(Error::TableFill)?;
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
		for table in self.tables {
			let table_id = table.table_id();
			let cols = immutable_witness_index_columns(table.cols);

			let mut count = 0;
			for (oracle_id_offset, col) in cols.into_iter().enumerate() {
				let oracle_id = first_oracle_id_in_table + oracle_id_offset;
				let log_capacity = if col.is_single_row {
					0
				} else {
					table.log_capacity
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

			// Every table partition has a step_down appended to the end of the table to support non-power of two height tables
			for log_values_per_row in table.selector_log_values_per_rows.into_iter() {
				let oracle_id = first_oracle_id_in_table + count;
				let size = statement.table_sizes[table_id] << log_values_per_row;
				let log_size = log2_ceil_usize(size);
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
	table: &'cs Table<F>,
	oracle_offset_lookup: Vec<OracleId>,
	selector_log_values_per_rows: Vec<usize>,
	cols: Vec<WitnessIndexColumn<'alloc, U>>,
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
	pub shape: ColumnShape,
	pub data: WitnessDataMut<'a, U>,
	pub is_single_row: bool,
}

#[derive(Debug)]
pub enum WitnessDataMut<'alloc, U: UnderlierType> {
	Owned(&'alloc mut [U]),
	SameAsOracleIndex(usize),
}

#[derive(Debug)]
pub enum RefCellData<'alloc, U: UnderlierType> {
	Owned(RefCell<&'alloc mut [U]>),
	SameAsOracleIndex(usize),
}

#[derive(Debug)]
pub struct ImmutableWitnessIndexColumn<'a, U: UnderlierType> {
	pub shape: ColumnShape,
	pub data: &'a [U],
	pub is_single_row: bool,
}

fn immutable_witness_index_columns<'a, U: UnderlierType>(
	cols: Vec<WitnessIndexColumn<'a, U>>,
) -> Vec<ImmutableWitnessIndexColumn<'a, U>> {
	let mut result: Vec<ImmutableWitnessIndexColumn<'a, U>> = Vec::with_capacity(cols.len());
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
	pub fn new(allocator: &'alloc bumpalo::Bump, table: &'cs Table<F>, table_size: usize) -> Self {
		// TODO: Make packed columns alias the same slice data
		let log_capacity = log2_ceil_usize(table_size);

		let mut min_log_cell_bits = U::LOG_BITS;

		let mut next_oracle_index = 0;
		let mut cols = Vec::new();
		let mut oracle_offset_lookup = Vec::with_capacity(table.columns.len());

		for col in &table.columns {
			let shape = col.shape;
			let data = match col.col {
				ColumnDef::Packed { col: inner_col, .. } => {
					WitnessDataMut::SameAsOracleIndex(oracle_offset_lookup[inner_col.table_index])
				}
				_ => {
					let log_cell_bits = col.shape.tower_height + col.shape.log_values_per_row;
					min_log_cell_bits = min_log_cell_bits.min(log_cell_bits);
					let log_col_bits = log_cell_bits + log_capacity;

					// TODO: Error instead of panic
					if log_col_bits < U::LOG_BITS {
						panic!("capacity is too low");
					}

					// TODO: Allocate uninitialized memory and avoid filling. That should be OK because
					// Underlier is Pod.
					WitnessDataMut::Owned(
						allocator.alloc_slice_fill_default(1 << (log_col_bits - U::LOG_BITS)),
					)
				}
			};
			cols.push(WitnessIndexColumn {
				shape,
				data,
				is_single_row: matches!(col.col, ColumnDef::RepeatingTransparent { .. }),
			});

			if matches!(col.col, ColumnDef::RepeatingTransparent { .. }) {
				cols.push(WitnessIndexColumn {
					shape,
					data: WitnessDataMut::SameAsOracleIndex(next_oracle_index),
					is_single_row: false,
				});
				next_oracle_index += 1;
				oracle_offset_lookup.push(next_oracle_index);
				next_oracle_index += 1;
			} else {
				oracle_offset_lookup.push(next_oracle_index);
				next_oracle_index += 1;
			}
		}

		Self {
			table,
			selector_log_values_per_rows: table.partitions.keys().collect(),
			cols,
			log_capacity,
			min_log_segment_size: U::LOG_BITS - min_log_cell_bits,
			oracle_offset_lookup,
		}
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
			oracle_offset_lookup: &self.oracle_offset_lookup,
		}
	}

	/// Returns an iterator over segments of witness index rows.
	pub fn segments(
		&mut self,
		log_size: usize,
	) -> impl Iterator<Item = TableWitnessIndexSegment<'_, U, F>> + '_ {
		assert!(log_size <= self.log_capacity);
		assert!(log_size >= self.min_log_segment_size);

		let self_ref = self as &Self;
		(0..1 << (self.log_capacity - log_size)).map(move |i| {
			let table = self_ref.table;
			let col_strides = self_ref
				.cols
				.iter()
				.map(|col| match &col.data {
					WitnessDataMut::SameAsOracleIndex(index) => {
						RefCellData::SameAsOracleIndex(*index)
					}
					WitnessDataMut::Owned(data) => {
						let log_cell_bits = col.shape.tower_height + col.shape.log_values_per_row;
						let log_stride = log_size + log_cell_bits - U::LOG_BITS;
						RefCellData::Owned(RefCell::new(unsafe {
							// Safety: The function borrows self mutably, so we have mutable access to
							// all columns and thus none can be borrowed by anyone else. The loop is
							// constructed to borrow disjoint segments of each column -- if this loop
							// were transposed, we would use `chunks_mut`.
							cast_slice_ref_to_mut(&data[i << log_stride..(i + 1) << log_stride])
						}))
					}
				})
				.collect();
			TableWitnessIndexSegment {
				table,
				cols: col_strides,
				log_size,
				oracle_offset_lookup: &self_ref.oracle_offset_lookup,
			}
		})
	}

	pub fn par_segments(
		&mut self,
		log_size: usize,
	) -> impl ParallelIterator<Item = TableWitnessIndexSegment<'_, U, F>> + '_ {
		assert!(log_size < self.log_capacity);
		assert!(log_size >= self.min_log_segment_size);

		// TODO: deduplicate closure between this and `segments`. It's kind of a tricky interface
		// because of the unsafe casting.
		let self_ref = self as &Self;
		(0..1 << (self.log_capacity - log_size))
			.into_par_iter()
			.map(move |start| {
				let table = self_ref.table;
				let col_strides = self_ref
					.cols
					.iter()
					.map(|col| match &col.data {
						WitnessDataMut::SameAsOracleIndex(index) => {
							RefCellData::SameAsOracleIndex(*index)
						}
						WitnessDataMut::Owned(data) => {
							let log_cell_bits =
								col.shape.tower_height + col.shape.log_values_per_row;
							let log_stride = log_size + log_cell_bits - U::LOG_BITS;
							RefCellData::Owned(RefCell::new(unsafe {
								// Safety: The function borrows self mutably, so we have mutable access to
								// all columns and thus none can be borrowed by anyone else. The loop is
								// constructed to borrow disjoint segments of each column -- if this loop
								// were transposed, we would use `chunks_mut`.
								cast_slice_ref_to_mut(
									&data[start << log_stride..(start + 1) << log_stride],
								)
							}))
						}
					})
					.collect();
				TableWitnessIndexSegment {
					table,
					cols: col_strides,
					log_size,
					oracle_offset_lookup: &self_ref.oracle_offset_lookup,
				}
			})
	}
}

/// A vertical segment of a table witness index.
///
/// This provides runtime-checked access to slices of the witness columns. This is used separately
/// from [`TableWitnessIndex`] so that witness population can be parallelized over segments.
#[derive(Debug, CopyGetters)]
pub struct TableWitnessIndexSegment<
	'alloc,
	U: UnderlierType = OptimalUnderlier,
	F: TowerField = B128,
> {
	oracle_offset_lookup: &'alloc [ColumnIndex],
	table: &'alloc Table<F>,
	cols: Vec<RefCellData<'alloc, U>>,
	#[get_copy = "pub"]
	log_size: usize,
}

impl<'alloc, U: UnderlierType, F: TowerField> TableWitnessIndexSegment<'alloc, U, F> {
	pub fn get<FSub: TowerField, const V: usize>(
		&self,
		col: Col<FSub, V>,
	) -> Result<Ref<[PackedType<U, FSub>]>, Error>
	where
		U: PackScalar<FSub>,
	{
		// TODO: Check consistency of static params with column info

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
		// TODO: Check consistency of static params with column info

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

	fn get_col_data(&self, table_index: ColumnIndex) -> Option<&RefCell<&'alloc mut [U]>> {
		self.get_col_data_by_oracle_offset(self.oracle_offset_lookup[table_index])
	}

	fn get_col_data_by_oracle_offset(
		&self,
		oracle_id: OracleId,
	) -> Option<&RefCell<&'alloc mut [U]>> {
		match self.cols.get(oracle_id) {
			Some(RefCellData::Owned(data)) => Some(data),
			Some(RefCellData::SameAsOracleIndex(id)) => self.get_col_data_by_oracle_offset(*id),
			None => None,
		}
	}
}

// TODO: clippy error (clippy::mut_from_ref): mutable borrow from immutable input(s)
#[allow(clippy::mut_from_ref)]
unsafe fn cast_slice_ref_to_mut<T>(slice: &[T]) -> &mut [T] {
	slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len())
}

/// A struct that can populate segments of a table witness using row descriptors.
pub trait TableFiller<U: UnderlierType = OptimalUnderlier, F: TowerField = B128> {
	/// A struct that specifies the row contents.
	type Event;

	/// Returns the table ID.
	fn id(&self) -> TableId;

	/// Fill the table witness with data derived from the given rows.
	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &'a mut TableWitnessIndexSegment<U, F>,
	) -> anyhow::Result<()>;
}

/// Fill a full table witness index using the given row data.
///
/// This function iterates through witness segments sequentially in a single thread.
pub fn fill_table_sequential<U: UnderlierType, F: TowerField, T: TableFiller<U, F>>(
	table: &T,
	rows: &[T::Event],
	witness: &mut TableWitnessIndex<U, F>,
) -> anyhow::Result<()> {
	ensure!(witness.capacity() >= rows.len(), "rows exceed witness capacity");

	let log_segment_size = witness.min_log_segment_size();

	let segment_size = 1 << log_segment_size;
	let n_full_chunks = rows.len() / segment_size;
	let n_extra_rows = rows.len() % segment_size;
	let max_full_chunk_index = n_full_chunks * segment_size;

	// Fill segments of the table with full chunks
	let mut segments_iter = witness.segments(log_segment_size);
	for (row_chunk, mut witness_segment) in
		iter::zip(rows[..max_full_chunk_index].chunks(segment_size), &mut segments_iter)
	{
		table.fill(row_chunk.iter(), &mut witness_segment)?;
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
		table.fill(repeating_rows, &mut witness_segment)?;
	}

	// TODO: copy a filled segment to the remaining segments

	Ok(())
}

// TODO: fill_table_parallel
// TODO: a streaming version that streams in rows and fills in a background thread pool.

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{
		arch::OptimalUnderlier128b,
		packed::{len_packed_slice, set_packed_slice},
	};

	use super::*;
	use crate::builder::{
		types::{B1, B32, B8},
		TableBuilder,
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
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &inner_table, table_size);
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
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &inner_table, table_size);

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
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &inner_table, table_size);

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
}
