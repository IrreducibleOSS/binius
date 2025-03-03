// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter, slice,
};

use binius_core::{transparent::step_down::StepDown, witness::MultilinearExtensionIndex};
use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	ExtensionField, TowerField,
};
use binius_math::MultilinearExtension;
use binius_maybe_rayon::prelude::*;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};
use getset::CopyGetters;

use super::{
	column::{Col, ColumnShape},
	error::Error,
	statement::Statement,
	table::{Table, TableId},
	types::{B1, B128, B16, B32, B64, B8},
	ColumnId,
};

/// Holds witness column data for all tables in a constraint system, indexed by column ID.
#[derive(Debug, Default, CopyGetters)]
pub struct WitnessIndex<'a, U: UnderlierType = OptimalUnderlier> {
	pub tables: Vec<TableWitnessIndex<'a, U>>,
}

impl<'a, U: UnderlierType> WitnessIndex<'a, U> {
	pub fn get_table(&mut self, table_id: TableId) -> Option<&mut TableWitnessIndex<'a, U>> {
		self.tables.get_mut(table_id)
	}

	pub fn fill_table_sequential<T: TableFiller<U>>(
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

	pub fn into_multilinear_extension_index<F>(
		self,
		statement: &Statement<F>,
	) -> MultilinearExtensionIndex<'a, U, F>
	where
		F: TowerField
			+ ExtensionField<B1>
			+ ExtensionField<B8>
			+ ExtensionField<B16>
			+ ExtensionField<B32>
			+ ExtensionField<B64>
			+ ExtensionField<B128>,
		U: PackScalar<F>
			+ PackScalar<B1>
			+ PackScalar<B8>
			+ PackScalar<B16>
			+ PackScalar<B32>
			+ PackScalar<B64>
			+ PackScalar<B128>,
	{
		let mut index = MultilinearExtensionIndex::new();
		let mut first_oracle_id_in_table = 0;
		for table in self.tables {
			let mut pack_factors = Vec::new();
			let mut count = 0;

			for partition in table.cols.into_iter() {
				let pack_factor = partition
					.first()
					.expect("There should be no empty partitions")
					.shape
					.pack_factor;
				pack_factors.push(pack_factor);

				for col in partition.into_iter() {
					let oracle_id = first_oracle_id_in_table + col.id.table_index;
					let n_vars = table.log_capacity + col.shape.pack_factor;
					let witness = match col.shape.tower_height {
						0 => MultilinearExtension::new(
							n_vars,
							PackedType::<U, B1>::from_underliers_ref(col.data),
						)
						.unwrap()
						.specialize_arc_dyn(),
						3 => MultilinearExtension::new(
							n_vars,
							PackedType::<U, B8>::from_underliers_ref(col.data),
						)
						.unwrap()
						.specialize_arc_dyn(),
						4 => MultilinearExtension::new(
							n_vars,
							PackedType::<U, B16>::from_underliers_ref(col.data),
						)
						.unwrap()
						.specialize_arc_dyn(),
						5 => MultilinearExtension::new(
							n_vars,
							PackedType::<U, B32>::from_underliers_ref(col.data),
						)
						.unwrap()
						.specialize_arc_dyn(),
						6 => MultilinearExtension::new(
							n_vars,
							PackedType::<U, B64>::from_underliers_ref(col.data),
						)
						.unwrap()
						.specialize_arc_dyn(),
						7 => MultilinearExtension::new(
							n_vars,
							PackedType::<U, B128>::from_underliers_ref(col.data),
						)
						.unwrap()
						.specialize_arc_dyn(),
						_ => {
							panic!("Unsupported tower height: {}", col.shape.tower_height);
						}
					};
					index.update_multilin_poly([(oracle_id, witness)]).unwrap();
					count += 1;
				}
			}

			// Every table partition has a step_down appended to the end of the table to support non-power of two height tables
			for pack_factor in pack_factors.into_iter() {
				let oracle_id = first_oracle_id_in_table + count;
				let size = statement.table_sizes[table.table_id] << pack_factor;
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

/// Holds witness column data for a table, indexed by column index.
#[derive(Debug, Default, CopyGetters)]
pub struct TableWitnessIndex<'a, U: UnderlierType = OptimalUnderlier> {
	table_id: TableId,
	cols: Vec<Vec<WitnessIndexColumn<'a, U>>>,
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
	pub id: ColumnId,
	pub shape: ColumnShape,
	pub data: &'a mut [U],
}

impl<'a, U: UnderlierType> TableWitnessIndex<'a, U> {
	pub fn new(
		allocator: &'a bumpalo::Bump,
		table: &Table<impl TowerField>,
		table_size: usize,
	) -> Self {
		// TODO: Make packed columns alias the same slice data
		let log_capacity = log2_ceil_usize(table_size);

		let mut min_log_cell_bits = U::LOG_BITS;

		let cols = table
			.partitions
			.iter()
			.map(|partition| {
				partition
					.columns
					.iter()
					.map(|&id| {
						let shape = table.columns[id.table_index].shape;
						let log_cell_bits = shape.tower_height + shape.pack_factor;
						min_log_cell_bits = min_log_cell_bits.min(log_cell_bits);
						let log_col_bits = log_cell_bits + log_capacity;

						// TODO: Error instead of panic
						if log_col_bits < U::LOG_BITS {
							panic!("capacity is too low");
						}
						// TODO: Allocate uninitialized memory and avoid filling. That should be OK because
						// Underlier is Pod.
						let col_data =
							allocator.alloc_slice_fill_default(1 << (log_col_bits - U::LOG_BITS));

						WitnessIndexColumn {
							id,
							shape,
							data: col_data,
						}
					})
					.collect()
			})
			.collect();
		Self {
			table_id: table.id,
			cols,
			log_capacity,
			min_log_segment_size: U::LOG_BITS - min_log_cell_bits,
		}
	}

	pub fn capacity(&self) -> usize {
		1 << self.log_capacity
	}

	/// Returns a witness index segment covering the entire table.
	pub fn full_segment(&mut self) -> TableWitnessIndexSegment<U> {
		let cols = self
			.cols
			.iter_mut()
			.map(|partition| {
				partition
					.iter_mut()
					.map(|col| RefCell::new(&mut *col.data))
					.collect()
			})
			.collect();
		TableWitnessIndexSegment {
			table_id: self.table_id,
			cols,
			log_size: self.log_capacity,
		}
	}

	/// Returns an iterator over segments of witness index rows.
	pub fn segments(
		&mut self,
		log_size: usize,
	) -> impl Iterator<Item = TableWitnessIndexSegment<'_, U>> + '_ {
		assert!(log_size <= self.log_capacity);
		assert!(log_size >= self.min_log_segment_size);

		let self_ref = self as &Self;
		(0..1 << (self.log_capacity - log_size)).map(move |i| {
			let table_id = self_ref.table_id;
			let col_strides = self_ref
				.cols
				.iter()
				.map(|partition| {
					partition
						.iter()
						.map(|col| {
							let log_cell_bits = col.shape.tower_height + col.shape.pack_factor;
							let log_stride = log_size + log_cell_bits - U::LOG_BITS;
							// Safety: The function borrows self mutably, so we have mutable access to
							// all columns and thus none can be borrowed by anyone else. The loop is
							// constructed to borrow disjoint segments of each column -- if this loop
							// were transposed, we would use `chunks_mut`.
							let col_stride = unsafe {
								cast_slice_ref_to_mut(
									&col.data[i << log_stride..(i + 1) << log_stride],
								)
							};
							RefCell::new(col_stride)
						})
						.collect()
				})
				.collect();
			TableWitnessIndexSegment {
				table_id,
				cols: col_strides,
				log_size,
			}
		})
	}

	pub fn par_segments(
		&mut self,
		log_size: usize,
	) -> impl ParallelIterator<Item = TableWitnessIndexSegment<'_, U>> + '_ {
		assert!(log_size < self.log_capacity);
		assert!(log_size >= self.min_log_segment_size);

		// TODO: deduplicate closure between this and `segments`. It's kind of a tricky interface
		// because of the unsafe casting.
		let self_ref = self as &Self;
		(0..1 << (self.log_capacity - log_size))
			.into_par_iter()
			.map(move |start| {
				let table_id = self_ref.table_id;
				let col_strides = self_ref
					.cols
					.iter()
					.map(|partition| {
						partition
							.iter()
							.map(|col| {
								let log_cell_bits = col.shape.tower_height + col.shape.pack_factor;
								let log_stride = log_size + log_cell_bits - U::LOG_BITS;
								// Safety: The function borrows self mutably, so we have mutable access to
								// all columns and thus none can be borrowed by anyone else. The loop is
								// constructed to borrow disjoint segments of each column -- if this loop
								// were transposed, we would use `chunks_mut`.
								let col_stride = unsafe {
									cast_slice_ref_to_mut(
										&col.data[start << log_stride..(start + 1) << log_stride],
									)
								};
								RefCell::new(col_stride)
							})
							.collect()
					})
					.collect();
				TableWitnessIndexSegment {
					table_id,
					cols: col_strides,
					log_size,
				}
			})
	}
}

/// A vertical segment of a table witness index.
///
/// This provides runtime-checked access to slices of the witness columns. This is used separately
/// from [`TableWitnessIndex`] so that witness population can be parallelized over segments.
#[derive(Debug, Default, CopyGetters)]
pub struct TableWitnessIndexSegment<'a, U: UnderlierType = OptimalUnderlier> {
	table_id: TableId,
	cols: Vec<Vec<RefCell<&'a mut [U]>>>,
	#[get_copy = "pub"]
	log_size: usize,
}

impl<U: UnderlierType> TableWitnessIndexSegment<'_, U> {
	pub fn get<F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<Ref<[PackedType<U, F>]>, Error>
	where
		U: PackScalar<F>,
	{
		// TODO: Check consistency of static params with column info

		if col.id.table_id != self.table_id {
			return Err(Error::TableMismatch {
				column_table_id: col.id.table_id,
				witness_table_id: self.table_id,
			});
		}

		let col = self
			.cols
			.get(col.id.partition_id)
			.and_then(|partition| partition.get(col.id.partition_index))
			.ok_or_else(|| Error::MissingColumn(col.id))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		Ok(Ref::map(col_ref, |x| <PackedType<U, F>>::from_underliers_ref(x)))
	}

	pub fn get_mut<F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<RefMut<[PackedType<U, F>]>, Error>
	where
		U: PackScalar<F>,
	{
		// TODO: Check consistency of static params with column info

		if col.id.table_id != self.table_id {
			return Err(Error::TableMismatch {
				column_table_id: col.id.table_id,
				witness_table_id: self.table_id,
			});
		}

		let col = self
			.cols
			.get(col.id.partition_id)
			.and_then(|partition| partition.get(col.id.partition_index))
			.ok_or_else(|| Error::MissingColumn(col.id))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |x| <PackedType<U, F>>::from_underliers_ref_mut(x)))
	}

	pub fn get_as<T: Pod, F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<Ref<[T]>, Error>
	where
		U: Pod,
	{
		let col = self
			.cols
			.get(col.id.partition_id)
			.and_then(|partition| partition.get(col.id.partition_index))
			.ok_or_else(|| Error::MissingColumn(col.id))?;
		let col_ref = col.try_borrow().map_err(Error::WitnessBorrow)?;
		Ok(Ref::map(col_ref, |x| must_cast_slice(x)))
	}

	pub fn get_mut_as<T: Pod, F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<RefMut<[T]>, Error>
	where
		U: Pod,
	{
		if col.id.table_id != self.table_id {
			return Err(Error::TableMismatch {
				column_table_id: col.id.table_id,
				witness_table_id: self.table_id,
			});
		}

		let col = self
			.cols
			.get(col.id.partition_id)
			.and_then(|partition| partition.get(col.id.partition_index))
			.ok_or_else(|| Error::MissingColumn(col.id))?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |x| must_cast_slice_mut(x)))
	}

	pub fn size(&self) -> usize {
		1 << self.log_size
	}
}

// TODO: clippy error (clippy::mut_from_ref): mutable borrow from immutable input(s)
#[allow(clippy::mut_from_ref)]
unsafe fn cast_slice_ref_to_mut<T>(slice: &[T]) -> &mut [T] {
	slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len())
}

/// A struct that can populate segments of a table witness using row descriptors.
pub trait TableFiller<U: UnderlierType = OptimalUnderlier> {
	/// A struct that specifies the row contents.
	type Event;

	/// Returns the table ID.
	fn id(&self) -> TableId;

	/// Fill the table witness with data derived from the given rows.
	fn fill(
		&self,
		rows: &[Self::Event],
		witness: &mut TableWitnessIndexSegment<U>,
	) -> anyhow::Result<()>;
}

/// Fill a full table witness index using the given row data.
///
/// This function iterates through witness segments sequentially in a single thread.
pub fn fill_table_sequential<U: UnderlierType, T: TableFiller<U>>(
	table: &T,
	rows: &[T::Event],
	witness: &mut TableWitnessIndex<U>,
) -> anyhow::Result<()> {
	let log_segment_size = witness.min_log_segment_size();

	// TODO: Handle the case when rows are not a multiple of the segment size
	assert_eq!(rows.len() % (1 << log_segment_size), 0);

	for (row_chunk, mut witness_segment) in
		iter::zip(rows.chunks(1 << log_segment_size), witness.segments(log_segment_size))
	{
		table.fill(row_chunk, &mut witness_segment)?;
	}

	Ok(())
}

// TODO: fill_table_parallel
// TODO: a streaming version that streams in rows and fills in a background thread pool.

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{arch::OptimalUnderlier128b, packed::len_packed_slice};

	use super::*;
	use crate::builder::types::{B1, B32, B8};

	#[test]
	fn test_table_witness_borrows() {
		let table_id = 0;
		let mut table = Table::<B128>::new(table_id, "table".to_string());
		let col0 = table.add_committed::<B1, 3>("col0");
		let col1 = table.add_committed::<B1, 5>("col1");
		let col2 = table.add_committed::<B8, 0>("col2");
		let col3 = table.add_committed::<B32, 0>("col3");

		let allocator = bumpalo::Bump::new();
		let table_size = 64;
		let mut index =
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &table, table_size);
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
		let mut table = Table::<B128>::new(table_id, "table".to_string());
		let col0 = table.add_committed::<B1, 3>("col0");
		let col1 = table.add_committed::<B1, 5>("col1");
		let col2 = table.add_committed::<B8, 0>("col2");
		let col3 = table.add_committed::<B32, 0>("col3");

		let allocator = bumpalo::Bump::new();
		let table_size = 64;
		let mut index =
			TableWitnessIndex::<OptimalUnderlier128b>::new(&allocator, &table, table_size);

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
}
