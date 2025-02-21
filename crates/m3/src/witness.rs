// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter,
	ops::{Deref, DerefMut},
	slice,
};

use binius_core::witness::MultilinearExtensionIndex;
use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	Field, PackedField, TowerField,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::iter::IterExtensions;
use bytemuck::{must_cast_slice, Pod};
use getset::CopyGetters;

use super::error::Error;
use crate::{
	builder::Col,
	constraint_system::{ColumnId, ColumnShape, TableId},
};

/// Runtime borrow checking on columns. Maybe read-write lock?
#[derive(Debug, Default, CopyGetters)]
pub struct WitnessIndex<'a, U: UnderlierType = OptimalUnderlier> {
	tables: Vec<TableWitnessIndex<'a, U>>,
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
		let table = self
			.get_table(table_id)
			.ok_or(Error::MissingTable { table_id })?;
		fill_table_sequential(table, rows, table).map_err(|err| Error::TableFill(Box::new(err)))?;
		Ok(())
	}

	pub fn into_multilinear_extension_index<F>(self) -> MultilinearExtensionIndex<'a, U, F>
	where
		F: Field,
		U: PackScalar<F>,
	{
		todo!()
	}
}

/// Runtime borrow checking on columns. Maybe read-write lock?
#[derive(Debug, Default, CopyGetters)]
pub struct TableWitnessIndex<'a, U: UnderlierType = OptimalUnderlier> {
	table_id: TableId,
	cols: Vec<(&'a mut [U], usize)>,
	#[get_copy = "pub"]
	log_capacity: usize,
	/// Binary logarithm of the mininimum segment size.
	///
	/// This is the minimum number of logical rows that can be put into one segment during
	/// iteration. It is the maximum number of logical rows occupied by a single underlier.
	#[get_copy = "pub"]
	min_log_segment_size: usize,
}

impl<'a, U: UnderlierType> TableWitnessIndex<'a, U> {
	pub fn new(
		bump: &'a bumpalo::Bump,
		table_id: TableId,
		column_info: &[ColumnShape],
		log_capacity: usize,
	) -> Self {
		// TODO: Make packed columns alias the same slice data

		let mut min_log_cell_bits = U::LOG_BITS;
		let cols = column_info
			.iter()
			.map(|info| {
				let log_cell_bits = info.tower_height + info.pack_factor;
				min_log_cell_bits = min_log_cell_bits.min(log_cell_bits);

				let log_col_bits = log_cell_bits + log_capacity;
				// TODO: Error instead of panic
				if log_col_bits < U::LOG_BITS {
					panic!("capacity is too low");
				}
				// TODO: Allocate uninitialized memory and avoid filling. That should be OK because
				// Underlier is Pod.
				let col_data = bump.alloc_slice_fill_default(1 << (log_col_bits - U::LOG_BITS));
				(col_data, log_cell_bits)
			})
			.collect();
		Self {
			table_id,
			cols,
			log_capacity,
			min_log_segment_size: U::LOG_BITS - min_log_cell_bits,
		}
	}

	pub fn capacity(&self) -> usize {
		1 << self.log_capacity
	}

	pub fn full_segment(&mut self) -> TableWitnessIndexSegment<U> {
		let cols = self
			.cols
			.iter_mut()
			.map(|(col, _)| RefCell::new(&mut **col))
			.collect::<Vec<_>>();
		TableWitnessIndexSegment {
			table_id: self.table_id,
			cols,
			log_size: self.log_capacity,
		}
	}

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
				.map(|(col_ref, log_cell_bits)| {
					let log_stride = log_size + log_cell_bits - U::LOG_BITS;
					// Safety: The function borrows self mutably, so we have mutable access to
					// all columns and thus none can be borrowed by anyone else. The loop is
					// constructed to borrow disjoint segments of each column -- if this loop
					// were transposed, we would use `chunks_mut`.
					let col_stride = unsafe {
						cast_slice_ref_to_mut(&col_ref[i << log_stride..(i + 1) << log_stride])
					};
					RefCell::new(col_stride)
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
					.map(|(col_ref, log_cell_bits)| {
						let log_stride = log_size + log_cell_bits - U::LOG_BITS;
						// Safety: The function borrows self mutably, so we have mutable access to
						// all columns and thus none can be borrowed by anyone else. The loop is
						// constructed to borrow disjoint segments of each column -- if this loop
						// were transposed, we would use `chunks_mut`.
						let col_stride = unsafe {
							cast_slice_ref_to_mut(
								&col_ref[start << log_stride..(start + 1) << log_stride],
							)
						};
						RefCell::new(col_stride)
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

/// Runtime borrow checking on columns. Maybe read-write lock?
#[derive(Debug, Default, CopyGetters)]
pub struct TableWitnessIndexSegment<'a, U: UnderlierType = OptimalUnderlier> {
	table_id: TableId,
	cols: Vec<RefCell<&'a mut [U]>>,
	#[get_copy = "pub"]
	log_size: usize,
}

impl<'a, U: UnderlierType> TableWitnessIndexSegment<'a, U> {
	pub fn get<F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<Ref<[PackedType<U, F>]>, Error>
	where
		U: PackScalar<F>,
	{
		// TODO: Check consistency of static params with column info

		if col.table_id != self.table_id {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table_id,
			});
		}

		// let table_col = self
		// 	.cols
		// 	.get(col.table_id)
		// 	.ok_or_else(|| Error::MissingTable {
		// 		table_id: col.table_id,
		// 	})?;
		let col = self.cols.get(col.index).ok_or_else(|| {
			Error::MissingColumn(ColumnId {
				table_id: col.table_id,
				index: col.index,
			})
		})?;
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

		if col.table_id != self.table_id {
			return Err(Error::TableMismatch {
				column_table_id: col.table_id,
				witness_table_id: self.table_id,
			});
		}

		let col = self.cols.get(col.index).ok_or_else(|| {
			Error::MissingColumn(ColumnId {
				table_id: col.table_id,
				index: col.index,
			})
		})?;
		let col_ref = col.try_borrow_mut().map_err(Error::WitnessBorrowMut)?;
		Ok(RefMut::map(col_ref, |x| <PackedType<U, F>>::from_underliers_ref_mut(x)))
	}

	// TODO: Something similar to WitnessEntry::as_slice<T: Pod> that casts the underliers to T
	pub fn get_as<F: TowerField, const V: usize, T: Pod>(
		&self,
		col: Col<F, V>,
	) -> Result<Ref<[T]>, Error>
	where
		U: Pod,
	{
		todo!()
	}

	// TODO: Something similar to WitnessEntry::as_slice<T: Pod> that casts the underliers to T
	pub fn get_mut_as<F: TowerField, const V: usize, T: Pod>(
		&self,
		col: Col<F, V>,
	) -> Result<RefMut<[T]>, Error>
	where
		U: Pod,
	{
		todo!()
	}

	pub fn size(&self) -> usize {
		1 << self.log_size
	}
}

unsafe fn cast_slice_ref_to_mut<T>(slice: &[T]) -> &mut [T] {
	slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len())
}

/// A struct that can populate segments of a table witness using row descriptors.
pub trait TableFiller<U: UnderlierType = OptimalUnderlier> {
	/// A struct that specifies the row contents.
	type Event;

	type Error;

	fn id(&self) -> TableId;

	/// Fill the table witness with data derived from the given rows.
	fn fill(
		&self,
		rows: &[Self::Event],
		witness: &mut TableWitnessIndexSegment<U>,
	) -> Result<(), Self::Error>;
}

pub fn fill_table_sequential<U: UnderlierType, T: TableFiller<U>>(
	table: &T,
	rows: &[T::Event],
	mut witness: &mut TableWitnessIndex<U>,
) -> Result<(), T::Error> {
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

// Also, fill_table_parallel
// Also, a streaming version that streams in rows and fills in a background thread pool.

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{arch::OptimalUnderlier128b, packed::len_packed_slice};

	use super::*;
	use crate::types::{B1, B32, B8};

	#[test]
	fn test_table_witness_borrows() {
		let table_id = 0;

		let col0 = <Col<B1, 3>>::new(ColumnId { table_id, index: 0 });
		let col1 = <Col<B1, 5>>::new(ColumnId { table_id, index: 1 });
		let col2 = <Col<B8>>::new(ColumnId { table_id, index: 2 });
		let col3 = <Col<B32>>::new(ColumnId { table_id, index: 3 });

		let col_shapes = [col0.shape(), col1.shape(), col2.shape(), col3.shape()];

		let bump = bumpalo::Bump::new();
		let log_capacity = 6;
		let table_id = 0;
		let mut index = TableWitnessIndex::<OptimalUnderlier128b>::new(
			&bump,
			table_id,
			&col_shapes,
			log_capacity,
		);
		let segment = index.full_segment();

		{
			let col0_ref0 = segment.get(col0).unwrap();
			let col0_ref1 = segment.get(col0).unwrap();
			assert_matches!(segment.get_mut(col0), Err(Error::WitnessBorrowMut(_)));
			drop(col0_ref0);

			let col1_ref = segment.get_mut(col1).unwrap();
			assert_matches!(segment.get(col1), Err(Error::WitnessBorrow(_)));
			drop(col1_ref);
		}

		assert_eq!(len_packed_slice(&*segment.get_mut(col0).unwrap()), 1 << 9);
		assert_eq!(len_packed_slice(&*segment.get_mut(col1).unwrap()), 1 << 11);
		assert_eq!(len_packed_slice(&*segment.get_mut(col2).unwrap()), 1 << 6);
		assert_eq!(len_packed_slice(&*segment.get_mut(col3).unwrap()), 1 << 6);
	}

	#[test]
	fn test_table_witness_segments() {
		let table_id = 0;

		let col0 = <Col<B1, 3>>::new(ColumnId { table_id, index: 0 });
		let col1 = <Col<B1, 5>>::new(ColumnId { table_id, index: 1 });
		let col2 = <Col<B8>>::new(ColumnId { table_id, index: 2 });
		let col3 = <Col<B32>>::new(ColumnId { table_id, index: 3 });

		let col_shapes = [col0.shape(), col1.shape(), col2.shape(), col3.shape()];

		let bump = bumpalo::Bump::new();
		let log_capacity = 6;
		let table_id = 0;
		let mut index = TableWitnessIndex::<OptimalUnderlier128b>::new(
			&bump,
			table_id,
			&col_shapes,
			log_capacity,
		);

		assert_eq!(index.min_log_segment_size(), 4);
		let mut iter = index.segments(5);
		let mut seg0 = iter.next().unwrap();
		let mut seg1 = iter.next().unwrap();
		assert!(iter.next().is_none());

		assert_eq!(len_packed_slice(&*seg0.get_mut(col0).unwrap()), 1 << 8);
		assert_eq!(len_packed_slice(&*seg0.get_mut(col1).unwrap()), 1 << 10);
		assert_eq!(len_packed_slice(&*seg0.get_mut(col2).unwrap()), 1 << 5);
		assert_eq!(len_packed_slice(&*seg0.get_mut(col3).unwrap()), 1 << 5);

		assert_eq!(len_packed_slice(&*seg1.get_mut(col0).unwrap()), 1 << 8);
		assert_eq!(len_packed_slice(&*seg1.get_mut(col1).unwrap()), 1 << 10);
		assert_eq!(len_packed_slice(&*seg1.get_mut(col2).unwrap()), 1 << 5);
		assert_eq!(len_packed_slice(&*seg1.get_mut(col3).unwrap()), 1 << 5);
	}
}
