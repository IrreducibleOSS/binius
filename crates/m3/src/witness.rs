// Copyright 2025 Irreducible Inc.

use std::{
	cell::{Ref, RefCell, RefMut},
	iter,
	ops::{Deref, DerefMut},
	slice,
};

use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	PackedField, TowerField,
};
use getset::CopyGetters;

use super::error::Error;
use crate::constraint_system::{Col, ColumnId, ColumnShape, TableId};

/// Runtime borrow checking on columns. Maybe read-write lock?
#[derive(Debug, Default, CopyGetters)]
pub struct WitnessIndex<'a, U: UnderlierType = OptimalUnderlier> {
	tables: Vec<TableWitnessIndex<'a, U>>,
}

/// Runtime borrow checking on columns. Maybe read-write lock?
#[derive(Debug, Default, CopyGetters)]
pub struct TableWitnessIndex<'a, U: UnderlierType = OptimalUnderlier> {
	table_id: TableId,
	cols: Vec<RefCell<&'a mut [U]>>,
	#[get_copy = "pub"]
	log_capacity: usize,
}

impl<'a, U: UnderlierType> TableWitnessIndex<'a, U> {
	pub fn new(
		bump: &'a bumpalo::Bump,
		table_id: TableId,
		column_info: &[ColumnShape],
		log_capacity: usize,
	) -> Self {
		// TODO: Make packed columns alias the same slice data

		let cols = column_info
			.iter()
			.map(|info| {
				let log_col_bits = info.tower_height + info.pack_factor + log_capacity;
				// TODO: Error instead of panic
				if log_col_bits < U::LOG_BITS {
					panic!("capacity is too low");
				}
				// TODO: Allocate uninitialized memory and avoid filling. That should be OK because
				// Underlier is Pod.
				let col_data = bump.alloc_slice_fill_default(1 << (log_col_bits - U::LOG_BITS));
				RefCell::new(col_data)
			})
			.collect();
		Self {
			table_id,
			cols,
			log_capacity,
		}
	}

	pub fn capacity(&self) -> usize {
		1 << self.log_capacity
	}

	pub fn min_log_segment_size(&self) -> usize {
		// This value is conservative, but that should be fine in practice since underliers
		// only go up to 512 bits.
		U::BITS
	}

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

	// This doesn't compile yet
	/*
	pub fn segments(&mut self, log_size: usize) -> impl Iterator<Item = TableWitnessIndex<U>> + '_ {
		assert!(log_size < self.log_capacity);

		//let col_refs = self.cols.iter().map(|col| col.try_borrow().expect("the method borrows self mutably; any column borrows would require a reference to self; these must be the only column borrows")).collect::<Vec<_>>();
		(0..1 << self.log_capacity)
			.step_by(1 << log_size)
			.map(move |start| {
				let col_strides = self.cols
					.iter()
					.map(|col| {
						let col_ref = col.try_borrow().expect("the method borrows self mutably; any column borrows would require a reference to self; these must be the only column borrows");
						// Safety: The function borrows self mutably, so we have mutable access to
						// all columns and thus none can be borrowed by anyone else. The loop is
						// constructed to borrow disjoint segments of each column -- if this loop
						// were transposed, we would use `chunks_mut`.
						let col_stride = unsafe {
							cast_slice_ref_to_mut(&col_ref[start..start + (1 << log_size)])
						};
						RefCell::new(col_stride)
					})
					.collect();
				TableWitnessIndex {
					table_id: self.table_id,
					cols: col_strides,
					log_capacity: log_size,
				}
			})
	}
	 */

	// pub fn par_segments(
	// 	&mut self,
	// 	log_size: usize,
	// ) -> impl Iterator<Item = TableWitnessIndexStride<U>> {
	// 	todo!()
	// }
}

unsafe fn cast_slice_ref_to_mut<T>(slice: &[T]) -> &mut [T] {
	slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len())
}

/*
/// Runtime borrow checking on columns. Maybe read-write lock?
#[derive(Debug, Default, CopyGetters)]
pub struct TableWitnessIndexStride<'a, U: UnderlierType = OptimalUnderlier> {
	//table_index: &'a TableWitnessIndex<'a, U>,
	cols: Vec<RefCell<&'a mut [U]>>,
	#[get_copy = "pub"]
	log_size: usize,
}

impl<'a, U: UnderlierType> TableWitnessIndexStride<'a, U> {
	pub fn get<F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<Ref<impl Deref<Target = [PackedType<U, F>]>>, Error>
	where
		U: PackScalar<F>,
	{
		let table_cols =
			self.table_index
				.cols
				.get(col.table_id)
				.ok_or_else(|| Error::MissingTable {
					table_id: col.table_id,
				})?;
		let col = table_cols.get(col.index).ok_or_else(|| {
			Error::MissingColumn(ColumnId {
				table_id: col.table_id,
				index: col.index,
			})
		})?;
		Ok(col.borrow())
	}

	pub fn get_mut<F: TowerField, const V: usize>(
		&self,
		col: Col<F, V>,
	) -> Result<RefMut<impl DerefMut<Target = [PackedType<U, F>]>>, Error>
	where
		U: PackScalar<F>,
	{
		let table_cols = self
			.cols
			.get(col.table_id)
			.ok_or_else(|| Error::MissingTable {
				table_id: col.table_id,
			})?;
		let col = table_cols.get(col.index).ok_or_else(|| {
			Error::MissingColumn(ColumnId {
				table_id: col.table_id,
				index: col.index,
			})
		})?;
		Ok(col.borrow_mut())
	}
}
*/

/// A struct that can populate segments of a table witness using row descriptors.
trait TableFiller<U: UnderlierType = OptimalUnderlier> {
	/// A struct that specifies the row contents.
	type Row;

	type Error: std::error::Error + Send + Sync + 'static;

	/// Fill the table witness with data derived from the given rows.
	fn fill(&self, rows: &[Self::Row], witness: TableWitnessIndex<U>) -> Result<(), Self::Error>;
}

fn fill_table_sequential<T: TableFiller, U: UnderlierType>(
	table: &mut T,
	rows: &[T::Row],
	mut witness: TableWitnessIndex<U>,
) -> Result<(), T::Error> {
	let log_segment_size = witness.min_log_segment_size();
	for (row_chunk, mut witness_segment) in
		iter::zip(rows.chunks(log_segment_size), witness.segments(log_segment_size))
	{
		table.fill(row_chunk, &mut witness_segment)?;
	}

	// TODO: Handle the case when rows are not a multiple of the segment size

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
		let index = TableWitnessIndex::<OptimalUnderlier128b>::new(
			&bump,
			table_id,
			&col_shapes,
			log_capacity,
		);

		{
			let col0_ref0 = index.get(col0).unwrap();
			let col0_ref1 = index.get(col0).unwrap();
			assert_matches!(index.get_mut(col0), Err(Error::WitnessBorrowMut(_)));
			drop(col0_ref0);

			let col1_ref = index.get_mut(col1).unwrap();
			assert_matches!(index.get(col1), Err(Error::WitnessBorrow(_)));
			drop(col1_ref);
		}

		assert_eq!(len_packed_slice(&*index.get_mut(col0).unwrap()), 1 << 9);
		assert_eq!(len_packed_slice(&*index.get_mut(col1).unwrap()), 1 << 11);
		assert_eq!(len_packed_slice(&*index.get_mut(col2).unwrap()), 1 << 6);
		assert_eq!(len_packed_slice(&*index.get_mut(col3).unwrap()), 1 << 6);
	}

	/*
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
		let index = TableWitnessIndex::<OptimalUnderlier128b>::new(
			&bump,
			table_id,
			&col_shapes,
			log_capacity,
		);

		assert_eq!(index.min_log_segment_size(), 4);
	}
	*/
}
