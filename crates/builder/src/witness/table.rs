// Copyright 2024-2025 Irreducible Inc.

// use core::slice::SlicePattern;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

// use anyhow::{anyhow, Error};
use binius_core::{
	constraint_system::{channel, Table, TableId},
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, MultilinearPolyVariant, OracleId},
	// witness::MultilinearWitness,
};
use binius_field::{
	arch::OptimalUnderlier,
	as_packed_field::{PackScalar, PackedType},
	underlier::{Divisible, UnderlierType, WithUnderlier},
	BinaryField128b as B128, BinaryField16b as B16, BinaryField1b as B1, BinaryField32b as B32,
	BinaryField64b as B64, BinaryField8b as B8, ExtensionField, Field, PackedField, TowerField,
};
use binius_math::MultilinearExtension;
use binius_utils::bail;
use bumpalo::Bump;
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};
use itertools::izip;

use super::{
	consistency::hydrate_constraint_system,
	error::Error,
	virtuals::fill_virtuals,
	witness::{TableEntry, WitnessBuilder},
};

type U = OptimalUnderlier;

type Boundary = channel::Boundary<B128>;
type ConstraintSystem = binius_core::constraint_system::ConstraintSystem<B128>;
type MultilinearExtensionIndex<'a> =
	binius_core::witness::MultilinearExtensionIndex<'a, OptimalUnderlier, B128>;
type MultilinearWitness<'a> = binius_core::witness::MultilinearWitness<'a, PackedType<U, B128>>;

// TRANSFORMER
pub trait OriginalFiller
where
	Self: Sized,
{
	// type Input;
	const ROWS_PER_INPUT: usize;
	fn name() -> impl ToString;
	fn inputs_per_batch() -> usize;
	fn populate<'a>(inputs: &[Self], underliers: &mut [&'a mut [U]]);
	// // fn col_infos(&self) -> Vec<>,
	// fn col_lens(&self) -> Vec<usize>;
	// fn rows_per_input(&self) -> usize;
}

// pub struct Info {}

// TABLE BUILDER

pub struct TableFiller<'a, Input: OriginalFiller> {
	// populator: Input,
	input_count: usize,
	batch: Vec<Input>,
	next_index_in_batch: usize,
	batch_count: usize,
	underliers: Vec<&'a mut [U]>,
	table_entry: Rc<RefCell<TableEntry<'a>>>,
	constraint_system: Rc<RefCell<&'a mut ConstraintSystem>>,
	boundaries: Rc<RefCell<Vec<Boundary>>>,
}

impl<'a, Input: OriginalFiller> TableFiller<'a, Input> {
	pub fn new(builder: &WitnessBuilder<'a>, input_count: usize) -> Self {
		let (bump, table_entry, constraint_system, boundaries) =
			builder.get_table_entry(Input::name()).unwrap();

		// we need to allocate the mutable ref.
		// this is at the table level.
		// we're supposed to allocate for all the originals.
		// we use the constraint system t
		// we need to access the oracles for this table.
		// that is, we need tables to oracles.

		// use the constraint system to go through every column, get it's size and then allocate
		//
		let underliers = vec![];
		// populator
		// 	.col_lens()
		// 	.into_iter()
		// 	.map(|col_len| bumpalo::vec![in bump; U::default(); col_len].into_bump_slice_mut())
		// 	.collect();

		let batch_size = Input::inputs_per_batch();

		Self {
			// populator,
			input_count,
			batch: Vec::with_capacity(batch_size),
			next_index_in_batch: 0,
			batch_count: 0,
			underliers,
			table_entry,
			constraint_system,
			boundaries,
		}
	}

	#[inline]
	pub fn append(&mut self, input: Input) {
		self.batch[self.next_index_in_batch] = input;
		self.next_index_in_batch += 1;

		if self.next_index_in_batch == Input::inputs_per_batch() {
			// we've reached a full batch
			// oh, we're supposed to now compute the slice on which to call the transformer.
			// recall we're supposed to pass a populator a list of inputs and mutable refs to the columns
			//

			// let underliers = self.underliers;
			// we need to make new mutable references basd on the current index.
			// what's the current index, and what's the current length?

			let inputs_index = self.batch_count * self.batch.len();
			// let underliers_index = input_index * underliers_per_input;

			let underliers = self.underliers.iter_mut().map(|slice| {
				// we need to compute for this column, where is the
				//
				let tower_height = 2;

				let elements_per_underlier = PackedType::<U, B32>::WIDTH;

				//
				let start = 2;
				let end = 4;
				let subslice = &mut slice[start..end];
				subslice
				//
			});

			// let underlier_indices = &mut self.underliers[0][start..end];

			// let x = underliers.iter_mut();
			Input::populate(&self.batch, underliers);

			// self.underliers = x.collect::<Vec<_>>();

			self.next_index_in_batch = 0;
			self.batch_count += 1;
		}
	}
}

impl<'a, Input: OriginalFiller> Drop for TableFiller<'a, Input> {
	fn drop(&mut self) {
		let mut table_entry = self.table_entry.borrow_mut();

		// set count
		let input_count = Input::inputs_per_batch() * self.batch_count + self.next_index_in_batch;
		let count = input_count * Input::ROWS_PER_INPUT;
		table_entry.count = Some(self.input_count);

		// set underliers
		table_entry.underliers = std::mem::take(&mut self.underliers)
			.into_iter()
			.map(|underliers| &*underliers)
			.collect();
	}
}

// EXAMPLE

// fn example() {
// 	let bump = Bump::new();
// 	let mut constraint_system = ConstraintSystem::new();
// 	let builder = WitnessBuilder::new(&bump, &mut constraint_system);

// 	let my_trans = MyTransformer {};
// 	let row = Row { xin: 2, yin: 3 };

// 	let mut table = TableBuilder::new(&builder, 3, my_trans);
// 	table.append(row);

// 	// here we are going through the trace just appending rows
// 	// it's my_trans that says how to transform each and how to fill the committed columns
// }

// struct Row {
// 	xin: u32,
// 	yin: u32,
// }

// #[derive(Default)]
// struct MyTransformer {
// 	//
// }

// fn as_mut_slice

// impl<U, FW, FS> EntryBuilder<'_, U, FW, FS>
// where
// 	U: PackScalar<FW> + PackScalar<FS> + Pod,
// 	FS: TowerField,
// 	FW: TowerField + ExtensionField<FS>,
// {
// 	#[inline]
// 	pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
// 		must_cast_slice_mut(self.underliers())
// 	}
// }

// struct MyRefs<'a> {
// 	left: &'a mut [u32],
// }

// impl Populator for MyTransformer {
// 	type Input = Row;
// 	fn transform<'a>(&self, input: &[Row], mut underliers: impl Iterator<Item = &'a mut [U]>) {
// 		// let mut x_underliers = vec![U::default(); 3];
// 		// let mut y_underliers = vec![U::default(); 3];
// 		// let unders = underliers.collect::<Vec<_>>();

// 		// so suppose here
// 		let col0 = underliers.next().unwrap().as_mut();
// 		let col1 = underliers.next().unwrap().as_mut();
// 		//
// 		//
// 		//
// 		// let x = left[0].as_mut();
// 		// let y = right[0].as_mut();
// 		// x[0] = 2;
// 		// y[0] = 4;
// 		// x[0] = 1;
// 		// let x = underliers.get_many_mut([0, 1]).unwrap();

// 		// ok this means

// 		// we could iterate across the columns I guess, and batch writes to each column.

// 		// **x = 30;

// 		// let x: &mut [u32] = must_cast_slice_mut::<U, u32>(x_underliers.as_mut_slice());
// 		// let y: &mut [u32] = must_cast_slice_mut::<U, u32>(y_underliers.as_mut_slice());

// 		// let z: &mut [u32] = underliers[2];

// 		// izip!(input.iter(), x.iter_mut(), y.iter_mut(), z.iter_mut()).map(|(input, x, y, z)| {
// 		// 	*x = input.xin;
// 		// 	*y = input.yin;
// 		// 	*z = *x & *y;
// 		// });
// 	}
// 	fn name(&self) -> impl ToString {
// 		"my_table"
// 	}
// 	fn col_lens(&self) -> Vec<usize> {
// 		let n_vars = 10;
// 		let log_len = n_vars - PackedType::<U, B32>::LOG_WIDTH;
// 		let len = 1 << log_len;

// 		vec![len, len]
// 	}
// 	fn inputs_per_batch(&self) -> usize {
// 		1
// 	}
// 	fn rows_per_input(&self) -> usize {
// 		32
// 	}
// }
