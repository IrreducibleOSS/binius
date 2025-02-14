// Copyright 2024-2025 Irreducible Inc.

pub mod table;
// use core::slice::SlicePattern;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

// use anyhow::{anyhow, Error};
use binius_core::{
	constraint_system::channel::Boundary,
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, MultilinearPolyVariant, OracleId},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
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

use crate::{constraint_system::ConstraintSystem, U};

// TABLE ENTRY

#[derive(Default)]
pub struct TableEntry<'a> {
	pub underliers: Vec<&'a [U]>,
	pub count: Option<usize>,
}

// WITNESS BUILDER

pub struct WitnessBuilder<'a> {
	bump: &'a Bump,
	constraint_system: Rc<RefCell<&'a mut ConstraintSystem>>,
	entries: Vec<Rc<RefCell<TableEntry<'a>>>>,
	boundaries: Rc<RefCell<Vec<Boundary<B128>>>>,
	// tables_to_oracles: Vec<Vec<OracleId>>,
	// virtuals: Vec<Arc<Ref
}

impl<'a> WitnessBuilder<'a> {
	// constraint system assumed uninitialized and consistent, ie output after build equivalent
	pub fn new(bump: &'a Bump, constraint_system: &'a mut ConstraintSystem) -> Self {
		let default_table_entry = Rc::new(RefCell::new(TableEntry::default()));
		let table_count = constraint_system.tables.len();

		constraint_system.fill_tables_to_oracles();

		Self {
			bump,
			constraint_system: Rc::new(RefCell::new(constraint_system)),
			entries: vec![default_table_entry; table_count],
			boundaries: Rc::new(RefCell::new(Vec::new())),
		}
	}

	// pub fn new_table_filler<Input: Filler>(
	// 	&self,
	// 	builder: &WitnessBuilder<'a>,
	// 	input_count: usize,
	// ) -> TableFiller<'a, Input> {
	// 	TableFiller::<Input>::new(builder, input_count)
	// }

	// tables can also collect boundary values,
	pub fn add_boundary(&mut self, boundary: Boundary) {
		self.boundaries.borrow_mut().push(boundary);
	}

	pub fn get_table_entry(
		&self,
		table_name: impl ToString,
	) -> Result<
		(
			&'a Bump,
			Rc<RefCell<TableEntry<'a>>>,
			Rc<RefCell<&'a mut ConstraintSystem>>, // this needs to change, shouldn't be mutable access here
			Rc<RefCell<Vec<Boundary>>>,
		),
		Error,
	> {
		let table_name = table_name.to_string();
		let constraint_system = self.constraint_system.clone();

		let borrowed_constraint_system = constraint_system.borrow();
		let table = borrowed_constraint_system
			.tables
			.iter()
			.find(|table| table.name == table_name)
			.ok_or(Error::MissingTable(table_name))?;

		let entry = self.entries[table.id].clone();
		let boundaries = self.boundaries.clone();

		Ok((self.bump, entry, constraint_system.clone(), boundaries))
	}

	pub fn build(self) -> Result<(Vec<Boundary>, MultilinearExtensionIndex<'a>), Error> {
		let Self {
			bump,
			constraint_system,
			entries,
			boundaries,
		} = self;

		let constraint_system = Rc::into_inner(constraint_system)
			.expect("dangling refs")
			.into_inner();

		let boundary_values = Rc::into_inner(boundaries)
			.ok_or(Error::RemainingRcRefs)?
			.into_inner();

		// for each table holds the ids of original oracles in that table (in increasing order)
		let tables_to_originals: Vec<Vec<OracleId>> = constraint_system
			.tables_to_oracles
			.as_ref()
			.expect("we computed it")
			.iter()
			.map(|table_oracle_ids| {
				table_oracle_ids
					.iter()
					.filter(|&&id| match constraint_system.oracles.oracle(id).variant {
						MultilinearPolyVariant::Committed => true,
						// MultilinearPolyVariant::Deterministic => true,
						_ => false,
					})
					.copied()
					.collect()
			})
			.collect();

		let tables_to_original_underliers =
			fill_in_committed_n_vars(constraint_system, entries, &tables_to_originals).unwrap();

		hydrate_constraint_system(constraint_system).unwrap();

		let original_ids_with_underliers = {
			let mut pairs = tables_to_originals
				.into_iter()
				.zip(tables_to_original_underliers)
				.flat_map(|(table_original_ids, table_underliers)| {
					table_original_ids.into_iter().zip(table_underliers)
				})
				.collect::<Vec<_>>();

			pairs.sort_by(|x, y| x.0.cmp(&y.0));
			pairs
		};

		let all_oracles_underliers =
			fill_virtuals(constraint_system, bump, original_ids_with_underliers);

		let extension = build_witness_extension(&*constraint_system, all_oracles_underliers);

		Ok((boundary_values, extension))
	}
}

// FILL ORIGINALS N_VARS

fn process_entry<'a>(
	table_id: TableId,
	entry: TableEntry<'a>,
	table_original_ids: &[OracleId],
	constraint_system: &mut ConstraintSystem,
) -> Result<Vec<&'a [U]>, Error> {
	// let table = &constraint_system.tables[table_id];
	// let committed_oracle_ids = table.get_committed_oracle_ids();

	// // for now we'll have n_vars at least PackedType::<U, B>::LOG_WIDTH
	// // worst case in TableEntry we add a field for n_vars
	// let n_vars = underliers.len().trailing_zeros() as usize + PackedType::<U, B>::LOG_WIDTH;
	let oracles = &mut constraint_system.oracles;

	let given_slice_count = entry.underliers.len();
	let expected_slice_count = table_original_ids.len();
	if given_slice_count != expected_slice_count {
		bail!(Error::IncorrectSliceCount(given_slice_count, expected_slice_count))
	}

	let my_ref_list = table_original_ids
		.iter()
		.zip(entry.underliers)
		.map(|(&table_original_id, underliers)| {
			//
			let tower_level = oracles.tower_level(table_original_id);
			//

			if !underliers.len().is_power_of_two() {
				// this should actually go in the caller ???
				// bail!(Error::NonPowerOfTwoLengthUnderliers(
				// 	"table name".to_string(),
				// 	"col name".to_string(),
				// 	underliers.len()
				// ));
			}

			// for now we'll have n_vars at least PackedType::<U, B>::LOG_WIDTH
			// worst case in TableEntry we add a field for n_vars
			// correctly compute this using tower level
			let n_vars = underliers.len().trailing_zeros() as usize + U::LOG_BITS;

			let oracle_ref = oracles.get_mut_ref(table_original_id).unwrap();
			oracle_ref.set_n_vars(n_vars);

			underliers
		})
		.collect::<Vec<_>>();

	Ok(my_ref_list)
}

fn fill_in_committed_n_vars<'a>(
	constraint_system: &mut ConstraintSystem,
	entries: Vec<Rc<RefCell<TableEntry<'a>>>>,
	tables_to_originals: &[Vec<OracleId>],
) -> Result<Vec<Vec<&'a [U]>>, Error> {
	let oracles = &constraint_system.oracles;
	// let tables_to_oracles = self.constraint_system.tables_to_oracles();

	let ref_list = entries
		.into_iter()
		.zip(tables_to_originals)
		.enumerate()
		.map(|(table_id, (entry, table_original_ids))| -> Vec<&[U]> {
			let entry = Rc::into_inner(entry)
				.ok_or(Error::RemainingRcRefs)
				.unwrap()
				.into_inner();

			let ref_list =
				process_entry(table_id, entry, table_original_ids, constraint_system).unwrap();

			// Ok(ref_list)
			ref_list
		})
		.collect::<Vec<_>>();

	Ok(ref_list)
}

// BUILD WITNESS EXTENSION

fn get_multilinear_witness<'a, B: TowerField>(
	// table_name: impl ToString,
	// col_name: impl ToString,
	underliers: &'a [U],
) -> Result<MultilinearWitness<'a>, Error>
where
	B128: ExtensionField<B>,
	U: PackScalar<B>,
{
	if !underliers.len().is_power_of_two() {
		// this should actually go in the caller
		bail!(Error::NonPowerOfTwoLengthUnderliers(
			"table name".to_string(),
			"col name".to_string(),
			underliers.len()
		));
	}

	// for now we'll have n_vars at least PackedType::<U, B>::LOG_WIDTH
	// worst case in TableEntry we add a field for n_vars
	let n_vars = underliers.len().trailing_zeros() as usize + PackedType::<U, B>::LOG_WIDTH;

	let packed = PackedType::<U, B>::from_underliers_ref(underliers);
	let multilinear_extension = MultilinearExtension::new(n_vars, packed)?;
	let multilinear_witness = multilinear_extension.specialize_arc_dyn();
	Ok(multilinear_witness)
}

// given the constraint system and the underliers, the refs for the oracles, make and return the extension
fn build_witness_extension<'a>(
	constraint_system: &ConstraintSystem,
	underliers: Vec<&'a [U]>,
) -> MultilinearExtensionIndex<'a> {
	let mut extension_index = MultilinearExtensionIndex::new();

	let witnesses = (0..constraint_system.oracles.size())
		.zip(underliers)
		.map(|(oracle_id, underliers)| {
			let tower_level = constraint_system.oracles.tower_level(oracle_id);

			let multilinear_witness = match tower_level {
				0 => get_multilinear_witness::<B1>(underliers),
				3 => get_multilinear_witness::<B8>(underliers),
				4 => get_multilinear_witness::<B16>(underliers),
				5 => get_multilinear_witness::<B32>(underliers),
				6 => get_multilinear_witness::<B64>(underliers),
				7 => get_multilinear_witness::<B128>(underliers),
				_ => panic!("argue this is unreachable"),
			}?;
			Ok((oracle_id, multilinear_witness))
		})
		.collect::<Result<Vec<_>, Error>>()
		.unwrap();

	extension_index.update_multilin_poly(witnesses).unwrap();

	extension_index
}
