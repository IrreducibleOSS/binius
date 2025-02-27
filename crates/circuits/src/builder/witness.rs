// Copyright 2024-2025 Irreducible Inc.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use anyhow::{anyhow, Error};
use binius_core::{
	oracle::{MultilinearOracleSet, OracleId},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::WithUnderlier,
	ExtensionField, PackedField, TowerField,
};
use binius_math::MultilinearExtension;
use binius_utils::bail;
use bytemuck::{must_cast_slice, must_cast_slice_mut, Pod};

use super::types::{F, U};

pub struct Builder<'arena> {
	bump: &'arena bumpalo::Bump,

	oracles: Rc<RefCell<MultilinearOracleSet<F>>>,

	#[allow(clippy::type_complexity)]
	entries: Rc<RefCell<Vec<Option<WitnessBuilderEntry<'arena>>>>>,
}

struct WitnessBuilderEntry<'arena> {
	witness: Result<MultilinearWitness<'arena, PackedType<U, F>>, binius_math::Error>,
	tower_level: usize,
	data: &'arena [U],
}

impl<'arena> Builder<'arena> {
	pub fn new(
		allocator: &'arena bumpalo::Bump,
		oracles: Rc<RefCell<MultilinearOracleSet<F>>>,
	) -> Self {
		Self {
			bump: allocator,
			oracles,
			entries: Rc::new(RefCell::new(Vec::new())),
		}
	}

	pub fn new_column<FS: TowerField>(&self, id: OracleId) -> EntryBuilder<'arena, FS>
	where
		U: PackScalar<FS>,
		F: ExtensionField<FS>,
	{
		let oracles = self.oracles.borrow();
		let log_rows = oracles.n_vars(id);
		let len = 1 << log_rows.saturating_sub(<PackedType<U, FS>>::LOG_WIDTH);
		let data = bumpalo::vec![in self.bump; U::default(); len].into_bump_slice_mut();
		EntryBuilder {
			_marker: PhantomData,
			log_rows,
			id,
			data: Some(data),
			entries: self.entries.clone(),
		}
	}

	pub fn new_column_with_default<FS: TowerField>(
		&self,
		id: OracleId,
		default: FS,
	) -> EntryBuilder<'arena, FS>
	where
		U: PackScalar<FS>,
		F: ExtensionField<FS>,
	{
		let oracles = self.oracles.borrow();
		let log_rows = oracles.n_vars(id);
		let len = 1 << log_rows.saturating_sub(<PackedType<U, FS>>::LOG_WIDTH);
		let default = WithUnderlier::to_underlier(PackedType::<U, FS>::broadcast(default));
		let data = bumpalo::vec![in self.bump; default; len].into_bump_slice_mut();
		EntryBuilder {
			_marker: PhantomData,
			log_rows,
			id,
			data: Some(data),
			entries: self.entries.clone(),
		}
	}

	pub fn get<FS>(&self, id: OracleId) -> Result<WitnessEntry<'arena, FS>, Error>
	where
		FS: TowerField,
		U: PackScalar<FS>,
		F: ExtensionField<FS>,
	{
		let entries = self.entries.borrow();
		let oracles = self.oracles.borrow();
		if !oracles.is_valid_oracle_id(id) {
			bail!(anyhow!("OracleId {id} does not exist in MultilinearOracleSet"));
		}
		let entry = entries
			.get(id)
			.and_then(|entry| entry.as_ref())
			.ok_or_else(|| anyhow!("Witness for {} is missing", oracles.label(id)))?;

		if entry.tower_level != FS::TOWER_LEVEL {
			bail!(anyhow!(
				"Provided tower level ({}) for {} does not match stored tower level {}.",
				FS::TOWER_LEVEL,
				oracles.label(id),
				entry.tower_level
			));
		}

		Ok(WitnessEntry {
			data: entry.data,
			log_rows: oracles.n_vars(id),
			_marker: PhantomData,
		})
	}

	pub fn set<FS: TowerField>(
		&self,
		id: OracleId,
		entry: WitnessEntry<'arena, FS>,
	) -> Result<(), Error>
	where
		U: PackScalar<FS>,
		F: ExtensionField<FS>,
	{
		let oracles = self.oracles.borrow();
		if !oracles.is_valid_oracle_id(id) {
			bail!(anyhow!("OracleId {id} does not exist in MultilinearOracleSet"));
		}
		let mut entries = self.entries.borrow_mut();
		if id >= entries.len() {
			entries.resize_with(id + 1, || None);
		}
		entries[id] = Some(WitnessBuilderEntry {
			data: entry.data,
			tower_level: FS::TOWER_LEVEL,
			witness: MultilinearExtension::new(entry.log_rows, entry.packed())
				.map(|x| x.specialize_arc_dyn()),
		});
		Ok(())
	}

	pub fn build(self) -> Result<MultilinearExtensionIndex<'arena, U, F>, Error> {
		let mut result = MultilinearExtensionIndex::new();
		let entries = Rc::into_inner(self.entries)
			.ok_or_else(|| anyhow!("Failed to build. There are still entries refs. Make sure there are no pending column insertions."))?
			.into_inner()
			.into_iter()
			.enumerate()
			.filter_map(|(id, entry)| entry.map(|entry| Ok((id, entry.witness?))))
			.collect::<Result<Vec<_>, Error>>()?;
		result.update_multilin_poly(entries)?;
		Ok(result)
	}
}

#[derive(Debug, Clone, Copy)]
pub struct WitnessEntry<'arena, FS: TowerField>
where
	U: PackScalar<FS>,
{
	data: &'arena [U],
	log_rows: usize,
	_marker: PhantomData<FS>,
}

impl<'arena, FS: TowerField> WitnessEntry<'arena, FS>
where
	U: PackScalar<FS>,
{
	#[inline]
	pub fn packed(&self) -> &'arena [PackedType<U, FS>] {
		WithUnderlier::from_underliers_ref(self.data)
	}

	#[inline]
	pub const fn as_slice<T: Pod>(&self) -> &'arena [T] {
		must_cast_slice(self.data)
	}

	pub const fn repacked<FE>(&self) -> WitnessEntry<'arena, FE>
	where
		FE: TowerField + ExtensionField<FS>,
		U: PackScalar<FE>,
	{
		WitnessEntry {
			data: self.data,
			log_rows: self.log_rows - <FE as ExtensionField<FS>>::LOG_DEGREE,
			_marker: PhantomData,
		}
	}

	pub const fn low_rows(&self) -> usize {
		self.log_rows
	}
}

pub struct EntryBuilder<'arena, FS>
where
	FS: TowerField,
	U: PackScalar<FS>,
	F: ExtensionField<FS>,
{
	_marker: PhantomData<FS>,
	#[allow(clippy::type_complexity)]
	entries: Rc<RefCell<Vec<Option<WitnessBuilderEntry<'arena>>>>>,
	id: OracleId,
	log_rows: usize,
	data: Option<&'arena mut [U]>,
}

impl<FS> EntryBuilder<'_, FS>
where
	FS: TowerField,
	U: PackScalar<FS>,
	F: ExtensionField<FS>,
{
	#[inline]
	pub fn packed(&mut self) -> &mut [PackedType<U, FS>] {
		PackedType::<U, FS>::from_underliers_ref_mut(self.underliers())
	}

	#[inline]
	pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
		must_cast_slice_mut(self.underliers())
	}

	#[inline]
	const fn underliers(&mut self) -> &mut [U] {
		self.data
			.as_mut()
			.expect("Should only be None after Drop::drop has run")
	}
}

impl<FS> Drop for EntryBuilder<'_, FS>
where
	FS: TowerField,
	U: PackScalar<FS>,
	F: ExtensionField<FS>,
{
	fn drop(&mut self) {
		let data = Option::take(&mut self.data).expect("data is always Some until this point");
		let mut entries = self.entries.borrow_mut();
		let id = self.id;
		if id >= entries.len() {
			entries.resize_with(id + 1, || None);
		}
		entries[id] = Some(WitnessBuilderEntry {
			data,
			tower_level: FS::TOWER_LEVEL,
			witness: MultilinearExtension::new(
				self.log_rows,
				PackedType::<U, FS>::from_underliers_ref(data),
			)
			.map(|x| x.specialize_arc_dyn()),
		})
	}
}
