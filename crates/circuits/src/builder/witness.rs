// Copyright 2024-2025 Irreducible Inc.

use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use anyhow::{Error, anyhow};
use binius_core::{
	oracle::{MultilinearOracleSet, OracleId},
	witness::{MultilinearExtensionIndex, MultilinearWitness},
};
use binius_field::{
	ExtensionField, PackedField, TowerField,
	as_packed_field::{PackScalar, PackedType},
	underlier::WithUnderlier,
};
use binius_math::MultilinearExtension;
use binius_utils::bail;
use bytemuck::{Pod, must_cast_slice, must_cast_slice_mut};

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
	nonzero_scalars_prefix: usize,
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
		let nonzero_scalars_prefix = 1 << self.oracles.borrow().n_vars(id);
		self.new_column_with_nonzero_scalars_prefix(id, nonzero_scalars_prefix)
	}

	pub fn new_column_with_nonzero_scalars_prefix<FS: TowerField>(
		&self,
		id: OracleId,
		nonzero_scalars_prefix: usize,
	) -> EntryBuilder<'arena, FS>
	where
		U: PackScalar<FS>,
		F: ExtensionField<FS>,
	{
		let oracles = self.oracles.borrow();
		let log_rows = oracles.n_vars(id);
		// TODO: validate nonzero_scalars_prefix
		let len = 1 << log_rows.saturating_sub(<PackedType<U, FS>>::LOG_WIDTH);
		let data = bumpalo::vec![in self.bump; U::default(); len].into_bump_slice_mut();
		EntryBuilder {
			_marker: PhantomData,
			entries: self.entries.clone(),
			id,
			log_rows,
			nonzero_scalars_prefix,
			data: Some(data),
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
		let nonzero_scalars_prefix = 1 << log_rows;
		let len = 1 << log_rows.saturating_sub(<PackedType<U, FS>>::LOG_WIDTH);
		let default = WithUnderlier::to_underlier(PackedType::<U, FS>::broadcast(default));
		let data = bumpalo::vec![in self.bump; default; len].into_bump_slice_mut();
		EntryBuilder {
			_marker: PhantomData,
			entries: self.entries.clone(),
			id,
			log_rows,
			nonzero_scalars_prefix,
			data: Some(data),
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
			.get(id.index())
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
			nonzero_scalars_prefix: entry.nonzero_scalars_prefix,
			_marker: PhantomData,
		})
	}

	pub fn set<FS: TowerField>(
		&self,
		oracle_id: OracleId,
		entry: WitnessEntry<'arena, FS>,
	) -> Result<(), Error>
	where
		U: PackScalar<FS>,
		F: ExtensionField<FS>,
	{
		let oracles = self.oracles.borrow();
		if !oracles.is_valid_oracle_id(oracle_id) {
			bail!(anyhow!("OracleId {oracle_id} does not exist in MultilinearOracleSet"));
		}
		let mut entries = self.entries.borrow_mut();
		let oracle_index = oracle_id.index();
		if oracle_index >= entries.len() {
			entries.resize_with(oracle_index + 1, || None);
		}
		entries[oracle_index] = Some(WitnessBuilderEntry {
			data: entry.data,
			nonzero_scalars_prefix: entry.nonzero_scalars_prefix,
			tower_level: FS::TOWER_LEVEL,
			witness: MultilinearExtension::new(entry.log_rows, entry.packed())
				.map(|x| x.specialize_arc_dyn()),
		});
		Ok(())
	}

	pub fn build(self) -> Result<MultilinearExtensionIndex<'arena, PackedType<U, F>>, Error> {
		let mut result = MultilinearExtensionIndex::new();
		let entries = Rc::into_inner(self.entries)
			.ok_or_else(|| anyhow!("Failed to build. There are still entries refs. Make sure there are no pending column insertions."))?
			.into_inner()
			.into_iter()
			.enumerate()
			.filter_map(|(index, entry)| entry.map(|entry| Ok((OracleId::from_index(index), entry.witness?, entry.nonzero_scalars_prefix))))
			.collect::<Result<Vec<_>, Error>>()?;
		result.update_multilin_poly_with_nonzero_scalars_prefixes(entries)?;
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
	nonzero_scalars_prefix: usize,
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
		let log_extension_degree = <FE as ExtensionField<FS>>::LOG_DEGREE;
		WitnessEntry {
			data: self.data,
			log_rows: self.log_rows - log_extension_degree,
			nonzero_scalars_prefix: self
				.nonzero_scalars_prefix
				.div_ceil(1 << log_extension_degree),
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
	nonzero_scalars_prefix: usize,
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
		let oracle_index = self.id.index();
		let nonzero_scalars_prefix = self.nonzero_scalars_prefix;
		if oracle_index >= entries.len() {
			entries.resize_with(oracle_index + 1, || None);
		}
		entries[oracle_index] = Some(WitnessBuilderEntry {
			data,
			nonzero_scalars_prefix,
			tower_level: FS::TOWER_LEVEL,
			witness: MultilinearExtension::new(
				self.log_rows,
				PackedType::<U, FS>::from_underliers_ref(data),
			)
			.map(|x| x.specialize_arc_dyn()),
		})
	}
}
