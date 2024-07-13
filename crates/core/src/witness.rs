// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::OracleId,
	polynomial::{
		util::PackingDeref, Error as PolynomialError, MultilinearExtension,
		MultilinearExtensionBorrowed, MultilinearPoly,
	},
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	ExtensionField, Field, PackedField, TowerField,
};
use std::{fmt::Debug, sync::Arc};

pub type MultilinearWitness<'a, P> = Arc<dyn MultilinearPoly<P> + Send + Sync + 'a>;

/// Data structure that indexes multilinear polynomial witnesses by oracle ID.
///
/// A [`crate::oracle::MultilinearOracleSet`] indexes multilinear polynomial oracles by assigning
/// unique, sequential  oracle IDs. This index stores the corresponding witnesses, as
/// [`MultilinearPoly`] trait objects. Not every oracle is required to have a stored witness -- in
/// some cases, only a derived multilinear witness is required.
///
/// *DEPRECATED*: See the [`MultilinearExtensionIndex`] documentation.
#[derive(Default, Debug)]
pub struct MultilinearWitnessIndex<'a, P: PackedField> {
	multilinears: Vec<Option<MultilinearWitness<'a, P>>>,
}

impl<'a, P> MultilinearWitnessIndex<'a, P>
where
	P: PackedField,
{
	pub fn new() -> Self {
		Self::default()
	}

	#[allow(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.multilinears.len()
	}

	pub fn get(&self, id: OracleId) -> Option<&MultilinearWitness<'a, P>> {
		self.multilinears.get(id)?.as_ref()
	}

	pub fn set(&mut self, id: OracleId, witness: MultilinearWitness<'a, P>) {
		if id >= self.multilinears.len() {
			self.multilinears.resize(id + 1, None);
		}
		self.multilinears[id] = Some(witness);
	}

	pub fn set_many(
		&mut self,
		witnesses: impl IntoIterator<Item = (OracleId, MultilinearWitness<'a, P>)>,
	) {
		for (id, witness) in witnesses {
			self.set(id, witness);
		}
	}
}

#[derive(Debug)]
struct MultilinearExtensionBacking<'a, U: UnderlierType> {
	underliers: ArcOrRef<'a, [U]>,
	tower_level: usize,
}

#[derive(Debug)]
struct MultilinearExtensionIndexEntry<'a, U: UnderlierType, F>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
{
	type_erased: MultilinearWitness<'a, PackedType<U, F>>,
	backing: Option<MultilinearExtensionBacking<'a, U>>,
}

/// Data structure that indexes multilinear extensions by oracle ID.
///
/// A [`crate::oracle::MultilinearOracleSet`] indexes multilinear polynomial oracles by assigning
/// unique, sequential oracle IDs. This struct is similar to [`MultilinearWitnessIndex`] in that
/// stores the corresponding multilinear extensions for each oracle. However, unlike
/// [`MultilinearWitnessIndex`], with [`MultilinearExtensionIndex`], the caller can get the
/// [`MultilinearExtension`] defined natively over a subfield. This is possible because the
/// [`MultilinearExtensionIndex::get`] method is generic over the subfield type and the struct
/// itself only stores the underlying data.
///
/// This does provide a superset of the functionality of [`MultilinearWitnessIndex`]. We plan to
/// entirely replace [`MultilinearWitnessIndex`] eventually, but it is kept around for backwards
/// compatibility.
#[derive(Default, Debug)]
pub struct MultilinearExtensionIndex<'a, U: UnderlierType, FW>
where
	U: UnderlierType + PackScalar<FW>,
	FW: Field,
{
	entries: Vec<Option<MultilinearExtensionIndexEntry<'a, U, FW>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("witness not found for oracle {id}")]
	MissingWitness { id: OracleId },
	#[error("witness for oracle id {id} does not have an explicit backing multilinear")]
	NoExplicitBackingMultilinearExtension { id: OracleId },
	#[error("oracle tower height does not match field parameter")]
	OracleTowerHeightMismatch {
		oracle_id: OracleId,
		oracle_level: usize,
		field_level: usize,
	},
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
}

impl<'a, U, FW> MultilinearExtensionIndex<'a, U, FW>
where
	U: UnderlierType + PackScalar<FW>,
	FW: Field,
{
	pub fn new() -> Self {
		Self::default()
	}

	pub fn get<FS>(
		&self,
		id: OracleId,
	) -> Result<MultilinearExtensionBorrowed<PackedType<U, FS>>, Error>
	where
		FS: TowerField,
		FW: ExtensionField<FS>,
		U: PackScalar<FS>,
	{
		let entry = self
			.entries
			.get(id)
			.ok_or(Error::MissingWitness { id })?
			.as_ref()
			.ok_or(Error::MissingWitness { id })?;

		let backing = entry
			.backing
			.as_ref()
			.ok_or(Error::NoExplicitBackingMultilinearExtension { id })?;

		if backing.tower_level != FS::TOWER_LEVEL {
			return Err(Error::OracleTowerHeightMismatch {
				oracle_id: id,
				oracle_level: backing.tower_level,
				field_level: FS::TOWER_LEVEL,
			});
		}

		let underliers_ref = backing.underliers.as_ref();

		let mle = MultilinearExtension::from_values_slice(
			PackedType::<U, FS>::from_underliers_ref(underliers_ref),
		)?;
		Ok(mle)
	}

	pub fn get_multilin_poly(
		&self,
		id: OracleId,
	) -> Result<MultilinearWitness<'a, PackedType<U, FW>>, Error> {
		let entry = self
			.entries
			.get(id)
			.ok_or(Error::MissingWitness { id })?
			.as_ref()
			.ok_or(Error::MissingWitness { id })?;
		Ok(entry.type_erased.clone())
	}

	pub fn update_owned<FS, Data>(
		self,
		witnesses: impl IntoIterator<Item = (OracleId, Data)>,
	) -> Result<MultilinearExtensionIndex<'a, U, FW>, Error>
	where
		FS: TowerField,
		FW: ExtensionField<FS>,
		U: PackScalar<FS> + Debug,
		Data: Into<Arc<[U]>>,
	{
		let MultilinearExtensionIndex { mut entries } = self;
		for (id, witness) in witnesses {
			if id >= entries.len() {
				entries.resize_with(id + 1, || None);
			}

			let witness = witness.into();
			let mle = MultilinearExtension::<_, PackingDeref<U, FS, _>>::from_underliers(
				witness.clone(),
			)?;
			let backing = MultilinearExtensionBacking {
				underliers: ArcOrRef::Arc(witness),
				tower_level: FS::TOWER_LEVEL,
			};
			entries[id] = Some(MultilinearExtensionIndexEntry {
				type_erased: mle.specialize_arc_dyn(),
				backing: Some(backing),
			});
		}
		Ok(MultilinearExtensionIndex { entries })
	}

	pub fn update_borrowed<'new, FS>(
		self,
		witnesses: impl IntoIterator<Item = (OracleId, &'new [U])>,
	) -> Result<MultilinearExtensionIndex<'new, U, FW>, Error>
	where
		'a: 'new,
		FS: TowerField,
		FW: ExtensionField<FS>,
		U: PackScalar<FS>,
	{
		let MultilinearExtensionIndex { mut entries } = self;
		for (id, witness) in witnesses {
			if id >= entries.len() {
				entries.resize_with(id + 1, || None);
			}

			let mle = MultilinearExtension::from_values_slice(
				PackedType::<U, FS>::from_underliers_ref(witness),
			)?;
			let backing = MultilinearExtensionBacking {
				underliers: ArcOrRef::Ref(witness),
				tower_level: FS::TOWER_LEVEL,
			};
			entries[id] = Some(MultilinearExtensionIndexEntry {
				type_erased: mle.specialize_arc_dyn(),
				backing: Some(backing),
			});
		}
		Ok(MultilinearExtensionIndex { entries })
	}

	pub fn update_multilin_poly(
		self,
		witnesses: impl IntoIterator<Item = (OracleId, MultilinearWitness<'a, PackedType<U, FW>>)>,
	) -> Result<MultilinearExtensionIndex<'a, U, FW>, Error> {
		let MultilinearExtensionIndex { mut entries } = self;
		for (id, witness) in witnesses {
			if id >= entries.len() {
				entries.resize_with(id + 1, || None);
			}

			entries[id] = Some(MultilinearExtensionIndexEntry {
				type_erased: witness,
				backing: None,
			});
		}
		Ok(MultilinearExtensionIndex { entries })
	}

	pub fn update_packed<'new, FS>(
		self,
		witnesses: impl IntoIterator<Item = (OracleId, &'new [PackedType<U, FS>])>,
	) -> Result<MultilinearExtensionIndex<'new, U, FW>, Error>
	where
		'a: 'new,
		FS: TowerField,
		FW: ExtensionField<FS>,
		U: PackScalar<FS>,
	{
		self.update_borrowed(
			witnesses.into_iter().map(|(oracle_id, packed)| {
				(oracle_id, <PackedType<U, FS>>::to_underliers_ref(packed))
			}),
		)
	}

	pub fn witness_index(&self) -> MultilinearWitnessIndex<'a, PackedType<U, FW>> {
		let mut index = MultilinearWitnessIndex::new();
		index.set_many(self.entries.iter().enumerate().flat_map(|(id, entry)| {
			let entry = entry.as_ref()?;
			Some((id, entry.type_erased.clone()))
		}));
		index
	}
}

#[derive(Debug)]
enum ArcOrRef<'a, T: ?Sized> {
	Arc(Arc<T>),
	Ref(&'a T),
}

impl<'a, T: ?Sized> AsRef<T> for ArcOrRef<'a, T> {
	fn as_ref(&self) -> &T {
		match self {
			Self::Arc(owned) => owned,
			Self::Ref(borrowed) => borrowed,
		}
	}
}
