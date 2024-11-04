// Copyright 2024 Irreducible Inc.

use crate::{oracle::OracleId, polynomial::Error as PolynomialError};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	ExtensionField, Field, PackedExtension, TowerField,
};
use binius_math::{
	MultilinearExtension, MultilinearExtensionBorrowed, MultilinearPoly, PackingDeref,
};
use binius_utils::bail;
use std::{fmt::Debug, sync::Arc};

pub type MultilinearWitness<'a, P> = Arc<dyn MultilinearPoly<P> + Send + Sync + 'a>;

/// Data structure that indexes multilinear extensions by oracle ID.
///
/// A [`crate::oracle::MultilinearOracleSet`] indexes multilinear polynomial oracles by assigning
/// unique, sequential oracle IDs. The caller can get the [`MultilinearExtension`] defined natively
/// over a subfield. This is possible because the [`MultilinearExtensionIndex::get`] method is
/// generic over the subfield type and the struct itself only stores the underlying data.
#[derive(Default, Debug)]
pub struct MultilinearExtensionIndex<'a, U: UnderlierType, FW>
where
	U: UnderlierType + PackScalar<FW>,
	FW: Field,
{
	entries: Vec<Option<MultilinearWitness<'a, PackedType<U, FW>>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("witness not found for oracle {id}")]
	MissingWitness { id: OracleId },
	#[error("witness for oracle id {id} does not have an explicit backing multilinear")]
	NoExplicitBackingMultilinearExtension { id: OracleId },
	#[error("log degree mismatch for oracle id {oracle_id}. field_log_extension_degree = {field_log_extension_degree} entry_log_extension_degree = {entry_log_extension_degree}")]
	OracleExtensionDegreeMismatch {
		oracle_id: OracleId,
		field_log_extension_degree: usize,
		entry_log_extension_degree: usize,
	},
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
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

		if entry.log_extension_degree() != FW::LOG_DEGREE {
			bail!(Error::OracleExtensionDegreeMismatch {
				oracle_id: id,
				field_log_extension_degree: FW::LOG_DEGREE,
				entry_log_extension_degree: entry.log_extension_degree()
			})
		}

		let evals = entry
			.packed_evals()
			.map(<PackedType<U, FW>>::cast_bases)
			.ok_or(Error::NoExplicitBackingMultilinearExtension { id })?;

		Ok(MultilinearExtension::from_values_slice(evals)?)
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
		Ok(entry.clone())
	}

	/// Whether has data for the given oracle id.
	pub fn has(&self, id: OracleId) -> bool {
		self.entries.get(id).map_or(false, Option::is_some)
	}

	pub fn set_owned<FS, Data>(
		&mut self,
		witnesses: impl IntoIterator<Item = (OracleId, Data)>,
	) -> Result<(), Error>
	where
		FS: TowerField,
		FW: ExtensionField<FS>,
		U: PackScalar<FS> + Debug,
		Data: Into<Arc<[U]>>,
	{
		for (id, witness) in witnesses {
			if id >= self.entries.len() {
				self.entries.resize_with(id + 1, || None);
			}

			let mle =
				MultilinearExtension::<_, PackingDeref<U, FS, _>>::from_underliers(witness.into())?;
			self.entries[id] = Some(mle.specialize_arc_dyn());
		}
		Ok(())
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
			entries[id] = Some(mle.specialize_arc_dyn());
		}
		Ok(MultilinearExtensionIndex { entries })
	}

	pub fn update_multilin_poly(
		&mut self,
		witnesses: impl IntoIterator<Item = (OracleId, MultilinearWitness<'a, PackedType<U, FW>>)>,
	) -> Result<(), Error> {
		for (id, witness) in witnesses {
			if id >= self.entries.len() {
				self.entries.resize_with(id + 1, || None);
			}
			self.entries[id] = Some(witness);
		}
		Ok(())
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
}
