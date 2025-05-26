// Copyright 2024-2025 Irreducible Inc.

use std::{cell::RefCell, fmt::Debug, sync::Arc};

use binius_compute::{
	ComputeLayer, ComputeMemory, FSlice,
	alloc::{BumpAllocator, ComputeAllocator},
};
use binius_field::{Field, PackedField};
use binius_math::MultilinearPoly;
use binius_utils::sparse_index::SparseIndex;

use crate::{oracle::OracleId, polynomial::Error as PolynomialError};

pub type MultilinearWitness<'a, P> = Arc<dyn MultilinearPoly<P> + Send + Sync + 'a>;

#[derive(Clone, Debug)]
pub struct IndexEntry<'a, P: PackedField> {
	pub multilin_poly: MultilinearWitness<'a, P>,
	pub nonzero_scalars_prefix: usize,
}

/// Data structure that indexes multilinear extensions by oracle ID.
///
/// A [`crate::oracle::MultilinearOracleSet`] indexes multilinear polynomial oracles by assigning
/// unique, sequential oracle IDs.
#[derive(Default, Debug)]
pub struct MultilinearExtensionIndex<'a, P>
where
	P: PackedField,
{
	entries: Vec<Option<IndexEntry<'a, P>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("witness not found for oracle {id}")]
	MissingWitness { id: OracleId },
	#[error("witness for oracle id {id} does not have an explicit backing multilinear")]
	NoExplicitBackingMultilinearExtension { id: OracleId },
	#[error(
		"log degree mismatch for oracle id {oracle_id}. field_log_extension_degree = {field_log_extension_degree} entry_log_extension_degree = {entry_log_extension_degree}"
	)]
	OracleExtensionDegreeMismatch {
		oracle_id: OracleId,
		field_log_extension_degree: usize,
		entry_log_extension_degree: usize,
	},
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Compute Layer error: {0}")]
	ComputeError(#[from] binius_compute::Error),
	#[error("Alloc error: {0}")]
	AllocError(#[from] binius_compute::alloc::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
}

impl<'a, P> MultilinearExtensionIndex<'a, P>
where
	P: PackedField,
{
	pub fn new() -> Self {
		Self::default()
	}

	pub fn get_index_entry(&self, id: OracleId) -> Result<IndexEntry<'a, P>, Error> {
		let entry = self
			.entries
			.get(id.index())
			.ok_or(Error::MissingWitness { id })?
			.as_ref()
			.ok_or(Error::MissingWitness { id })?;
		Ok(entry.clone())
	}

	pub fn get_multilin_poly(&self, id: OracleId) -> Result<MultilinearWitness<'a, P>, Error> {
		Ok(self.get_index_entry(id)?.multilin_poly)
	}

	/// Whether has data for the given oracle id.
	pub fn has(&self, id: OracleId) -> bool {
		self.entries.get(id.index()).is_some_and(Option::is_some)
	}

	pub fn update_multilin_poly(
		&mut self,
		witnesses: impl IntoIterator<Item = (OracleId, MultilinearWitness<'a, P>)>,
	) -> Result<(), Error> {
		self.update_multilin_poly_with_nonzero_scalars_prefixes(witnesses.into_iter().map(
			|(id, multilin_poly)| {
				let nonzero_scalars_prefix = 1 << multilin_poly.n_vars();
				(id, multilin_poly, nonzero_scalars_prefix)
			},
		))
	}

	pub fn update_multilin_poly_with_nonzero_scalars_prefixes(
		&mut self,
		witnesses: impl IntoIterator<Item = (OracleId, MultilinearWitness<'a, P>, usize)>,
	) -> Result<(), Error> {
		for (id, multilin_poly, nonzero_scalars_prefix) in witnesses {
			let id_index = id.index();
			if id_index >= self.entries.len() {
				self.entries.resize_with(id_index + 1, || None);
			}
			// TODO: validate nonzero_scalars_prefix
			self.entries[id_index] = Some(IndexEntry {
				multilin_poly,
				nonzero_scalars_prefix,
			});
		}
		Ok(())
	}
}

pub struct HalMultilinearExtensionIndex<'a, 'alloc, F: Field, Hal: ComputeLayer<F>> {
	indexes: RefCell<SparseIndex<FSlice<'a, F, Hal>>>,
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	hal: &'a Hal,
}

impl<'a, 'alloc, F: Field, Hal: ComputeLayer<F>> HalMultilinearExtensionIndex<'a, 'alloc, F, Hal> {
	pub fn new(dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>, hal: &'a Hal) -> Self {
		Self {
			indexes: RefCell::new(SparseIndex::default()),
			dev_alloc,
			hal,
		}
	}

	pub fn lazy_get<'b, P: PackedField<Scalar = F>>(
		&self,
		id: OracleId,
		witness_index: &MultilinearExtensionIndex<'b, P>,
	) -> Result<FSlice<'a, F, Hal>, Error> {
		if !self.has(id) {
			self.update_multilin_polys_from_witness(&[id], witness_index)?;
		}

		self.get_multilin_poly(id)
	}

	pub fn get_multilin_poly(&self, id: OracleId) -> Result<FSlice<'a, F, Hal>, Error> {
		self.indexes
			.borrow()
			.get(id.index())
			.ok_or(Error::MissingWitness { id })
			.copied()
			.map_err(Error::from)
	}

	pub fn update_multilin_polys_from_witness<'b, P>(
		&self,
		ids: &[OracleId],
		witness_index: &MultilinearExtensionIndex<'b, P>,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F>,
	{
		for id in ids {
			if !self.indexes.borrow().contains_key(id.index()) {
				let poly = witness_index.get_multilin_poly(*id)?;
				let n_vars = poly.n_vars();

				let evals = (0..1 << n_vars)
					.map(|i| poly.evaluate_on_hypercube(i).expect("correct size"))
					.collect::<Vec<_>>();

				self.update_multilin_poly([(*id, &evals[..])])?;
			}
		}
		Ok(())
	}

	/// Whether has data for the given oracle id.
	pub fn has(&self, id: OracleId) -> bool {
		self.indexes.borrow().contains_key(id.index())
	}

	pub fn update_multilin_poly<'b>(
		&self,
		witnesses: impl IntoIterator<Item = (OracleId, &'b [F])>,
	) -> Result<(), Error> {
		for (id, evals) in witnesses {
			let mut buffer = self.dev_alloc.alloc(evals.len())?;
			self.hal.copy_h2d(evals, &mut buffer)?;
			self.indexes
				.borrow_mut()
				.set(id.index(), Hal::DevMem::to_const(buffer));
		}
		Ok(())
	}
}
