// Copyright 2024 Ulvetanna Inc.

use crate::{oracle::OracleId, polynomial::MultilinearPoly};
use binius_field::PackedField;
use std::sync::Arc;

pub type MultilinearWitness<'a, P> = Arc<dyn MultilinearPoly<P> + Send + Sync + 'a>;

/// Data structure that indexes multilinear polynomial witnesses by oracle ID.
///
/// A [`crate::oracle::MultilinearOracleSet`] indexes multilinear polynomial oracles by assigning
/// unique, sequential  oracle IDs. This index stores the corresponding witnesses, as
/// [`MultilinearPoly`] trait objects. Not every oracle is required to have a stored witness -- in
/// some cases, only a derived multilinear witness is required.
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
