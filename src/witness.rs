// Copyright 2024 Ulvetanna Inc.

use crate::{field::PackedField, oracle::OracleId, polynomial::MultilinearPoly};
use std::{marker::PhantomData, sync::Arc};

pub type MultilinearWitness<'a, P> = Arc<dyn MultilinearPoly<P> + Send + Sync + 'a>;

/// Data structure that indexes multilinear polynomial witnesses by oracle ID.
///
/// A [`crate::oracle::MultilinearOracleSet`] indexes multilinear polynomial oracles by assigning
/// unique, sequential  oracle IDs. This index stores the corresponding witnesses, as
/// [`MultilinearPoly`] trait objects. Not every oracle is required to have a stored witness -- in
/// some cases, only a derived multilinear witness is required.
#[derive(Debug)]
pub struct MultilinearWitnessIndex<'a, P: PackedField> {
	multilinears: Vec<Option<MultilinearWitness<'a, P>>>,
	_p_marker: PhantomData<P>,
}

impl<'a, P> MultilinearWitnessIndex<'a, P>
where
	P: PackedField,
{
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			multilinears: Vec::new(),
			_p_marker: PhantomData,
		}
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
}
