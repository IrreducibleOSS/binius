// Copyright 2023-2025 Irreducible Inc.

use std::{
	hash::Hash,
	ops::{Deref, Range},
	slice,
	sync::Arc,
};

use binius_field::Field;
use bytes::{Buf, BufMut};

use super::error::Error;
use crate::{
	oracle::OracleId,
	transcript::{TranscriptReader, TranscriptWriter},
};

/// This struct represents a claim to be verified through the evalcheck protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalcheckMultilinearClaim<F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub id: OracleId,
	/// Evaluation Point
	pub eval_point: EvalPoint<F>,
	/// Claimed Evaluation
	pub eval: F,
}

#[repr(u8)]
#[derive(Debug)]
enum EvalcheckNumerics {
	NewClaim = 1,
	DuplicateClaim,
}

/// The proof output of a given claim which may recursively contain proofs of subclaims arising from a given claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalcheckProof {
	NewClaim,
	DuplicateClaim(usize),
}

impl EvalcheckNumerics {
	const fn from(x: u8) -> Result<Self, Error> {
		match x {
			1 => Ok(Self::NewClaim),
			2 => Ok(Self::DuplicateClaim),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

/// Serializes the `EvalcheckProof` into the transcript
pub fn serialize_evalcheck_proof<B: BufMut>(
	transcript: &mut TranscriptWriter<B>,
	evalcheck: &EvalcheckProof,
) {
	match evalcheck {
		EvalcheckProof::NewClaim => {
			transcript.write_bytes(&[EvalcheckNumerics::NewClaim as u8]);
		}
		EvalcheckProof::DuplicateClaim(index) => {
			transcript.write_bytes(&[EvalcheckNumerics::DuplicateClaim as u8]);
			transcript.write(index);
		}
	}
}

/// Deserializes the `EvalcheckProof` object from the given transcript.
pub fn deserialize_evalcheck_proof<B: Buf>(
	transcript: &mut TranscriptReader<B>,
) -> Result<EvalcheckProof, Error> {
	let mut ty = 0;
	transcript.read_bytes(slice::from_mut(&mut ty))?;
	let as_enum = EvalcheckNumerics::from(ty)?;

	match as_enum {
		EvalcheckNumerics::NewClaim => Ok(EvalcheckProof::NewClaim),
		EvalcheckNumerics::DuplicateClaim => {
			let index = transcript.read()?;
			Ok(EvalcheckProof::DuplicateClaim(index))
		}
	}
}

/// Data structure for efficiently querying and inserting evaluations of claims.
///
/// Equivalent to a `HashMap<(OracleId, EvalPoint<F>), T>` but uses vectors of vectors to store the data.
/// This data structure is more memory efficient for small number of evaluation points and OracleIds which
/// are grouped together.
pub struct EvalPointOracleIdMap<T: Clone, F: Field> {
	data: Vec<Vec<(EvalPoint<F>, T)>>,
}

impl<T: Clone, F: Field> EvalPointOracleIdMap<T, F> {
	pub fn new() -> Self {
		Self {
			data: Default::default(),
		}
	}

	/// Query the first value found for an evaluation point for a given oracle id.
	///
	/// Returns `None` if no value is found.
	pub fn get(&self, id: OracleId, eval_point: &[F]) -> Option<&T> {
		self.data
			.get(id)?
			.iter()
			.find(|(ep, _)| **ep == *eval_point)
			.map(|(_, val)| val)
	}

	/// Insert a new evaluation point for a given oracle id.
	///
	/// We do not replace existing values.
	pub fn insert(&mut self, id: OracleId, eval_point: EvalPoint<F>, val: T) {
		if id >= self.data.len() {
			self.data.resize(id + 1, Vec::new());
		}

		self.data[id].push((eval_point, val))
	}

	/// Flatten the data structure into a vector of values.
	pub fn flatten(mut self) -> Vec<T> {
		self.data.reverse();

		std::mem::take(&mut self.data)
			.into_iter()
			.flatten()
			.map(|(_, val)| val)
			.collect::<Vec<_>>()
	}

	pub fn clear(&mut self) {
		self.data.clear()
	}
}

impl<T: Clone, F: Field> Default for EvalPointOracleIdMap<T, F> {
	fn default() -> Self {
		Self {
			data: Default::default(),
		}
	}
}

/// A wrapper struct for evaluation points.
#[derive(Debug, Clone, Eq)]
pub struct EvalPoint<F: Field> {
	data: Arc<[F]>,
	range: Range<usize>,
}

impl<F: Field> PartialEq for EvalPoint<F> {
	fn eq(&self, other: &Self) -> bool {
		self.data[self.range.clone()] == other.data[other.range.clone()]
	}
}

impl<F: Field> Hash for EvalPoint<F> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		self.data[self.range.clone()].hash(state)
	}
}

impl<F: Field> EvalPoint<F> {
	pub fn slice(&self, range: Range<usize>) -> Self {
		assert!(self.range.len() >= range.len());

		let new_range = self.range.start + range.start..self.range.start + range.end;

		Self {
			data: self.data.clone(),
			range: new_range,
		}
	}

	pub fn to_vec(&self) -> Vec<F> {
		self.data.to_vec()
	}
}

impl<F: Field> From<Vec<F>> for EvalPoint<F> {
	fn from(data: Vec<F>) -> Self {
		let range = 0..data.len();
		Self {
			data: data.into(),
			range,
		}
	}
}

impl<F: Field> From<&[F]> for EvalPoint<F> {
	fn from(data: &[F]) -> Self {
		let range = 0..data.len();
		Self {
			data: data.into(),
			range,
		}
	}
}

impl<F: Field> Deref for EvalPoint<F> {
	type Target = [F];

	fn deref(&self) -> &Self::Target {
		&self.data[self.range.clone()]
	}
}
