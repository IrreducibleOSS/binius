// Copyright 2023-2025 Irreducible Inc.

use std::{
	hash::Hash,
	ops::{Deref, Range},
	slice,
	sync::Arc,
};

use binius_field::{Field, TowerField};
use bytes::{Buf, BufMut};

use super::error::Error;
use crate::{
	oracle::OracleId,
	transcript::{TranscriptReader, TranscriptWriter},
};

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
	Transparent = 1,
	Committed,
	Shifted,
	Packed,
	Repeating,
	LinearCombination,
	ZeroPadded,
	CompositeMLE,
	Projected,
	DuplicateClaim,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalcheckProof<F: Field> {
	Transparent,
	Committed,
	Shifted,
	Packed,
	Repeating(Box<EvalcheckProof<F>>),
	LinearCombination {
		subproofs: Vec<(Option<F>, EvalcheckProof<F>)>,
	},
	ZeroPadded(F, Box<EvalcheckProof<F>>),
	CompositeMLE,
	Projected(Box<EvalcheckProof<F>>),
	DuplicateClaim(usize),
}

impl<F: Field> EvalcheckProof<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> EvalcheckProof<FI> {
		match self {
			Self::Transparent => EvalcheckProof::Transparent,
			Self::Committed => EvalcheckProof::Committed,
			Self::Shifted => EvalcheckProof::Shifted,
			Self::Packed => EvalcheckProof::Packed,
			Self::Repeating(proof) => EvalcheckProof::Repeating(Box::new(proof.isomorphic())),
			Self::LinearCombination { subproofs } => EvalcheckProof::LinearCombination {
				subproofs: subproofs
					.into_iter()
					.map(|(eval, proof)| (eval.map(|eval| eval.into()), proof.isomorphic()))
					.collect(),
			},
			Self::ZeroPadded(eval, proof) => {
				EvalcheckProof::ZeroPadded(eval.into(), Box::new(proof.isomorphic()))
			}
			Self::CompositeMLE => EvalcheckProof::CompositeMLE,
			Self::Projected(proof) => EvalcheckProof::Projected(Box::new(proof.isomorphic())),
			Self::DuplicateClaim(index) => EvalcheckProof::DuplicateClaim(index),
		}
	}
}

impl EvalcheckNumerics {
	const fn from(x: u8) -> Result<Self, Error> {
		match x {
			1 => Ok(Self::Transparent),
			2 => Ok(Self::Committed),
			3 => Ok(Self::Shifted),
			4 => Ok(Self::Packed),
			5 => Ok(Self::Repeating),
			6 => Ok(Self::LinearCombination),
			7 => Ok(Self::ZeroPadded),
			8 => Ok(Self::CompositeMLE),
			9 => Ok(Self::Projected),
			10 => Ok(Self::DuplicateClaim),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

/// Serializes the `EvalcheckProof` into the transcript
pub fn serialize_evalcheck_proof<B: BufMut, F: TowerField>(
	transcript: &mut TranscriptWriter<B>,
	evalcheck: &EvalcheckProof<F>,
) {
	match evalcheck {
		EvalcheckProof::Transparent => {
			transcript.write_bytes(&[EvalcheckNumerics::Transparent as u8]);
		}
		EvalcheckProof::Committed => {
			transcript.write_bytes(&[EvalcheckNumerics::Committed as u8]);
		}
		EvalcheckProof::Shifted => {
			transcript.write_bytes(&[EvalcheckNumerics::Shifted as u8]);
		}
		EvalcheckProof::Packed => {
			transcript.write_bytes(&[EvalcheckNumerics::Packed as u8]);
		}
		EvalcheckProof::Repeating(inner) => {
			transcript.write_bytes(&[EvalcheckNumerics::Repeating as u8]);
			serialize_evalcheck_proof(transcript, inner);
		}
		EvalcheckProof::LinearCombination { subproofs } => {
			transcript.write_bytes(&[EvalcheckNumerics::LinearCombination as u8]);
			let len_u64 = subproofs.len() as u64;
			transcript.write_bytes(&len_u64.to_le_bytes());
			for (scalar, subproof) in subproofs {
				serialize_evalcheck_proof(transcript, subproof);
				if let Some(scalar) = scalar {
					transcript.write_scalar(*scalar);
				}
			}
		}
		EvalcheckProof::ZeroPadded(val, subproof) => {
			transcript.write_bytes(&[EvalcheckNumerics::ZeroPadded as u8]);
			transcript.write_scalar(*val);
			serialize_evalcheck_proof(transcript, subproof);
		}
		EvalcheckProof::CompositeMLE => {
			transcript.write_bytes(&[EvalcheckNumerics::CompositeMLE as u8]);
		}
		EvalcheckProof::DuplicateClaim(index) => {
			transcript.write_bytes(&[EvalcheckNumerics::DuplicateClaim as u8]);
			transcript.write(index);
		}
		EvalcheckProof::Projected(subproof) => {
			transcript.write_bytes(&[EvalcheckNumerics::Projected as u8]);
			serialize_evalcheck_proof(transcript, subproof);
		}
	}
}

/// Deserializes the `EvalcheckProof` object from the given transcript.
pub fn deserialize_evalcheck_proof<B: Buf, F: TowerField>(
	transcript: &mut TranscriptReader<B>,
) -> Result<EvalcheckProof<F>, Error> {
	let mut ty = 0;
	transcript.read_bytes(slice::from_mut(&mut ty))?;
	let as_enum = EvalcheckNumerics::from(ty)?;

	match as_enum {
		EvalcheckNumerics::Transparent => Ok(EvalcheckProof::Transparent),
		EvalcheckNumerics::Committed => Ok(EvalcheckProof::Committed),
		EvalcheckNumerics::Shifted => Ok(EvalcheckProof::Shifted),
		EvalcheckNumerics::Packed => Ok(EvalcheckProof::Packed),
		EvalcheckNumerics::Repeating => {
			let inner = deserialize_evalcheck_proof(transcript)?;
			Ok(EvalcheckProof::Repeating(Box::new(inner)))
		}
		EvalcheckNumerics::LinearCombination => {
			let mut len = [0u8; 8];
			transcript.read_bytes(&mut len)?;
			let len = u64::from_le_bytes(len) as usize;
			let mut subproofs = Vec::new();
			for _ in 0..len {
				let subproof = deserialize_evalcheck_proof(transcript)?;

				let scalar = if let EvalcheckProof::DuplicateClaim(_) = subproof {
					None
				} else {
					Some(transcript.read_scalar()?)
				};

				subproofs.push((scalar, subproof));
			}
			Ok(EvalcheckProof::LinearCombination { subproofs })
		}
		EvalcheckNumerics::ZeroPadded => {
			let scalar = transcript.read_scalar()?;
			let subproof = deserialize_evalcheck_proof(transcript)?;
			Ok(EvalcheckProof::ZeroPadded(scalar, Box::new(subproof)))
		}
		EvalcheckNumerics::CompositeMLE => Ok(EvalcheckProof::CompositeMLE),
		EvalcheckNumerics::DuplicateClaim => {
			let index = transcript.read()?;
			Ok(EvalcheckProof::DuplicateClaim(index))
		}
		EvalcheckNumerics::Projected => {
			let subproof = deserialize_evalcheck_proof(transcript)?;
			Ok(EvalcheckProof::Projected(Box::new(subproof)))
		}
	}
}

pub struct EvalPointOracleIdMap<T: Clone, F: Field> {
	data: Vec<Vec<(EvalPoint<F>, T)>>,
}

impl<T: Clone, F: Field> EvalPointOracleIdMap<T, F> {
	pub fn new() -> Self {
		Self {
			data: Default::default(),
		}
	}

	pub fn get(&self, id: OracleId, eval_point: &[F]) -> Option<&T> {
		self.data
			.get(id)?
			.iter()
			.find(|(ep, _)| **ep == *eval_point)
			.map(|(_, val)| val)
	}

	pub fn insert(&mut self, id: OracleId, eval_point: EvalPoint<F>, val: T) {
		if id >= self.data.len() {
			self.data.resize(id + 1, Vec::new());
		}

		self.data[id].push((eval_point, val))
	}

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
