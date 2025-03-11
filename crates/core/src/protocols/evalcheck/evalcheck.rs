// Copyright 2023-2025 Irreducible Inc.

use std::{
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

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalcheckProof<F: Field> {
	Transparent,
	Committed,
	Shifted,
	Packed,
	Repeating(Box<EvalcheckProof<F>>),
	LinearCombination {
		subproofs: Vec<(F, EvalcheckProof<F>)>,
	},
	ZeroPadded(F, Box<EvalcheckProof<F>>),
	CompositeMLE,
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
					.map(|(eval, proof)| (eval.into(), proof.isomorphic()))
					.collect(),
			},
			Self::ZeroPadded(eval, proof) => {
				EvalcheckProof::ZeroPadded(eval.into(), Box::new(proof.isomorphic()))
			}
			Self::CompositeMLE => EvalcheckProof::CompositeMLE,
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
				transcript.write_scalar(*scalar);
				serialize_evalcheck_proof(transcript, subproof)
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
			let mut subproofs: Vec<(F, EvalcheckProof<F>)> = Vec::new();
			for _ in 0..len {
				let scalar = transcript.read_scalar()?;
				let subproof = deserialize_evalcheck_proof(transcript)?;
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
}

impl<T: Clone, F: Field> Default for EvalPointOracleIdMap<T, F> {
	fn default() -> Self {
		Self {
			data: Default::default(),
		}
	}
}

#[derive(Debug, Clone)]
pub struct EvalPoint<F> {
	data: Arc<[F]>,
	range: Range<usize>,
}

impl<F: Field> PartialEq for EvalPoint<F> {
	fn eq(&self, other: &Self) -> bool {
		self.data[self.range.clone()] == other.data[other.range.clone()]
	}
}

impl<F: Clone> EvalPoint<F> {
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

impl<F> From<Vec<F>> for EvalPoint<F> {
	fn from(data: Vec<F>) -> Self {
		let range = 0..data.len();
		Self {
			data: data.into(),
			range,
		}
	}
}

impl<F: Clone> From<&[F]> for EvalPoint<F> {
	fn from(data: &[F]) -> Self {
		let range = 0..data.len();
		Self {
			data: data.into(),
			range,
		}
	}
}

impl<F> Deref for EvalPoint<F> {
	type Target = [F];

	fn deref(&self) -> &Self::Target {
		&self.data[self.range.clone()]
	}
}
