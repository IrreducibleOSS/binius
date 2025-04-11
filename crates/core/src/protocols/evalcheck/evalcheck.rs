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
	transcript::{read_u64, write_u64, TranscriptReader, TranscriptWriter},
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
enum SubproofNumerics {
	ExistingClaim = 1,
	NewProof,
}

impl SubproofNumerics {
	const fn from(x: u8) -> Result<Self, Error> {
		match x {
			1 => Ok(Self::ExistingClaim),
			2 => Ok(Self::NewProof),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Subproof<F: Field> {
	ExistingClaim(usize),
	NewProof {
		proof: Option<EvalcheckProof<F>>,
		eval: F,
	},
}

#[repr(u8)]
#[derive(Debug, PartialEq)]
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
	Duplicated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalcheckProof<F: Field> {
	Transparent,
	Committed,
	Shifted,
	Packed,
	Repeating(Box<Option<EvalcheckProof<F>>>),
	LinearCombination { subproofs: Vec<Subproof<F>> },
	ZeroPadded(F, Box<Option<EvalcheckProof<F>>>),
	CompositeMLE,
	Projected(Box<Option<EvalcheckProof<F>>>),
}

impl<F: Field> EvalcheckProof<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> EvalcheckProof<FI> {
		match self {
			Self::Transparent => EvalcheckProof::Transparent,
			Self::Committed => EvalcheckProof::Committed,
			Self::Shifted => EvalcheckProof::Shifted,
			Self::Packed => EvalcheckProof::Packed,
			Self::Repeating(proof) => {
				EvalcheckProof::Repeating(Box::new(proof.map(|proof| proof.isomorphic())))
			}
			Self::LinearCombination { subproofs } => EvalcheckProof::LinearCombination {
				subproofs: subproofs
					.into_iter()
					.map(|subclaim| match subclaim {
						Subproof::ExistingClaim(id) => Subproof::ExistingClaim(id),
						Subproof::NewProof { proof, eval } => Subproof::NewProof {
							proof: proof.map(|proof| proof.isomorphic()),
							eval: eval.into(),
						},
					})
					.collect(),
			},
			Self::ZeroPadded(eval, proof) => EvalcheckProof::ZeroPadded(
				eval.into(),
				Box::new(proof.map(|proof| proof.isomorphic())),
			),
			Self::CompositeMLE => EvalcheckProof::CompositeMLE,
			Self::Projected(proof) => {
				EvalcheckProof::Projected(Box::new(proof.map(|proof| proof.isomorphic())))
			}
		}
	}
}

impl EvalcheckNumerics {
	fn from(x: u8) -> Result<Self, Error> {
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
			10 => Ok(Self::Duplicated),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

#[repr(u8)]
#[derive(Debug)]
enum EvalcheckProofAdviceNumerics {
	HandleClaim = 1,
	DuplicateClaim,
}

impl EvalcheckProofAdviceNumerics {
	const fn from(x: u8) -> Result<Self, Error> {
		match x {
			1 => Ok(Self::HandleClaim),
			2 => Ok(Self::DuplicateClaim),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalcheckProofAdvice {
	HandleClaim,
	DuplicateClaim(usize),
}

pub fn serialize_advice<B: BufMut>(
	transcript: &mut TranscriptWriter<B>,
	advice: &EvalcheckProofAdvice,
) {
	match advice {
		EvalcheckProofAdvice::HandleClaim => {
			transcript.write_bytes(&[EvalcheckProofAdviceNumerics::HandleClaim as u8]);
		}
		EvalcheckProofAdvice::DuplicateClaim(idx) => {
			transcript.write_bytes(&[EvalcheckProofAdviceNumerics::DuplicateClaim as u8]);
			transcript.write(idx);
		}
	}
}

pub fn deserialize_advice<B: Buf>(
	transcript: &mut TranscriptReader<B>,
) -> Result<EvalcheckProofAdvice, Error> {
	let mut ty = 0;
	transcript.read_bytes(slice::from_mut(&mut ty))?;

	let as_enum = EvalcheckProofAdviceNumerics::from(ty)?;

	match as_enum {
		EvalcheckProofAdviceNumerics::HandleClaim => Ok(EvalcheckProofAdvice::HandleClaim),
		EvalcheckProofAdviceNumerics::DuplicateClaim => {
			let idx = transcript.read()?;
			Ok(EvalcheckProofAdvice::DuplicateClaim(idx))
		}
	}
}

/// Serializes the `EvalcheckProof` into the transcript
pub fn serialize_evalcheck_proof<B: BufMut, F: TowerField>(
	transcript: &mut TranscriptWriter<B>,
	evalcheck: &Option<EvalcheckProof<F>>,
) {
	if evalcheck.is_none() {
		transcript.write_bytes(&[EvalcheckNumerics::Duplicated as u8]);
		return;
	}

	match evalcheck.clone().unwrap() {
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
			serialize_evalcheck_proof(transcript, &inner);
		}
		EvalcheckProof::LinearCombination { subproofs } => {
			transcript.write_bytes(&[EvalcheckNumerics::LinearCombination as u8]);
			let len_u64 = subproofs.len() as u64;
			transcript.write_bytes(&len_u64.to_le_bytes());
			for subclaim in subproofs {
				match subclaim {
					Subproof::ExistingClaim(id) => {
						transcript.write_bytes(&[SubproofNumerics::ExistingClaim as u8]);
						write_u64(transcript, id as u64);
					}
					Subproof::NewProof { proof, eval } => {
						transcript.write_bytes(&[SubproofNumerics::NewProof as u8]);
						transcript.write(&eval);
						serialize_evalcheck_proof(transcript, &proof);
					}
				}
			}
		}
		EvalcheckProof::ZeroPadded(val, subproof) => {
			transcript.write_bytes(&[EvalcheckNumerics::ZeroPadded as u8]);
			transcript.write_scalar(val);
			serialize_evalcheck_proof(transcript, &subproof);
		}
		EvalcheckProof::CompositeMLE => {
			transcript.write_bytes(&[EvalcheckNumerics::CompositeMLE as u8]);
		}
		EvalcheckProof::Projected(subproof) => {
			transcript.write_bytes(&[EvalcheckNumerics::Projected as u8]);
			serialize_evalcheck_proof(transcript, &subproof);
		}
	}
}

/// Deserializes the `EvalcheckProof` object from the given transcript.
pub fn deserialize_evalcheck_proof<B: Buf, F: TowerField>(
	transcript: &mut TranscriptReader<B>,
) -> Result<Option<EvalcheckProof<F>>, Error> {
	let mut ty = 0;
	transcript.read_bytes(slice::from_mut(&mut ty))?;
	let as_enum = EvalcheckNumerics::from(ty)?;

	if as_enum == EvalcheckNumerics::Duplicated {
		return Ok(None);
	}

	let proof = match as_enum {
		EvalcheckNumerics::Transparent => EvalcheckProof::Transparent,
		EvalcheckNumerics::Committed => EvalcheckProof::Committed,
		EvalcheckNumerics::Shifted => EvalcheckProof::Shifted,
		EvalcheckNumerics::Packed => EvalcheckProof::Packed,
		EvalcheckNumerics::Repeating => {
			let inner = deserialize_evalcheck_proof(transcript)?;
			EvalcheckProof::Repeating(Box::new(inner))
		}
		EvalcheckNumerics::LinearCombination => {
			let mut len = [0u8; 8];
			transcript.read_bytes(&mut len)?;
			let len = u64::from_le_bytes(len) as usize;
			let mut subproofs = Vec::with_capacity(len);
			for _ in 0..len {
				let variant: u8 = transcript.read()?;
				let subproof = match SubproofNumerics::from(variant)? {
					SubproofNumerics::ExistingClaim => {
						Subproof::ExistingClaim(read_u64(transcript)? as usize)
					}
					SubproofNumerics::NewProof => {
						let eval = transcript.read()?;
						let proof = deserialize_evalcheck_proof(transcript)?;
						Subproof::NewProof { eval, proof }
					}
				};
				subproofs.push(subproof);
			}
			EvalcheckProof::LinearCombination { subproofs }
		}
		EvalcheckNumerics::ZeroPadded => {
			let scalar = transcript.read_scalar()?;
			let subproof = deserialize_evalcheck_proof(transcript)?;
			EvalcheckProof::ZeroPadded(scalar, Box::new(subproof))
		}
		EvalcheckNumerics::CompositeMLE => EvalcheckProof::CompositeMLE,
		EvalcheckNumerics::Projected => {
			let subproof = deserialize_evalcheck_proof(transcript)?;
			EvalcheckProof::Projected(Box::new(subproof))
		}
		EvalcheckNumerics::Duplicated => unreachable!(),
	};

	Ok(Some(proof))
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
