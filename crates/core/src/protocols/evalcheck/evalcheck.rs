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

// #[derive(Debug, Clone)]
// pub struct EvalcheckProof<F: Field>(pub Vec<EvalcheckProofEnum<F>>);

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
}

#[repr(u8)]
#[derive(Debug)]
enum EvalcheckProofAdviceNumerics {
	HandleClaim = 1,
	DuplicateClaim,
}

#[repr(u8)]
#[derive(Debug)]
enum SubclaimNumerics {
	ExistingClaim = 1,
	NewClaim,
}

pub type ProofIndex = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalcheckProofAdvice {
	HandleClaim,
	DuplicateClaim(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Subclaim<F: Field> {
	ExistingClaim(ProofIndex),
	NewClaim(F),
}

/// A struct for storing the EvalcheckProofs, for proofs referencing inner evalchecks the index is
/// is index into `EvalcheckProof` struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalcheckProofEnum<F: Field> {
	Transparent,
	Committed,
	Shifted,
	Packed,
	Repeating,
	LinearCombination { subproofs: Vec<Subclaim<F>> },
	ZeroPadded(F),
	CompositeMLE,
	Projected,
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
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

impl SubclaimNumerics {
	const fn from(x: u8) -> Result<Self, Error> {
		match x {
			1 => Ok(Self::ExistingClaim),
			2 => Ok(Self::NewClaim),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
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

/// Serializes the `EvalcheckProof` into the transcript
pub fn serialize_evalcheck_proof<B: BufMut, F: TowerField>(
	transcript: &mut TranscriptWriter<B>,
	evalcheck: &EvalcheckProofEnum<F>,
) {
	match evalcheck {
		EvalcheckProofEnum::Transparent => {
			transcript.write_bytes(&[EvalcheckNumerics::Transparent as u8]);
		}
		EvalcheckProofEnum::Committed => {
			transcript.write_bytes(&[EvalcheckNumerics::Committed as u8]);
		}
		EvalcheckProofEnum::Shifted => {
			transcript.write_bytes(&[EvalcheckNumerics::Shifted as u8]);
		}
		EvalcheckProofEnum::Packed => {
			transcript.write_bytes(&[EvalcheckNumerics::Packed as u8]);
		}
		EvalcheckProofEnum::Repeating => {
			transcript.write_bytes(&[EvalcheckNumerics::Repeating as u8]);
		}
		EvalcheckProofEnum::LinearCombination { subproofs } => {
			transcript.write_bytes(&[EvalcheckNumerics::LinearCombination as u8]);
			write_u64(transcript, subproofs.len() as u64);
			for subproof in subproofs {
				match subproof {
					Subclaim::ExistingClaim(index) => {
						transcript.write_bytes(&[SubclaimNumerics::ExistingClaim as u8]);
						write_u64(transcript, *index as u64);
					}
					Subclaim::NewClaim(eval) => {
						transcript.write_bytes(&[SubclaimNumerics::NewClaim as u8]);
						transcript.write_scalar(*eval);
					}
				}
			}
		}
		EvalcheckProofEnum::ZeroPadded(val) => {
			transcript.write_bytes(&[EvalcheckNumerics::ZeroPadded as u8]);
			transcript.write_scalar(*val);
		}
		EvalcheckProofEnum::CompositeMLE => {
			transcript.write_bytes(&[EvalcheckNumerics::CompositeMLE as u8]);
		}
		EvalcheckProofEnum::Projected => {
			transcript.write_bytes(&[EvalcheckNumerics::Projected as u8]);
		}
	}
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

/// Deserializes the `EvalcheckProof` object from the given transcript.
pub fn deserialize_evalcheck_proof<B: Buf, F: TowerField>(
	transcript: &mut TranscriptReader<B>,
) -> Result<EvalcheckProofEnum<F>, Error> {
	let mut ty = 0;
	transcript.read_bytes(slice::from_mut(&mut ty))?;
	let as_enum = EvalcheckNumerics::from(ty)?;

	match as_enum {
		EvalcheckNumerics::Transparent => Ok(EvalcheckProofEnum::Transparent),
		EvalcheckNumerics::Committed => Ok(EvalcheckProofEnum::Committed),
		EvalcheckNumerics::Shifted => Ok(EvalcheckProofEnum::Shifted),
		EvalcheckNumerics::Packed => Ok(EvalcheckProofEnum::Packed),
		EvalcheckNumerics::Repeating => Ok(EvalcheckProofEnum::Repeating),
		EvalcheckNumerics::LinearCombination => {
			let len = read_u64(transcript)? as usize;
			let mut subproofs = Vec::with_capacity(len);
			for _ in 0..len {
				let variant: u8 = transcript.read()?;
				let subproof = match SubclaimNumerics::from(variant)? {
					SubclaimNumerics::ExistingClaim => {
						Subclaim::ExistingClaim(read_u64(transcript)? as usize)
					}
					SubclaimNumerics::NewClaim => {
						let new_eval = transcript.read()?;
						Subclaim::NewClaim(new_eval)
					}
				};
				subproofs.push(subproof);
			}
			Ok(EvalcheckProofEnum::LinearCombination { subproofs })
		}
		EvalcheckNumerics::ZeroPadded => {
			let scalar = transcript.read_scalar()?;
			Ok(EvalcheckProofEnum::ZeroPadded(scalar))
		}
		EvalcheckNumerics::CompositeMLE => Ok(EvalcheckProofEnum::CompositeMLE),
		EvalcheckNumerics::Projected => Ok(EvalcheckProofEnum::Projected),
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

#[derive(Debug, thiserror::Error)]
pub enum InsertionError {
	#[error("Value already exists for this slot")]
	ValueExists,
}

// Issues with this struct, It should be a Vec<(EvalPoint<F>, Vec<T>)> for better caching
// What happens if the user tries to insert a different T to same (OracleId, EvalPoint<F>) combo?
// Find only returns the first EvalPoint<F> it could find, maybe better to return filtered
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

	pub fn get_mut(&mut self, id: OracleId, eval_point: &[F]) -> Option<&mut T> {
		self.data
			.get_mut(id)?
			.iter_mut()
			.find(|(ep, _)| ep.as_ref() == eval_point)
			.map(|(_, val)| val)
	}

	pub fn insert_with_duplication(&mut self, id: OracleId, eval_point: EvalPoint<F>, val: T) {
		if id >= self.data.len() {
			self.data.resize(id + 1, Vec::new());
		}

		self.data[id].push((eval_point, val))
	}

	pub fn insert(
		&mut self,
		id: OracleId,
		eval_point: EvalPoint<F>,
		val: T,
	) -> Result<(), InsertionError> {
		if self.get(id, eval_point.as_ref()).is_some() {
			return Err(InsertionError::ValueExists);
		}
		self.insert_with_duplication(id, eval_point, val);
		Ok(())
	}

	pub fn into_flatten(mut self) -> Vec<T> {
		self.flatten()
	}

	pub fn flatten(&mut self) -> Vec<T> {
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
