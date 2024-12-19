// Copyright 2023-2024 Irreducible Inc.

use super::error::Error;
use crate::{
	oracle::{CommittedId, MultilinearPolyOracle, OracleId},
	transcript::{CanRead, CanWrite},
};
use binius_field::{Field, TowerField};
use std::slice;

#[derive(Debug, Clone)]
pub struct EvalcheckMultilinearClaim<F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: MultilinearPolyOracle<F>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
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
}

impl<F: Field> EvalcheckProof<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> EvalcheckProof<FI> {
		match self {
			EvalcheckProof::Transparent => EvalcheckProof::Transparent,
			EvalcheckProof::Committed => EvalcheckProof::Committed,
			EvalcheckProof::Shifted => EvalcheckProof::Shifted,
			EvalcheckProof::Packed => EvalcheckProof::Packed,
			EvalcheckProof::Repeating(proof) => {
				EvalcheckProof::Repeating(Box::new(proof.isomorphic()))
			}
			EvalcheckProof::LinearCombination { subproofs } => EvalcheckProof::LinearCombination {
				subproofs: subproofs
					.into_iter()
					.map(|(eval, proof)| (eval.into(), proof.isomorphic()))
					.collect::<Vec<_>>(),
			},
			EvalcheckProof::ZeroPadded(eval, proof) => {
				EvalcheckProof::ZeroPadded(eval.into(), Box::new(proof.isomorphic()))
			}
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommittedEvalClaim<F: Field> {
	pub id: CommittedId,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
}

impl EvalcheckNumerics {
	fn from(x: u8) -> Result<Self, Error> {
		match x {
			1 => Ok(EvalcheckNumerics::Transparent),
			2 => Ok(EvalcheckNumerics::Committed),
			3 => Ok(EvalcheckNumerics::Shifted),
			4 => Ok(EvalcheckNumerics::Packed),
			5 => Ok(EvalcheckNumerics::Repeating),
			6 => Ok(EvalcheckNumerics::LinearCombination),
			7 => Ok(EvalcheckNumerics::ZeroPadded),
			_ => Err(Error::EvalcheckSerializationError),
		}
	}
}

/// Serializes the `EvalcheckProof` into the transcript
pub fn serialize_evalcheck_proof<Transcript: CanWrite, F: TowerField>(
	transcript: &mut Transcript,
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
			serialize_evalcheck_proof(transcript, inner.as_ref());
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
			serialize_evalcheck_proof(transcript, subproof.as_ref());
		}
	}
}

/// Deserializes the `EvalcheckProof` object from the given transcript.
pub fn deserialize_evalcheck_proof<Transcript: CanRead, F: TowerField>(
	transcript: &mut Transcript,
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
	}
}

pub struct EvalPointOracleIdMap<T: Clone, F: Field> {
	data: Vec<Vec<(Vec<F>, T)>>,
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

	pub fn insert(&mut self, id: OracleId, eval_point: Vec<F>, val: T) {
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
