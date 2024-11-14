// Copyright 2023-2024 Irreducible Inc.

use super::error::Error;
use crate::{
	oracle::{BatchId, CommittedBatch, CommittedId, MultilinearPolyOracle},
	transcript::{CanRead, CanWrite},
};
use binius_field::{Field, TowerField};
use binius_utils::bail;
use std::slice;
use tracing::instrument;

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

/// A batched PCS claim where all member polynomials have the same query (can be verified directly)
#[derive(Debug)]
pub struct SameQueryPcsClaim<F: Field> {
	/// Common evaluation point
	pub eval_point: Vec<F>,
	/// Vector of individual claimed evaluations (in batch_ref.idx_in_batch order)
	pub evals: Vec<F>,
}

/// A mutable structure which keeps track of PCS claims for polynomial batches, potentially over
/// several evalcheck/sumcheck calls
#[derive(Debug)]
pub struct BatchCommittedEvalClaims<F: Field> {
	/// Number of polynomials in each batch
	batch_lengths: Vec<usize>,
	/// Claims accumulated for each batch
	claims_by_batch: Vec<Vec<CommittedEvalClaim<F>>>,
}

impl<F: Field> BatchCommittedEvalClaims<F> {
	/// Creates a new PCS claims accumulator.
	/// `batches` is a nested array listing which committed ids belong to which batch, for example
	/// `[[1, 2], [3, 4]]` batches polys 1 & 2 into first batch and 3 and 4 into second batch. Order
	/// within batch is important.
	pub fn new(batches: &[CommittedBatch]) -> Self {
		let batch_lengths = batches.iter().map(|batch| batch.n_polys).collect();
		let claims_by_batch = vec![vec![]; batches.len()];

		Self {
			batch_lengths,
			claims_by_batch,
		}
	}

	/// Insert a new claim into the batch.
	pub fn insert(&mut self, claim: CommittedEvalClaim<F>) {
		let claims_by_batch = &mut self.claims_by_batch[claim.id.batch_id];
		if !claims_by_batch.contains(&claim) {
			claims_by_batch.push(claim);
		}
	}

	pub fn n_batches(&self) -> usize {
		self.claims_by_batch.len()
	}

	/// Extract a same query claim, if possible (hence the Option in happy path)
	#[instrument(skip_all, name = "evalcheck::extract_pcs_claim", level = "debug")]
	pub fn try_extract_same_query_pcs_claim(
		&self,
		batch_id: BatchId,
	) -> Result<Option<SameQueryPcsClaim<F>>, Error> {
		let claims = self
			.claims_by_batch
			.get(batch_id)
			.ok_or(Error::UnknownBatchId(batch_id))?;

		// batches cannot be empty
		let first = claims.first().ok_or(Error::EmptyBatch(batch_id))?;

		// all evaluation points should match
		if claims
			.iter()
			.any(|claim| claim.eval_point != first.eval_point)
		{
			return Ok(None);
		}

		// assemble the evals vector according to idx_in_batch of each poly
		let mut evals: Vec<Option<F>> = vec![None; self.batch_lengths[batch_id]];

		for claim in claims {
			let opt_other_eval = evals[claim.id.index].replace(claim.eval);

			// if two claims somehow end pointing into the same slot, check that they don't conflict
			if opt_other_eval.map_or(false, |other_eval| other_eval != claim.eval) {
				bail!(Error::ConflictingEvals(batch_id));
			}
		}

		// strip the inner Option
		let evals = evals
			.into_iter()
			.collect::<Option<Vec<_>>>()
			.ok_or(Error::MissingEvals(batch_id))?;

		let eval_point = first.eval_point.clone();

		Ok(Some(SameQueryPcsClaim { eval_point, evals }))
	}

	/// Take out potentially non-same-query claims of a batch for additional processing - one example
	/// would be an extra sumcheck round to convert non-same-query claims into same query claims
	pub fn take_claims(&mut self, batch_id: BatchId) -> Result<Vec<CommittedEvalClaim<F>>, Error> {
		let claims = self
			.claims_by_batch
			.get_mut(batch_id)
			.ok_or(Error::UnknownBatchId(batch_id))?;

		Ok(std::mem::take(claims))
	}
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
