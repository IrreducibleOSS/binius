// Copyright 2023 Ulvetanna Inc.

use super::error::Error;
use crate::oracle::{
	BatchId, CommittedBatch, CommittedId, CompositePolyOracle, MultilinearPolyOracle,
};
use binius_field::Field;
use tracing::instrument;

#[derive(Debug, Clone)]
pub struct EvalcheckClaim<F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: CompositePolyOracle<F>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
	/// Whether the evaluation point is random
	pub is_random_point: bool,
}

#[derive(Debug, Clone)]
pub struct EvalcheckMultilinearClaim<F: Field> {
	/// Virtual Polynomial Oracle for which the evaluation is claimed
	pub poly: MultilinearPolyOracle<F>,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
	/// Whether the evaluation point is random
	pub is_random_point: bool,
}

#[derive(Debug)]
pub enum EvalcheckProof<F: Field> {
	Transparent,
	Committed,
	Shifted,
	Packed,
	Repeating(Box<EvalcheckProof<F>>),
	Interleaved {
		eval1: F,
		eval2: F,
		subproof1: Box<EvalcheckProof<F>>,
		subproof2: Box<EvalcheckProof<F>>,
	},
	Merged {
		eval1: F,
		eval2: F,
		subproof1: Box<EvalcheckProof<F>>,
		subproof2: Box<EvalcheckProof<F>>,
	},
	Composite {
		subproofs: Vec<(F, EvalcheckProof<F>)>,
	},
}

#[derive(Debug, Clone)]
pub struct CommittedEvalClaim<F: Field> {
	pub id: CommittedId,
	/// Evaluation Point
	pub eval_point: Vec<F>,
	/// Claimed Evaluation
	pub eval: F,
	/// Whether the evaluation point is random
	pub is_random_point: bool,
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
		self.claims_by_batch[claim.id.batch_id].push(claim);
	}

	pub fn n_batches(&self) -> usize {
		self.claims_by_batch.len()
	}

	/// Extract a same query claim, if possible (hence the Option in happy path)
	#[instrument(skip_all, name = "evalcheck::extract_pcs_claim")]
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

		// PCS requires random queries, thus abort when non-random one is found
		if claims.iter().any(|claim| !claim.is_random_point) {
			return Ok(None);
		}

		// assemble the evals vector according to idx_in_batch of each poly
		let mut evals: Vec<Option<F>> = vec![None; self.batch_lengths[batch_id]];

		for claim in claims {
			let opt_other_eval = evals[claim.id.index].replace(claim.eval);

			// if two claims somehow end pointing into the same slot, check that they don't conflict
			if opt_other_eval.map_or(false, |other_eval| other_eval != claim.eval) {
				return Err(Error::ConflictingEvals(batch_id));
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
