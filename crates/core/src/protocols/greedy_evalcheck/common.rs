// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::BatchId,
	protocols::{
		evalcheck::{EvalcheckProof, SameQueryPcsClaim},
		sumcheck::SumcheckBatchProof,
	},
};
use binius_field::Field;

#[derive(Debug, Default)]
pub struct GreedyEvalcheckProof<F: Field> {
	pub initial_evalcheck_proofs: Vec<EvalcheckProof<F>>,
	pub virtual_opening_proofs: Vec<(SumcheckBatchProof<F>, Vec<EvalcheckProof<F>>)>,
	#[allow(clippy::type_complexity)]
	pub batch_opening_proof: Vec<Option<(SumcheckBatchProof<F>, Vec<EvalcheckProof<F>>)>>,
}

#[derive(Debug)]
pub struct GreedyEvalcheckProveOutput<F: Field> {
	pub same_query_claims: Vec<(BatchId, SameQueryPcsClaim<F>)>,
	pub proof: GreedyEvalcheckProof<F>,
}
