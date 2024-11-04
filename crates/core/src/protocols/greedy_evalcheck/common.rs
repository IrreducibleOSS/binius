// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::BatchId,
	protocols::{
		evalcheck::{EvalcheckProof, SameQueryPcsClaim},
		sumcheck::Proof,
	},
};
use binius_field::Field;

#[derive(Debug, Clone, Default)]
pub struct GreedyEvalcheckProof<F: Field> {
	pub initial_evalcheck_proofs: Vec<EvalcheckProof<F>>,
	pub virtual_opening_proofs: Vec<(Proof<F>, Vec<EvalcheckProof<F>>)>,
	pub batch_opening_proof: (Proof<F>, Vec<EvalcheckProof<F>>),
}

impl<F: Field> GreedyEvalcheckProof<F> {
	fn isomorphic_vec<FI: Field + From<F>>(vec: Vec<EvalcheckProof<F>>) -> Vec<EvalcheckProof<FI>> {
		vec.into_iter().map(|x| x.isomorphic()).collect()
	}

	pub fn isomorphic<FI: Field + From<F>>(self) -> GreedyEvalcheckProof<FI> {
		GreedyEvalcheckProof {
			initial_evalcheck_proofs: Self::isomorphic_vec(self.initial_evalcheck_proofs),
			virtual_opening_proofs: self
				.virtual_opening_proofs
				.into_iter()
				.map(|(proof, evalcheck_proof)| {
					(proof.isomorphic(), Self::isomorphic_vec(evalcheck_proof))
				})
				.collect(),
			batch_opening_proof: (
				self.batch_opening_proof.0.isomorphic(),
				Self::isomorphic_vec(self.batch_opening_proof.1),
			),
		}
	}
}

#[derive(Debug)]
pub struct GreedyEvalcheckProveOutput<F: Field> {
	pub same_query_claims: Vec<(BatchId, SameQueryPcsClaim<F>)>,
	pub proof: GreedyEvalcheckProof<F>,
}
