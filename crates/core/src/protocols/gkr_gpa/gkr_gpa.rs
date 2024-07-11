// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::MultilinearPolyOracle,
	protocols::{
		abstract_sumcheck::ReducedClaim, evalcheck::EvalcheckMultilinearClaim,
		gkr_sumcheck::GkrSumcheckBatchProof,
	},
	witness::MultilinearWitness,
};
use binius_field::Field;

pub const GKR_SUMCHECK_DEGREE: usize = 2;

#[derive(Debug, Clone)]
pub struct GrandProductClaim<F: Field> {
	/// Oracle to the multilinear polynomial
	pub poly: MultilinearPolyOracle<F>,
	/// Claimed Product
	pub product: F,
}

pub type GrandProductWitness<'a, PW> = MultilinearWitness<'a, PW>;

/// LayerClaim is a claim about the evaluation of the kth layer-multilinear at a specific evaluation point
///
/// Notation:
/// * The kth layer-multilinear is the multilinear polynomial whose evaluations are the intermediate values of the kth
/// layer of the evaluated product circuit.
pub type LayerClaim<F> = ReducedClaim<F>;

/// BatchLayerProof is the proof that reduces the kth layer of a batch
/// of product circuits to the (k+1)th layer
///
/// Notation:
/// * The kth layer-multilinear is the multilinear polynomial whose evaluations are the intermediate values of the kth
/// layer of the evaluated product circuit.
/// * $r'_k$ is challenge generated during the k-variate sumcheck reduction from layer k to layer k+1
#[derive(Debug, Clone)]
pub struct BatchLayerProof<F: Field> {
	/// The proof of the batched sumcheck reduction (on $k$ variables)
	/// None for the zeroth to first layer reduction
	/// Some for the subsequent reductions
	pub gkr_sumcheck_batch_proof: GkrSumcheckBatchProof<F>,
	/// The evaluations of the appropriate (k+1)th layer-multilinear at
	/// evaluation point $(r'_k, 0)$
	pub zero_evals: Vec<F>,
	/// The evaluations of the appropriate (k+1)th layer-multilinear at
	/// evaluation point $(r'_k, 1)$
	pub one_evals: Vec<F>,
}

#[derive(Debug, Clone, Default)]
pub struct GrandProductBatchProof<F: Field> {
	pub batch_layer_proofs: Vec<BatchLayerProof<F>>,
}

#[derive(Debug, Default)]
pub struct GrandProductBatchProveOutput<F: Field> {
	// Reduced evalcheck claims for all the initial grand product claims
	pub evalcheck_multilinear_claims: Vec<EvalcheckMultilinearClaim<F>>,
	// The batch proof
	pub proof: GrandProductBatchProof<F>,
}
