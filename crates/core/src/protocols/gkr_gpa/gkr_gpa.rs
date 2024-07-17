// Copyright 2024 Ulvetanna Inc.

use super::Error;
use crate::{
	oracle::MultilinearPolyOracle,
	protocols::{
		abstract_sumcheck::ReducedClaim, evalcheck::EvalcheckMultilinearClaim,
		gkr_sumcheck::GkrSumcheckBatchProof,
	},
	witness::MultilinearWitness,
};
use binius_field::{Field, PackedField};
use rayon::prelude::*;

pub const GKR_SUMCHECK_DEGREE: usize = 2;

type LayerEvals<'a, FW> = &'a [FW];
type LayerHalfEvals<'a, FW> = (&'a [FW], &'a [FW]);

#[derive(Debug, Clone)]
pub struct GrandProductClaim<F: Field> {
	/// Oracle to the multilinear polynomial
	pub poly: MultilinearPolyOracle<F>,
	/// Claimed Product
	pub product: F,
}

#[derive(Debug, Clone)]
pub struct GrandProductWitness<'a, PW: PackedField> {
	poly: MultilinearWitness<'a, PW>,
	circuit_evals: Vec<Vec<PW::Scalar>>,
}

impl<'a, PW: PackedField> GrandProductWitness<'a, PW> {
	pub fn new(poly: MultilinearWitness<'a, PW>) -> Result<Self, Error> {
		// Compute the circuit layers from bottom to top
		let input_layer = (0..1 << poly.n_vars())
			.into_par_iter()
			.map(|i| poly.evaluate_on_hypercube(i))
			.collect::<Result<Vec<_>, _>>()?;
		let mut all_layers = vec![input_layer];
		for curr_n_vars in (0..poly.n_vars()).rev() {
			let layer_below = all_layers.last().expect("layers is not empty by invariant");
			let new_layer = (0..1 << curr_n_vars)
				.into_par_iter()
				.map(|i| {
					let left = layer_below[i];
					let right = layer_below[i + (1 << curr_n_vars)];
					left * right
				})
				.collect();
			all_layers.push(new_layer);
		}

		// Reverse the layers
		all_layers.reverse();
		Ok(Self {
			poly,
			circuit_evals: all_layers,
		})
	}

	/// Returns the base-two log of the number of inputs to the GKR Grand Product Circuit
	pub fn n_vars(&self) -> usize {
		self.poly.n_vars()
	}

	/// Returns the evaluation of the GKR Grand Product Circuit
	pub fn grand_product_evaluation(&self) -> PW::Scalar {
		// By invariant, we will have n_vars + 1 layers, and the ith layer will have 2^i elements.
		// Therefore, this 2-D array access is safe.
		self.circuit_evals[0][0]
	}

	pub fn ith_layer_evals(&self, i: usize) -> Result<LayerEvals<'_, PW::Scalar>, Error> {
		let max_layer_idx = self.n_vars();
		if i > max_layer_idx {
			return Err(Error::InvalidLayerIndex);
		}
		Ok(&self.circuit_evals[i])
	}

	/// Returns the evaluations of the ith layer of the GKR Grand Product Circuit, split into two halves
	/// REQUIRES: 0 <= i < n_vars
	pub fn ith_layer_eval_halves(&self, i: usize) -> Result<LayerHalfEvals<'_, PW::Scalar>, Error> {
		if i == 0 {
			return Err(Error::CannotSplitOutputLayerIntoHalves);
		}
		let layer = self.ith_layer_evals(i)?;
		let half = layer.len() / 2;
		debug_assert_eq!(half, 1 << (i - 1));
		Ok((&layer[..half], &layer[half..]))
	}
}

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
