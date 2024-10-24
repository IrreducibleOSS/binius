// Copyright 2024 Ulvetanna Inc.

use super::Error;
use crate::{protocols::sumcheck::Proof as SumcheckBatchProof, witness::MultilinearWitness};
use binius_field::{Field, PackedField};
use binius_utils::bail;
use rayon::prelude::*;

type LayerEvals<'a, FW> = &'a [FW];
type LayerHalfEvals<'a, FW> = (&'a [FW], &'a [FW]);

#[derive(Debug, Clone)]
pub struct GrandProductClaim<F: Field> {
	pub n_vars: usize,
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

	/// Returns the base-two log of the number of inputs to the GKR grand product circuit
	pub fn n_vars(&self) -> usize {
		self.poly.n_vars()
	}

	/// Returns the evaluation of the GKR grand product circuit
	pub fn grand_product_evaluation(&self) -> PW::Scalar {
		// By invariant, we will have n_vars + 1 layers, and the ith layer will have 2^i elements.
		// Therefore, this 2-D array access is safe.
		self.circuit_evals[0][0]
	}

	pub fn ith_layer_evals(&self, i: usize) -> Result<LayerEvals<'_, PW::Scalar>, Error> {
		let max_layer_idx = self.n_vars();
		if i > max_layer_idx {
			bail!(Error::InvalidLayerIndex);
		}
		Ok(&self.circuit_evals[i])
	}

	/// Returns the evaluations of the ith layer of the GKR grand product circuit, split into two halves
	/// REQUIRES: 0 <= i < n_vars
	pub fn ith_layer_eval_halves(&self, i: usize) -> Result<LayerHalfEvals<'_, PW::Scalar>, Error> {
		if i == 0 {
			bail!(Error::CannotSplitOutputLayerIntoHalves);
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
///   layer of the evaluated product circuit.
#[derive(Debug)]
pub struct LayerClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
}

#[derive(Debug, Clone, Default)]
pub struct GrandProductBatchProof<F: Field> {
	pub batch_layer_proofs: Vec<SumcheckBatchProof<F>>,
}

#[derive(Debug, Default)]
pub struct GrandProductBatchProveOutput<F: Field> {
	// Reduced evalcheck claims for all the initial grand product claims
	pub final_layer_claims: Vec<LayerClaim<F>>,
	// The batch proof
	pub proof: GrandProductBatchProof<F>,
}
