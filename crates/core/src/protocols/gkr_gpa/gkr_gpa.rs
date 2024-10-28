// Copyright 2024 Irreducible Inc.

use super::Error;
use crate::{protocols::sumcheck::Proof as SumcheckBatchProof, witness::MultilinearWitness};
use binius_field::{Field, PackedField};
use binius_utils::bail;
use bytemuck::zeroed_vec;
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
	circuit_evals: Vec<Vec<PW>>,
}

impl<'a, PW: PackedField> GrandProductWitness<'a, PW> {
	pub fn new(poly: MultilinearWitness<'a, PW>) -> Result<Self, Error> {
		if PW::LOG_WIDTH != 0 {
			todo!("currently only supports packed fields with width 1");
		}

		// Compute the circuit layers from bottom to top
		// TODO: Why does this fully copy the input layer?
		const LOG_CHUNK_SIZE: usize = 12;
		let log_chunk_size = poly.n_vars().min(LOG_CHUNK_SIZE);
		let mut input_layer = zeroed_vec(1 << poly.n_vars());
		input_layer
			.par_chunks_mut(1 << log_chunk_size)
			.enumerate()
			.for_each(|(i, chunk)| {
				poly.subcube_evals(log_chunk_size, i, 0, chunk).expect(
					"index is between 0 and 2^{n_vars - log_chunk_size}; \
					log_embedding degree is 0",
				)
			});

		let mut all_layers = vec![input_layer];
		for curr_n_vars in (0..poly.n_vars()).rev() {
			let layer_below = all_layers.last().expect("layers is not empty by invariant");
			let (left_half, right_half) = layer_below.split_at(1 << curr_n_vars);

			let mut new_layer = zeroed_vec(1 << curr_n_vars);
			new_layer
				.par_iter_mut()
				.zip(left_half.par_iter().zip(right_half.par_iter()))
				.for_each(|(out_i, (left_i, right_i))| {
					*out_i = *left_i * *right_i;
				});
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
		self.circuit_evals[0][0].get(0)
	}

	pub fn ith_layer_evals(&self, i: usize) -> Result<LayerEvals<'_, PW>, Error> {
		let max_layer_idx = self.n_vars();
		if i > max_layer_idx {
			bail!(Error::InvalidLayerIndex);
		}
		Ok(&self.circuit_evals[i])
	}

	/// Returns the evaluations of the ith layer of the GKR grand product circuit, split into two halves
	/// REQUIRES: 0 <= i < n_vars
	pub fn ith_layer_eval_halves(&self, i: usize) -> Result<LayerHalfEvals<'_, PW>, Error> {
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
