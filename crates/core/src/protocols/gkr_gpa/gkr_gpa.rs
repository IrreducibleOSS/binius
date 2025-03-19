// Copyright 2024-2025 Irreducible Inc.

use std::slice;

use binius_field::{packed::get_packed_slice, Field, PackedField};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use tracing::{debug_span, instrument};

use super::{packed_field_storage::PackedFieldStorage, Error};
use crate::witness::MultilinearWitness;

type LayerEvals<'a, PW> = &'a [PW];
type LayerHalfEvals<'a, PW> = (PackedFieldStorage<'a, PW>, PackedFieldStorage<'a, PW>);

#[derive(Debug, Clone)]
pub struct GrandProductClaim<F: Field> {
	pub n_vars: usize,
	/// Claimed Product
	pub product: F,
}

impl<F: Field> GrandProductClaim<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> GrandProductClaim<FI> {
		GrandProductClaim {
			n_vars: self.n_vars,
			product: self.product.into(),
		}
	}
}

#[derive(Debug, Clone)]
pub struct GrandProductWitness<PW: PackedField> {
	n_vars: usize,
	/// Triangular 2D array of the binary tree of products of the input layer.
	circuit_evals: Vec<Vec<PW>>,
}

impl<PW: PackedField> GrandProductWitness<PW> {
	#[instrument(skip_all, level = "debug", name = "GrandProductWitness::new")]
	pub fn new(poly: MultilinearWitness<PW>) -> Result<Self, Error> {
		// Compute the circuit layers from bottom to top
		// TODO: Why does this fully copy the input layer?
		let mut input_layer = zeroed_vec(1 << poly.n_vars().saturating_sub(PW::LOG_WIDTH));

		if poly.n_vars() >= PW::LOG_WIDTH {
			const LOG_CHUNK_SIZE: usize = 12;
			let log_chunk_size = (poly.n_vars() - PW::LOG_WIDTH).min(LOG_CHUNK_SIZE);
			input_layer
				.par_chunks_mut(1 << (log_chunk_size - PW::LOG_WIDTH))
				.enumerate()
				.for_each(|(i, chunk)| {
					poly.subcube_evals(log_chunk_size, i, 0, chunk).expect(
						"index is between 0 and 2^{n_vars - log_chunk_size}; \
						log_embedding degree is 0",
					)
				});
		} else {
			poly.subcube_evals(poly.n_vars(), 0, 0, slice::from_mut(&mut input_layer[0]))
				.expect(
					"index is between 0 and 2^{n_vars - log_chunk_size}; log_embedding degree is 0",
				);
		}

		let mut all_layers = vec![input_layer];
		debug_span!("constructing_layers").in_scope(|| {
			for curr_n_vars in (0..poly.n_vars()).rev() {
				let layer_below = all_layers.last().expect("layers is not empty by invariant");
				let mut new_layer = zeroed_vec(1 << curr_n_vars.saturating_sub(PW::LOG_WIDTH));

				if curr_n_vars >= PW::LOG_WIDTH {
					let (left_half, right_half) =
						layer_below.split_at(1 << (curr_n_vars - PW::LOG_WIDTH));

					new_layer
						.par_iter_mut()
						.zip(left_half.par_iter().zip(right_half.par_iter()))
						.for_each(|(out_i, (left_i, right_i))| {
							*out_i = *left_i * *right_i;
						});
				} else {
					let new_layer = &mut new_layer[0];
					let len = 1 << curr_n_vars;
					for i in 0..len {
						new_layer.set(
							i,
							get_packed_slice(layer_below, i)
								* get_packed_slice(layer_below, len + i),
						);
					}
				}

				all_layers.push(new_layer);
			}
		});

		// Reverse the layers
		all_layers.reverse();
		Ok(Self {
			n_vars: poly.n_vars(),
			circuit_evals: all_layers,
		})
	}

	/// Returns the base-two log of the number of inputs to the GKR grand product circuit
	pub const fn n_vars(&self) -> usize {
		self.n_vars
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

		if layer.len() > 1 {
			let half = layer.len() / 2;
			debug_assert_eq!(half << PW::LOG_WIDTH, 1 << (i - 1));

			Ok((layer[..half].into(), layer[half..].into()))
		} else {
			let layer_size = 1 << (i - 1);

			let first_half = PackedFieldStorage::new_inline(layer[0].iter().take(layer_size))?;
			let second_half =
				PackedFieldStorage::new_inline(layer[0].iter().skip(layer_size).take(layer_size))?;

			Ok((first_half, second_half))
		}
	}
}

/// LayerClaim is a claim about the evaluation of the kth layer-multilinear at a specific evaluation point
///
/// Notation:
/// * The kth layer-multilinear is the multilinear polynomial whose evaluations are the intermediate values of the kth
///   layer of the evaluated product circuit.
#[derive(Debug, Clone, Default)]
pub struct LayerClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
}

impl<F: Field> LayerClaim<F> {
	pub fn isomorphic<FI: Field>(self) -> LayerClaim<FI>
	where
		F: Into<FI>,
	{
		LayerClaim {
			eval_point: self.eval_point.into_iter().map(Into::into).collect(),
			eval: self.eval.into(),
		}
	}
}

#[derive(Debug, Default)]
pub struct GrandProductBatchProveOutput<F: Field> {
	// Reduced evalcheck claims for all the initial grand product claims
	pub final_layer_claims: Vec<LayerClaim<F>>,
}
