// Copyright 2024-2025 Irreducible Inc.

use binius_field::{packed::get_packed_slice, Field, PackedField};
use binius_maybe_rayon::prelude::*;
use bytemuck::zeroed_vec;
use tracing::{debug_span, instrument};

use super::Error;

#[derive(Debug, Clone)]
pub struct GrandProductClaim<F: Field> {
	pub n_vars: usize,
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
pub struct GrandProductWitness<P: PackedField> {
	n_vars: usize,
	circuit_layers: Vec<Vec<P>>,
}

impl<P: PackedField> GrandProductWitness<P> {
	#[instrument(skip_all, level = "debug", name = "GrandProductWitness::new")]
	pub fn new(n_vars: usize, input_layer: Vec<P>) -> Result<Self, Error> {
		// TODO: validate

		let mut circuit_layers = Vec::with_capacity(n_vars + 1);

		circuit_layers.push(input_layer);
		debug_span!("constructing_layers").in_scope(|| {
			for layer_n_vars in (0..n_vars).rev() {
				let prev_layer = circuit_layers
					.last()
					.expect("all_layers is not empty by invariant");
				let max_layer_len = 1 << layer_n_vars.saturating_sub(P::LOG_WIDTH);
				let mut layer = zeroed_vec(prev_layer.len().min(max_layer_len));

				if layer_n_vars >= P::LOG_WIDTH {
					let packed_len = 1 << (layer_n_vars - P::LOG_WIDTH);
					let pivot = prev_layer.len().saturating_sub(packed_len);

					if pivot > 0 {
						let (evals_0, evals_1) = prev_layer.split_at(packed_len);
						(layer.as_mut_slice(), evals_0, evals_1)
							.into_par_iter()
							.for_each(|(product, &eval_0, &eval_1)| {
								*product = eval_0 * eval_1;
							});
					}

					layer[pivot..]
						.copy_from_slice(&prev_layer[pivot..packed_len.min(prev_layer.len())]);
				} else if prev_layer.len() > 0 {
					let layer = layer
						.first_mut()
						.expect("layer.len() >= 1 iff prev_layer.len() >= 1");
					for i in 0..1 << layer_n_vars {
						let product = get_packed_slice(prev_layer, i)
							* get_packed_slice(prev_layer, i | 1 << layer_n_vars);
						layer.set(i, product);
					}
				}

				circuit_layers.push(layer);
			}
		});

		Ok(Self {
			n_vars,
			circuit_layers,
		})
	}

	/// Returns the base-two log of the number of inputs to the GKR grand product circuit
	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Returns the evaluation of the GKR grand product circuit
	pub fn grand_product_evaluation(&self) -> P::Scalar {
		let first_layer = self.circuit_layers.last().expect("always n_vars+1 layers");
		let first_packed = first_layer.get(0).copied().unwrap_or(P::one());
		first_packed.get(0)
	}

	/// TODO: comment
	pub fn into_circuit_layers(self) -> Vec<Vec<P>> {
		self.circuit_layers
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
