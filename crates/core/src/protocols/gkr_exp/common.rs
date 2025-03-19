// Copyright 2025 Irreducible Inc.

use binius_field::Field;

use crate::protocols::sumcheck::prove::eq_ind::{EqIndSumcheckProver, EqIndSumcheckProverBuilder};

/// LayerClaim is a claim about the evaluation of the kth layer-multilinear at a specific evaluation point
///
/// Notation:
/// * The kth layer-multilinear is the multilinear polynomial whose evaluations are the intermediate values of the kth
///   layer of the evaluated circuit.
#[derive(Debug, Clone, Default)]
pub struct LayerClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
}

/// ExpClaim is a claim about the evaluation of the first layer-multilinear at a specific evaluation point.
#[derive(Clone)]
pub struct ExpClaim<F: Field> {
	pub eval_point: Vec<F>,
	pub eval: F,
	/// The number of bits used to represent the integer.
	pub exponent_bit_width: usize,
	pub n_vars: usize,
	/// - `true`: Indicates that the dynamic base is used
	/// - `false`: Indicates that the constant base is used.
	pub uses_dynamic_base: bool,
}

impl<F: Field> From<ExpClaim<F>> for LayerClaim<F> {
	fn from(value: ExpClaim<F>) -> Self {
		Self {
			eval: value.eval,
			eval_point: value.eval_point,
		}
	}
}

pub struct BaseExpReductionOutput<F: Field> {
	/// Reduced evalcheck claims for every prover for each layer.
	///
	/// The first dimension of the vector represents each layer,
	/// and the second dimension represents the LayerClaims.
	///
	/// Since [super::batch_prove] works with exponents of different widths and different types of base,
	/// the length of each layer can vary.
	pub layers_claims: Vec<Vec<LayerClaim<F>>>,
}

pub type GKRExpProverBuilder<'a, P, Backend> = EqIndSumcheckProverBuilder<'a, P, Backend>;

pub type GKRExpProver<'a, FDomain, P, Composition, M, Backend> =
	EqIndSumcheckProver<'a, FDomain, P, Composition, M, Backend>;
