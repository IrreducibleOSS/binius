// Copyright 2025 Irreducible Inc.

use binius_field::Field;

use crate::protocols::gkr_gpa::gpa_sumcheck::prove::GPAProver;

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

impl<F: Field> LayerClaim<F> {
	pub fn isomorphic<FI>(self) -> LayerClaim<FI>
	where
		F: Into<FI>,
		FI: Field,
	{
		let Self { eval_point, eval } = self;

		LayerClaim {
			eval_point: eval_point.into_iter().map(|x| x.into()).collect::<Vec<_>>(),
			eval: eval.into(),
		}
	}
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
	pub constant_base: Option<F>,
}

impl<F: Field> From<ExpClaim<F>> for LayerClaim<F> {
	fn from(value: ExpClaim<F>) -> Self {
		Self {
			eval: value.eval,
			eval_point: value.eval_point,
		}
	}
}

impl<F: Field> ExpClaim<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> ExpClaim<FI> {
		let Self {
			eval_point,
			eval,
			exponent_bit_width,
			n_vars,
			constant_base,
		} = self;

		ExpClaim {
			eval_point: eval_point.into_iter().map(|x| x.into()).collect::<Vec<_>>(),
			eval: eval.into(),
			exponent_bit_width,
			n_vars,
			constant_base: constant_base.map(|base| base.into()),
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

impl<F: Field> BaseExpReductionOutput<F> {
	pub fn isomorphic<FI>(self) -> BaseExpReductionOutput<FI>
	where
		F: Into<FI>,
		FI: Field,
	{
		let Self { layers_claims } = self;

		BaseExpReductionOutput {
			layers_claims: layers_claims
				.into_iter()
				.map(|claims| {
					claims
						.into_iter()
						.map(|claim| claim.isomorphic())
						.collect::<Vec<_>>()
				})
				.collect::<Vec<_>>(),
		}
	}
}

pub type GKRExpProver<'a, FDomain, P, Composition, M, Backend> =
	GPAProver<'a, FDomain, P, Composition, M, Backend>;
