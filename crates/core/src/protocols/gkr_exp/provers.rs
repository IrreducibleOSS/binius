// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, ExtensionField, PackedField};
use binius_utils::bail;

use super::{
	common::{ExpClaim, LayerClaim},
	compositions::{ExpCompositions, ProverExpComposition},
	error::Error,
	utils::first_layer_inverse,
	witness::BaseExpWitness,
};
use crate::{
	composition::{FixedDimIndexCompositions, IndexComposition},
	protocols::sumcheck::CompositeSumClaim,
	witness::MultilinearWitness,
};

pub trait ExpProver<'a, P: PackedField> {
	fn exponent_bit_width(&self) -> usize;

	fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}

	/// return the eval_point of the internal [LayerClaim].
	fn layer_claim_eval_point(&self) -> &[P::Scalar];

	/// return [CompositeSumClaim] and multilinears that it contains,
	/// If the prover does not participate in the sumcheck for this layer,
	/// the function returns `None`.
	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<Option<CompositeSumClaimWithMultilinears<'a, P>>, Error>;

	/// return a tuple of the number of multilinears used by this prover for this layer.
	fn layer_n_multilinears(&self, layer_no: usize) -> usize;

	/// return a tuple of the number of sumcheck claims used by this prover for this layer.
	fn layer_n_claims(&self, layer_no: usize) -> usize;

	/// update prover internal [LayerClaim] and return the exponent bit [LayerClaim]
	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> Vec<LayerClaim<P::Scalar>>;
}

struct ExpCommonProver<'a, P: PackedField> {
	witness: BaseExpWitness<'a, P>,
	current_layer_claim: LayerClaim<P::Scalar>,
}

impl<'a, P: PackedField> ExpCommonProver<'a, P> {
	fn new(witness: BaseExpWitness<'a, P>, claim: ExpClaim<P::Scalar>) -> Self {
		Self {
			witness,
			current_layer_claim: claim.into(),
		}
	}

	pub fn exponent_bit_width(&self) -> usize {
		self.witness.exponent.len()
	}

	fn current_layer_single_bit_output_layers_data(
		&self,
		layer_no: usize,
	) -> MultilinearWitness<'a, P> {
		let index = self.witness.single_bit_output_layers_data.len() - layer_no - 2;

		self.witness.single_bit_output_layers_data[index].clone()
	}

	pub fn eval_point(&self) -> &[P::Scalar] {
		&self.current_layer_claim.eval_point
	}

	fn current_layer_exponent_bit(&self, index: usize) -> MultilinearWitness<'a, P> {
		self.witness.exponent[index].clone()
	}

	pub fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}
}

pub struct GeneratorExpProver<'a, P: PackedField, FBase>(
	ExpCommonProver<'a, P>,
	PhantomData<FBase>,
);

impl<'a, P: PackedField, FBase> GeneratorExpProver<'a, P, FBase> {
	pub fn new(witness: BaseExpWitness<'a, P>, claim: &ExpClaim<P::Scalar>) -> Result<Self, Error> {
		if witness.base.is_some() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(ExpCommonProver::new(witness, claim.clone()), PhantomData))
	}
}

impl<'a, P, FBase> ExpProver<'a, P> for GeneratorExpProver<'a, P, FBase>
where
	P::Scalar: BinaryField + ExtensionField<FBase>,
	P: PackedField,
	FBase: BinaryField,
{
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width()
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<Option<CompositeSumClaimWithMultilinears<'a, P>>, Error> {
		if self.0.is_last_layer(layer_no) {
			return Ok(None);
		}

		let internal_layer_index = self.exponent_bit_width() - 1 - layer_no;

		let this_layer_input = self.0.current_layer_single_bit_output_layers_data(layer_no);

		let exponent_bit = self
			.0
			.current_layer_exponent_bit(internal_layer_index)
			.clone();

		let this_layer_multilinears = vec![this_layer_input, exponent_bit];

		let base_power_constant =
			P::Scalar::from(FBase::MULTIPLICATIVE_GENERATOR.pow(1 << internal_layer_index));

		let composition = IndexComposition::new(
			composite_claims_n_multilinears,
			[multilinears_index, multilinears_index + 1],
			ExpCompositions::GeneratorBase {
				base_power_constant,
			},
		)?;

		let composition = FixedDimIndexCompositions::Bivariate(composition);

		let this_layer_composite_claim = CompositeSumClaim {
			sum: self.0.current_layer_claim.eval,
			composition,
		};

		Ok(Some(CompositeSumClaimWithMultilinears {
			claim: this_layer_composite_claim,
			multilinears: this_layer_multilinears,
		}))
	}

	fn layer_claim_eval_point(&self) -> &[<P as PackedField>::Scalar] {
		self.0.eval_point()
	}

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> Vec<LayerClaim<P::Scalar>> {
		let exponent_bit_claim = if self.is_last_layer(layer_no) {
			// the evaluation of the last exponent bit can be uniquely calculated from the previous exponentiation layer claim.
			// a_0(x) = (V_0(x) - 1)/(g - 1)
			let LayerClaim { eval_point, eval } = self.0.current_layer_claim.clone();

			LayerClaim::<P::Scalar> {
				eval_point,
				eval: first_layer_inverse::<FBase, _>(eval),
			}
		} else {
			let n_vars = self.layer_claim_eval_point().len();

			let layer_eval = multilinear_evals[0];

			let exponent_bit_eval = multilinear_evals[1];

			let eval_point = &r[r.len() - n_vars..];
			if !self.is_last_layer(layer_no) {
				self.0.current_layer_claim = LayerClaim {
					eval: layer_eval,
					eval_point: eval_point.to_vec(),
				};
			}

			LayerClaim {
				eval: exponent_bit_eval,
				eval_point: eval_point.to_vec(),
			}
		};

		vec![exponent_bit_claim]
	}

	fn layer_n_multilinears(&self, layer_no: usize) -> usize {
		if self.is_last_layer(layer_no) {
			0
		} else {
			// this_layer_input, exponent_bit
			2
		}
	}

	fn layer_n_claims(&self, layer_no: usize) -> usize {
		if self.is_last_layer(layer_no) {
			0
		} else {
			1
		}
	}
}

pub struct DynamicBaseExpProver<'a, P: PackedField>(ExpCommonProver<'a, P>);

impl<'a, P: PackedField> DynamicBaseExpProver<'a, P> {
	pub fn new(witness: BaseExpWitness<'a, P>, claim: &ExpClaim<P::Scalar>) -> Result<Self, Error> {
		if witness.base.is_none() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(ExpCommonProver::new(witness, claim.clone())))
	}
}

pub struct CompositeSumClaimWithMultilinears<'a, P: PackedField> {
	pub claim: CompositeSumClaim<P::Scalar, ProverExpComposition<P::Scalar>>,
	pub multilinears: Vec<MultilinearWitness<'a, P>>,
}

impl<'a, P: PackedField> ExpProver<'a, P> for DynamicBaseExpProver<'a, P> {
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width()
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<Option<CompositeSumClaimWithMultilinears<'a, P>>, Error> {
		let base = self
			.0
			.witness
			.base
			.clone()
			.expect("DynamicBase witness must contain base");

		let exponent_bit = self.0.current_layer_exponent_bit(layer_no);

		let (composition, this_layer_multilinears) = if self.0.is_last_layer(layer_no) {
			let this_layer_multilinears = vec![base, exponent_bit];

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[multilinears_index, multilinears_index + 1],
				ExpCompositions::DynamicBaseLastLayer,
			)?;
			let composition = FixedDimIndexCompositions::Bivariate(composition);
			(composition, this_layer_multilinears)
		} else {
			let this_layer_input = self
				.0
				.current_layer_single_bit_output_layers_data(layer_no)
				.clone();

			let this_layer_multilinears = vec![this_layer_input, exponent_bit, base];
			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[
					multilinears_index,
					multilinears_index + 1,
					multilinears_index + 2,
				],
				ExpCompositions::DynamicBase,
			)?;
			let composition = FixedDimIndexCompositions::Trivariate(composition);
			(composition, this_layer_multilinears)
		};

		let this_layer_composite_claim = CompositeSumClaim {
			sum: self.0.current_layer_claim.eval,
			composition,
		};

		Ok(Some(CompositeSumClaimWithMultilinears {
			claim: this_layer_composite_claim,
			multilinears: this_layer_multilinears,
		}))
	}

	fn layer_claim_eval_point(&self) -> &[<P as PackedField>::Scalar] {
		self.0.eval_point()
	}

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> Vec<LayerClaim<P::Scalar>> {
		let n_vars = self.0.eval_point().len();
		let eval_point = &r[r.len() - n_vars..];

		let mut claims = Vec::with_capacity(2);

		let exponent_bit_eval = multilinear_evals[1];

		let exponent_bit_claim = LayerClaim {
			eval: exponent_bit_eval,
			eval_point: eval_point.to_vec(),
		};

		claims.push(exponent_bit_claim);

		if self.is_last_layer(layer_no) {
			let base_eval = multilinear_evals[0];

			let base_claim = LayerClaim {
				eval: base_eval,
				eval_point: eval_point.to_vec(),
			};
			claims.push(base_claim)
		} else {
			let layer_eval = multilinear_evals[0];

			self.0.current_layer_claim = LayerClaim {
				eval: layer_eval,
				eval_point: eval_point.to_vec(),
			};

			let base_eval = multilinear_evals[2];

			let base_claim = LayerClaim {
				eval: base_eval,
				eval_point: eval_point.to_vec(),
			};

			claims.push(base_claim)
		}

		claims
	}

	fn layer_n_multilinears(&self, layer_no: usize) -> usize {
		if self.is_last_layer(layer_no) {
			// base, exponent_bit
			2
		} else {
			// this_layer_input, exponent_bit, base
			3
		}
	}

	fn layer_n_claims(&self, _layer_no: usize) -> usize {
		1
	}
}
