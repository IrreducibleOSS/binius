// Copyright 2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, ExtensionField, Field, PackedField};
use binius_math::EvaluationOrder;
use binius_utils::bail;

use super::{
	common::{ExpClaim, LayerClaim},
	compositions::{ExpCompositions, VerifierExpComposition},
	error::Error,
	utils::first_layer_inverse,
};
use crate::{
	composition::{FixedDimIndexCompositions, IndexComposition},
	protocols::sumcheck::{zerocheck::ExtraProduct, CompositeSumClaim},
};

pub trait ExpVerifier<F: Field> {
	fn exponent_bit_width(&self) -> usize;

	fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}

	/// return the `eval_point` of the internal [ExpClaim].
	fn layer_claim_eval_point(&self) -> &[F];

	/// return [CompositeSumClaim] and multilinears that it contains,
	/// If the verifier does not participate in the sumcheck for this layer,
	/// the function returns `None`.
	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
		eq_ind_index: usize,
	) -> Result<Option<CompositeSumClaim<F, VerifierExpComposition<F>>>, Error>;

	/// return a tuple of the number of multilinears used by this verifier for this layer.
	fn layer_n_multilinears(&self, layer_no: usize) -> usize;

	/// return a tuple of the number of sumcheck claims used by this verifier for this layer.
	fn layer_n_claims(&self, layer_no: usize) -> usize;

	/// update verifier internal [ExpClaim] and return the [LayerClaim]s of multilinears,
	/// excluding `this_layer_input`.
	fn finish_layer(
		&mut self,
		evaluation_order: EvaluationOrder,
		layer_no: usize,
		multilinear_evals: &[F],
		r: &[F],
	) -> Vec<LayerClaim<F>>;
}

pub struct GeneratorExpVerifier<F: Field, FBase>(ExpClaim<F>, PhantomData<FBase>);

impl<F: Field, FBase> GeneratorExpVerifier<F, FBase> {
	pub fn new(claim: &ExpClaim<F>) -> Result<Self, Error> {
		if claim.uses_dynamic_base {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(claim.clone(), PhantomData))
	}
}

impl<F, FBase> ExpVerifier<F> for GeneratorExpVerifier<F, FBase>
where
	FBase: BinaryField,
	F: BinaryField + ExtensionField<FBase>,
{
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width
	}

	fn layer_claim_eval_point(&self) -> &[F] {
		&self.0.eval_point
	}

	fn finish_layer(
		&mut self,
		evaluation_order: EvaluationOrder,
		layer_no: usize,
		multilinear_evals: &[F],
		r: &[F],
	) -> Vec<LayerClaim<F>> {
		let exponent_bit_claim = if self.is_last_layer(layer_no) {
			// the evaluation of the last exponent bit can be uniquely calculated from the previous exponentiation layer claim.
			// a_0(x) = (V_0(x) - 1)/(g - 1)
			LayerClaim {
				eval_point: self.0.eval_point.clone(),
				eval: first_layer_inverse::<FBase, _>(self.0.eval),
			}
		} else {
			let n_vars = self.layer_claim_eval_point().len();

			let layer_eval = multilinear_evals[0];

			let exponent_bit_eval = multilinear_evals[1];

			let eval_point = match evaluation_order {
				EvaluationOrder::LowToHigh => r[r.len() - n_vars..].to_vec(),
				EvaluationOrder::HighToLow => r[..n_vars].to_vec(),
			};

			if !self.is_last_layer(layer_no) {
				self.0.eval = layer_eval;
				self.0.eval_point = eval_point.clone();
			}

			LayerClaim {
				eval: exponent_bit_eval,
				eval_point,
			}
		};

		vec![exponent_bit_claim]
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
		eq_ind_index: usize,
	) -> Result<Option<CompositeSumClaim<F, VerifierExpComposition<F>>>, Error> {
		if self.is_last_layer(layer_no) {
			Ok(None)
		} else {
			let internal_layer_index = self.exponent_bit_width() - 1 - layer_no;

			let base_power_constant =
				F::from(FBase::MULTIPLICATIVE_GENERATOR.pow(1 << internal_layer_index));

			let this_layer_input_index = multilinears_index;
			let exponent_bit_index = multilinears_index + 1;

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[this_layer_input_index, exponent_bit_index, eq_ind_index],
				ExtraProduct {
					inner: ExpCompositions::ConstantBase {
						base_power_constant,
					},
				},
			)?;

			let this_round_composite_claim = CompositeSumClaim {
				sum: self.0.eval,
				composition: FixedDimIndexCompositions::Trivariate(composition),
			};

			Ok(Some(this_round_composite_claim))
		}
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

pub struct ExpDynamicVerifier<F: Field>(ExpClaim<F>);

impl<F: Field> ExpDynamicVerifier<F> {
	pub fn new(claim: &ExpClaim<F>) -> Result<Self, Error> {
		if !claim.uses_dynamic_base {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(claim.clone()))
	}
}

impl<F: Field> ExpVerifier<F> for ExpDynamicVerifier<F> {
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width
	}

	fn layer_claim_eval_point(&self) -> &[F] {
		&self.0.eval_point
	}

	fn finish_layer(
		&mut self,
		evaluation_order: EvaluationOrder,
		layer_no: usize,
		multilinear_evals: &[F],
		r: &[F],
	) -> Vec<LayerClaim<F>> {
		let n_vars = self.layer_claim_eval_point().len();

		let eval_point = match evaluation_order {
			EvaluationOrder::LowToHigh => r[r.len() - n_vars..].to_vec(),
			EvaluationOrder::HighToLow => r[..n_vars].to_vec(),
		};

		let mut claims = Vec::with_capacity(2);

		let exponent_bit_eval = multilinear_evals[1];

		let exponent_bit_claim = LayerClaim {
			eval: exponent_bit_eval,
			eval_point: eval_point.clone(),
		};

		claims.push(exponent_bit_claim);

		if self.is_last_layer(layer_no) {
			let base_eval = multilinear_evals[0];

			let base_claim = LayerClaim {
				eval: base_eval,
				eval_point,
			};
			claims.push(base_claim)
		} else {
			let layer_eval = multilinear_evals[0];

			self.0.eval = layer_eval;
			self.0.eval_point = eval_point.clone();

			let base_eval = multilinear_evals[2];

			let base_claim = LayerClaim {
				eval: base_eval,
				eval_point,
			};

			claims.push(base_claim)
		}

		claims
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
		eq_ind_index: usize,
	) -> Result<Option<CompositeSumClaim<F, VerifierExpComposition<F>>>, Error> {
		let composition = if self.is_last_layer(layer_no) {
			let base_index = multilinears_index;
			let exponent_bit_index = multilinears_index + 1;

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[base_index, exponent_bit_index, eq_ind_index],
				ExtraProduct {
					inner: ExpCompositions::DynamicBaseLastLayer,
				},
			)?;

			FixedDimIndexCompositions::Trivariate(composition)
		} else {
			let this_layer_input_index = multilinears_index;
			let exponent_bit_index = multilinears_index + 1;
			let base_index = multilinears_index + 2;

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[
					this_layer_input_index,
					exponent_bit_index,
					base_index,
					eq_ind_index,
				],
				ExtraProduct {
					inner: ExpCompositions::DynamicBase,
				},
			)?;

			FixedDimIndexCompositions::Quadrivariate(composition)
		};

		let this_round_composite_claim = CompositeSumClaim {
			sum: self.0.eval,
			composition,
		};

		Ok(Some(this_round_composite_claim))
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
