// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, Field, PackedField};
use binius_math::EvaluationOrder;
use binius_utils::bail;

use super::{
	common::{ExpClaim, LayerClaim},
	compositions::{ExpCompositions, IndexedExpComposition},
	error::Error,
	utils::first_layer_inverse,
};
use crate::{
	composition::{FixedDimIndexCompositions, IndexComposition},
	protocols::sumcheck::CompositeSumClaim,
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
	) -> Result<Option<CompositeSumClaim<F, IndexedExpComposition<F>>>, Error>;

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

pub struct StaticBaseExpVerifier<F: Field>(ExpClaim<F>);

impl<F: Field> StaticBaseExpVerifier<F> {
	pub fn new(claim: &ExpClaim<F>) -> Result<Self, Error> {
		if claim.static_base.is_none() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(claim.clone()))
	}
}

impl<F> ExpVerifier<F> for StaticBaseExpVerifier<F>
where
	F: BinaryField,
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
			// the evaluation of the last exponent bit can be uniquely calculated from the previous
			// exponentiation layer claim. a_0(x) = (V_0(x) - 1)/(g - 1)

			let base = self.0.static_base.expect("static_base exist");

			LayerClaim {
				eval_point: self.0.eval_point.clone(),
				eval: first_layer_inverse(self.0.eval, base),
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
	) -> Result<Option<CompositeSumClaim<F, IndexedExpComposition<F>>>, Error> {
		if self.is_last_layer(layer_no) {
			Ok(None)
		} else {
			let internal_layer_index = self.exponent_bit_width() - 1 - layer_no;

			let base_power_static = self
				.0
				.static_base
				.expect("static_base exist")
				.pow(1 << internal_layer_index);

			let this_layer_input_index = multilinears_index;
			let exponent_bit_index = multilinears_index + 1;

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[this_layer_input_index, exponent_bit_index],
				ExpCompositions::StaticBase { base_power_static },
			)?;

			let this_round_composite_claim = CompositeSumClaim {
				sum: self.0.eval,
				composition: FixedDimIndexCompositions::Bivariate(composition),
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
		if self.is_last_layer(layer_no) { 0 } else { 1 }
	}
}

pub struct DynamicExpVerifier<F: Field>(ExpClaim<F>);

impl<F: Field> DynamicExpVerifier<F> {
	pub fn new(claim: &ExpClaim<F>) -> Result<Self, Error> {
		if claim.static_base.is_some() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(claim.clone()))
	}
}

impl<F: Field> ExpVerifier<F> for DynamicExpVerifier<F> {
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
	) -> Result<Option<CompositeSumClaim<F, IndexedExpComposition<F>>>, Error> {
		let composition = if self.is_last_layer(layer_no) {
			let base_index = multilinears_index;
			let exponent_bit_index = multilinears_index + 1;

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[base_index, exponent_bit_index],
				ExpCompositions::DynamicBaseLastLayer,
			)?;

			FixedDimIndexCompositions::Bivariate(composition)
		} else {
			let this_layer_input_index = multilinears_index;
			let exponent_bit_index = multilinears_index + 1;
			let base_index = multilinears_index + 2;

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[this_layer_input_index, exponent_bit_index, base_index],
				ExpCompositions::DynamicBase,
			)?;

			FixedDimIndexCompositions::Trivariate(composition)
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
