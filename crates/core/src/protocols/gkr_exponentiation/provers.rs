// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, ExtensionField, PackedField};
use binius_utils::bail;

use super::{
	common::ExponentiationClaim,
	compositions::{ExponentiationCompositions, ProverExponentiationComposition},
	error::Error,
	utils::first_layer_inverse,
	witness::BaseExponentWitness,
};
use crate::{
	composition::{FixedDimIndexCompositions, IndexComposition},
	protocols::{gkr_gpa::LayerClaim, sumcheck::CompositeSumClaim},
	witness::MultilinearWitness,
};

pub trait ExponentiationProver<'a, P: PackedField> {
	fn exponent_bit_width(&self) -> usize;

	fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}

	// return the eval_point of the current layer claim.
	fn layer_claim_eval_point(&self) -> &[P::Scalar];

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<ProverLayerClaimMeta<'a, P>, Error>;

	// return a tuple of the number of multilinears and the sumcheck claims used by this prover for the current layer.
	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize);

	// update the current exponentiation layer claim and return the exponent bit layer claim
	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> LayerClaim<P::Scalar>;
}

struct ExponentiationCommonProver<'a, P: PackedField> {
	witness: BaseExponentWitness<'a, P>,
	current_layer_claim: LayerClaim<P::Scalar>,
}

impl<'a, P: PackedField> ExponentiationCommonProver<'a, P> {
	fn new(witness: BaseExponentWitness<'a, P>, claim: ExponentiationClaim<P::Scalar>) -> Self {
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

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> LayerClaim<P::Scalar> {
		let n_vars = self.eval_point().len();
		let eval_point = &r[r.len() - n_vars..];

		if !self.is_last_layer(layer_no) {
			self.current_layer_claim = LayerClaim {
				eval: multilinear_evals[0],
				eval_point: eval_point.to_vec(),
			};
		}

		LayerClaim {
			eval: multilinear_evals[1],
			eval_point: eval_point.to_vec(),
		}
	}
}

pub struct GeneratorExponentiationProver<'a, P: PackedField, FBase>(
	ExponentiationCommonProver<'a, P>,
	PhantomData<FBase>,
);

impl<'a, P: PackedField, FBase> GeneratorExponentiationProver<'a, P, FBase> {
	pub fn new(
		witness: BaseExponentWitness<'a, P>,
		claim: &ExponentiationClaim<P::Scalar>,
	) -> Result<Self, Error> {
		if witness.base.is_some() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(ExponentiationCommonProver::new(witness, claim.clone()), PhantomData))
	}
}

impl<'a, P, FBase> ExponentiationProver<'a, P> for GeneratorExponentiationProver<'a, P, FBase>
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
	) -> Result<ProverLayerClaimMeta<'a, P>, Error> {
		if self.0.is_last_layer(layer_no) {
			return Ok(ProverLayerClaimMeta {
				claim: None,
				multilinears: Vec::new(),
			});
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
			ExponentiationCompositions::GeneratorBase {
				base_power_constant,
			},
		)?;

		let composition = FixedDimIndexCompositions::Bivariate(composition);

		let this_layer_composite_claim = CompositeSumClaim {
			sum: self.0.current_layer_claim.eval,
			composition,
		};

		Ok(ProverLayerClaimMeta {
			claim: Some(this_layer_composite_claim),
			multilinears: this_layer_multilinears,
		})
	}

	fn layer_claim_eval_point(&self) -> &[<P as PackedField>::Scalar] {
		self.0.eval_point()
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		if self.is_last_layer(layer_no) {
			(0, 0)
		} else {
			// this_layer_input, exponent_bit
			(2, 1)
		}
	}

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> LayerClaim<P::Scalar> {
		if self.0.is_last_layer(layer_no) {
			// the evaluation of the last exponent bit can be uniquely calculated from the previous exponentiation layer claim.
			// a_0(x) = (V_0(x) - 1)/(g - 1)
			let LayerClaim { eval_point, eval } = self.0.current_layer_claim.clone();

			let exponent_claim = LayerClaim::<P::Scalar> {
				eval_point,
				eval: first_layer_inverse::<FBase, _>(eval),
			};
			return exponent_claim;
		}

		self.0.finish_layer(layer_no, multilinear_evals, r)
	}
}

pub struct DynamicBaseExponentiationProver<'a, P: PackedField>(ExponentiationCommonProver<'a, P>);

impl<'a, P: PackedField> DynamicBaseExponentiationProver<'a, P> {
	pub fn new(
		witness: BaseExponentWitness<'a, P>,
		claim: &ExponentiationClaim<P::Scalar>,
	) -> Result<Self, Error> {
		if witness.base.is_none() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(ExponentiationCommonProver::new(witness, claim.clone())))
	}
}

pub struct ProverLayerClaimMeta<'a, P: PackedField> {
	pub claim: Option<CompositeSumClaim<P::Scalar, ProverExponentiationComposition<P::Scalar>>>,
	pub multilinears: Vec<MultilinearWitness<'a, P>>,
}

impl<'a, P: PackedField> ExponentiationProver<'a, P> for DynamicBaseExponentiationProver<'a, P> {
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width()
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<ProverLayerClaimMeta<'a, P>, Error> {
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
				ExponentiationCompositions::DynamicBaseLastLayer,
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
				ExponentiationCompositions::DynamicBase,
			)?;
			let composition = FixedDimIndexCompositions::Trivariate(composition);
			(composition, this_layer_multilinears)
		};

		let this_layer_composite_claim = CompositeSumClaim {
			sum: self.0.current_layer_claim.eval,
			composition,
		};

		Ok(ProverLayerClaimMeta {
			claim: Some(this_layer_composite_claim),
			multilinears: this_layer_multilinears,
		})
	}

	fn layer_claim_eval_point(&self) -> &[<P as PackedField>::Scalar] {
		self.0.eval_point()
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		if self.is_last_layer(layer_no) {
			// Base, exponent_bit
			(2, 1)
		} else {
			// this_layer_input, exponent_bit, Base
			(3, 1)
		}
	}

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> LayerClaim<P::Scalar> {
		self.0.finish_layer(layer_no, multilinear_evals, r)
	}
}
