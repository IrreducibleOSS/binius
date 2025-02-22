// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, ExtensionField, PackedField};
use binius_utils::bail;

use super::{
	common::GeneratorExponentClaim, compositions::ExponentCompositions, utils::first_layer_inverse,
	witness::GeneratorExponentWitness,
};
use crate::{
	composition::{ComplexIndexComposition, IndexComposition},
	protocols::{gkr_gpa::LayerClaim, gkr_int_mul::error::Error, sumcheck::CompositeSumClaim},
	witness::MultilinearWitness,
};

pub enum Provers<'a, P: PackedField, FGenerator> {
	StaticProver(StaticProver<'a, P, FGenerator>),
	DynamicProver(DynamicProver<'a, P>),
}

impl<'a, P: PackedField, FGenerator> Provers<'a, P, FGenerator> {
	pub fn new(
		witness: GeneratorExponentWitness<'a, P>,
		claim: &GeneratorExponentClaim<P::Scalar>,
	) -> Self {
		let is_dynamic_prover = witness.generator.is_some();

		let prover = CommonProver::new(witness, claim.clone());

		if is_dynamic_prover {
			Self::DynamicProver(DynamicProver(prover))
		} else {
			Self::StaticProver(StaticProver(prover, PhantomData))
		}
	}
}

impl<'a, P, FGenerator> GeneratorProver<'a, P> for Provers<'a, P, FGenerator>
where
	FGenerator: BinaryField,
	P::Scalar: BinaryField + ExtensionField<FGenerator>,
	P: PackedField,
{
	fn exponent_bit_width(&self) -> usize {
		match self {
			Self::StaticProver(prover) => prover.exponent_bit_width(),
			Self::DynamicProver(prover) => prover.exponent_bit_width(),
		}
	}

	fn get_layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<ProverLayerClaimMeta<'a, P>, Error> {
		match self {
			Self::StaticProver(prover) => prover.get_layer_composite_sum_claim(
				layer_no,
				composite_claims_n_multilinears,
				multilinears_index,
			),
			Self::DynamicProver(prover) => prover.get_layer_composite_sum_claim(
				layer_no,
				composite_claims_n_multilinears,
				multilinears_index,
			),
		}
	}

	fn eval_point(&self) -> &[<P as PackedField>::Scalar] {
		match self {
			Self::StaticProver(prover) => prover.eval_point(),
			Self::DynamicProver(prover) => prover.eval_point(),
		}
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		match self {
			Self::StaticProver(prover) => prover.layer_n_multilinears_n_claims(layer_no),
			Self::DynamicProver(prover) => prover.layer_n_multilinears_n_claims(layer_no),
		}
	}

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> LayerClaim<P::Scalar> {
		match self {
			Self::StaticProver(prover) => prover.finish_layer(layer_no, multilinear_evals, r),
			Self::DynamicProver(prover) => prover.finish_layer(layer_no, multilinear_evals, r),
		}
	}
}

pub trait GeneratorProver<'a, P: PackedField> {
	fn exponent_bit_width(&self) -> usize;

	fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}

	fn eval_point(&self) -> &[P::Scalar];

	fn get_layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<ProverLayerClaimMeta<'a, P>, Error>;

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize);

	fn finish_layer(
		&mut self,
		layer_no: usize,
		multilinear_evals: &[P::Scalar],
		r: &[P::Scalar],
	) -> LayerClaim<P::Scalar>;
}

struct CommonProver<'a, P: PackedField> {
	witness: GeneratorExponentWitness<'a, P>,
	current_layer_claim: LayerClaim<P::Scalar>,
}

impl<'a, P: PackedField> CommonProver<'a, P> {
	fn new(
		witness: GeneratorExponentWitness<'a, P>,
		claim: GeneratorExponentClaim<P::Scalar>,
	) -> Self {
		Self {
			witness,
			current_layer_claim: claim.into(),
		}
	}

	pub fn exponent_bit_width(&self) -> usize {
		self.witness.exponent.len()
	}

	pub fn current_layer_single_bit_output_layers_data(
		&self,
		layer_no: usize,
	) -> MultilinearWitness<'a, P> {
		let index = self.witness.single_bit_output_layers_data.len() - layer_no - 2;

		self.witness.single_bit_output_layers_data[index].clone()
	}

	pub fn eval_point(&self) -> &[P::Scalar] {
		&self.current_layer_claim.eval_point
	}

	pub fn current_layer_exponent_bit(&self, index: usize) -> MultilinearWitness<'a, P> {
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
		if self.is_last_layer(layer_no) {
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

pub struct StaticProver<'a, P: PackedField, FGenerator>(
	CommonProver<'a, P>,
	PhantomData<FGenerator>,
);

impl<'a, P: PackedField, FGenerator> StaticProver<'a, P, FGenerator> {
	pub fn new(
		witness: GeneratorExponentWitness<'a, P>,
		claim: &GeneratorExponentClaim<P::Scalar>,
	) -> Result<Self, Error> {
		if witness.generator.is_some() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(CommonProver::new(witness, claim.clone()), PhantomData))
	}
}

impl<'a, P: PackedField, FGenerator: BinaryField> GeneratorProver<'a, P>
	for StaticProver<'a, P, FGenerator>
where
	P::Scalar: BinaryField + ExtensionField<FGenerator>,
{
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width()
	}

	fn get_layer_composite_sum_claim(
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

		let this_round_input = self.0.current_layer_single_bit_output_layers_data(layer_no);

		let exponent_bit = self.0.current_layer_exponent_bit(internal_layer_index);

		let this_round_multilinears = vec![this_round_input.clone(), exponent_bit.clone()];

		let generator_power_constant =
			P::Scalar::from(FGenerator::MULTIPLICATIVE_GENERATOR.pow(1 << internal_layer_index));

		let composition = IndexComposition::new(
			composite_claims_n_multilinears,
			[multilinears_index, multilinears_index + 1],
			ExponentCompositions::StaticGenerator {
				generator_power_constant,
			},
		)?;

		let composition = ComplexIndexComposition::Bivariate(composition);

		let this_round_composite_claim = CompositeSumClaim {
			sum: self.0.current_layer_claim.eval,
			composition,
		};

		Ok(ProverLayerClaimMeta {
			claim: Some(this_round_composite_claim),
			multilinears: this_round_multilinears,
		})
	}

	fn eval_point(&self) -> &[<P as PackedField>::Scalar] {
		self.0.eval_point()
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		if self.is_last_layer(layer_no) {
			(0, 0)
		} else {
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
			let LayerClaim { eval_point, eval } = self.0.current_layer_claim.clone();

			let exponent_claim = LayerClaim::<P::Scalar> {
				eval_point,
				eval: first_layer_inverse::<FGenerator, _>(eval),
			};
			return exponent_claim;
		}

		self.0.finish_layer(layer_no, multilinear_evals, r)
	}
}

pub struct DynamicProver<'a, P: PackedField>(CommonProver<'a, P>);

impl<'a, P: PackedField> DynamicProver<'a, P> {
	pub fn new(
		witness: GeneratorExponentWitness<'a, P>,
		claim: &GeneratorExponentClaim<P::Scalar>,
	) -> Result<Self, Error> {
		if witness.generator.is_none() {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(CommonProver::new(witness, claim.clone())))
	}
}

#[allow(clippy::type_complexity)]
pub struct ProverLayerClaimMeta<'a, P: PackedField> {
	pub claim: Option<
		CompositeSumClaim<P::Scalar, ComplexIndexComposition<ExponentCompositions<P::Scalar>>>,
	>,
	pub multilinears: Vec<MultilinearWitness<'a, P>>,
}

impl<'a, P: PackedField> GeneratorProver<'a, P> for DynamicProver<'a, P> {
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width()
	}

	fn get_layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
	) -> Result<ProverLayerClaimMeta<'a, P>, Error> {
		let generator = self
			.0
			.witness
			.generator
			.clone()
			.expect("dynamic generator witness must contain generator");

		let exponent_bit = self.0.current_layer_exponent_bit(layer_no);

		let (composition, this_round_multilinears) = if self.0.is_last_layer(layer_no) {
			let this_round_multilinears = vec![generator, exponent_bit];

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[multilinears_index, multilinears_index + 1],
				ExponentCompositions::DynamicGeneratorLastLayer,
			)?;
			let composition = ComplexIndexComposition::Bivariate(composition);
			(composition, this_round_multilinears)
		} else {
			let this_round_input = self
				.0
				.current_layer_single_bit_output_layers_data(layer_no)
				.clone();

			let this_round_multilinears = vec![this_round_input, exponent_bit, generator];
			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[
					multilinears_index,
					multilinears_index + 1,
					multilinears_index + 2,
				],
				ExponentCompositions::DynamicGenerator,
			)?;
			let composition = ComplexIndexComposition::Trivariate(composition);
			(composition, this_round_multilinears)
		};

		let this_round_composite_claim = CompositeSumClaim {
			sum: self.0.current_layer_claim.eval,
			composition,
		};

		Ok(ProverLayerClaimMeta {
			claim: Some(this_round_composite_claim),
			multilinears: this_round_multilinears,
		})
	}

	fn eval_point(&self) -> &[<P as PackedField>::Scalar] {
		self.0.eval_point()
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		if self.is_last_layer(layer_no) {
			(2, 1)
		} else {
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
