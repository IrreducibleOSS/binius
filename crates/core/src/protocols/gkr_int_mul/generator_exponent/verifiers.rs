// Copyright 2024-2025 Irreducible Inc.

use std::marker::PhantomData;

use binius_field::{BinaryField, ExtensionField, Field, PackedField};
use binius_utils::bail;

use super::{
	common::ExponentiationClaim,
	compositions::{ExponentiationCompositions, VerifierExponentiationComposition},
	utils::first_layer_inverse,
};
use crate::{
	composition::{IndexComposition, PresizedIndexCompositions},
	protocols::{
		gkr_gpa::LayerClaim,
		gkr_int_mul::error::Error,
		sumcheck::{zerocheck::ExtraProduct, CompositeSumClaim},
	},
};

pub trait ExponentiationVerifier<F: Field> {
	fn exponent_bit_width(&self) -> usize;

	fn is_last_layer(&self, layer_no: usize) -> bool {
		self.exponent_bit_width() - 1 - layer_no == 0
	}

	// return the eval_point of the current layer claim.
	fn layer_claim_eval_point(&self) -> &[F];

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
		eq_ind_index: usize,
	) -> Result<Option<CompositeSumClaim<F, VerifierExponentiationComposition<F>>>, Error>;

	// return a tuple of the number of multilinears and the sumcheck claims used by this verifier for the current layer
	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize);

	// update the current exponentiation layer claim and return the exponent bit layer claim
	fn finish_layer(&mut self, layer_no: usize, multilinear_evals: &[F], r: &[F]) -> LayerClaim<F>;
}

pub struct ExponentiationStaticVerifier<F: Field, FGenerator>(
	ExponentiationClaim<F>,
	PhantomData<FGenerator>,
);

impl<F: Field, FGenerator> ExponentiationStaticVerifier<F, FGenerator> {
	pub fn new(claim: &ExponentiationClaim<F>) -> Result<Self, Error> {
		if claim.with_dynamic_generator {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(claim.clone(), PhantomData))
	}
}

impl<F, FGenerator> ExponentiationVerifier<F> for ExponentiationStaticVerifier<F, FGenerator>
where
	FGenerator: BinaryField,
	F: BinaryField + ExtensionField<FGenerator>,
{
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width
	}

	fn layer_claim_eval_point(&self) -> &[F] {
		&self.0.eval_point
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		if self.is_last_layer(layer_no) {
			(0, 0)
		} else {
			// this_round_input, exponent_bit
			(2, 1)
		}
	}

	fn finish_layer(&mut self, layer_no: usize, multilinear_evals: &[F], r: &[F]) -> LayerClaim<F> {
		if self.is_last_layer(layer_no) {
			// the evaluation of the last exponent bit can be uniquely calculated from the previous exponentiation layer claim.
			// a_0(x) = (V_0(x) - 1)/(g - 1)

			LayerClaim {
				eval_point: self.0.eval_point.clone(),
				eval: first_layer_inverse::<FGenerator, _>(self.0.eval),
			}
		} else {
			let n_vars = self.layer_claim_eval_point().len();

			let eval_point = &r[r.len() - n_vars..];
			if !self.is_last_layer(layer_no) {
				self.0.eval = multilinear_evals[0];
				self.0.eval_point = eval_point.to_vec();
			}

			LayerClaim {
				eval: multilinear_evals[1],
				eval_point: eval_point.to_vec(),
			}
		}
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
		eq_ind_index: usize,
	) -> Result<Option<CompositeSumClaim<F, VerifierExponentiationComposition<F>>>, Error> {
		if self.is_last_layer(layer_no) {
			Ok(None)
		} else {
			let internal_layer_index = self.exponent_bit_width() - 1 - layer_no;

			let generator_power_constant =
				F::from(FGenerator::MULTIPLICATIVE_GENERATOR.pow(1 << internal_layer_index));

			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[multilinears_index, multilinears_index + 1, eq_ind_index],
				ExtraProduct {
					inner: ExponentiationCompositions::StaticGenerator {
						generator_power_constant,
					},
				},
			)?;

			let this_round_composite_claim = CompositeSumClaim {
				sum: self.0.eval,
				composition: PresizedIndexCompositions::Trivariate(composition),
			};

			Ok(Some(this_round_composite_claim))
		}
	}
}

pub struct ExponentiationDynamicVerifier<F: Field>(ExponentiationClaim<F>);

impl<F: Field> ExponentiationDynamicVerifier<F> {
	pub fn new(claim: &ExponentiationClaim<F>) -> Result<Self, Error> {
		if !claim.with_dynamic_generator {
			bail!(Error::IncorrectWitnessType);
		}

		Ok(Self(claim.clone()))
	}
}

impl<F: Field> ExponentiationVerifier<F> for ExponentiationDynamicVerifier<F> {
	fn exponent_bit_width(&self) -> usize {
		self.0.exponent_bit_width
	}

	fn layer_claim_eval_point(&self) -> &[F] {
		&self.0.eval_point
	}

	fn layer_n_multilinears_n_claims(&self, layer_no: usize) -> (usize, usize) {
		if self.is_last_layer(layer_no) {
			// generator, exponent_bit
			(2, 1)
		} else {
			// this_round_input, exponent_bit, generator
			(3, 1)
		}
	}

	fn finish_layer(&mut self, layer_no: usize, multilinear_evals: &[F], r: &[F]) -> LayerClaim<F> {
		let n_vars = self.layer_claim_eval_point().len();

		let eval_point = &r[r.len() - n_vars..];
		if !self.is_last_layer(layer_no) {
			self.0.eval = multilinear_evals[0];
			self.0.eval_point = eval_point.to_vec();
		}

		LayerClaim {
			eval: multilinear_evals[1],
			eval_point: eval_point.to_vec(),
		}
	}

	fn layer_composite_sum_claim(
		&self,
		layer_no: usize,
		composite_claims_n_multilinears: usize,
		multilinears_index: usize,
		eq_ind_index: usize,
	) -> Result<Option<CompositeSumClaim<F, VerifierExponentiationComposition<F>>>, Error> {
		let composition = if self.is_last_layer(layer_no) {
			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[multilinears_index, multilinears_index + 1, eq_ind_index],
				ExtraProduct {
					inner: ExponentiationCompositions::DynamicGeneratorLastLayer,
				},
			)?;

			PresizedIndexCompositions::Trivariate(composition)
		} else {
			let composition = IndexComposition::new(
				composite_claims_n_multilinears,
				[
					multilinears_index,
					multilinears_index + 1,
					multilinears_index + 2,
					eq_ind_index,
				],
				ExtraProduct {
					inner: ExponentiationCompositions::DynamicGenerator,
				},
			)?;

			PresizedIndexCompositions::Quadrivariate(composition)
		};

		let this_round_composite_claim = CompositeSumClaim {
			sum: self.0.eval,
			composition,
		};

		Ok(Some(this_round_composite_claim))
	}
}
