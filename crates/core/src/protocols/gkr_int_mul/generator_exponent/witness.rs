// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::min, slice};

use binius_field::{
	ext_base_op_par, BinaryField, BinaryField1b, ExtensionField, PackedExtension, PackedField,
};
use binius_math::{MLEEmbeddingAdapter, MultilinearExtension};
use binius_maybe_rayon::{
	prelude::{IndexedParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use binius_utils::bail;
use bytemuck::zeroed_vec;

use crate::{
	protocols::{gkr_int_mul::error::Error, sumcheck::equal_n_vars_check},
	witness::MultilinearWitness,
};

#[derive(Clone)]
pub struct GeneratorExponentWitness<'a, P: PackedField> {
	pub exponent: Vec<MultilinearWitness<'a, P>>,
	pub single_bit_output_layers_data: Vec<MultilinearWitness<'a, P>>,
	pub generator: Option<MultilinearWitness<'a, P>>,
}

fn copy_witness_into_vec<P, PE>(poly: &MultilinearWitness<PE>) -> Vec<P>
where
	P: PackedField,
	PE: PackedExtension<P::Scalar, PackedSubfield = P>,
	PE::Scalar: ExtensionField<P::Scalar>,
{
	let mut input_layer: Vec<P> = zeroed_vec(1 << poly.n_vars().saturating_sub(P::LOG_WIDTH));

	let log_degree = PE::Scalar::LOG_DEGREE;
	if poly.n_vars() >= P::LOG_WIDTH {
		let log_chunk_size = min(poly.n_vars() - P::LOG_WIDTH, 12);
		input_layer
			.par_chunks_mut(1 << log_chunk_size)
			.enumerate()
			.for_each(|(i, chunk)| {
				poly.subcube_evals(
					log_chunk_size + P::LOG_WIDTH,
					i,
					log_degree,
					PE::cast_exts_mut(chunk),
				)
				.expect("")
			});
	} else {
		poly.subcube_evals(
			poly.n_vars(),
			0,
			log_degree,
			PE::cast_exts_mut(slice::from_mut(&mut input_layer[0])),
		)
		.expect("index is between 0 and 2^{n_vars - log_chunk_size}; log_embedding degree is 0");
	}

	input_layer
}

fn evaluate_single_bit_output_packed<PBits, PGenerator>(
	exponent_bit: &[PBits],
	generator: Generator<PGenerator>,
	previous_single_bit_output: &[PGenerator],
) -> Vec<PGenerator>
where
	PBits: PackedField<Scalar = BinaryField1b>,
	PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	PGenerator::Scalar: BinaryField,
{
	debug_assert_eq!(
		PBits::WIDTH * exponent_bit.len(),
		PGenerator::WIDTH * previous_single_bit_output.len()
	);

	let mut result = previous_single_bit_output.to_vec();

	let _ = ext_base_op_par(&mut result, exponent_bit, |i, prev_out, exp_bit_broadcasted| {
		let (generator, prev_out) = match &generator {
			Generator::Static(g) => (*g, prev_out),
			Generator::Dynamic(g) => (g[i], prev_out.square()),
		};

		prev_out
			* (PGenerator::cast_ext(
				PGenerator::cast_base(generator - PGenerator::one()) * exp_bit_broadcasted,
			) + PGenerator::one())
	});

	result
}

enum Generator<'a, PGenerator: PackedField> {
	Static(PGenerator),
	Dynamic(&'a [PGenerator]),
}

fn evaluate_first_layer_output_packed<PBits, PGenerator>(
	exponent_bit: &[PBits],
	generator: Generator<PGenerator>,
) -> Vec<PGenerator>
where
	PBits: PackedField<Scalar = BinaryField1b>,
	PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	PGenerator::Scalar: BinaryField,
{
	let mut result = vec![PGenerator::zero(); exponent_bit.len() * PGenerator::Scalar::DEGREE];

	let _ = ext_base_op_par(&mut result, exponent_bit, |i, _, exp_bit_broadcasted| {
		let generator = match &generator {
			Generator::Static(g) => *g,
			Generator::Dynamic(g) => g[i],
		};

		PGenerator::cast_ext(
			PGenerator::cast_base(generator - PGenerator::one()) * exp_bit_broadcasted,
		) + PGenerator::one()
	});

	result
}

impl<'a, P> GeneratorExponentWitness<'a, P>
where
	P: PackedField,
{
	pub fn new_with_static_generator<PBits, PGenerator>(
		exponent: Vec<MultilinearWitness<'a, P>>,
	) -> Result<Self, Error>
	where
		PBits: PackedField<Scalar = BinaryField1b>,
		PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
		PGenerator::Scalar: BinaryField,
		P: PackedExtension<PBits::Scalar, PackedSubfield = PBits>
			+ PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>,
		P::Scalar: ExtensionField<PGenerator::Scalar>,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExponent)
		}

		equal_n_vars_check(&exponent)?;

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		let mut packed_generator_power_constant =
			PGenerator::broadcast(PGenerator::Scalar::MULTIPLICATIVE_GENERATOR);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed::<PBits, PGenerator>(
			&copy_witness_into_vec(&exponent[0]),
			Generator::Static(packed_generator_power_constant),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			packed_generator_power_constant = packed_generator_power_constant.square();

			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				&copy_witness_into_vec(&exponent[layer_idx_from_left]),
				Generator::Static(packed_generator_power_constant),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::new(exponent[0].n_vars(), single_bit_output_layers_data)
					.map(MLEEmbeddingAdapter::<PGenerator, P>::from)
					.map(|mle| mle.upcast_arc_dyn())
			})
			.collect::<Result<Vec<_>, binius_math::Error>>()?;

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			generator: None,
		})
	}

	pub fn new_with_dynamic_generator<PBits, PGenerator>(
		exponent: Vec<MultilinearWitness<'a, P>>,
		generator: MultilinearWitness<'a, P>,
	) -> Result<Self, Error>
	where
		PBits: PackedField<Scalar = BinaryField1b>,
		PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
		PGenerator::Scalar: BinaryField,
		P: PackedExtension<PBits::Scalar, PackedSubfield = PBits>
			+ PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>,
		P::Scalar: ExtensionField<PGenerator::Scalar>,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExponent)
		}

		equal_n_vars_check(&exponent)?;

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		let generator_evals = copy_witness_into_vec::<PGenerator, P>(&generator);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed::<PBits, PGenerator>(
			&copy_witness_into_vec(&exponent[exponent_bit_width - 1]),
			Generator::Dynamic(&generator_evals),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				&copy_witness_into_vec(&exponent[exponent_bit_width - layer_idx_from_left - 1]),
				Generator::Dynamic(&generator_evals),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::from_values(single_bit_output_layers_data)
					.map(MLEEmbeddingAdapter::<PGenerator, P>::from)
					.map(|mle| mle.upcast_arc_dyn())
			})
			.collect::<Result<Vec<_>, binius_math::Error>>()?;

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			generator: Some(generator),
		})
	}

	pub fn with_dynamic_generator(&self) -> bool {
		self.generator.is_some()
	}

	pub fn n_vars(&self) -> usize {
		self.exponent[0].n_vars()
	}

	pub fn exponentiation_result_witness(&self) -> MultilinearWitness<'a, P> {
		self.single_bit_output_layers_data
			.last()
			.expect("witness contains single_bit_output_layers_data")
			.clone()
	}
}
