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
use bytemuck::zeroed_vec;

use crate::{
	protocols::{gkr_gpa::Error, sumcheck::equal_n_vars_check},
	witness::MultilinearWitness,
};

#[derive(Clone)]
pub struct GeneratorExponentWitness<'a, PGenerator: PackedField, P: PackedField> {
	pub exponent: Vec<MultilinearWitness<'a, P>>,
	pub single_bit_output_layers_data: Vec<Vec<PGenerator>>,
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

impl<'a, PGenerator, P> GeneratorExponentWitness<'a, PGenerator, P>
where
	PGenerator: PackedField,
	PGenerator::Scalar: BinaryField,
	P: PackedExtension<PGenerator::Scalar, PackedSubfield = PGenerator>,
	P::Scalar: BinaryField + ExtensionField<PGenerator::Scalar>,
{
	pub fn new_with_static_generator<PBits>(
		exponent: Vec<MultilinearWitness<'a, P>>,
	) -> Result<Self, Error>
	where
		PBits: PackedField<Scalar = BinaryField1b>,
		PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
		P: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	{
		equal_n_vars_check(&exponent)?;

		let exponent_bit_width = exponent.len();

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

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			generator: None,
		})
	}

	pub fn new_with_dynamic_generator<PBits>(
		exponent: Vec<MultilinearWitness<'a, P>>,
		generator_evals: &[PGenerator],
	) -> Result<Self, Error>
	where
		PBits: PackedField<Scalar = BinaryField1b>,
		PGenerator: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
		P: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	{
		equal_n_vars_check(&exponent)?;

		let exponent_bit_width = exponent.len();

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed::<PBits, PGenerator>(
			&copy_witness_into_vec(&exponent[exponent_bit_width - 1]),
			Generator::Dynamic(generator_evals),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				&copy_witness_into_vec(&exponent[exponent_bit_width - layer_idx_from_left - 1]),
				Generator::Dynamic(generator_evals),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let n_vars = exponent[0].n_vars();

		let generator = MultilinearExtension::new(n_vars, generator_evals.to_vec())
			.map(MLEEmbeddingAdapter::<PGenerator, P>::from)?
			.upcast_arc_dyn();

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			generator: Some(generator),
		})
	}

	pub fn exponentiation_result(&self) -> Option<&[PGenerator]> {
		self.single_bit_output_layers_data
			.last()
			.map(|value| value.as_slice())
	}
}
