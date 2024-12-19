// Copyright 2024 Irreducible Inc.

use std::{array, cmp::min, slice};

use crate::{protocols::gkr_gpa::Error, witness::MultilinearWitness};
use binius_field::{
	ext_base_op_par, BinaryField, BinaryField1b, ExtensionField, Field, PackedExtension,
	PackedField, PackedFieldIndexable,
};
use bytemuck::zeroed_vec;
use rayon::{
	prelude::{IndexedParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};

#[allow(dead_code)]
pub struct GeneratorExponentWitness<
	'a,
	P: PackedField,
	PE: PackedField,
	PChallenge: PackedField,
	const EXPONENT_BIT_WIDTH: usize,
> {
	pub exponent: [MultilinearWitness<'a, PChallenge>; EXPONENT_BIT_WIDTH],
	pub exponent_data: [Vec<P>; EXPONENT_BIT_WIDTH],
	pub single_bit_output_layers_data: [Vec<PE>; EXPONENT_BIT_WIDTH],
}

fn copy_witness_into_vec<P, PE>(poly: &MultilinearWitness<PE>) -> Vec<P>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	PE: PackedExtension<P::Scalar, PackedSubfield = P>,
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

fn evaluate_single_bit_output_packed<P, PE>(
	exponent_bit: &[P],
	generator_power_constant: PE::Scalar,
	previous_single_bit_output: &[PE],
) -> Vec<PE>
where
	P: PackedField<Scalar = BinaryField1b>,
	PE: PackedField,
	PE: PackedFieldIndexable,
	PE: PackedExtension<P::Scalar, PackedSubfield = P>,
	PE::Scalar: ExtensionField<BinaryField1b>,
	PE::Scalar: BinaryField,
{
	debug_assert_eq!(P::WIDTH * exponent_bit.len(), PE::WIDTH * previous_single_bit_output.len());

	let mut result = previous_single_bit_output.to_vec();

	let packed_generator_power = PE::broadcast(generator_power_constant);

	let _ = ext_base_op_par(&mut result, exponent_bit, |prev_out, exp_bit_broadcasted| {
		prev_out
			* (PE::cast_ext(
				PE::cast_base(packed_generator_power - PE::one()) * exp_bit_broadcasted,
			) + PE::one())
	});

	result
}

fn evaluate_first_layer_output_packed<P, PE>(
	exponent_bit: &[P],
	generator_power_constant: PE::Scalar,
) -> Vec<PE>
where
	P: PackedField<Scalar = BinaryField1b>,
	PE: PackedField,
	PE: PackedFieldIndexable,
	PE: PackedExtension<P::Scalar, PackedSubfield = P>,
	PE::Scalar: ExtensionField<BinaryField1b>,
{
	let mut result = vec![PE::zero(); exponent_bit.len() * PE::Scalar::DEGREE];

	let packed_generator_power = PE::broadcast(generator_power_constant);

	let _ = ext_base_op_par(&mut result, exponent_bit, |_, exp_bit_broadcasted| {
		PE::cast_ext(PE::cast_base(packed_generator_power - PE::one()) * exp_bit_broadcasted)
			+ PE::one()
	});

	result
}

impl<'a, P, PE, PChallenge, const EXPONENT_BIT_WIDTH: usize>
	GeneratorExponentWitness<'a, P, PE, PChallenge, EXPONENT_BIT_WIDTH>
where
	P: PackedField<Scalar = BinaryField1b>,
	PE: PackedField,
	PE: PackedFieldIndexable,
	PE: PackedExtension<P::Scalar, PackedSubfield = P>,
	PE::Scalar: ExtensionField<BinaryField1b> + BinaryField,
	PChallenge: PackedField,
	PChallenge: PackedExtension<P::Scalar, PackedSubfield = P>,
	PChallenge::Scalar: ExtensionField<BinaryField1b>,
{
	#[allow(dead_code)]
	pub fn new(
		exponent: [MultilinearWitness<'a, PChallenge>; EXPONENT_BIT_WIDTH],
	) -> Result<Self, Error> {
		let num_rows = 1 << exponent[0].n_vars();

		let exponent_data = array::from_fn(|i| copy_witness_into_vec(&exponent[i]));

		let mut single_bit_output_layers_data =
			array::from_fn(|_| vec![PE::zero(); num_rows / PE::WIDTH]);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed::<P, PE>(
			&exponent_data[0],
			PE::Scalar::MULTIPLICATIVE_GENERATOR,
		);

		for layer_idx_from_left in 1..EXPONENT_BIT_WIDTH {
			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				&exponent_data[layer_idx_from_left],
				PE::Scalar::MULTIPLICATIVE_GENERATOR.pow([1 << layer_idx_from_left]),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			)
		}

		Ok(Self {
			exponent,
			exponent_data,
			single_bit_output_layers_data,
		})
	}
}