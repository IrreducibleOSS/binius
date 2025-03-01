// Copyright 2025 Irreducible Inc.

use std::{cmp::min, slice};

use binius_field::{
	ext_base_op_par, BinaryField, BinaryField1b, ExtensionField, Field, PackedExtension,
	PackedField,
};
use binius_math::{MLEEmbeddingAdapter, MultilinearExtension};
use binius_maybe_rayon::{
	prelude::{IndexedParallelIterator, ParallelIterator},
	slice::ParallelSliceMut,
};
use binius_utils::bail;
use bytemuck::zeroed_vec;

use super::error::Error;
use crate::{protocols::sumcheck::equal_n_vars_check, witness::MultilinearWitness};

#[derive(Clone)]
pub struct BaseExpWitness<'a, P: PackedField, FBase: Field> {
	/// Multilinears that represent an integers by its bits.
	pub exponent: Vec<MultilinearWitness<'a, P>>,
	/// Circuit layer-multilinears
	pub single_bit_output_layers_data: Vec<MultilinearWitness<'a, P>>,
	/// The base to be used for exponentiation.
	pub base: BaseWitness<'a, P, FBase>,
}

#[derive(Clone)]
pub enum BaseWitness<'a, P: PackedField, FBase: Field> {
	Constant(FBase),
	Dynamic(MultilinearWitness<'a, P>),
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

fn evaluate_single_bit_output_packed<PBits, PBase>(
	exponent_bit: &[PBits],
	base: BaseEvals<PBase>,
	previous_single_bit_output: &[PBase],
) -> Vec<PBase>
where
	PBits: PackedField<Scalar = BinaryField1b>,
	PBase: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	PBase::Scalar: BinaryField,
{
	debug_assert_eq!(
		PBits::WIDTH * exponent_bit.len(),
		PBase::WIDTH * previous_single_bit_output.len()
	);

	let mut result = previous_single_bit_output.to_vec();

	let _ = ext_base_op_par(&mut result, exponent_bit, |i, prev_out, exp_bit_broadcasted| {
		let (base, prev_out) = match &base {
			BaseEvals::Constant(g) => (*g, prev_out),
			BaseEvals::Dynamic(g) => (g[i], prev_out.square()),
		};

		prev_out
			* (PBase::cast_ext(PBase::cast_base(base - PBase::one()) * exp_bit_broadcasted)
				+ PBase::one())
	});

	result
}

enum BaseEvals<'a, PBase: PackedField> {
	Constant(PBase),
	Dynamic(&'a [PBase]),
}

fn evaluate_first_layer_output_packed<PBits, PBase>(
	exponent_bit: &[PBits],
	base: BaseEvals<PBase>,
) -> Vec<PBase>
where
	PBits: PackedField<Scalar = BinaryField1b>,
	PBase: PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
	PBase::Scalar: BinaryField,
{
	let mut result = vec![PBase::zero(); exponent_bit.len() * PBase::Scalar::DEGREE];

	let _ = ext_base_op_par(&mut result, exponent_bit, |i, _, exp_bit_broadcasted| {
		let base = match &base {
			BaseEvals::Constant(g) => *g,
			BaseEvals::Dynamic(g) => g[i],
		};

		PBase::cast_ext(PBase::cast_base(base - PBase::one()) * exp_bit_broadcasted) + PBase::one()
	});

	result
}

impl<'a, P, FBase> BaseExpWitness<'a, P, FBase>
where
	P: PackedField,
	FBase: BinaryField,
{
	/// Constructs a witness where the base is the constant [BinaryField].
	pub fn new_with_constant_base<PBits, PBase>(
		exponent: Vec<MultilinearWitness<'a, P>>,
		base: PBase::Scalar,
	) -> Result<Self, Error>
	where
		PBits: PackedField<Scalar = BinaryField1b>,
		PBase: PackedField<Scalar = FBase> + PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
		P: PackedExtension<PBits::Scalar, PackedSubfield = PBits>
			+ PackedExtension<PBase::Scalar, PackedSubfield = PBase>,
		P::Scalar: ExtensionField<PBase::Scalar>,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExp)
		}

		if exponent.len() > PBase::Scalar::N_BITS {
			bail!(Error::SmallBaseField)
		}

		equal_n_vars_check(&exponent)?;

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		let mut packed_base_power_constant = PBase::broadcast(base);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed::<PBits, PBase>(
			&copy_witness_into_vec(&exponent[0]),
			BaseEvals::Constant(packed_base_power_constant),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			packed_base_power_constant = packed_base_power_constant.square();

			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				&copy_witness_into_vec(&exponent[layer_idx_from_left]),
				BaseEvals::Constant(packed_base_power_constant),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::new(exponent[0].n_vars(), single_bit_output_layers_data)
					.map(MLEEmbeddingAdapter::<PBase, P>::from)
					.map(|mle| mle.upcast_arc_dyn())
			})
			.collect::<Result<Vec<_>, binius_math::Error>>()?;

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			base: BaseWitness::Constant(base),
		})
	}

	/// Constructs a witness with a specified multilinear base.
	pub fn new_with_dynamic_base<PBits, PBase>(
		exponent: Vec<MultilinearWitness<'a, P>>,
		base: MultilinearWitness<'a, P>,
	) -> Result<Self, Error>
	where
		PBits: PackedField<Scalar = BinaryField1b>,
		PBase: PackedField<Scalar = FBase> + PackedExtension<PBits::Scalar, PackedSubfield = PBits>,
		P: PackedExtension<PBits::Scalar, PackedSubfield = PBits>
			+ PackedExtension<PBase::Scalar, PackedSubfield = PBase>,
		P::Scalar: ExtensionField<PBase::Scalar>,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExp)
		}

		if exponent.len() > PBase::Scalar::N_BITS {
			bail!(Error::SmallBaseField)
		}

		equal_n_vars_check(&exponent)?;

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		let base_evals = copy_witness_into_vec::<PBase, P>(&base);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed::<PBits, PBase>(
			&copy_witness_into_vec(&exponent[exponent_bit_width - 1]),
			BaseEvals::Dynamic(&base_evals),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				&copy_witness_into_vec(&exponent[exponent_bit_width - layer_idx_from_left - 1]),
				BaseEvals::Dynamic(&base_evals),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::from_values(single_bit_output_layers_data)
					.map(MLEEmbeddingAdapter::<PBase, P>::from)
					.map(|mle| mle.upcast_arc_dyn())
			})
			.collect::<Result<Vec<_>, binius_math::Error>>()?;

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			base: BaseWitness::Dynamic(base),
		})
	}

	pub const fn uses_dynamic_base(&self) -> bool {
		match self.base {
			BaseWitness::Constant(_) => false,
			BaseWitness::Dynamic(_) => true,
		}
	}

	pub fn n_vars(&self) -> usize {
		self.exponent[0].n_vars()
	}

	/// Returns the multilinear that corresponds to the exponentiation of the base to an integers.
	pub fn exponentiation_result_witness(&self) -> MultilinearWitness<'a, P> {
		self.single_bit_output_layers_data
			.last()
			.expect("single_bit_output_layers_data not empty")
			.clone()
	}
}
