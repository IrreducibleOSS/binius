// Copyright 2025 Irreducible Inc.

use std::{mem, sync::Arc, time::Instant};

use binius_field::{BinaryField, PackedField};
use binius_math::{MLEDirectAdapter, MultilinearExtension};
use binius_maybe_rayon::{
	iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator},
	prelude::ParallelIterator,
};
use binius_utils::bail;
use tracing::instrument;

use super::error::Error;
use crate::{protocols::sumcheck::equal_n_vars_check, witness::MultilinearWitness};

#[derive(Clone)]
pub struct BaseExpWitness<'a, P: PackedField> {
	/// Multilinears that represent an integers by its bits.
	pub exponent: Vec<MultilinearWitness<'a, P>>,
	/// Circuit layer-multilinears
	pub single_bit_output_layers_data: Vec<MultilinearWitness<'a, P>>,
	/// The base to be used for exponentiation.
	pub base: BaseWitness<'a, P>,
}

impl<'a, P: PackedField> Drop for BaseExpWitness<'a, P> {
	fn drop(&mut self) {

		println!("{}", self.single_bit_output_layers_data[0].n_vars());

		let start = Instant::now();
		self.single_bit_output_layers_data.iter().for_each(|arc| println!("{}", Arc::strong_count(arc)));
		
		std::mem::drop(std::mem::take(&mut self.single_bit_output_layers_data));
		let duration = start.elapsed();
		println!("Время выполнения drop : {:?}", duration);
	}
}

#[derive(Clone)]
pub enum BaseWitness<'a, P: PackedField> {
	Constant(P::Scalar),
	Dynamic(MultilinearWitness<'a, P>),
}

#[instrument(skip_all, name = "gkr_exp::evaluate_single_bit_output_packed")]
fn evaluate_single_bit_output_packed<P>(
	exponent_bit_witness: MultilinearWitness<P>,
	base: Base<P>,
	previous_single_bit_output: &[P],
) -> Vec<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	let one = P::one();

	previous_single_bit_output
		.into_par_iter()
		.enumerate()
		.map(|(i, &prev_out)| {
			let (base, prev_out) = match &base {
				Base::Constant(g) => (*g, prev_out),
				Base::Dynamic(poly) => (
					poly.packed_evals()
						.expect("base and exponent_bit_witness have the same width")[i],
					prev_out.square(),
				),
			};

			// TODO: fix when MLEEmbeddingAdapter::packed_evals() works correct
			let exponent_bit =
				P::from_fn(|j| {
					exponent_bit_witness
					.evaluate_on_hypercube(P::WIDTH * i + j)
					.expect("previous_single_bit_output and exponent_bit_witness have the same width")
				});
			prev_out * (one + exponent_bit * (one + base))
		})
		.collect::<Vec<_>>()
}

enum Base<'a, P: PackedField> {
	Constant(P),
	Dynamic(MultilinearWitness<'a, P>),
}

fn evaluate_first_layer_output_packed<P>(
	exponent_bit_witness: MultilinearWitness<P>,
	base: Base<P>,
) -> Vec<P>
where
	P: PackedField,
	P::Scalar: BinaryField,
{
	let one = P::one();

	(0..1 << exponent_bit_witness.n_vars().saturating_sub(P::LOG_WIDTH))
		.into_par_iter()
		.map(|i| {
			let base = match &base {
				Base::Constant(g) => *g,
				Base::Dynamic(poly) => poly
					.packed_evals()
					.expect("base and exponent_bit_witness have the same width")[i],
			};

			// TODO: fix when MLEEmbeddingAdapter::packed_evals() works correct
			let exponent_bit = P::from_fn(|j| {
				exponent_bit_witness
					.evaluate_on_hypercube(P::WIDTH * i + j)
					.expect("eval on the hypercube exists")
			});

			one + exponent_bit * (one + base)
		})
		.collect::<Vec<_>>()
}

impl<'a, P> BaseExpWitness<'a, P>
where
	P: PackedField,
{
	/// Constructs a witness where the base is the constant [BinaryField].
	#[instrument(skip_all, name = "gkr_exp::new_with_constant_base")]
	pub fn new_with_constant_base(
		exponent: Vec<MultilinearWitness<'a, P>>,
		base: P::Scalar,
	) -> Result<Self, Error>
	where
		P: PackedField,
		P::Scalar: BinaryField,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExp)
		}

		if exponent.len() > P::Scalar::N_BITS {
			bail!(Error::SmallBaseField)
		}

		equal_n_vars_check(&exponent)?;

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		let mut packed_base_power_constant = P::broadcast(base);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed(
			exponent[0].clone(),
			Base::Constant(packed_base_power_constant),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			packed_base_power_constant = packed_base_power_constant.square();

			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				exponent[layer_idx_from_left].clone(),
				Base::Constant(packed_base_power_constant),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::new(exponent[0].n_vars(), single_bit_output_layers_data)
					.map(MLEDirectAdapter::<P>::from)
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
	#[instrument(skip_all, name = "gkr_exp::new_with_dynamic_base")]
	pub fn new_with_dynamic_base(
		exponent: Vec<MultilinearWitness<'a, P>>,
		base: MultilinearWitness<'a, P>,
	) -> Result<Self, Error>
	where
		P: PackedField,
		P::Scalar: BinaryField,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExp)
		}

		if exponent.len() > P::Scalar::N_BITS {
			bail!(Error::SmallBaseField)
		}

		equal_n_vars_check(&exponent)?;

		if exponent[0].n_vars() != base.n_vars() {
			bail!(Error::NumberOfVariablesMismatch)
		}

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed(
			exponent[exponent_bit_width - 1].clone(),
			Base::Dynamic(base.clone()),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				exponent[exponent_bit_width - layer_idx_from_left - 1].clone(),
				Base::Dynamic(base.clone()),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::new(exponent[0].n_vars(), single_bit_output_layers_data)
					.map(MLEDirectAdapter::<P>::from)
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
