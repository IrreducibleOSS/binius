// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField, RepackedExtension};
use binius_math::MultilinearExtension;
use binius_maybe_rayon::{
	iter::{IndexedParallelIterator, IntoParallelIterator},
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

#[derive(Clone)]
pub enum BaseWitness<'a, P: PackedField> {
	Static(P::Scalar),
	Dynamic(MultilinearWitness<'a, P>),
}

#[instrument(skip_all, name = "gkr_exp::evaluate_single_bit_output_packed")]
fn evaluate_single_bit_output_packed<P, PExpBase>(
	exponent_bit_witness: MultilinearWitness<P>,
	base: Base<P, PExpBase>,
	previous_single_bit_output: &[PExpBase],
) -> Vec<PExpBase>
where
	P: RepackedExtension<PExpBase>,
	PExpBase: PackedField,
{
	let one = PExpBase::one();

	previous_single_bit_output
		.into_par_iter()
		.enumerate()
		.map(|(i, &prev_out)| {
			let (base, prev_out) = match &base {
				Base::Static(g) => (*g, prev_out),
				Base::Dynamic(poly) => (
					P::cast_bases(
						poly.packed_evals()
							.expect("base and exponent_bit_witness have the same width"),
					)[i],
					prev_out.square(),
				),
			};

			let exponent_bit = PExpBase::from_fn(|j| {
				let ext_bit = exponent_bit_witness
					.evaluate_on_hypercube(PExpBase::WIDTH * i + j)
					.expect("eval on the hypercube exists");

				if ext_bit == P::Scalar::one() {
					PExpBase::Scalar::one()
				} else {
					PExpBase::Scalar::zero()
				}
			});

			prev_out * (one + exponent_bit * (one + base))
		})
		.collect::<Vec<_>>()
}

enum Base<'a, P: PackedField, PExpBase: PackedField> {
	Static(PExpBase),
	Dynamic(MultilinearWitness<'a, P>),
}

fn evaluate_first_layer_output_packed<P, PExpBase>(
	exponent_bit_witness: MultilinearWitness<P>,
	base: Base<P, PExpBase>,
) -> Vec<PExpBase>
where
	P: RepackedExtension<PExpBase>,
	PExpBase: PackedField,
{
	let one = PExpBase::one();

	(0..1
		<< exponent_bit_witness
			.n_vars()
			.saturating_sub(PExpBase::LOG_WIDTH))
		.into_par_iter()
		.map(|i| {
			let base = match &base {
				Base::Static(g) => *g,
				Base::Dynamic(poly) => P::cast_bases(
					poly.packed_evals()
						.expect("base and exponent_bit_witness have the same width"),
				)[i],
			};

			let exponent_bit = PExpBase::from_fn(|j| {
				let ext_bit = exponent_bit_witness
					.evaluate_on_hypercube(PExpBase::WIDTH * i + j)
					.expect("eval on the hypercube exists");

				if ext_bit == P::Scalar::one() {
					PExpBase::Scalar::one()
				} else {
					PExpBase::Scalar::zero()
				}
			});

			one + exponent_bit * (one + base)
		})
		.collect::<Vec<_>>()
}

impl<'a, P> BaseExpWitness<'a, P>
where
	P: PackedField,
{
	/// Constructs a witness where the base is the static [BinaryField].
	#[instrument(skip_all, name = "gkr_exp::new_with_static_base")]
	pub fn new_with_static_base<PExpBase>(
		exponent: Vec<MultilinearWitness<'a, P>>,
		base: PExpBase::Scalar,
	) -> Result<Self, Error>
	where
		P: RepackedExtension<PExpBase>,
		PExpBase::Scalar: BinaryField,
		PExpBase: PackedField,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExp)
		}

		if exponent.len() > PExpBase::Scalar::N_BITS {
			bail!(Error::SmallBaseField)
		}

		equal_n_vars_check(&exponent)?;

		let mut single_bit_output_layers_data = vec![Vec::new(); exponent_bit_width];

		let mut packed_base_power_static = PExpBase::broadcast(base);

		single_bit_output_layers_data[0] = evaluate_first_layer_output_packed(
			exponent[0].clone(),
			Base::Static(packed_base_power_static),
		);

		for layer_idx_from_left in 1..exponent_bit_width {
			packed_base_power_static = packed_base_power_static.square();

			single_bit_output_layers_data[layer_idx_from_left] = evaluate_single_bit_output_packed(
				exponent[layer_idx_from_left].clone(),
				Base::Static(packed_base_power_static),
				&single_bit_output_layers_data[layer_idx_from_left - 1],
			);
		}

		let single_bit_output_layers_data = single_bit_output_layers_data
			.into_iter()
			.map(|single_bit_output_layers_data| {
				MultilinearExtension::new(exponent[0].n_vars(), single_bit_output_layers_data)
					.map(|me| me.specialize_arc_dyn())
			})
			.collect::<Result<Vec<_>, binius_math::Error>>()?;

		Ok(Self {
			exponent,
			single_bit_output_layers_data,
			base: BaseWitness::Static(base.into()),
		})
	}

	/// Constructs a witness with a specified multilinear base.
	///
	/// # Requirements
	/// For efficiency, the internal base witness data is required to be `PExpBase`.
	#[instrument(skip_all, name = "gkr_exp::new_with_dynamic_base")]
	pub fn new_with_dynamic_base<PExpBase>(
		exponent: Vec<MultilinearWitness<'a, P>>,
		base: MultilinearWitness<'a, P>,
	) -> Result<Self, Error>
	where
		P: RepackedExtension<PExpBase>,
		PExpBase::Scalar: BinaryField,
		PExpBase: PackedField,
	{
		let exponent_bit_width = exponent.len();

		if exponent_bit_width == 0 {
			bail!(Error::EmptyExp)
		}

		if exponent.len() > PExpBase::Scalar::N_BITS {
			bail!(Error::SmallBaseField)
		}

		equal_n_vars_check(&exponent)?;

		if exponent[0].n_vars() != base.n_vars() {
			bail!(Error::NumberOfVariablesMismatch)
		}

		let base_evals: &[PExpBase] = P::cast_bases(base.packed_evals().unwrap());
		if base_evals.len() != 1 << (exponent[0].n_vars().saturating_sub(PExpBase::LOG_WIDTH)) {
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
					.map(|me| me.specialize_arc_dyn())
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
			BaseWitness::Static(_) => false,
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
