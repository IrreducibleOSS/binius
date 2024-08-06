// Copyright 2024 Ulvetanna Inc.

use std::mem::size_of;
use std::rc::Rc;
use std::slice::from_raw_parts;
use std::sync::{Arc, Mutex};
use itertools::Itertools;
use linerate_binius_sumcheck::{Descriptor, SumcheckFpgaAccelerator};
use tracing::instrument;
use crate::{eq_ind_reducer::EqIndReducer, utils::tensor_product, Error};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField};
use binius_math::EvaluationDomain;

/// Describes the shape of the zerocheck computation.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundParameters {
	pub round: usize,
	pub n_vars: usize,
	pub cols: usize,
	pub degree: usize,
	pub small_field_width: Option<usize>,
}

/// Represents input data of the computation round.
#[derive(Clone, Debug)]
pub struct ZerocheckRoundInput<'a, F, PW, FDomain>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
	FDomain: Field,
{
	pub zc_challenges: &'a [F],
	pub eq_ind: &'a [PW],
	pub query: Option<&'a [PW]>,
	pub current_round_sum: F,
	pub mixing_challenge: F,
	pub domain: &'a EvaluationDomain<FDomain>,
	pub underlier_data: Option<Vec<Option<Vec<u8>>>>,
}

/// A callback interface to handle the zerocheck computation on the CPU.
pub trait ZerocheckCpuBackendHelper<F, PW, FDomain>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
	FDomain: Field,
{
	fn handle_zerocheck_round(
		&mut self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW, FDomain>,
	) -> Result<Vec<PW::Scalar>, Error>;

	#[instrument(skip_all)]
	fn remove_smaller_domain_optimization(&mut self);
}

// TODO: This shouldn't do any `unsafe` operations. Improve the trait bounds.
#[instrument(skip_all)]
pub(crate) fn run_zerocheck<F, PW, FDomain>(
	accelerator: Option<Arc<Mutex<SumcheckFpgaAccelerator>>>,
	params: &ZerocheckRoundParameters,
	input: &ZerocheckRoundInput<F, PW, FDomain>,
) -> Result<Option<Vec<PW::Scalar>>, Error>
where
	F: Field,
	PW: PackedField + PackedExtension<FDomain>,
	PW::Scalar: From<F> + Into<F> + ExtensionField<FDomain>,
	FDomain: Field,
{
	if let (Some(accelerator), Some(tensor), Some(small_field_width)) = (accelerator, input.query, params.small_field_width) {
		let linerate_descriptor = Descriptor {
			trace_matrix: build_trace_matrix(params, input)?,
			rows: 1 << params.n_vars,
			cols: params.cols,
			small_field_width,
			// TODO: Code should be included in the ZerocheckRoundParameters.
			// Until symbolic evaluation is available, we're doing our best.
			code: Rc::new(get_compiled_code(params)?),
			degree: params.degree,
		};

		assert_eq!(size_of::<u128>(), size_of::<PW::Scalar>());
		assert_eq!(size_of::<u128>(), size_of::<F>());
		let tensor =
			unsafe { from_raw_parts(tensor.as_ptr() as *const u128, tensor.len() * PW::WIDTH) };

		let mixing_challenge = unsafe {
			let mixing_challenge = PW::Scalar::from(input.mixing_challenge);
			let ptr = &mixing_challenge as *const PW::Scalar as *const u128;
			*ptr
		};

		let prefix_query_len = SumcheckFpgaAccelerator::INTERLEAVE.ilog2() as usize;
		assert_eq!(1 << prefix_query_len, SumcheckFpgaAccelerator::INTERLEAVE);
		assert_eq!(input.zc_challenges.len(), params.n_vars - 1);
		let zc_challenges_pw: Vec<PW::Scalar> = input.zc_challenges
			[params.round..params.round + prefix_query_len]
			.iter()
			.map(|x| PW::Scalar::from(*x))
			.collect_vec();
		let prefix_eq_ind_scalar: Vec<PW::Scalar> =
			tensor_product::<PW::Scalar>(&zc_challenges_pw)?;
		let prefix_eq_ind = unsafe {
			from_raw_parts(prefix_eq_ind_scalar.as_ptr() as *const u128, prefix_eq_ind_scalar.len())
		};

		let eq_ind = unsafe {
			from_raw_parts(
				input.eq_ind.as_ptr() as *const PW::Scalar,
				input.eq_ind.len() * PW::WIDTH,
			)
		};
		let aggregator = EqIndReducer::<PW::Scalar>::new(eq_ind);

		let mut evals = {
			let mut accelerator_lock = accelerator.lock().unwrap();
			accelerator_lock
				.zerocheck(
					&linerate_descriptor,
					params.round,
					tensor,
					&mixing_challenge,
					prefix_eq_ind,
					aggregator,
				)
				.map_err(Error::LinerateSumcheckError)?
		};

		let zero_evaluation = if params.round == 0 {
			assert_eq!(input.current_round_sum, F::ZERO);
			PW::Scalar::ZERO
		} else {
			assert_eq!(input.zc_challenges.len(), params.n_vars - 1);
			let alpha = PW::Scalar::from(input.zc_challenges[params.round - 1]);
			let alpha_bar = PW::Scalar::ONE - alpha;
			let one_evaluation = evals[0];
			let zero_evaluation_numerator =
				PW::Scalar::from(input.current_round_sum) - one_evaluation * alpha;
			let zero_evaluation_denominator_inv = alpha_bar.invert().unwrap();
			zero_evaluation_numerator * zero_evaluation_denominator_inv
		};
		evals.insert(0, zero_evaluation);
		let coeffs = input.domain.interpolate(&evals)?;
		return Ok(Some(coeffs[1..].to_vec()));
	}
	Ok(None)
}

#[instrument(skip_all)]
fn build_trace_matrix<F, PW, FDomain>(
	params: &ZerocheckRoundParameters,
	input: &ZerocheckRoundInput<F, PW, FDomain>,
) -> Result<Rc<Vec<u8>>, Error>
where
	F: Field,
	PW: PackedField,
	PW::Scalar: From<F> + Into<F>,
	FDomain: Field,
{
	let cols = params.cols;
	let rows = 1 << params.n_vars;
	let small_field_width = params.small_field_width.ok_or(Error::MissingData)?;
	assert_eq!(0, (cols * rows * small_field_width) % 8);
	let bytes = (cols * rows * small_field_width) / 8;
	let underlier_data = input
		.underlier_data
		.as_ref()
		.ok_or(Error::MissingData)?;
	let mut trace_matrix = vec![0u8; bytes];
	match small_field_width {
		1 => {
			for row in 0..rows {
				for (col, underlier_col) in underlier_data.iter().enumerate() {
					let underlier_col = underlier_col.as_ref().ok_or(Error::MissingData)?;
					let v = (underlier_col[row / 8] & (1 << (row % 8))) >> (row % 8);

					let bit = row * cols + col;
					trace_matrix[bit / 8] |= v << (bit % 8);
				}
			}
		}
		_ => unimplemented!(),
	}
	Ok(Rc::new(trace_matrix))
}

const KECCAKF_CODE: &[u8] = include_bytes!("../zerocheck_code/zerocheck_keccakf_n4.bin");
const U32ADD_CODE: &[u8] = include_bytes!("../zerocheck_code/zerocheck_u32add_n4.bin");

#[instrument(skip_all)]
fn get_compiled_code(params: &ZerocheckRoundParameters) -> Result<Vec<u8>, Error> {
	match (params.degree, params.cols) {
		(2, 5) => Ok(U32ADD_CODE.to_vec()),
		(2, 117) => Ok(KECCAKF_CODE.to_vec()),
		_ => Err(Error::UnavailableCompiledCode),
	}
}
