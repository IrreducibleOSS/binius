// Copyright 2024 Ulvetanna Inc.

use crate::{
	immutable_slice::VecOrImmutableSlice,
	tensor_product::run_tensor_product,
	utils::tensor_product,
	zerocheck::{run_zerocheck, ZerocheckCpuBackendHelper, ZerocheckRoundParameters, ZerocheckRoundInput},
	ComputationBackend, CpuBackend, Error,
};
use binius_field::{packed::iter_packed_slice, AESTowerField128b, BinaryField128b, ExtensionField, Field, PackedExtension, PackedField};
use itertools::Itertools;
use linerate_binius_sumcheck::SumcheckFpgaAccelerator;
use linerate_binius_tensor_product::TensorProductFpgaAccelerator;
use std::{
	collections::HashMap,
	sync::{Arc, Mutex},
};
use tracing::{debug, instrument};

#[derive(Clone, Debug)]
pub struct LinerateBackend {
	cpu: CpuBackend,

	tensor_product: HashMap<String, Arc<Mutex<TensorProductFpgaAccelerator>>>,
	sumcheck: HashMap<(String, usize, usize), Arc<Mutex<SumcheckFpgaAccelerator>>>,
}

pub fn make_linerate_backend() -> LinerateBackend {
	LinerateBackend::new(CpuBackend)
}

impl LinerateBackend {
	#[allow(dead_code)]
	fn new(cpu_backend: CpuBackend) -> Self {
		debug!("LinerateBackend::new");
		let mut tensor_product = HashMap::new();
		if let Some(accelerator) =
			new_tensor_product_accelerator_from_env("TENSOR_PRODUCT_FPT").unwrap()
		{
			tensor_product.insert(format!("{:?}", BinaryField128b::ONE), accelerator);
		}
		if let Some(accelerator) =
			new_tensor_product_accelerator_from_env("TENSOR_PRODUCT_AES").unwrap()
		{
			tensor_product.insert(format!("{:?}", AESTowerField128b::ONE), accelerator);
		}
		let mut sumcheck = HashMap::new();
		if let Some(accelerator) = new_sumcheck_accelerator_from_env("SUMCHECK_FPT_2_1").unwrap() {
			sumcheck.insert((format!("{:?}", BinaryField128b::ONE), 2, 1), accelerator);
		}
		debug!(?tensor_product, ?sumcheck, "LinerateBackend::new()");
		Self {
			cpu: cpu_backend,
			tensor_product,
			sumcheck,
		}
	}
}

fn new_tensor_product_accelerator_from_env(
	var: &str,
) -> Result<Option<Arc<Mutex<TensorProductFpgaAccelerator>>>, Error> {
	Ok(match std::env::var(var) {
		Ok(card) => Some(Arc::new(Mutex::new(
			TensorProductFpgaAccelerator::new(&card).map_err(Error::LinerateTensorProductError)?,
		))),
		Err(_) => None,
	})
}

fn new_sumcheck_accelerator_from_env(
	var: &str,
) -> Result<Option<Arc<Mutex<SumcheckFpgaAccelerator>>>, Error> {
	Ok(match std::env::var(var) {
		Ok(card) => Some(Arc::new(Mutex::new(
			SumcheckFpgaAccelerator::new(&card).map_err(Error::LinerateSumcheckError)?,
		))),
		Err(_) => None,
	})
}

impl ComputationBackend for LinerateBackend {
	#[instrument(skip_all)]
	fn tensor_product_full_query<P: PackedField>(
		&self,
		query: &[P::Scalar],
	) -> Result<VecOrImmutableSlice<P>, Error> {
		let result =
			run_tensor_product::<P>(self.get_tensor_product_accelerator::<P::Scalar>(), query)?;
		match result {
			Some(result) => {
				if cfg!(feature = "debug_validation") {
					let expected: Vec<P> = tensor_product(query).unwrap();
					if let Some(pos) = iter_packed_slice(&result[..])
						.zip(iter_packed_slice(&expected[..]))
						.find_position(|(got, expected)| got != expected)
					{
						panic!("pos={}, got={:?}, expected={:?}", pos.0, pos.1 .0, pos.1 .1)
					};
				}
				Ok(result)
			}
			None => Ok(self.cpu.tensor_product_full_query::<P>(query)?),
		}
	}

	#[instrument(skip_all)]
	fn zerocheck_compute_round_coeffs<F, PW, FDomain>(
		&self,
		params: &ZerocheckRoundParameters,
		input: &ZerocheckRoundInput<F, PW, FDomain>,
		cpu_handler: &mut dyn ZerocheckCpuBackendHelper<F, PW, FDomain>,
	) -> Result<Vec<PW::Scalar>, Error>
	where
		F: Field,
		PW: PackedField + PackedExtension<FDomain>,
		PW::Scalar: From<F> + Into<F> + ExtensionField<FDomain>,
		FDomain: Field,
	{
		if params.round < SumcheckFpgaAccelerator::NUM_ROUNDS {
			if let Some(small_field_width) = params.small_field_width {
				let maybe_accelerator = self
					.get_sumcheck_accelerator::<PW::Scalar>(params.degree, small_field_width);
				if let Some(coeffs) = run_zerocheck::<F, PW, FDomain>(maybe_accelerator, params, input)? {
					cpu_handler.remove_smaller_domain_optimization();
					return Ok(coeffs);
				}
			}
		}
		// No computation was attempted by LinerateBackend, therefore fall back to CpuBackend.
		self.cpu
			.zerocheck_compute_round_coeffs::<F, PW, FDomain>(params, input, cpu_handler)
	}
}

impl LinerateBackend {
	fn get_tensor_product_accelerator<F: Field>(
		&self,
	) -> Option<Arc<Mutex<TensorProductFpgaAccelerator>>> {
		let one = format!("{:?}", F::ONE);
		debug!(one, "get_tensor_product_accelerator");
		self.tensor_product.get(&one).cloned()
	}

	fn get_sumcheck_accelerator<F: Field>(
		&self,
		degree: usize,
		small_field_width: usize,
	) -> Option<Arc<Mutex<SumcheckFpgaAccelerator>>> {
		let one = format!("{:?}", F::ONE);
		debug!(one, degree, small_field_width, "get_sumcheck_accelerator");
		self.sumcheck
			.get(&(one, degree, small_field_width))
			.cloned()
	}
}
