// Copyright 2024 Ulvetanna Inc.

use crate::{immutable_slice::VecOrImmutableSlice, utils::tensor_product, Error};
use binius_field::PackedField;
use linerate_binius_tensor_product::TensorProductFpgaAccelerator;
use std::{
	slice::from_raw_parts,
	sync::{Arc, Mutex},
};
use tracing::trace;

const MIN_BENEFICIAL_N: usize = 14;

pub(crate) fn run_tensor_product<P: PackedField>(
	accelerator: Option<Arc<Mutex<TensorProductFpgaAccelerator>>>,
	query: &[P::Scalar],
) -> Result<Option<VecOrImmutableSlice<P>>, Error> {
	let mut result = None;
	if let Some(accelerator) = accelerator {
		if TensorProductFpgaAccelerator::n_supported(query.len()) && query.len() >= MIN_BENEFICIAL_N
		{
			let intermediate: Vec<P> =
				tensor_product(&query[..TensorProductFpgaAccelerator::INTERMEDIATE_ROUND])?;
			// TODO: Avoid unsafe, use trait bounds.
			let intermediate_f = unsafe {
				from_raw_parts(
					intermediate.as_ptr() as *const P::Scalar,
					intermediate.len() * P::WIDTH,
				)
			};

			let mut accelerator_lock = accelerator.lock().unwrap();
			let expansion = accelerator_lock
				.compute::<P::Scalar, P>(query, intermediate_f)
				.map_err(Error::LinerateTensorProductError)?;
			trace!(?query, ?intermediate_f, expansion = ?expansion[..32].to_vec());
			result = Some(VecOrImmutableSlice::IS(expansion));
		}
	}
	Ok(result)
}
