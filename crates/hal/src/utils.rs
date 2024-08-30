// Copyright 2024 Ulvetanna Inc.

use crate::Error;
use binius_field::{packed::set_packed_slice, Field, PackedField};
use binius_math::tensor_prod_eq_ind;
use bytemuck::zeroed_vec;

/// A simple interface to compute the tensor product of a full query.
pub(crate) fn tensor_product<P: PackedField>(query: &[P::Scalar]) -> Result<Vec<P>, Error> {
	let n = query.len();
	let len = ((1 << n) / P::WIDTH).max(1);
	let mut buffer = zeroed_vec::<P>(len);
	set_packed_slice(&mut buffer, 0, P::Scalar::ONE);
	tensor_prod_eq_ind(0, &mut buffer[..], query).map_err(Error::MathError)?;
	Ok(buffer)
}
