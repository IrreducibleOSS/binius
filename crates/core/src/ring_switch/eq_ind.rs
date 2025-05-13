// Copyright 2024-2025 Irreducible Inc.

use std::{any::TypeId, iter, marker::PhantomData, sync::Arc};

use binius_field::{
	BinaryField1b, ExtensionField, Field, PackedExtension, PackedField, TowerField,
	byte_iteration::{
		ByteIteratorCallback, can_iterate_bytes, create_partial_sums_lookup_tables, iterate_bytes,
	},
	util::inner_product_unchecked,
};
use binius_math::{MultilinearExtension, tensor_prod_eq_ind};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;

use super::error::Error;
use crate::{
	polynomial::{Error as PolynomialError, MultivariatePoly},
	tensor_algebra::TensorAlgebra,
};

/// Information about the row-batching coefficients.
#[derive(Debug)]
pub struct RowBatchCoeffs<F> {
	coeffs: Vec<F>,
	/// This is a lookup table for the partial sums of the coefficients
	/// that is used to efficiently fold with 1-bit coefficients.
	partial_sums_lookup_table: Vec<F>,
	query: Vec<F>,
}

impl<F: Field> RowBatchCoeffs<F> {
	pub fn new(coeffs: Vec<F>, query: Vec<F>) -> Self {
		let partial_sums_lookup_table = if coeffs.len() >= 8 {
			create_partial_sums_lookup_tables(coeffs.as_slice())
		} else {
			Vec::new()
		};

		Self {
			coeffs,
			partial_sums_lookup_table,
			query,
		}
	}

	pub fn coeffs(&self) -> &[F] {
		&self.coeffs
	}
}

/// The multilinear function $A$ from [DP24] Section 5.
///
/// The function $A$ is $\ell':= \ell - \kappa$-variate and depends on the last $\ell'$ coordinates
/// of the evaluation point as well as the $\kappa$ mixing challenges.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug, Clone)]
pub struct RingSwitchEqInd<FSub, F> {
	/// $z_{\kappa}, \ldots, z_{\ell-1}$
	z_vals: Arc<[F]>,
	row_batch_coeffs: Arc<RowBatchCoeffs<F>>,
	mixing_coeff: F,
	_marker: PhantomData<FSub>,
}

impl<FSub, F> RingSwitchEqInd<FSub, F>
where
	FSub: Field,
	F: ExtensionField<FSub>,
{
	pub fn new(
		z_vals: Arc<[F]>,
		row_batch_coeffs: Arc<RowBatchCoeffs<F>>,
		mixing_coeff: F,
	) -> Result<Self, Error> {
		if row_batch_coeffs.coeffs.len() < F::DEGREE {
			bail!(Error::InvalidArgs(
				"RingSwitchEqInd::new expects row_batch_coeffs length greater than or equal to \
				the extension degree"
					.into()
			));
		}

		Ok(Self {
			z_vals,
			row_batch_coeffs,
			mixing_coeff,
			_marker: PhantomData,
		})
	}

	pub fn multilinear_extension<P: PackedField<Scalar = F>>(
		&self,
	) -> Result<MultilinearExtension<P>, Error> {
		let mut evals = zeroed_vec::<P>(1 << self.z_vals.len().saturating_sub(P::LOG_WIDTH));
		evals[0].set(0, self.mixing_coeff);
		tensor_prod_eq_ind(0, &mut evals, &self.z_vals)?;

		evals.par_iter_mut().for_each(|val| {
			*val = P::from_scalars(
				val.iter()
					.map(|v| inner_product_subfield(v, &self.row_batch_coeffs)),
			);
		});
		Ok(MultilinearExtension::new(self.z_vals.len(), evals)?)
	}
}

#[inline(always)]
fn inner_product_subfield<FSub, F>(value: F, row_batch_coeffs: &RowBatchCoeffs<F>) -> F
where
	FSub: Field,
	F: ExtensionField<FSub>,
{
	if TypeId::of::<FSub>() == TypeId::of::<BinaryField1b>() && can_iterate_bytes::<F>() {
		// Special case when we are folding with 1-bit coefficients.
		// Use partial sums lookup table to speed up the computation.

		struct Callback<'a, F> {
			partial_sums_lookup: &'a [F],
			result: F,
		}

		impl<F: Field> ByteIteratorCallback for Callback<'_, F> {
			#[inline(always)]
			fn call(&mut self, iter: impl Iterator<Item = u8>) {
				for (byte_index, byte) in iter.enumerate() {
					self.result += self.partial_sums_lookup[(byte_index << 8) + byte as usize];
				}
			}
		}

		let mut callback = Callback {
			partial_sums_lookup: &row_batch_coeffs.partial_sums_lookup_table,
			result: F::ZERO,
		};
		iterate_bytes(std::slice::from_ref(&value), &mut callback);

		callback.result
	} else {
		// fall back to the general case
		inner_product_unchecked(row_batch_coeffs.coeffs.iter().copied(), F::iter_bases(&value))
	}
}

impl<FSub, F> MultivariatePoly<F> for RingSwitchEqInd<FSub, F>
where
	FSub: TowerField,
	F: TowerField + PackedField<Scalar = F> + PackedExtension<FSub>,
{
	fn n_vars(&self) -> usize {
		self.z_vals.len()
	}

	fn degree(&self) -> usize {
		self.n_vars()
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		if query.len() != self.n_vars() {
			bail!(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars()
			});
		};

		let tensor_eval = iter::zip(&*self.z_vals, query).fold(
			<TensorAlgebra<FSub, F>>::from_vertical(self.mixing_coeff),
			|eval, (&vert_i, &hztl_i)| {
				// This formula is specific to characteristic 2 fields
				// Here we know that $h v + (1 - h) (1 - v) = 1 + h + v$.
				let vert_scaled = eval.clone().scale_vertical(vert_i);
				let hztl_scaled = eval.clone().scale_horizontal(hztl_i);
				eval + &vert_scaled + &hztl_scaled
			},
		);

		let folded_eval = tensor_eval.fold_vertical(&self.row_batch_coeffs.coeffs);
		Ok(folded_eval)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField8b, BinaryField128b};
	use binius_math::MultilinearQuery;
	use iter::repeat_with;
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;

	#[test]
	fn test_evaluation_consistency() {
		type FS = BinaryField8b;
		type F = BinaryField128b;
		let kappa = <TensorAlgebra<FS, F>>::kappa();
		let ell = 10;
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = ell - kappa;
		let z_vals = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Arc<[_]>>();

		// Changed to a query being expanded

		let row_batch_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(kappa)
			.collect::<Vec<_>>();

		let row_batch_coeffs = Arc::new(RowBatchCoeffs::new(
			MultilinearQuery::<F, _>::expand(&row_batch_challenges).into_expansion(),
			row_batch_challenges,
		));

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();
		let eval_query = MultilinearQuery::<F>::expand(&eval_point);

		let mixing_coeff = <F as Field>::random(&mut rng);

		let rs_eq = RingSwitchEqInd::<FS, _>::new(z_vals, row_batch_coeffs, mixing_coeff).unwrap();
		let mle = rs_eq.multilinear_extension::<F>().unwrap();

		let val1 = rs_eq.evaluate(&eval_point).unwrap();
		let val2 = mle.evaluate(&eval_query).unwrap();
		assert_eq!(val1, val2);
	}
}
