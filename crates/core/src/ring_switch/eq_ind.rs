// Copyright 2024-2025 Irreducible Inc.

use std::{iter, marker::PhantomData, sync::Arc};

use binius_field::{
	util::inner_product_unchecked, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_math::{tensor_prod_eq_ind, MultilinearExtension};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use rayon::prelude::*;

use super::error::Error;
use crate::{
	polynomial::{Error as PolynomialError, MultivariatePoly},
	tensor_algebra::TensorAlgebra,
};

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
	row_batch_coeffs: Arc<[F]>,
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
		row_batch_coeffs: Arc<[F]>,
		mixing_coeff: F,
	) -> Result<Self, Error> {
		if row_batch_coeffs.len() < F::DEGREE {
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

	pub fn multilinear_extension<P: PackedFieldIndexable<Scalar = F>>(
		&self,
	) -> Result<MultilinearExtension<P>, Error> {
		let mut evals = zeroed_vec::<P>(1 << self.z_vals.len().saturating_sub(P::LOG_WIDTH));
		evals[0].set(0, self.mixing_coeff);
		tensor_prod_eq_ind(0, &mut evals, &self.z_vals)?;
		P::unpack_scalars_mut(&mut evals)
			.par_iter_mut()
			.for_each(|val| {
				let vert = *val;
				*val = inner_product_unchecked(
					self.row_batch_coeffs.iter().copied(),
					ExtensionField::<FSub>::iter_bases(&vert),
				);
			});
		Ok(MultilinearExtension::from_values(evals)?)
	}
}

impl<FSub, F> MultivariatePoly<F> for RingSwitchEqInd<FSub, F>
where
	FSub: TowerField,
	F: TowerField + PackedField<Scalar = F> + ExtensionField<FSub> + PackedExtension<FSub>,
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

		let folded_eval = tensor_eval.fold_vertical(&self.row_batch_coeffs);
		Ok(folded_eval)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128b, BinaryField8b};
	use binius_math::MultilinearQuery;
	use iter::repeat_with;
	use rand::{prelude::StdRng, SeedableRng};

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

		let row_batch_coeffs = repeat_with(|| <F as Field>::random(&mut rng))
			.take(1 << kappa)
			.collect::<Arc<[_]>>();

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();
		let eval_query = MultilinearQuery::<F>::expand(&eval_point);

		let mixing_coeff = <F as Field>::random(&mut rng);

		let rs_eq =
			RingSwitchEqInd::<FS, _>::new(z_vals.clone(), row_batch_coeffs, mixing_coeff).unwrap();
		let mle = rs_eq.multilinear_extension::<F>().unwrap();

		let val1 = rs_eq.evaluate(&eval_point).unwrap();
		let val2 = mle.evaluate(&eval_query).unwrap();
		assert_eq!(val1, val2);
	}
}
