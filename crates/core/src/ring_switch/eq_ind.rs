// Copyright 2024 Irreducible, Inc

use super::error::Error;
use crate::{
	polynomial::{Error as PolynomialError, MultivariatePoly},
	tensor_algebra::TensorAlgebra,
};
use binius_field::{
	util::inner_product_unchecked, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::MultilinearExtension;
use binius_utils::bail;
use rayon::prelude::*;
use std::{iter, marker::PhantomData, sync::Arc};

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

	pub fn multilinear_extension<P, Backend>(
		&self,
		backend: &Backend,
	) -> Result<MultilinearExtension<P, Backend::Vec<P>>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
		Backend: ComputationBackend,
	{
		// TODO: Deduplicate the computation of this expansion operation across ring switch EQ INDs
		// sharing `z_vals`. It's not dire, because each element is multiplied by a mixing
		// coefficient, so it at must doubles the total number of multiplications.
		let mut evals = backend.tensor_product_full_query(self.z_vals.as_ref())?;
		P::unpack_scalars_mut(&mut evals)
			.par_iter_mut()
			.for_each(|val| {
				let vert = *val * self.mixing_coeff;
				*val = inner_product_unchecked(
					self.row_batch_coeffs.iter().copied(),
					ExtensionField::<FSub>::iter_bases(&vert),
				);
			});
		Ok(MultilinearExtension::from_values_generic(evals)?)
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
	use super::*;
	use binius_field::{BinaryField128b, BinaryField8b};
	use binius_hal::make_portable_backend;
	use binius_math::MultilinearQuery;
	use iter::repeat_with;
	use rand::{prelude::StdRng, SeedableRng};

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

		let backend = make_portable_backend();

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();
		let eval_query = MultilinearQuery::<F>::expand(&eval_point);

		let mixing_coeff = <F as Field>::random(&mut rng);

		let rs_eq =
			RingSwitchEqInd::<FS, _>::new(z_vals.clone(), row_batch_coeffs, mixing_coeff).unwrap();
		let mle = rs_eq.multilinear_extension::<F, _>(&backend).unwrap();

		let val1 = rs_eq.evaluate(&eval_point).unwrap();
		let val2 = mle.evaluate(&eval_query).unwrap();
		assert_eq!(val1, val2);
	}
}
