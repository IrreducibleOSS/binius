// Copyright 2024 Irreducible Inc.

use crate::{
	polynomial::{Error, MultivariatePoly},
	tensor_algebra::TensorAlgebra,
};
use binius_field::{
	util::inner_product_unchecked, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend, ComputationBackendExt};
use binius_math::MultilinearExtension;
use binius_utils::bail;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::{iter, marker::PhantomData};
use tracing::instrument;

/// This struct provides functionality for the multilinear function $A$ from [DP24] Section 5.
///
/// The function $A$ is $\ell':= \ell - \kappa$-variate and depends on the last $\ell'$ coordinates
/// of the evaluation point as well as the $\kappa$ mixing challenges.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub struct RingSwitchEqInd<FS, F>
where
	FS: Field,
	F: ExtensionField<FS>,
{
	n_vars: usize,
	r_evals: Vec<F>, // r_{\kappa}, \ldots, r_{\ell-1} (in particular, we forget about r_0, \ldots, r_{\kappa-1})
	r_mixing_challenges: Vec<F>, // r''_0, \ldots, r''_{\kappa-1}
	_phantom: PhantomData<FS>,
}

impl<FS, F> RingSwitchEqInd<FS, F>
where
	FS: Field,
	F: ExtensionField<FS> + PackedField<Scalar = F> + PackedExtension<FS>,
{
	pub fn new(r_evals: Vec<F>, r_mixing_challenges: Vec<F>) -> Result<Self, Error> {
		let n_vars = r_evals.len();

		if r_mixing_challenges.len() != <TensorAlgebra<FS, F>>::kappa() {
			bail!(Error::RingSwitchWrongLength {
				expected: <TensorAlgebra<FS, F>>::kappa(),
				actual: r_mixing_challenges.len()
			});
		}
		Ok(Self {
			n_vars,
			r_evals,
			r_mixing_challenges,
			_phantom: PhantomData,
		})
	}

	#[instrument("RingSwitchEqInd::multilinear_extension", skip_all, level = "trace")]
	pub fn multilinear_extension<P, Backend>(
		&self,
		backend: &Backend,
	) -> Result<MultilinearExtension<P, Backend::Vec<P>>, Error>
	where
		P: PackedFieldIndexable<Scalar = F>,
		Backend: ComputationBackend,
	{
		let r_evals = &self.r_evals;
		let r_mixing_challenges = &self.r_mixing_challenges;
		let expanded_mixing_coeffs = backend.multilinear_query(r_mixing_challenges)?;
		let mut evals = backend.tensor_product_full_query(r_evals)?;
		P::unpack_scalars_mut(&mut evals)
			.par_iter_mut()
			.for_each(|val| {
				let vert = *val;
				*val = inner_product_unchecked(
					expanded_mixing_coeffs.expansion().iter().copied(),
					ExtensionField::<FS>::iter_bases(&vert),
				);
			});
		Ok(MultilinearExtension::from_values_generic(evals)?)
	}
}

impl<FS, F> MultivariatePoly<F> for RingSwitchEqInd<FS, F>
where
	FS: TowerField,
	F: ExtensionField<FS> + PackedField<Scalar = F> + PackedExtension<FS> + TowerField,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.n_vars
	}

	fn evaluate(&self, query: &[F]) -> Result<F, Error> {
		// query is typically going to be `r_sumcheck_challenges`
		let n_vars = MultivariatePoly::<F>::n_vars(self);
		if query.len() != n_vars {
			bail!(Error::IncorrectQuerySize { expected: n_vars });
		};
		let r_evals = &self.r_evals;
		let r_mixing_challenges = &self.r_mixing_challenges;

		let tensor_eval = iter::zip(r_evals.iter().copied(), query.iter().copied()).fold(
			TensorAlgebra::one(),
			|eval, (vert_i, hztl_i)| {
				// This formula is specific to characteristic 2 fields
				// Here we know that $h v + (1 - h) (1 - v) = 1 + h + v$.
				let vert_scaled = eval.clone().scale_vertical(vert_i);
				let hztl_scaled = eval.clone().scale_horizontal(hztl_i);
				eval + &vert_scaled + &hztl_scaled
			},
		);
		// Use the portable CPU backend because the size of the hypercube is small.
		let backend = make_portable_backend();
		let expanded_mixing_coeffs = &backend
			.tensor_product_full_query(r_mixing_challenges)
			.expect("F extension degree is less than 2^31");
		let folded_eval = inner_product_unchecked::<F, _>(
			tensor_eval.transpose().vertical_elems().iter().copied(),
			expanded_mixing_coeffs.iter().copied(),
		);
		Ok(folded_eval)
	}

	// realistically this will be 7.
	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use binius_field::{BinaryField128b, BinaryField8b};
	use iter::repeat_with;
	use rand::{prelude::StdRng, SeedableRng};

	#[test]
	fn test_ring_switch_eq_ind() {
		type FS = BinaryField8b;
		type F = BinaryField128b;
		let kappa = <TensorAlgebra<FS, F>>::kappa();
		let ell = 10;
		let mut rng = StdRng::seed_from_u64(0);

		let n_vars = ell - kappa;
		let r_evals = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let r_mixing_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(kappa)
			.collect::<Vec<_>>();
		let backend = make_portable_backend();

		let r_sumcheck_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let rs_eq =
			RingSwitchEqInd::<FS, _>::new(r_evals.clone(), r_mixing_challenges.clone()).unwrap();

		let val1 = rs_eq.evaluate(&r_sumcheck_challenges).unwrap();

		let partial_evals = rs_eq.multilinear_extension::<F, _>(&backend).unwrap();
		let val2 = partial_evals
			.evaluate(
				&backend
					.multilinear_query::<F>(&r_sumcheck_challenges)
					.unwrap(),
			)
			.unwrap();

		assert_eq!(val1, val2);
	}
}
