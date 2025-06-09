// Copyright 2024-2025 Irreducible Inc.

use std::{iter, marker::PhantomData, sync::Arc};

use binius_compute::{
	ComputeLayer, ComputeLayerExecutor, ComputeMemory, FSlice, SizedSlice, SubfieldSlice,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_utils::bail;

use super::error::Error;
use crate::{
	compute_math::tensor_prod_eq_ind,
	polynomial::{Error as PolynomialError, MultivariatePoly},
	tensor_algebra::TensorAlgebra,
};

/// Information about the row-batching coefficients.
#[derive(Debug)]
pub struct RowBatchCoeffs<F> {
	coeffs: Vec<F>,
}

impl<F: Field> RowBatchCoeffs<F> {
	pub fn new(coeffs: Vec<F>) -> Self {
		Self { coeffs }
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

	pub fn multilinear_extension<'a, 'alloc, Hal: ComputeLayer<F>>(
		&self,
		hal: &'a Hal,
		exec: &mut Hal::Exec<'a>,
		dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
		host_alloc: &'a HostBumpAllocator<'a, F>,
		tower_level: usize,
	) -> Result<FSlice<'a, F, Hal>, Error> {
		let evals = tensor_prod_eq_ind(
			0,
			&[self.mixing_coeff],
			&self.z_vals,
			exec,
			hal,
			dev_alloc,
			host_alloc,
		)?;

		let subfield_vector = SubfieldSlice::new(evals, tower_level);

		let extension_degree = <F as ExtensionField<FSub>>::DEGREE;

		let mut row_batching_query_expansion = dev_alloc.alloc(extension_degree)?;

		hal.copy_h2d(
			&self.row_batch_coeffs.coeffs()[0..extension_degree],
			&mut row_batching_query_expansion,
		)?;

		let mut mle = dev_alloc.alloc(evals.len())?;

		exec.fold_right(
			subfield_vector,
			Hal::DevMem::as_const(&row_batching_query_expansion),
			&mut mle,
		)?;

		Ok(Hal::DevMem::to_const(mle))
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
				expected: self.n_vars(),
				actual: query.len(),
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
	use binius_fast_compute::{layer::FastCpuLayer, memory::PackedMemorySliceMut};
	use binius_field::{BinaryField8b, BinaryField128b, tower::CanonicalTowerFamily};
	use binius_math::MultilinearQuery;
	use bytemuck::zeroed_vec;
	use iter::repeat_with;
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::compute_math::eq_ind_partial_eval;

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

		let row_batch_challenges = repeat_with(|| <F as Field>::random(&mut rng))
			.take(kappa)
			.collect::<Vec<_>>();

		let row_batch_coeffs = Arc::new(RowBatchCoeffs::new(
			MultilinearQuery::<F, _>::expand(&row_batch_challenges).into_expansion(),
		));

		let eval_point = repeat_with(|| <F as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let mixing_coeff = <F as Field>::random(&mut rng);

		let hal = FastCpuLayer::<CanonicalTowerFamily, BinaryField128b>::default();

		let mut dev_mem = zeroed_vec(1 << 10);
		let mut host_mem = zeroed_vec(1 << 10);

		let dev_mem = PackedMemorySliceMut::new_slice(&mut dev_mem);

		let host_alloc = HostBumpAllocator::new(&mut host_mem);
		let dev_alloc = BumpAllocator::<_, _>::new(dev_mem);

		let rs_eq = RingSwitchEqInd::<FS, _>::new(z_vals, row_batch_coeffs, mixing_coeff).unwrap();

		let val1 = rs_eq.evaluate(&eval_point).unwrap();

		hal.execute(|exec| {
			let mle = rs_eq
				.multilinear_extension(
					&hal,
					exec,
					&dev_alloc,
					&host_alloc,
					BinaryField8b::TOWER_LEVEL,
				)
				.unwrap();

			let mle = SubfieldSlice::new(mle, BinaryField128b::TOWER_LEVEL);

			let query =
				eq_ind_partial_eval(&hal, exec, &eval_point, &dev_alloc, &host_alloc).unwrap();

			let val2 = exec.inner_product(mle, query).unwrap();

			assert_eq!(val1, val2);
			Ok(vec![])
		})
		.unwrap();
	}
}
