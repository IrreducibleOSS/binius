// Copyright 2024-2025 Irreducible Inc.

use std::{iter, marker::PhantomData, sync::Arc};

use binius_compute::{
	ComputeLayer, ComputeLayerExecutor, ComputeMemory, SizedSlice, SubfieldSlice,
	alloc::ComputeAllocator, cpu::CpuMemory,
};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_utils::bail;

use super::error::Error;
use crate::{
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

pub struct RingSwitchEqIndPrecompute<'a, F: Field, Mem: ComputeMemory<F>> {
	evals: Mem::FSliceMut<'a>,
	row_batching_query_expansion: Mem::FSlice<'a>,
	mle: Mem::FSliceMut<'a>,
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

	pub fn precompute_values<'a, Hal: ComputeLayer<F>, HostAllocatorType, DeviceAllocatorType>(
		z_vals: Arc<[F]>,
		row_batch_coeffs: Arc<RowBatchCoeffs<F>>,
		mixing_coeff: F,
		kappa: usize,
		hal: &Hal,
		dev_alloc: &'a DeviceAllocatorType,
		host_alloc: &HostAllocatorType,
	) -> Result<RingSwitchEqIndPrecompute<'a, F, Hal::DevMem>, Error>
	where
		HostAllocatorType: ComputeAllocator<F, CpuMemory>,
		DeviceAllocatorType: ComputeAllocator<F, Hal::DevMem>,
	{
		let extension_degree = 1 << (kappa);

		let mut row_batching_query_expansion = dev_alloc.alloc(extension_degree)?;

		hal.copy_h2d(
			&row_batch_coeffs.coeffs()[0..extension_degree],
			&mut row_batching_query_expansion,
		)?;

		let row_batching_query_expansion = Hal::DevMem::to_const(row_batching_query_expansion);

		let n_vars = z_vals.len();
		let mut evals = dev_alloc.alloc(1 << n_vars)?;

		{
			let host_val = host_alloc.alloc(1)?;
			host_val[0] = mixing_coeff;
			let mut dev_val = Hal::DevMem::slice_power_of_two_mut(&mut evals, 1);
			hal.copy_h2d(host_val, &mut dev_val)?;
		}

		let mle = dev_alloc.alloc(evals.len())?;

		Ok(RingSwitchEqIndPrecompute {
			evals,
			row_batching_query_expansion,
			mle,
		})
	}

	pub fn multilinear_extension<
		'a,
		Mem: ComputeMemory<F>,
		Exec: ComputeLayerExecutor<F, DevMem = Mem>,
	>(
		&self,
		precompute: RingSwitchEqIndPrecompute<'a, F, Mem>,
		exec: &mut Exec,
		tower_level: usize,
	) -> Result<Mem::FSlice<'a>, Error> {
		let RingSwitchEqIndPrecompute {
			mut evals,
			row_batching_query_expansion,
			mut mle,
		} = precompute;

		exec.tensor_expand(0, &self.z_vals, &mut evals)?;

		let subfield_vector = SubfieldSlice::new(Mem::as_const(&evals), tower_level);

		exec.fold_right(subfield_vector, row_batching_query_expansion, &mut mle)?;

		Ok(Mem::to_const(mle))
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
	use binius_compute::{ComputeData, ComputeHolder, cpu::layer::CpuLayerHolder};
	use binius_field::{BinaryField8b, BinaryField128b};
	use binius_math::{MultilinearQuery, eq_ind_partial_eval};
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

		let mut compute_holder = CpuLayerHolder::new(1 << 10, 1 << 10);

		let compute_data = compute_holder.to_data();

		let ComputeData {
			hal,
			dev_alloc,
			host_alloc,
			..
		} = compute_data;

		let precompute = RingSwitchEqInd::<FS, _>::precompute_values(
			z_vals.clone(),
			row_batch_coeffs.clone(),
			mixing_coeff,
			kappa,
			hal,
			&dev_alloc,
			&host_alloc,
		)
		.unwrap();

		let rs_eq = RingSwitchEqInd::<FS, _>::new(z_vals, row_batch_coeffs, mixing_coeff).unwrap();

		let val1 = rs_eq.evaluate(&eval_point).unwrap();

		hal.execute(|exec| {
			let mle = rs_eq
				.multilinear_extension(precompute, exec, BinaryField8b::TOWER_LEVEL)
				.unwrap();

			let mle = SubfieldSlice::new(mle, BinaryField128b::TOWER_LEVEL);

			let query = eq_ind_partial_eval(&eval_point);

			let val2 = exec.inner_product(mle, &query).unwrap();

			assert_eq!(val1, val2);
			Ok(vec![])
		})
		.unwrap();
	}
}
