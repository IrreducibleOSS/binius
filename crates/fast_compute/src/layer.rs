// Copyright 2025 Irreducible Inc.

use std::{cell::RefCell, marker::PhantomData};

use binius_compute::{
	each_tower_subfield,
	layer::{ComputeLayer, Error, FSlice, FSliceMut, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SubfieldSlice},
};
use binius_field::{
	ExtensionField, Field, PackedExtension, PackedField,
	tower::{PackedTop, TowerFamily},
	unpack_if_possible, unpack_if_possible_mut,
	util::inner_product_par,
};
use binius_math::{ArithExpr, tensor_prod_eq_ind};
use binius_maybe_rayon::{
	iter::{
		IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
		ParallelIterator,
	},
	slice::{ParallelSlice, ParallelSliceMut},
};
use binius_ntt::AdditiveNTT;
use bytemuck::zeroed_vec;
use itertools::izip;

use crate::{arith_circuit::ArithCircuitPoly, memory::PackedMemory};

#[derive(Debug)]
pub struct FastCpuExecutor;

/// Optimized CPU implementation of the compute layer.
#[derive(Debug, Default)]
pub struct FastCpuLayer<T: TowerFamily, P: PackedTop<T>>(PhantomData<(T, P)>);

impl<T: TowerFamily, P: PackedTop<T>> ComputeLayer<T::B128> for FastCpuLayer<T, P> {
	type Exec = FastCpuExecutor;
	type KernelExec = FastCpuExecutor;
	type DevMem = PackedMemory<P>;
	type OpValue = T::B128;
	type KernelValue = T::B128;
	type ExprEval = ArithCircuitPoly<T::B128>;

	fn host_alloc(&self, n: usize) -> impl AsMut<[T::B128]> + '_ {
		zeroed_vec(n)
	}

	fn copy_h2d(
		&self,
		src: &[T::B128],
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		if src.len() != dst.len() {
			return Err(Error::InputValidation(
				"precondition: src and dst buffers must have the same length".to_string(),
			));
		}

		unpack_if_possible_mut(
			dst.data,
			|scalars| {
				scalars.copy_from_slice(src);
				Ok(())
			},
			|packed| {
				src.par_chunks_exact(P::WIDTH)
					.zip(packed.par_iter_mut())
					.for_each(|(input, output)| {
						*output = PackedField::from_scalars(input.iter().copied());
					});

				Ok(())
			},
		)
	}

	fn copy_d2h(&self, src: FSlice<'_, T::B128, Self>, dst: &mut [T::B128]) -> Result<(), Error> {
		if src.len() != dst.len() {
			return Err(Error::InputValidation(
				"precondition: src and dst buffers must have the same length".to_string(),
			));
		}

		let dst = RefCell::new(dst);
		unpack_if_possible(
			src.data,
			|scalars| {
				dst.borrow_mut().copy_from_slice(scalars);
				Ok(())
			},
			|packed: &[P]| {
				(*dst.borrow_mut())
					.par_chunks_exact_mut(P::WIDTH)
					.zip(packed.par_iter())
					.for_each(|(output, input)| {
						for (input, output) in input.iter().zip(output.iter_mut()) {
							*output = input;
						}
					});

				for (input, output) in
					PackedField::iter_slice(packed).zip(dst.borrow_mut().iter_mut())
				{
					*output = input;
				}
				Ok(())
			},
		)
	}

	fn copy_d2d(
		&self,
		src: FSlice<'_, T::B128, Self>,
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		if src.len() != dst.len() {
			return Err(Error::InputValidation(
				"precondition: src and dst buffers must have the same length".to_string(),
			));
		}

		dst.data.copy_from_slice(src.data);

		Ok(())
	}

	fn execute(
		&self,
		f: impl FnOnce(&mut Self::Exec) -> Result<Vec<Self::OpValue>, Error>,
	) -> Result<Vec<T::B128>, Error> {
		f(&mut FastCpuExecutor)
	}

	fn inner_product(
		&self,
		_exec: &mut Self::Exec,
		a_in: SubfieldSlice<'_, T::B128, Self::DevMem>,
		b_in: FSlice<'_, T::B128, Self>,
	) -> Result<Self::OpValue, Error> {
		if a_in.slice.len() << (<T::B128 as ExtensionField<T::B1>>::LOG_DEGREE - a_in.tower_level)
			!= b_in.len()
		{
			return Err(Error::InputValidation(
				"precondition: a_in and b_in must have the same length".to_string(),
			));
		}

		fn inner_product_par_impl<FSub: Field, P: PackedExtension<FSub>>(
			a_in: &[P],
			b_in: &[P],
		) -> P::Scalar {
			inner_product_par(b_in, PackedExtension::cast_bases(a_in))
		}

		let result = each_tower_subfield!(
			a_in.tower_level,
			T,
			inner_product_par_impl::<_, P>(a_in.slice.data, b_in.data)
		);

		Ok(result)
	}

	fn tensor_expand(
		&self,
		_exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		tensor_prod_eq_ind(log_n, data.data, coordinates)
			.map_err(|_| Error::InputValidation("tensor dimensions are invalid".to_string()))
	}

	#[inline(always)]
	fn kernel_decl_value(
		&self,
		_exec: &mut Self::KernelExec,
		init: T::B128,
	) -> Result<Self::KernelValue, Error> {
		Ok(init)
	}

	fn compile_expr(&self, expr: &ArithExpr<T::B128>) -> Result<Self::ExprEval, Error> {
		let expr = ArithCircuitPoly::new(expr.into());
		Ok(expr)
	}

	fn accumulate_kernels(
		&self,
		_exec: &mut Self::Exec,
		_map: impl for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, T::B128, Self::DevMem>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		_mem_maps: Vec<KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		unimplemented!()
	}

	fn fold_left<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		_mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		_vec: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
		_out: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		unimplemented!()
	}

	fn fold_right<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		_mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		_vec: <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSlice<'_>,
		_out: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		unimplemented!()
	}

	fn sum_composition_evals(
		&self,
		_exec: &mut Self::KernelExec,
		_log_len: usize,
		_inputs: &[FSlice<'_, T::B128, Self>],
		_composition: &Self::ExprEval,
		_batch_coeff: T::B128,
		_accumulator: &mut T::B128,
	) -> Result<(), Error> {
		unimplemented!()
	}

	fn kernel_add(
		&self,
		_exec: &mut Self::KernelExec,
		log_len: usize,
		src1: FSlice<'_, T::B128, Self>,
		src2: FSlice<'_, T::B128, Self>,
		dst: &mut FSliceMut<'_, T::B128, Self>,
	) -> Result<(), Error> {
		if src1.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"src1 length must be equal to 2^log_len".to_string(),
			));
		}
		if src2.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"src2 length must be equal to 2^log_len".to_string(),
			));
		}
		if dst.len() != 1 << log_len {
			return Err(Error::InputValidation(
				"dst length must be equal to 2^log_len".to_string(),
			));
		}

		for (dst_i, &src1_i, &src2_i) in izip!(dst.data.iter_mut(), src1.data, src2.data) {
			*dst_i = src1_i + src2_i;
		}

		Ok(())
	}

	fn fri_fold<FSub>(
		&self,
		_exec: &mut Self::Exec,
		_ntt: &impl AdditiveNTT<FSub>,
		_log_len: usize,
		_log_batch_size: usize,
		_challenges: &[T::B128],
		_data_in: FSlice<T::B128, Self>,
		_data_out: &mut FSliceMut<T::B128, Self>,
	) -> Result<(), Error>
	where
		FSub: binius_field::BinaryField,
		T::B128: binius_field::ExtensionField<FSub>,
	{
		unimplemented!()
	}

	fn extrapolate_line(
		&self,
		_exec: &mut Self::Exec,
		evals_0: &mut FSliceMut<T::B128, Self>,
		evals_1: FSlice<T::B128, Self>,
		z: T::B128,
	) -> Result<(), Error> {
		if evals_0.len() != evals_1.len() {
			return Err(Error::InputValidation(
				"precondition: evals_0 and evals_1 must have the same length".to_string(),
			));
		}

		evals_0
			.data
			.par_iter_mut()
			.zip(evals_1.data.par_iter())
			.for_each(|(x0, x1)| *x0 += (*x1 - *x0) * z);
		Ok(())
	}
}
