// Copyright 2025 Irreducible Inc.

use core::slice;
use std::{cell::RefCell, marker::PhantomData, mem::MaybeUninit};

use binius_compute::{
	layer::{ComputeLayer, Error, FSlice, FSliceMut, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SubfieldSlice},
};
use binius_field::{
	tower::{PackedTop, TowerFamily},
	unpack_if_possible, unpack_if_possible_mut,
	util::inner_product_par,
	ExtensionField, PackedExtension, PackedField,
};
use binius_math::{fold_left, fold_right, tensor_prod_eq_ind, ArithExpr};
use binius_ntt::AdditiveNTT;
use binius_utils::checked_arithmetics::strict_log_2;
use bytemuck::zeroed_vec;

use crate::{arith_circuit::ArithCircuitPoly, memory::PackedMemory};

#[derive(Debug)]
pub struct FastCpuExecutor;

#[derive(Debug, Default)]
pub struct CpuLayer<T: TowerFamily, P: PackedTop<T>>(PhantomData<(T, P)>);

impl<T: TowerFamily, P: PackedTop<T>> ComputeLayer<T::B128> for CpuLayer<T, P> {
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
		debug_assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);

		unpack_if_possible_mut(
			dst.data,
			|scalars| {
				scalars.copy_from_slice(src);
				Ok(())
			},
			|packed| {
				let mut iter = src.iter().copied();
				for p in packed {
					*p = PackedField::from_scalars(&mut iter)
				}
				Ok(())
			},
		)
	}

	fn copy_d2h(&self, src: FSlice<'_, T::B128, Self>, dst: &mut [T::B128]) -> Result<(), Error> {
		debug_assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);

		let dst = RefCell::new(dst);
		unpack_if_possible(
			src.data,
			|scalars| {
				dst.borrow_mut().copy_from_slice(scalars);
				Ok(())
			},
			|packed: &[P]| {
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
		debug_assert_eq!(
			src.len(),
			dst.len(),
			"precondition: src and dst buffers must have the same length"
		);

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
		a_edeg: usize,
		a_in: FSlice<'_, T::B128, Self>,
		b_in: FSlice<'_, T::B128, Self>,
	) -> Result<Self::OpValue, Error> {
		debug_assert_eq!(
			a_in.len() << (<T::B128 as ExtensionField<T::B1>>::LOG_DEGREE - a_edeg),
			b_in.len(),
			"precondition: a_in and b_in must have the same length"
		);

		let result = match a_edeg {
			0 => inner_product_par(b_in.data, PackedExtension::<T::B1>::cast_bases(a_in.data)),
			3 => inner_product_par(b_in.data, PackedExtension::<T::B8>::cast_bases(a_in.data)),
			4 => inner_product_par(b_in.data, PackedExtension::<T::B16>::cast_bases(a_in.data)),
			5 => inner_product_par(b_in.data, PackedExtension::<T::B32>::cast_bases(a_in.data)),
			6 => inner_product_par(b_in.data, PackedExtension::<T::B64>::cast_bases(a_in.data)),
			7 => inner_product_par(b_in.data, PackedExtension::<T::B128>::cast_bases(a_in.data)),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of a_edeg: {}",
					a_edeg
				)))
			}
		};

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
		exec: &mut Self::Exec,
		map: impl for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<KernelBuffer<'a, T::B128, Self::DevMem>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		mem_maps: Vec<KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		todo!()
	}

	fn fold_left<'a>(
		&'a self,
		_exec: &'a mut Self::Exec,
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		let log_evals_size = strict_log_2(mat.subfield_len()).ok_or_else(|| {
			Error::InputValidation("the length of `mat` must be a power of 2".to_string())
		})?;
		let log_query_size = strict_log_2(vec.len()).ok_or_else(|| {
			Error::InputValidation("the length of `vec` must be a power of 2".to_string())
		})?;

		// Safety: `P` must not implement `Drop` to safely use `MaybeUninit` for uninitialized
		// memory.
		assert!(!std::mem::needs_drop::<P>(), "`P` must not implement Drop");
		let out = unsafe {
			slice::from_raw_parts_mut(out.data.as_ptr() as *mut MaybeUninit<P>, out.data.len())
		};

		let result = match mat.tower_level {
			0 => fold_left(
				PackedExtension::<T::B1>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			3 => fold_left(
				PackedExtension::<T::B8>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			4 => fold_left(
				PackedExtension::<T::B16>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			5 => fold_left(
				PackedExtension::<T::B32>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			6 => fold_left(
				PackedExtension::<T::B64>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			7 => fold_left(
				PackedExtension::<T::B128>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out,
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of mat.tower_level: {}",
					mat.tower_level
				)))
			}
		};

		result
			.map_err(|_| Error::InputValidation("the input data dimensions are wrong".to_string()))
	}

	fn fold_right<'a>(
		&'a self,
		exec: &'a mut Self::Exec,
		mat: SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		let log_evals_size = strict_log_2(mat.subfield_len()).ok_or_else(|| {
			Error::InputValidation("the length of `mat` must be a power of 2".to_string())
		})?;
		let log_query_size = strict_log_2(vec.len()).ok_or_else(|| {
			Error::InputValidation("the length of `vec` must be a power of 2".to_string())
		})?;

		let result = match mat.tower_level {
			0 => fold_right(
				PackedExtension::<T::B1>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			3 => fold_right(
				PackedExtension::<T::B8>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			4 => fold_right(
				PackedExtension::<T::B16>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			5 => fold_right(
				PackedExtension::<T::B32>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			6 => fold_right(
				PackedExtension::<T::B64>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			7 => fold_right(
				PackedExtension::<T::B128>::cast_bases(mat.slice.data),
				log_evals_size,
				vec.data,
				log_query_size,
				out.data,
			),
			_ => {
				return Err(Error::InputValidation(format!(
					"unsupported value of mat.tower_level: {}",
					mat.tower_level
				)))
			}
		};

		result
			.map_err(|_| Error::InputValidation("the input data dimensions are wrong".to_string()))
	}

	fn sum_composition_evals(
		&self,
		exec: &mut Self::KernelExec,
		log_len: usize,
		inputs: &[FSlice<'_, T::B128, Self>],
		composition: &Self::ExprEval,
		batch_coeff: T::B128,
		accumulator: &mut Self::KernelValue,
	) -> Result<(), Error> {
		todo!()
	}

	fn fri_fold<FSub>(
		&self,
		exec: &mut Self::Exec,
		ntt: &impl AdditiveNTT<FSub>,
		log_len: usize,
		log_batch_size: usize,
		challenges: &[T::B128],
		data_in: FSlice<T::B128, Self>,
		data_out: &mut FSliceMut<T::B128, Self>,
	) -> Result<(), Error>
	where
		FSub: binius_field::BinaryField,
		T::B128: binius_field::ExtensionField<FSub>,
	{
		todo!()
	}
}
