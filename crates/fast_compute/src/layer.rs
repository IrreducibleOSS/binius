// Copyright 2025 Irreducible Inc.

use std::{cell::RefCell, marker::PhantomData};

use binius_compute::{
	layer::{ComputeLayer, Error, FSlice, FSliceMut},
	memory::SizedSlice,
};
use binius_field::{tower::TowerFamily, unpack_if_possible, unpack_if_possible_mut, PackedField};
use binius_math::ArithExpr;
use binius_ntt::AdditiveNTT;
use bytemuck::zeroed_vec;

use crate::{arith_circuit::ArithCircuitPoly, memory::PackedMemory};

#[derive(Debug)]
pub struct FastCpuExecutor;

#[derive(Debug, Default)]
pub struct CpuLayer<T: TowerFamily, P: PackedField<Scalar = T::B128>>(PhantomData<(T, P)>);

impl<T: TowerFamily, P: PackedField<Scalar = T::B128>> ComputeLayer<T::B128> for CpuLayer<T, P> {
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
		exec: &mut Self::Exec,
		a_edeg: usize,
		a_in: FSlice<'_, T::B128, Self>,
		b_in: FSlice<'_, T::B128, Self>,
	) -> Result<Self::OpValue, Error> {
		todo!()
	}

	fn tensor_expand(
		&self,
		exec: &mut Self::Exec,
		log_n: usize,
		coordinates: &[T::B128],
		data: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		todo!()
	}

	fn kernel_decl_value(
		&self,
		exec: &mut Self::KernelExec,
		init: T::B128,
	) -> Result<Self::KernelValue, Error> {
		todo!()
	}

	fn compile_expr(&self, expr: &ArithExpr<T::B128>) -> Result<Self::ExprEval, Error> {
		todo!()
	}

	fn accumulate_kernels(
		&self,
		exec: &mut Self::Exec,
		map: impl for<'a> Fn(
			&'a mut Self::KernelExec,
			usize,
			Vec<binius_compute::layer::KernelBuffer<'a, T::B128, Self::DevMem>>,
		) -> Result<Vec<Self::KernelValue>, Error>,
		mem_maps: Vec<binius_compute::layer::KernelMemMap<'_, T::B128, Self::DevMem>>,
	) -> Result<Vec<Self::OpValue>, Error> {
		todo!()
	}

	fn fold_left<'a>(
		&'a self,
		exec: &'a mut Self::Exec,
		mat: binius_compute::memory::SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		todo!()
	}

	fn fold_right<'a>(
		&'a self,
		exec: &'a mut Self::Exec,
		mat: binius_compute::memory::SubfieldSlice<'_, T::B128, Self::DevMem>,
		vec: <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSlice<'_>,
		out: &mut <Self::DevMem as binius_compute::memory::ComputeMemory<T::B128>>::FSliceMut<'_>,
	) -> Result<(), Error> {
		todo!()
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
