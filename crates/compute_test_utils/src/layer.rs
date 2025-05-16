// Copyright 2025 Irreducible Inc.

use std::{iter::repeat_with, mem::MaybeUninit};

use binius_compute::{
	alloc::{BumpAllocator, ComputeAllocator},
	layer::{ComputeLayer, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SlicesBatch, SubfieldSlice},
};
use binius_fast_compute::fri::fold_interleaved;
use binius_field::{BinaryField, ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_math::{ArithExpr, MultilinearExtension, MultilinearQuery, tensor_prod_eq_ind};
use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::zeroed_vec;
use rand::{SeedableRng, prelude::StdRng};

pub fn test_generic_single_tensor_expand<F: Field, C: ComputeLayer<F>>(
	compute: C,
	device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let coordinates = repeat_with(|| F::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	// Allocate buffer to be device mapped
	let mut buffer = compute.host_alloc(1 << n_vars);
	let buffer = buffer.as_mut();
	for (i, x_i) in buffer.iter_mut().enumerate() {
		if i >= 4 {
			*x_i = F::ZERO;
		} else {
			*x_i = F::random(&mut rng);
		}
	}
	let mut buffer_clone = buffer.to_vec();

	// Copy the buffer to device slice
	let (mut buffer_slice, _device_memory) = C::DevMem::split_at_mut(device_memory, buffer.len());
	compute.copy_h2d(buffer, &mut buffer_slice).unwrap();

	// Run the HAL operation
	compute
		.execute(|exec| {
			compute.tensor_expand(exec, 2, &coordinates[2..], &mut buffer_slice)?;
			Ok(vec![])
		})
		.unwrap();

	// Copy the buffer back to host
	let buffer_slice = C::DevMem::as_const(&buffer_slice);
	compute.copy_d2h(buffer_slice, buffer).unwrap();

	// Compute the expected result and compare
	tensor_prod_eq_ind(2, &mut buffer_clone, &coordinates[2..]).unwrap();
	assert_eq!(buffer, buffer_clone);
}

pub fn test_generic_single_inner_product<
	F2: TowerField,
	F: Field + PackedExtension<F2> + ExtensionField<F2>,
	C: ComputeLayer<F>,
>(
	compute: C,
	device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	// Allocate buffers a and b to be device mapped
	let mut a_buffer = compute.host_alloc(1 << (n_vars - F::LOG_DEGREE));
	let a_buffer = a_buffer.as_mut();
	for x_i in a_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let a = a_buffer.to_vec();

	let mut b_buffer = compute.host_alloc(1 << n_vars);
	let b_buffer = b_buffer.as_mut();
	for x_i in b_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let b = b_buffer.to_vec();

	// Copy a and b to device (creating F-slices)
	let (mut a_slice, device_memory) = C::DevMem::split_at_mut(device_memory, a_buffer.len());
	compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
	let a_slice = C::DevMem::as_const(&a_slice);

	let (mut b_slice, _device_memory) = C::DevMem::split_at_mut(device_memory, b_buffer.len());
	compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
	let b_slice = C::DevMem::as_const(&b_slice);

	// Run the HAL operation to compute the inner product
	let actual = compute
		.execute(|exec| Ok(vec![compute.inner_product(exec, F2::TOWER_LEVEL, a_slice, b_slice)?]))
		.unwrap()
		.remove(0);

	// Compute the expected value and compare
	let expected = std::iter::zip(PackedField::iter_slice(F::cast_bases(&a)), &b)
		.map(|(a_i, &b_i)| b_i * a_i)
		.sum::<F>();
	assert_eq!(actual, expected);
}

pub fn test_generic_multiple_multilinear_evaluations<
	F1: TowerField,
	F2: TowerField,
	F: Field
		+ PackedField<Scalar = F>
		+ PackedExtension<F1>
		+ ExtensionField<F1>
		+ PackedExtension<F2>
		+ ExtensionField<F2>,
	C: ComputeLayer<F>,
>(
	compute: C,
	device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	// Allocate buffers to be device mapped
	let mut mle1_buffer = compute.host_alloc(1 << (n_vars - <F as ExtensionField<F1>>::LOG_DEGREE));
	let mle1_buffer = mle1_buffer.as_mut();
	for x_i in mle1_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let mle1 = mle1_buffer.to_vec();

	let mut mle2_buffer = compute.host_alloc(1 << (n_vars - <F as ExtensionField<F2>>::LOG_DEGREE));
	let mle2_buffer = mle2_buffer.as_mut();
	for x_i in mle2_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let mle2 = mle2_buffer.to_vec();

	let mut eq_ind_buffer = compute.host_alloc(1 << n_vars);
	let eq_ind_buffer = eq_ind_buffer.as_mut();
	for x_i in eq_ind_buffer.iter_mut() {
		*x_i = F::ZERO;
	}

	// Copy data to device (creating F-slices)
	let (mut mle1_slice, device_memory) = C::DevMem::split_at_mut(device_memory, mle1_buffer.len());
	compute.copy_h2d(mle1_buffer, &mut mle1_slice).unwrap();
	let mle1_slice = C::DevMem::as_const(&mle1_slice);

	let (mut mle2_slice, device_memory) = C::DevMem::split_at_mut(device_memory, mle2_buffer.len());
	compute.copy_h2d(mle2_buffer, &mut mle2_slice).unwrap();
	let mle2_slice = C::DevMem::as_const(&mle2_slice);

	let (mut eq_ind_slice, _device_memory) =
		C::DevMem::split_at_mut(device_memory, eq_ind_buffer.len());
	compute.copy_h2d(eq_ind_buffer, &mut eq_ind_slice).unwrap();

	let coordinates = repeat_with(|| <F as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	// Run the HAL operation
	let results = compute
		.execute(|exec| {
			{
				// Swap first element on the device buffer
				let mut first_elts = <C::DevMem as ComputeMemory<F>>::slice_mut(
					&mut eq_ind_slice,
					..C::DevMem::MIN_SLICE_LEN,
				);
				let mut buffer = zeroed_vec::<F>(C::DevMem::MIN_SLICE_LEN);
				compute.copy_d2h(
					<C::DevMem as ComputeMemory<F>>::as_const(&first_elts),
					&mut buffer,
				)?;
				buffer[0].set(0, F::ONE);
				compute.copy_h2d(&buffer, &mut first_elts)?;
			}

			compute.tensor_expand(exec, 0, &coordinates, &mut eq_ind_slice)?;

			let eq_ind = <C::DevMem as ComputeMemory<F>>::as_const(&eq_ind_slice);
			let (eval1, eval2) = compute.join(
				exec,
				|exec| compute.inner_product(exec, F1::TOWER_LEVEL, mle1_slice, eq_ind),
				|exec| compute.inner_product(exec, F2::TOWER_LEVEL, mle2_slice, eq_ind),
			)?;
			Ok(vec![eval1, eval2])
		})
		.unwrap();
	let (eval1, eval2) = TryInto::<[F; 2]>::try_into(results)
		.expect("expected two evaluations")
		.into();

	// Compute the expected value
	let query = MultilinearQuery::<F>::expand(&coordinates);
	let expected_eval1 =
		MultilinearExtension::new(n_vars, <F as PackedExtension<F1>>::cast_bases(&mle1))
			.unwrap()
			.evaluate(&query)
			.unwrap();
	let expected_eval2 =
		MultilinearExtension::new(n_vars, <F as PackedExtension<F2>>::cast_bases(&mle2))
			.unwrap()
			.evaluate(&query)
			.unwrap();

	// Copy eq_ind back from the device
	let eq_ind_slice = C::DevMem::as_const(&eq_ind_slice);
	compute.copy_d2h(eq_ind_slice, eq_ind_buffer).unwrap();

	// Compare the results
	assert_eq!(eq_ind_buffer, query.into_expansion());
	assert_eq!(eval1, expected_eval1);
	assert_eq!(eval2, expected_eval2);
}

pub fn test_generic_single_inner_product_using_kernel_accumulator<
	F: Field,
	C: ComputeLayer<F, ExprEval: Sync> + Sync,
>(
	compute: C,
	device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'_>,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);
	let log_min_chunk_size = 3;

	// Allocate buffers a and b to be device mapped
	let mut a_buffer = compute.host_alloc(1 << n_vars);
	let a_buffer = a_buffer.as_mut();
	for x_i in a_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let a = a_buffer.to_vec();

	let mut b_buffer = compute.host_alloc(1 << n_vars);
	let b_buffer = b_buffer.as_mut();
	for x_i in b_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let b = b_buffer.to_vec();

	// Copy a and b to device (creating F-slices)
	let (mut a_slice, device_memory) = C::DevMem::split_at_mut(device_memory, a_buffer.len());
	compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
	let a_slice = C::DevMem::as_const(&a_slice);

	let (mut b_slice, _device_memory) = C::DevMem::split_at_mut(device_memory, b_buffer.len());
	compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
	let b_slice = C::DevMem::as_const(&b_slice);

	// Run the HAL operation to compute the inner product
	let arith = ArithExpr::Var(0) * ArithExpr::Var(1);
	let eval = compute.compile_expr(&arith).unwrap();
	let [actual] = compute
		.execute(|exec| {
			let a_slice = KernelMemMap::Chunked {
				data: a_slice,
				log_min_chunk_size,
			};
			let b_slice = KernelMemMap::Chunked {
				data: b_slice,
				log_min_chunk_size,
			};
			let results = compute.accumulate_kernels(
				exec,
				|kernel_exec, _log_chunks, kernel_data| {
					let kernel_data = kernel_data
						.iter()
						.map(|buf| buf.to_ref())
						.collect::<Vec<_>>();
					let row_len = kernel_data[0].len();
					let slice_batch = SlicesBatch::new(kernel_data, row_len);
					let mut res = compute.kernel_decl_value(kernel_exec, F::ZERO)?;
					compute
						.sum_composition_evals(kernel_exec, &slice_batch, &eval, F::ONE, &mut res)
						.unwrap();
					Ok(vec![res])
				},
				vec![a_slice, b_slice],
			)?;
			assert_eq!(results.len(), 1);
			Ok(results)
		})
		.unwrap()
		.try_into()
		.unwrap();

	// Compute the expected value and compare
	let expected = std::iter::zip(PackedField::iter_slice(F::cast_bases(&a)), &b)
		.map(|(a_i, &b_i)| b_i * a_i)
		.sum::<F>();
	assert_eq!(actual, expected);
}

pub fn test_generic_kernel_add<'a, F: Field, C: ComputeLayer<F, ExprEval: Sync> + Sync>(
	compute: C,
	device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'a>,
	log_len: usize,
) where
	C::DevMem: 'a,
{
	let mut rng = StdRng::seed_from_u64(0);
	let log_min_chunk_size = 3;

	let device_allocator =
		BumpAllocator::<'a, F, <C as ComputeLayer<F>>::DevMem>::new(device_memory);

	// Allocate buffers a and b to be device mapped
	let mut a_buffer = compute.host_alloc(1 << log_len);
	let a_buffer = a_buffer.as_mut();
	for x_i in a_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let a = a_buffer.to_vec();

	let mut b_buffer = compute.host_alloc(1 << log_len);
	let b_buffer = b_buffer.as_mut();
	for x_i in b_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let b = b_buffer.to_vec();

	// Copy a and b to device (creating F-slices)
	let mut a_slice = device_allocator.alloc(1 << log_len).unwrap();
	compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
	let a_slice = C::DevMem::as_const(&a_slice);

	let mut b_slice = device_allocator.alloc(1 << log_len).unwrap();
	compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
	let b_slice = C::DevMem::as_const(&b_slice);

	// Run the HAL operation to compute the a + b
	let arith = ArithExpr::Var(0);
	let eval = compute.compile_expr(&arith).unwrap();
	let [actual] = compute
		.execute(|exec| {
			let a_slice = KernelMemMap::Chunked {
				data: a_slice,
				log_min_chunk_size,
			};
			let b_slice = KernelMemMap::Chunked {
				data: b_slice,
				log_min_chunk_size,
			};
			let c_slice = KernelMemMap::Local { log_size: log_len };
			let results = compute.accumulate_kernels(
				exec,
				|kernel_exec, _log_chunks, kernel_data| {
					let [a, b, c] = kernel_data
						.try_into()
						.unwrap_or_else(|_| panic!("expected 3 buffers"));
					let a = a.to_ref();
					let b = b.to_ref();
					let mut c = match c {
						KernelBuffer::Mut(c) => c,
						_ => unreachable!(),
					};
					let log_len = checked_log_2(a.len());
					compute.kernel_add(kernel_exec, log_len, a, b, &mut c)?;
					let c = C::DevMem::as_const(&c);
					let mut res = compute.kernel_decl_value(kernel_exec, F::ZERO)?;
					compute.sum_composition_evals(
						kernel_exec,
						&SlicesBatch::new(vec![c], c.len()),
						&eval,
						F::ONE,
						&mut res,
					)?;
					Ok(vec![res])
				},
				vec![a_slice, b_slice, c_slice],
			)?;
			assert_eq!(results.len(), 1);
			Ok(results)
		})
		.unwrap()
		.try_into()
		.unwrap();

	// Compute the expected value and compare
	let expected = a.iter().chain(b.iter()).sum::<F>();
	assert_eq!(actual, expected);
}

pub fn test_generic_fri_fold<'a, F, FSub, C>(
	compute: C,
	device_memory: <C::DevMem as ComputeMemory<F>>::FSliceMut<'a>,
	log_len: usize,
	log_batch_size: usize,
	log_fold_challenges: usize,
) where
	F: TowerField + ExtensionField<FSub>,
	FSub: BinaryField,
	C: ComputeLayer<F>,
	C::DevMem: 'a,
{
	let mut rng = StdRng::seed_from_u64(0);

	let ntt = binius_ntt::SingleThreadedNTT::<FSub>::new(log_len).unwrap();

	// Allocate buffers to be device mapped
	let mut data_in = compute.host_alloc(1 << (log_len + log_batch_size));
	let data_in = data_in.as_mut();
	for x_i in data_in.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let data_in = data_in.to_vec();

	// Copy the buffer to device slice
	let device_allocator =
		BumpAllocator::<'a, F, <C as ComputeLayer<F>>::DevMem>::new(device_memory);
	let mut data_in_slice = device_allocator.alloc(data_in.len()).unwrap();
	compute.copy_h2d(&data_in, &mut data_in_slice).unwrap();
	let data_in_slice = C::DevMem::as_const(&data_in_slice);

	let mut data_out = compute.host_alloc(1 << (log_len - log_fold_challenges));
	let data_out = data_out.as_mut();
	for x_i in data_out.iter_mut() {
		*x_i = <F as Field>::ZERO;
	}
	let mut data_out_slice = device_allocator.alloc(data_out.len()).unwrap();
	compute.copy_h2d(data_out, &mut data_out_slice).unwrap();

	// Create out slice

	let challenges = repeat_with(|| <F as Field>::random(&mut rng))
		.take(log_batch_size + log_fold_challenges)
		.collect::<Vec<_>>();

	// Run the HAL operation
	compute
		.execute(|exec| {
			compute.fri_fold(
				exec,
				&ntt,
				log_len,
				log_batch_size,
				&challenges,
				data_in_slice,
				&mut data_out_slice,
			)?;
			Ok(vec![])
		})
		.unwrap();

	// Copy the buffer back to host
	let data_out_slice = C::DevMem::as_const(&data_out_slice);
	compute.copy_d2h(data_out_slice, data_out).unwrap();

	// Compute the expected result and compare
	let expected_result = fold_interleaved(&ntt, &data_in, &challenges, log_len, log_batch_size);
	assert_eq!(data_out, &expected_result);
}

pub fn test_generic_single_left_fold<
	'a,
	'b,
	F: Field + TowerField,
	F2: ExtensionField<F> + TowerField,
	C,
>(
	compute: &'b C,
	device_memory: <C::DevMem as ComputeMemory<F2>>::FSliceMut<'a>,
	log_evals_size: usize,
	log_query_size: usize,
) where
	C: ComputeLayer<F2>,
	'a: 'b,
	<C as ComputeLayer<F2>>::DevMem: 'a,
{
	let mut rng = StdRng::seed_from_u64(0);

	let num_f_per_f2 = size_of::<F2>() / size_of::<F>();
	let log_evals_size_f2 = log_evals_size - num_f_per_f2.ilog2() as usize;
	let evals = repeat_with(|| F2::random(&mut rng))
		.take(1 << log_evals_size_f2)
		.collect::<Vec<_>>();
	let query = repeat_with(|| F2::random(&mut rng))
		.take(1 << log_query_size)
		.collect::<Vec<_>>();
	let mut out = compute.host_alloc(1 << (log_evals_size - log_query_size));
	let out = out.as_mut();
	for x_i in out.iter_mut() {
		*x_i = F2::random(&mut rng);
	}

	let device_allocator =
		BumpAllocator::<'a, F2, <C as ComputeLayer<F2>>::DevMem>::new(device_memory);
	let mut out_slice = device_allocator.alloc(out.len()).unwrap();
	let mut evals_slice = device_allocator.alloc(evals.len()).unwrap();
	let mut query_slice = device_allocator.alloc(query.len()).unwrap();
	compute.copy_h2d(out, &mut out_slice).unwrap();
	compute
		.copy_h2d(evals.as_slice(), &mut evals_slice)
		.unwrap();
	compute
		.copy_h2d(query.as_slice(), &mut query_slice)
		.unwrap();
	let const_evals_slice = <C as ComputeLayer<F2>>::DevMem::as_const(&evals_slice);
	let const_query_slice = <C as ComputeLayer<F2>>::DevMem::as_const(&query_slice);
	let evals_slice_with_tower_level =
		SubfieldSlice::<'_, F2, <C as ComputeLayer<F2>>::DevMem>::new(
			const_evals_slice,
			F::TOWER_LEVEL,
		);
	compute
		.execute(|exec| {
			compute
				.fold_left(exec, evals_slice_with_tower_level, const_query_slice, &mut out_slice)
				.unwrap();
			Ok(vec![])
		})
		.unwrap();
	compute
		.copy_d2h(<C as ComputeLayer<F2>>::DevMem::as_const(&out_slice), out)
		.unwrap();

	let mut expected_out = out.iter().map(|x| MaybeUninit::new(*x)).collect::<Vec<_>>();
	let evals_as_f1_slice = evals
		.iter()
		.flat_map(<F2 as ExtensionField<F>>::iter_bases)
		.collect::<Vec<_>>();
	binius_math::fold_left(
		&evals_as_f1_slice,
		log_evals_size,
		&query,
		log_query_size,
		expected_out.as_mut_slice(),
	)
	.unwrap();
	let expected_out = expected_out
		.iter()
		.map(|x| unsafe { x.assume_init() })
		.collect::<Vec<_>>();
	assert_eq!(out.len(), expected_out.len());
	assert_eq!(out, expected_out);
}

pub fn test_generic_single_right_fold<
	'a,
	'b,
	F: Field + TowerField,
	F2: ExtensionField<F> + TowerField,
	C,
>(
	compute: &'b C,
	device_memory: <C::DevMem as ComputeMemory<F2>>::FSliceMut<'a>,
	log_evals_size: usize,
	log_query_size: usize,
) where
	C: ComputeLayer<F2>,
	'a: 'b,
	<C as ComputeLayer<F2>>::DevMem: 'a,
{
	let mut rng = StdRng::seed_from_u64(0);

	let num_f_per_f2 = size_of::<F2>() / size_of::<F>();
	let log_evals_size_f2 = log_evals_size - num_f_per_f2.ilog2() as usize;
	let evals = repeat_with(|| F2::random(&mut rng))
		.take(1 << log_evals_size_f2)
		.collect::<Vec<_>>();
	let query = repeat_with(|| F2::random(&mut rng))
		.take(1 << log_query_size)
		.collect::<Vec<_>>();
	let mut out = compute.host_alloc(1 << (log_evals_size - log_query_size));
	let out = out.as_mut();
	for x_i in out.iter_mut() {
		*x_i = F2::random(&mut rng);
	}

	let device_allocator =
		BumpAllocator::<'a, F2, <C as ComputeLayer<F2>>::DevMem>::new(device_memory);
	let mut out_slice = device_allocator.alloc(out.len()).unwrap();
	let mut evals_slice = device_allocator.alloc(evals.len()).unwrap();
	let mut query_slice = device_allocator.alloc(query.len()).unwrap();
	compute.copy_h2d(out, &mut out_slice).unwrap();
	compute
		.copy_h2d(evals.as_slice(), &mut evals_slice)
		.unwrap();
	compute
		.copy_h2d(query.as_slice(), &mut query_slice)
		.unwrap();
	let const_evals_slice = <C as ComputeLayer<F2>>::DevMem::as_const(&evals_slice);
	let const_query_slice = <C as ComputeLayer<F2>>::DevMem::as_const(&query_slice);
	let evals_slice_with_tower_level =
		SubfieldSlice::<'_, F2, <C as ComputeLayer<F2>>::DevMem>::new(
			const_evals_slice,
			F::TOWER_LEVEL,
		);
	compute
		.execute(|exec| {
			compute
				.fold_right(exec, evals_slice_with_tower_level, const_query_slice, &mut out_slice)
				.unwrap();
			Ok(vec![])
		})
		.unwrap();
	compute
		.copy_d2h(<C as ComputeLayer<F2>>::DevMem::as_const(&out_slice), out)
		.unwrap();

	let mut expected_out = out.iter().map(|_| F2::ZERO).collect::<Vec<_>>();
	let evals_as_f1_slice = evals
		.iter()
		.flat_map(<F2 as ExtensionField<F>>::iter_bases)
		.collect::<Vec<_>>();
	binius_math::fold_right(
		&evals_as_f1_slice,
		log_evals_size,
		&query,
		log_query_size,
		expected_out.as_mut_slice(),
	)
	.unwrap();
	assert_eq!(out.len(), expected_out.len());
	assert_eq!(out, expected_out);
}
