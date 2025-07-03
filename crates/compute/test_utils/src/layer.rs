// Copyright 2025 Irreducible Inc.

use std::{iter, iter::repeat_with, mem::MaybeUninit};

use binius_compute::{
	ComputeData, ComputeHolder, ComputeLayerExecutor, KernelExecutor, KernelMem,
	alloc::ComputeAllocator,
	layer::{ComputeLayer, KernelBuffer, KernelMemMap},
	memory::{ComputeMemory, SizedSlice, SlicesBatch, SubfieldSlice},
};
use binius_core::composition::BivariateProduct;
use binius_field::{BinaryField, ExtensionField, Field, PackedExtension, PackedField, TowerField};
use binius_math::{
	ArithCircuit, CompositionPoly, MultilinearExtension, MultilinearQuery, extrapolate_line_scalar,
	tensor_prod_eq_ind,
};
use binius_ntt::fri::fold_interleaved;
use binius_utils::checked_arithmetics::checked_log_2;
use itertools::Itertools;
use rand::{Rng, SeedableRng, prelude::StdRng};

pub fn test_generic_single_tensor_expand<
	F: Field,
	C: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, C>,
>(
	mut compute_data: ComputeHolderType,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	let coordinates = repeat_with(|| F::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	let buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for (i, x_i) in buffer.iter_mut().enumerate() {
		if i >= 4 {
			*x_i = F::ZERO;
		} else {
			*x_i = F::random(&mut rng);
		}
	}
	let mut buffer_clone = buffer.to_vec();

	// Copy the buffer to device slice
	let mut buffer_slice = dev_alloc.alloc(buffer.len()).unwrap();
	compute.copy_h2d(buffer, &mut buffer_slice).unwrap();

	// Run the HAL operation
	compute
		.execute(|exec| {
			exec.tensor_expand(2, &coordinates[2..], &mut buffer_slice)?;
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
	ComputeHolderType: ComputeHolder<F, C>,
>(
	mut compute_holder: ComputeHolderType,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);
	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	// Allocate buffers a and b to be device mapped
	let a_buffer = host_alloc.alloc(1 << (n_vars - F::LOG_DEGREE)).unwrap();
	for x_i in a_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let a = a_buffer.to_vec();

	let b_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in b_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let b = b_buffer.to_vec();

	// Copy a and b to device (creating F-slices)
	let mut a_slice = dev_alloc.alloc(a_buffer.len()).unwrap();
	compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
	let a_slice = C::DevMem::as_const(&a_slice);

	let mut b_slice = dev_alloc.alloc(b_buffer.len()).unwrap();
	compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
	let b_slice = C::DevMem::as_const(&b_slice);

	// Run the HAL operation to compute the inner product
	let a_subslice = SubfieldSlice::new(a_slice, F2::TOWER_LEVEL);
	let actual = compute
		.execute(|exec| Ok(vec![exec.inner_product(a_subslice, b_slice)?]))
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
	ComputeDataHolderType: ComputeHolder<F, C>,
>(
	mut compute_data: ComputeDataHolderType,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);
	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	// Allocate buffers to be device mapped
	let mle1_buffer = host_alloc
		.alloc(1 << (n_vars - <F as ExtensionField<F1>>::LOG_DEGREE))
		.unwrap();
	for x_i in mle1_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let mle1 = mle1_buffer.to_vec();

	let mle2_buffer = host_alloc
		.alloc(1 << (n_vars - <F as ExtensionField<F2>>::LOG_DEGREE))
		.unwrap();
	for x_i in mle2_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let mle2 = mle2_buffer.to_vec();

	let eq_ind_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in eq_ind_buffer.iter_mut() {
		*x_i = F::ZERO;
	}

	// Copy data to device (creating F-slices)
	let mut mle1_slice = dev_alloc.alloc(mle1_buffer.len()).unwrap();
	compute.copy_h2d(mle1_buffer, &mut mle1_slice).unwrap();
	let mle1_slice = C::DevMem::as_const(&mle1_slice);

	let mut mle2_slice = dev_alloc.alloc(mle2_buffer.len()).unwrap();
	compute.copy_h2d(mle2_buffer, &mut mle2_slice).unwrap();
	let mle2_slice = C::DevMem::as_const(&mle2_slice);

	let mut eq_ind_slice = dev_alloc.alloc(eq_ind_buffer.len()).unwrap();
	compute.copy_h2d(eq_ind_buffer, &mut eq_ind_slice).unwrap();

	let coordinates = repeat_with(|| <F as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	// Run the HAL operation
	let results = compute
		.execute(|exec| {
			{
				// Swap first element on the device buffer
				let mut first_elt =
					<C::DevMem as ComputeMemory<F>>::slice_mut(&mut eq_ind_slice, ..1);
				compute.copy_h2d(&[F::ONE], &mut first_elt)?;
			}

			exec.tensor_expand(0, &coordinates, &mut eq_ind_slice)?;

			let eq_ind = <C::DevMem as ComputeMemory<F>>::as_const(&eq_ind_slice);
			let (eval1, eval2) = exec.join(
				|exec| exec.inner_product(SubfieldSlice::new(mle1_slice, F1::TOWER_LEVEL), eq_ind),
				|exec| exec.inner_product(SubfieldSlice::new(mle2_slice, F2::TOWER_LEVEL), eq_ind),
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

pub fn test_generic_map_with_multilinear_evaluations<
	F: Field + TowerField + PackedField<Scalar = F>,
	C: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, C>,
>(
	mut compute_data: ComputeHolderType,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);
	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	// Allocate buffers to be device mapped
	let mle1_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in mle1_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let mle1 = mle1_buffer.to_vec();

	let mle2_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in mle2_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let mle2 = mle2_buffer.to_vec();

	let eq_ind_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in eq_ind_buffer.iter_mut() {
		*x_i = F::ZERO;
	}

	// Copy data to device (creating F-slices)
	let mut mle1_slice = dev_alloc.alloc(mle1_buffer.len()).unwrap();
	compute.copy_h2d(mle1_buffer, &mut mle1_slice).unwrap();
	let mle1_slice = C::DevMem::as_const(&mle1_slice);

	let mut mle2_slice = dev_alloc.alloc(mle2_buffer.len()).unwrap();
	compute.copy_h2d(mle2_buffer, &mut mle2_slice).unwrap();
	let mle2_slice = C::DevMem::as_const(&mle2_slice);

	let mut eq_ind_slice = dev_alloc.alloc(eq_ind_buffer.len()).unwrap();
	compute.copy_h2d(eq_ind_buffer, &mut eq_ind_slice).unwrap();

	let coordinates = repeat_with(|| <F as Field>::random(&mut rng))
		.take(n_vars)
		.collect::<Vec<_>>();

	// Run the HAL operation
	let results = compute
		.execute(|exec| {
			{
				// Swap first element on the device buffer
				let mut first_elt =
					<C::DevMem as ComputeMemory<F>>::slice_mut(&mut eq_ind_slice, ..1);
				compute.copy_h2d(&[F::ONE], &mut first_elt)?;
			}

			exec.tensor_expand(0, &coordinates, &mut eq_ind_slice)?;

			let eq_ind = <C::DevMem as ComputeMemory<F>>::as_const(&eq_ind_slice);
			let res = exec.map([mle1_slice, mle2_slice].iter(), |exec, iter_item| {
				exec.inner_product(SubfieldSlice::new(*iter_item, F::TOWER_LEVEL), eq_ind)
			})?;
			Ok(res)
		})
		.unwrap();
	let (eval1, eval2) = TryInto::<[F; 2]>::try_into(results)
		.expect("expected two evaluations")
		.into();

	// Compute the expected value
	let query = MultilinearQuery::<F>::expand(&coordinates);
	let expected_eval1 = MultilinearExtension::new(n_vars, mle1)
		.unwrap()
		.evaluate(&query)
		.unwrap();
	let expected_eval2 = MultilinearExtension::new(n_vars, mle2)
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
	C: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, C>,
>(
	mut compute_data: ComputeHolderType,
	n_vars: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);
	let log_min_chunk_size = 3;

	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	// Allocate buffers a and b to be device mapped
	let a_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in a_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let a = a_buffer.to_vec();

	let b_buffer = host_alloc.alloc(1 << n_vars).unwrap();
	for x_i in b_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let b = b_buffer.to_vec();

	// Copy a and b to device (creating F-slices)
	let mut a_slice = dev_alloc.alloc(a_buffer.len()).unwrap();
	compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
	let a_slice = C::DevMem::as_const(&a_slice);

	let mut b_slice = dev_alloc.alloc(b_buffer.len()).unwrap();
	compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
	let b_slice = C::DevMem::as_const(&b_slice);

	// Run the HAL operation to compute the inner product
	let arith = ArithCircuit::var(0) * ArithCircuit::var(1);
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
			let results = exec.accumulate_kernels(
				|kernel_exec, _log_chunks, kernel_data| {
					let kernel_data = kernel_data
						.iter()
						.map(|buf| buf.to_ref())
						.collect::<Vec<_>>();
					let row_len = kernel_data[0].len();
					let slice_batch = SlicesBatch::new(kernel_data, row_len);
					let mut res = kernel_exec.decl_value(F::ZERO)?;
					kernel_exec
						.sum_composition_evals(&slice_batch, &eval, F::ONE, &mut res)
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

pub fn test_generic_kernel_add<
	F: Field,
	C: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, C>,
>(
	mut compute_data: ComputeHolderType,
	log_len: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);
	let log_min_chunk_size = 3;

	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	// Allocate buffers a and b to be device mapped
	let a_buffer = host_alloc.alloc(1 << log_len).unwrap();
	for x_i in a_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let a = a_buffer.to_vec();

	let b_buffer = host_alloc.alloc(1 << log_len).unwrap();
	for x_i in b_buffer.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}
	let b = b_buffer.to_vec();

	// Copy a and b to device (creating F-slices)
	let mut a_slice = dev_alloc.alloc(a_buffer.len()).unwrap();
	compute.copy_h2d(a_buffer, &mut a_slice).unwrap();
	let a_slice = C::DevMem::as_const(&a_slice);

	let mut b_slice = dev_alloc.alloc(b_buffer.len()).unwrap();
	compute.copy_h2d(b_buffer, &mut b_slice).unwrap();
	let b_slice = C::DevMem::as_const(&b_slice);

	// Run the HAL operation to compute the a + b
	let arith = ArithCircuit::var(0);
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
			let results = exec.accumulate_kernels(
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
					kernel_exec.add(log_len, a, b, &mut c)?;
					let c = <KernelMem<F, C>>::as_const(&c);
					let mut res = kernel_exec.decl_value(F::ZERO)?;
					kernel_exec.sum_composition_evals(
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

pub fn test_generic_fri_fold<F, FSub, C, ComputeHolderType: ComputeHolder<F, C>>(
	mut compute_data: ComputeHolderType,
	log_len: usize,
	log_batch_size: usize,
	log_fold_challenges: usize,
) where
	F: TowerField + ExtensionField<FSub>,
	FSub: BinaryField,
	C: ComputeLayer<F>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	let ntt = binius_ntt::SingleThreadedNTT::<FSub>::new(log_len).unwrap();

	// Allocate buffers to be device mapped
	let data_in = host_alloc.alloc(1 << (log_len + log_batch_size)).unwrap();
	for x_i in data_in.iter_mut() {
		*x_i = <F as Field>::random(&mut rng);
	}

	// Copy the buffer to device slice
	let mut data_in_slice = dev_alloc.alloc(data_in.len()).unwrap();
	compute.copy_h2d(data_in, &mut data_in_slice).unwrap();
	let data_in_slice = C::DevMem::as_const(&data_in_slice);

	let data_out = host_alloc
		.alloc(1 << (log_len - log_fold_challenges))
		.unwrap();
	for x_i in data_out.iter_mut() {
		*x_i = <F as Field>::ZERO;
	}
	let mut data_out_slice = dev_alloc.alloc(data_out.len()).unwrap();
	compute.copy_h2d(data_out, &mut data_out_slice).unwrap();

	let challenges = repeat_with(|| <F as Field>::random(&mut rng))
		.take(log_batch_size + log_fold_challenges)
		.collect::<Vec<_>>();

	// Run the HAL operation
	compute
		.execute(|exec| {
			exec.fri_fold(
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
	let expected_result = fold_interleaved(&ntt, data_in, &challenges, log_len, log_batch_size);
	assert_eq!(data_out, &expected_result);
}

pub fn test_generic_single_left_fold<
	F: Field + TowerField,
	F2: ExtensionField<F> + TowerField,
	C: ComputeLayer<F2>,
	ComputeHolderType: ComputeHolder<F2, C>,
>(
	mut compute_holder: ComputeHolderType,
	log_evals_size: usize,
	log_query_size: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	let num_f_per_f2 = size_of::<F2>() / size_of::<F>();
	let log_evals_size_f2 = log_evals_size - num_f_per_f2.ilog2() as usize;
	let evals = host_alloc.alloc(1 << log_evals_size_f2).unwrap();
	for eval in evals.iter_mut() {
		*eval = F2::random(&mut rng);
	}
	let query = host_alloc.alloc(1 << log_query_size).unwrap();
	for query_elem in query.iter_mut() {
		*query_elem = F2::random(&mut rng);
	}
	let out = host_alloc
		.alloc(1 << (log_evals_size - log_query_size))
		.unwrap();
	for x_i in out.iter_mut() {
		*x_i = F2::random(&mut rng);
	}

	let mut out_slice = dev_alloc.alloc(out.len()).unwrap();
	let mut evals_slice = dev_alloc.alloc(evals.len()).unwrap();
	let mut query_slice = dev_alloc.alloc(query.len()).unwrap();
	compute.copy_h2d(out, &mut out_slice).unwrap();
	compute.copy_h2d(evals, &mut evals_slice).unwrap();
	compute.copy_h2d(query, &mut query_slice).unwrap();
	let const_evals_slice = C::DevMem::as_const(&evals_slice);
	let const_query_slice = C::DevMem::as_const(&query_slice);
	let evals_slice_with_tower_level =
		SubfieldSlice::<'_, F2, C::DevMem>::new(const_evals_slice, F::TOWER_LEVEL);
	compute
		.execute(|exec| {
			exec.fold_left(evals_slice_with_tower_level, const_query_slice, &mut out_slice)
				.unwrap();
			Ok(vec![])
		})
		.unwrap();
	compute
		.copy_d2h(C::DevMem::as_const(&out_slice), out)
		.unwrap();

	let mut expected_out = out.iter().map(|x| MaybeUninit::new(*x)).collect::<Vec<_>>();
	let evals_as_f1_slice = evals
		.iter()
		.flat_map(<F2 as ExtensionField<F>>::iter_bases)
		.collect::<Vec<_>>();
	binius_math::fold_left(
		&evals_as_f1_slice,
		log_evals_size,
		query,
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
	F: Field + TowerField,
	F2: ExtensionField<F> + TowerField,
	C: ComputeLayer<F2>,
	ComputeHolderType: ComputeHolder<F2, C>,
>(
	mut compute_data: ComputeHolderType,
	log_evals_size: usize,
	log_query_size: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal: compute,
		host_alloc,
		dev_alloc,
		..
	} = compute_data.to_data();

	let num_f_per_f2 = size_of::<F2>() / size_of::<F>();
	let log_evals_size_f2 = log_evals_size - num_f_per_f2.ilog2() as usize;
	let evals = repeat_with(|| F2::random(&mut rng))
		.take(1 << log_evals_size_f2)
		.collect::<Vec<_>>();
	let query = repeat_with(|| F2::random(&mut rng))
		.take(1 << log_query_size)
		.collect::<Vec<_>>();
	let out = host_alloc
		.alloc(1 << (log_evals_size - log_query_size))
		.unwrap();
	for x_i in out.iter_mut() {
		*x_i = F2::random(&mut rng);
	}

	let mut out_slice = dev_alloc.alloc(out.len()).unwrap();
	let mut evals_slice = dev_alloc.alloc(evals.len()).unwrap();
	let mut query_slice = dev_alloc.alloc(query.len()).unwrap();
	compute.copy_h2d(out, &mut out_slice).unwrap();
	compute
		.copy_h2d(evals.as_slice(), &mut evals_slice)
		.unwrap();
	compute
		.copy_h2d(query.as_slice(), &mut query_slice)
		.unwrap();
	let const_evals_slice = C::DevMem::as_const(&evals_slice);
	let const_query_slice = C::DevMem::as_const(&query_slice);
	let evals_slice_with_tower_level =
		SubfieldSlice::<'_, F2, C::DevMem>::new(const_evals_slice, F::TOWER_LEVEL);
	compute
		.execute(|exec| {
			exec.fold_right(evals_slice_with_tower_level, const_query_slice, &mut out_slice)
				.unwrap();
			Ok(vec![])
		})
		.unwrap();
	compute
		.copy_d2h(C::DevMem::as_const(&out_slice), out)
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

pub fn test_extrapolate_line<
	F: Field,
	Hal: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, Hal>,
>(
	mut compute_holder: ComputeHolderType,
	log_len: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	let evals_0_host = host_alloc.alloc(1 << log_len).unwrap();
	let evals_1_host = host_alloc.alloc(1 << log_len).unwrap();
	let result_host = host_alloc.alloc(1 << log_len).unwrap();

	evals_0_host.fill_with(|| F::random(&mut rng));
	evals_1_host.fill_with(|| F::random(&mut rng));

	let mut evals_0_dev = dev_alloc.alloc(1 << log_len).unwrap();
	let mut evals_1_dev = dev_alloc.alloc(1 << log_len).unwrap();
	hal.copy_h2d(evals_0_host, &mut evals_0_dev).unwrap();
	hal.copy_h2d(evals_1_host, &mut evals_1_dev).unwrap();

	let z = F::random(&mut rng);

	let _ = hal
		.execute(|exec| {
			exec.extrapolate_line(&mut evals_0_dev, Hal::DevMem::as_const(&evals_1_dev), z)?;
			Ok(Vec::new())
		})
		.unwrap();

	hal.copy_d2h(Hal::DevMem::as_const(&evals_0_dev), result_host)
		.unwrap();

	let expected_result = iter::zip(evals_0_host, evals_1_host)
		.map(|(x0, x1)| extrapolate_line_scalar(*x0, *x1, z))
		.collect::<Vec<_>>();
	assert_eq!(result_host, &expected_result);
}

pub fn test_generic_compute_composite<
	F: Field,
	Hal: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, Hal>,
>(
	mut compute_holder: ComputeHolderType,
	log_len: usize,
) {
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	let input_0_host = host_alloc.alloc(1 << log_len).unwrap();
	let input_1_host = host_alloc.alloc(1 << log_len).unwrap();
	let output_host = host_alloc.alloc(1 << log_len).unwrap();

	input_0_host.fill_with(|| F::random(&mut rng));
	input_1_host.fill_with(|| F::random(&mut rng));

	let mut input_0_dev = dev_alloc.alloc(1 << log_len).unwrap();
	let mut input_1_dev = dev_alloc.alloc(1 << log_len).unwrap();
	let mut output_dev = dev_alloc.alloc(1 << log_len).unwrap();

	hal.copy_h2d(input_0_host, &mut input_0_dev).unwrap();
	hal.copy_h2d(input_1_host, &mut input_1_dev).unwrap();

	let input_0_dev = Hal::DevMem::as_const(&input_0_dev);
	let input_1_dev = Hal::DevMem::as_const(&input_1_dev);

	let bivariate_product_expr = hal
		.compile_expr(&CompositionPoly::<F>::expression(&BivariateProduct::default()))
		.unwrap();

	let inputs = SlicesBatch::new(vec![input_0_dev, input_1_dev], 1 << log_len);

	hal.execute(|exec| {
		exec.compute_composite(&inputs, &mut output_dev, &bivariate_product_expr)
			.unwrap();
		Ok(vec![])
	})
	.unwrap();

	hal.copy_d2h(Hal::DevMem::as_const(&output_dev), output_host)
		.unwrap();

	for (i, output) in output_host.iter_mut().enumerate() {
		assert_eq!(*output, input_0_host[i] * input_1_host[i])
	}
}

pub fn test_map_kernels<F, Hal, ComputeHolderType>(
	mut compute_holder: ComputeHolderType,
	log_len: usize,
) where
	F: Field,
	Hal: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, Hal>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	let input_1_host = host_alloc.alloc(1 << log_len).unwrap();

	input_1_host.fill_with(|| F::random(&mut rng));

	let input_2_host = host_alloc.alloc(1 << log_len).unwrap();

	input_2_host.fill_with(|| {
		if rng.random_bool(0.5) {
			F::ONE
		} else {
			F::ZERO
		}
	});

	let output_host = host_alloc.alloc(1 << log_len).unwrap();

	let mut input_1_dev = dev_alloc.alloc(1 << log_len).unwrap();
	let mut input_2_dev = dev_alloc.alloc(1 << log_len).unwrap();

	hal.copy_h2d(input_1_host, &mut input_1_dev).unwrap();
	hal.copy_h2d(input_2_host, &mut input_2_dev).unwrap();

	let mem_map = vec![
		KernelMemMap::ChunkedMut {
			data: Hal::DevMem::to_owned_mut(&mut input_1_dev),
			log_min_chunk_size: 0,
		},
		KernelMemMap::Chunked {
			data: Hal::DevMem::as_const(&input_2_dev),
			log_min_chunk_size: 0,
		},
	];

	hal.execute(|exec| {
		exec.map_kernels(
			|local_exec, log_chunks, mut buffers| {
				let log_chunk_size = log_len - log_chunks;

				let Ok([KernelBuffer::Mut(input_1), KernelBuffer::Ref(input_2)]) =
					TryInto::<&mut [_; 2]>::try_into(buffers.as_mut_slice())
				else {
					panic!(
						"exec_kernels did not create the mapped buffers struct according to the mapping"
					);
				};
				local_exec.add_assign(log_chunk_size, *input_2, input_1)?;

				Ok(())
			},
			mem_map,
		)
		.unwrap();
		Ok(vec![])
	})
	.unwrap();

	hal.copy_d2h(Hal::DevMem::as_const(&input_1_dev), output_host)
		.unwrap();

	for (i, output) in output_host.iter_mut().enumerate() {
		assert_eq!(*output, input_1_host[i] + input_2_host[i]);
	}
}

pub fn test_generic_pairwise_product_reduce<F, Hal, ComputeHolderType>(
	mut compute_holder: ComputeHolderType,
	log_len: usize,
) where
	F: Field,
	Hal: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, Hal>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let ComputeData {
		hal,
		host_alloc,
		dev_alloc,
		..
	} = compute_holder.to_data();

	let input = host_alloc.alloc(1 << log_len).unwrap();
	input.fill_with(|| F::random(&mut rng));

	let mut round_outputs = Vec::new();
	let mut current_len = input.len() / 2;
	while current_len >= 1 {
		round_outputs.push(dev_alloc.alloc(current_len).unwrap());
		current_len /= 2;
	}

	let mut dev_input = dev_alloc.alloc(input.len()).unwrap();
	hal.copy_h2d(input, &mut dev_input).unwrap();
	hal.execute(|exec| {
		exec.pairwise_product_reduce(Hal::DevMem::as_const(&dev_input), round_outputs.as_mut())
			.unwrap();
		Ok(vec![])
	})
	.unwrap();
	let mut working_input = input.to_vec();
	let mut round_idx = 0;
	while working_input.len() >= 2 {
		let mut round_results = (0..working_input.len() / 2).map(|_| F::ZERO).collect_vec();
		for idx in 0..round_results.len() {
			round_results[idx] = working_input[idx * 2] * working_input[idx * 2 + 1];
		}

		let actual_round_result = host_alloc.alloc(working_input.len() / 2).unwrap();
		hal.copy_d2h(Hal::DevMem::as_const(&round_outputs[round_idx]), actual_round_result)
			.unwrap();
		assert_eq!(round_results, actual_round_result);

		working_input = round_results;
		round_idx += 1;
	}
}
