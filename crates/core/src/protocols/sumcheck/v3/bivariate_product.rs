// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_compute::{ComputeLayer, ComputeMemory, FSlice, KernelBuffer, KernelMemMap};
use binius_field::{TowerField, util::powers};
use binius_math::{ArithExpr, CompositionPoly};

use super::error::Error;
use crate::composition::{BivariateProduct, IndexComposition};

/// Calculates the evaluations of the products of pairs of partially specialized multilinear
/// polynomials for sumcheck.
///
/// This performs round evaluation for a special case sumcheck prover for a sumcheck over bivariate
/// products of multilinear polynomials, defined over the same field as the sumcheck challenges,
/// using high-to-low variable binding order.
///
/// In more details, this function takes a slice of multilinear polynomials over a large field `F`
/// and a description of pairs of them, and computes the hypercube sum of the evaluations of these
/// pairs of multilinears at select points. The evaluation points are 0 and the "infinity" point.
/// The meaning of the infinity evaluation point is described in the documentation of
/// [`binius_math::EvaluationDomain`].
///
/// The evaluations are batched by mixing with the powers of a batch coefficient.
///
/// ## Mathematical Definition
///
/// Let $\alpha$ be the batching coefficient, $P_0, \ldots, P_{m-1}$ be the multilinears, and
/// $t \in \left(\mathbb{N}^2 \right)^k$ be a sequence of tuples of indices. This returns two
/// values:
///
/// $$
/// z_1 = \sum_{i=0}^{k-1} \alpha^i \sum_{v \in B_{m-1}} P_{t_{0,i}}(v || 0) P_{t_{1,i}}(v || 0)
/// \\\\ z_\infty = \sum_{i=0}^{k-1} \alpha^i \sum_{v \in B_{m-1}} P_{t_{0,i}}(v || \infty)
/// P_{t_{1,i}}(v || \infty) \\
/// $$
///
/// ## Returns
///
/// Returns the batched, summed evaluations at 0 and infinity.
pub fn calculate_round_evals<'a, F: TowerField, HAL: ComputeLayer<F>>(
	hal: &HAL,
	n_vars: usize,
	batch_coeff: F,
	multilins: &[FSlice<'a, F, HAL>],
	compositions: &[IndexComposition<BivariateProduct, 2>],
) -> Result<[F; 2], Error> {
	// TODO: if n_vars is too small, transfer data and fall back to the CPU impl

	let prod_evaluators = compositions
		.iter()
		.map(|composition| {
			let prod_expr = CompositionPoly::<F>::expression(&composition);
			hal.compile_expr(&ArithExpr::from(prod_expr))
		})
		.collect::<Result<Vec<_>, _>>()?;

	// n_vars - 1 is the number of variables in the halves of the split multilinear.
	let split_n_vars = n_vars - 1;
	let kernel_mappings = multilins
		.iter()
		.copied()
		.flat_map(|multilin| {
			let (lo_half, hi_half) = HAL::DevMem::split_at(multilin, 1 << split_n_vars);
			[
				KernelMemMap::Chunked {
					data: lo_half,
					log_min_chunk_size: 0,
				},
				KernelMemMap::Chunked {
					data: hi_half,
					log_min_chunk_size: 0,
				},
				// Evaluations of the multilinear at the extra evaluation point
				KernelMemMap::Local {
					log_size: split_n_vars,
				},
			]
		})
		.collect();

	let batch_coeffs = powers(batch_coeff)
		.take(compositions.len())
		.collect::<Vec<_>>();

	let evals = hal.execute(|exec| {
		hal.accumulate_kernels(
			exec,
			|local_exec, log_chunks, mut buffers| {
				let log_chunk_size = split_n_vars - log_chunks;

				// Compute the composite evaluations at the point ONE.
				let mut acc_1 = hal.kernel_decl_value(local_exec, F::ZERO)?;
				{
					let eval_1s = (0..multilins.len())
						.map(|i| buffers[i * 3 + 1].to_ref())
						.collect::<Vec<_>>();
					for (&batch_coeff, evaluator) in iter::zip(&batch_coeffs, &prod_evaluators) {
						hal.sum_composition_evals(
							local_exec,
							log_chunk_size,
							&eval_1s,
							evaluator,
							batch_coeff,
							&mut acc_1,
						)?;
					}
				}

				// Extrapolate the multilinear evaluations at the point Infinity.
				for group in buffers.chunks_mut(3) {
					let Ok(
						[
							KernelBuffer::Ref(evals_0),
							KernelBuffer::Ref(evals_1),
							KernelBuffer::Mut(evals_inf),
						],
					) = TryInto::<&mut [_; 3]>::try_into(group)
					else {
						panic!(
							"exec_kernels did not create the mapped buffers struct according to the mapping"
						);
					};
					hal.kernel_add(local_exec, log_chunk_size, *evals_0, *evals_1, evals_inf)?;
				}

				// Compute the composite evaluations at the point Infinity.
				let mut acc_inf = hal.kernel_decl_value(local_exec, F::ZERO)?;
				let eval_infs = (0..multilins.len())
					.map(|i| buffers[i * 3 + 2].to_ref())
					.collect::<Vec<_>>();
				for (&batch_coeff, evaluator) in iter::zip(&batch_coeffs, &prod_evaluators) {
					hal.sum_composition_evals(
						local_exec,
						log_chunk_size,
						&eval_infs,
						evaluator,
						batch_coeff,
						&mut acc_inf,
					)?;
				}

				Ok(vec![acc_1, acc_inf])
			},
			kernel_mappings,
		)
	})?;
	let evals = TryInto::<[F; 2]>::try_into(evals).expect("kernel returns two values");
	Ok(evals)
}

#[cfg(test)]
mod tests {
	use binius_compute::{
		FSliceMut,
		alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
		cpu::CpuLayer,
	};
	use binius_field::{
		BinaryField1b, BinaryField128b, Field, PackedBinaryField1x128b, PackedField,
		tower::CanonicalTowerFamily, util::inner_product_unchecked,
	};
	use binius_hal::make_portable_backend;
	use binius_math::{
		CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomain, EvaluationOrder,
		InterpolationDomain, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
	};
	use bytemuck::must_cast_slice;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		composition::BivariateProduct,
		polynomial::MultilinearComposite,
		protocols::sumcheck::{
			CompositeSumClaim, immediate_switchover_heuristic,
			prove::{RegularSumcheckProver, SumcheckProver},
		},
	};

	fn compute_composite_sum<P, M, Composition>(
		multilinears: &[M],
		composition: Composition,
	) -> P::Scalar
	where
		P: PackedField,
		M: MultilinearPoly<P> + Send + Sync,
		Composition: CompositionPoly<P>,
	{
		let n_vars = multilinears
			.first()
			.map(|multilinear| multilinear.n_vars())
			.unwrap_or_default();
		for multilinear in multilinears {
			assert_eq!(multilinear.n_vars(), n_vars);
		}

		let multilinears = multilinears.iter().collect::<Vec<_>>();
		let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
		(0..(1 << n_vars))
			.map(|j| witness.evaluate_on_hypercube(j).unwrap())
			.sum()
	}

	fn generic_test_calculate_round_evals<Hal: ComputeLayer<BinaryField128b>>(
		hal: &Hal,
		dev_mem: FSliceMut<BinaryField128b, Hal>,
	) {
		type F = BinaryField128b;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let mut host_mem = hal.host_alloc(3 * (1 << n_vars));
		let host_alloc = HostBumpAllocator::<F>::new(host_mem.as_mut());

		let evals_1 = host_alloc.alloc(1 << n_vars).unwrap();
		let evals_2 = host_alloc.alloc(1 << n_vars).unwrap();
		let evals_3 = host_alloc.alloc(1 << n_vars).unwrap();

		evals_1.fill_with(|| <F as Field>::random(&mut rng));
		evals_2.fill_with(|| <F as Field>::random(&mut rng));
		evals_3.fill_with(|| <F as Field>::random(&mut rng));

		let dev_alloc = BumpAllocator::<F, Hal::DevMem>::new(dev_mem);
		let mut evals_1_dev = dev_alloc.alloc(1 << n_vars).unwrap();
		let mut evals_2_dev = dev_alloc.alloc(1 << n_vars).unwrap();
		let mut evals_3_dev = dev_alloc.alloc(1 << n_vars).unwrap();
		hal.copy_h2d(evals_1, &mut evals_1_dev).unwrap();
		hal.copy_h2d(evals_2, &mut evals_2_dev).unwrap();
		hal.copy_h2d(evals_3, &mut evals_3_dev).unwrap();

		let mle_1 = MultilinearExtension::new(
			n_vars,
			must_cast_slice::<_, PackedBinaryField1x128b>(evals_1),
		)
		.unwrap();
		let mle_2 = MultilinearExtension::new(
			n_vars,
			must_cast_slice::<_, PackedBinaryField1x128b>(evals_2),
		)
		.unwrap();
		let mle_3 = MultilinearExtension::new(
			n_vars,
			must_cast_slice::<_, PackedBinaryField1x128b>(evals_3),
		)
		.unwrap();

		let multilins = [mle_1, mle_2, mle_3]
			.into_iter()
			.map(MLEDirectAdapter::from)
			.collect::<Vec<_>>();

		let batch_coeff = <BinaryField128b as Field>::random(&mut rng);

		let indexed_compositions = [
			IndexComposition::new(3, [0, 1], BivariateProduct::default()).unwrap(),
			IndexComposition::new(3, [1, 2], BivariateProduct::default()).unwrap(),
		];

		let sums = indexed_compositions
			.iter()
			.map(|composition| compute_composite_sum(&multilins, composition))
			.collect::<Vec<_>>();
		let sum = inner_product_unchecked(powers(batch_coeff), sums.iter().copied());

		let result = calculate_round_evals(
			hal,
			n_vars,
			batch_coeff,
			&[
				Hal::DevMem::as_const(&evals_1_dev),
				Hal::DevMem::as_const(&evals_2_dev),
				Hal::DevMem::as_const(&evals_3_dev),
			],
			&indexed_compositions,
		);
		assert!(result.is_ok());
		let evals = result.unwrap();
		assert_eq!(evals.len(), 2);

		let interpolation_domain = InterpolationDomain::from(
			EvaluationDomain::<BinaryField1b>::from_points(
				vec![BinaryField1b::ZERO, BinaryField1b::ONE],
				true,
			)
			.expect("domain is valid"),
		);

		let evals = [sum - evals[0], evals[0], evals[1]];
		let coeffs = interpolation_domain.interpolate(&evals).unwrap();

		let backend = make_portable_backend();
		let mut prover = RegularSumcheckProver::<BinaryField1b, _, _, _, _>::new(
			EvaluationOrder::HighToLow,
			multilins,
			iter::zip(indexed_compositions, sums)
				.map(|(composition, sum)| CompositeSumClaim { composition, sum }),
			DefaultEvaluationDomainFactory::default(),
			immediate_switchover_heuristic,
			&backend,
		)
		.unwrap();
		let expected_coeffs = prover.execute(batch_coeff).unwrap();

		assert_eq!(coeffs, expected_coeffs.0);
	}

	#[test]
	fn test_calculate_round_evals() {
		type Hal = CpuLayer<CanonicalTowerFamily>;
		type F = BinaryField128b;

		let hal = Hal::default();
		let mut dev_mem = vec![F::ZERO; 1 << 10];
		generic_test_calculate_round_evals(&hal, &mut dev_mem)
	}
}
