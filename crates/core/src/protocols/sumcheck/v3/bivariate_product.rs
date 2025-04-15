// Copyright 2025 Irreducible Inc.

use std::iter;

use binius_compute::{ComputeLayer, KernelBuffer, KernelMemMap};
use binius_field::{util::powers, TowerField};
use binius_math::ArithExpr;

use super::error::Error;
use crate::composition::{BivariateProduct, IndexComposition};

/// Returns the evaluations at 0 and infinity.
pub fn calculate_round_evals<'a, F: TowerField, HAL: ComputeLayer<F>>(
	hal: &HAL,
	n_vars: usize,
	batch_coeff: F,
	multilins: &[HAL::FSlice<'a>],
	compositions: &[IndexComposition<BivariateProduct, 2>],
) -> Result<[F; 2], Error> {
	// if n_vars is too small, transfer data and fall back to the CPU impl

	let prod_evaluators = compositions
		.iter()
		.map(|composition| {
			let [idx0, idx1] = composition.indices();
			let prod_expr = ArithExpr::Var(*idx0) * ArithExpr::Var(*idx1);
			hal.compile_expr(&prod_expr)
		})
		.collect::<Result<Vec<_>, _>>()?;

	// n_vars - 1 is the number of variables in the halves of the split multilinear.
	let split_n_vars = n_vars - 1;
	let kernel_mappings = multilins
		.iter()
		.copied()
		.flat_map(|multilin| {
			let (lo_half, hi_half) = HAL::split_at(multilin, 1 << split_n_vars);
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
				let eval_1s = (0..multilins.len())
					.map(|i| {
						let KernelBuffer::Ref(buffer) = &buffers[i * 3 + 1] else {
							panic!("exec_kernels did not create the mapped buffers struct according to the mapping")
						};
						*buffer
					})
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

				// Extrapolate the multilinear evaluations at the point Infinity.
				for group in buffers.chunks_mut(3) {
					let Ok(
						[KernelBuffer::Ref(evals_0), KernelBuffer::Ref(evals_1), KernelBuffer::Mut(evals_inf)],
					) = TryInto::<&mut [_; 3]>::try_into(group)
					else {
						panic!("exec_kernels did not create the mapped buffers struct according to the mapping");
					};
					hal.kernel_add(local_exec, log_chunk_size, *evals_0, *evals_1, evals_inf)?;
				}

				// Compute the composite evaluations at the point Infinity.
				let mut acc_inf = hal.kernel_decl_value(local_exec, F::ZERO)?;
				let eval_infs = (0..multilins.len())
					.map(|i| {
						let KernelBuffer::Mut(buffer) = &buffers[i * 3 + 2] else {
							panic!("exec_kernels did not create the mapped buffers struct according to the mapping")
						};
						HAL::as_const(buffer)
					})
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
	use std::iter::repeat_with;

	use binius_compute::{cpu::CpuLayer, tower::CanonicalTowerFamily};
	use binius_field::{
		util::inner_product_unchecked, BinaryField128b, BinaryField1b, Field,
		PackedBinaryField1x128b, PackedField, PackedFieldIndexable,
	};
	use binius_hal::make_portable_backend;
	use binius_math::{
		CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomain, EvaluationOrder,
		InterpolationDomain, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
	};
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;
	use crate::{
		composition::BivariateProduct,
		polynomial::MultilinearComposite,
		protocols::sumcheck::{
			immediate_switchover_heuristic,
			prove::{RegularSumcheckProver, SumcheckProver},
			CompositeSumClaim,
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

	#[test]
	fn test_calculate_round_evals() {
		type HAL = CpuLayer<CanonicalTowerFamily>;
		type F = BinaryField128b;

		let hal = HAL::default();
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let evals_1 = repeat_with(|| PackedBinaryField1x128b::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>();
		let evals_2 = repeat_with(|| PackedBinaryField1x128b::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>();
		let evals_3 = repeat_with(|| PackedBinaryField1x128b::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>();
		let mle_1 = MultilinearExtension::new(n_vars, evals_1.as_slice()).unwrap();
		let mle_2 = MultilinearExtension::new(n_vars, evals_2.as_slice()).unwrap();
		let mle_3 = MultilinearExtension::new(n_vars, evals_3.as_slice()).unwrap();
		let multilins = [mle_1, mle_2, mle_3].map(MLEDirectAdapter::from).to_vec();

		let batch_coeff = <F as Field>::random(&mut rng);

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
			&hal,
			n_vars,
			batch_coeff,
			&[
				PackedFieldIndexable::unpack_scalars(&evals_1),
				PackedFieldIndexable::unpack_scalars(&evals_2),
				PackedFieldIndexable::unpack_scalars(&evals_3),
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
}
