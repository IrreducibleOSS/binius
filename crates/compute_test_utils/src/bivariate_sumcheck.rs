// Copyright 2025 Irreducible Inc.

use std::{
	iter::{self, repeat_with},
	slice,
};

use binius_compute::{
	ComputeLayer, ComputeMemory, FSliceMut,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
};
use binius_core::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::HasherChallenger,
	polynomial::MultilinearComposite,
	protocols::sumcheck::{
		BatchSumcheckOutput, CompositeSumClaim, SumcheckClaim,
		front_loaded::BatchVerifier,
		immediate_switchover_heuristic,
		prove::{RegularSumcheckProver, SumcheckProver, front_loaded::BatchProver},
		v3::bivariate_product::{BivariateSumcheckProver, calculate_round_evals},
	},
	transcript::ProverTranscript,
};
use binius_field::{
	BinaryField1b, BinaryField128b, Field, PackedBinaryField1x128b, PackedField, TowerField,
	util::{inner_product_unchecked, powers},
};
use binius_hal::make_portable_backend;
use binius_hash::groestl::Groestl256;
use binius_math::{
	CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomain, EvaluationOrder,
	InterpolationDomain, MLEDirectAdapter, MultilinearExtension, MultilinearPoly, MultilinearQuery,
};
use bytemuck::must_cast_slice;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub fn generic_test_calculate_round_evals<Hal: ComputeLayer<BinaryField128b>>(
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

	let mle_1 =
		MultilinearExtension::new(n_vars, must_cast_slice::<_, PackedBinaryField1x128b>(evals_1))
			.unwrap();
	let mle_2 =
		MultilinearExtension::new(n_vars, must_cast_slice::<_, PackedBinaryField1x128b>(evals_2))
			.unwrap();
	let mle_3 =
		MultilinearExtension::new(n_vars, must_cast_slice::<_, PackedBinaryField1x128b>(evals_3))
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

pub fn generic_test_bivariate_sumcheck_prove_verify<F, Hal>(hal: &Hal, dev_mem: FSliceMut<F, Hal>)
where
	F: TowerField,
	Hal: ComputeLayer<F>,
{
	let n_vars = 8;
	let n_multilins = 8;
	let n_compositions = 8;

	let mut rng = StdRng::seed_from_u64(0);

	let evals = repeat_with(|| {
		repeat_with(|| F::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>()
	})
	.take(n_multilins)
	.collect::<Vec<_>>();
	let compositions = repeat_with(|| {
		// Choose 2 distinct indices at random
		let idx0 = rng.gen_range(0..n_multilins);
		let idx1 = (idx0 + rng.gen_range(0..n_multilins - 1)) % n_multilins;
		IndexComposition::new(n_multilins, [idx0, idx1], BivariateProduct::default()).unwrap()
	})
	.take(n_compositions)
	.collect::<Vec<_>>();

	let multilins = evals
		.iter()
		.map(|evals| {
			let mle = MultilinearExtension::new(n_vars, evals.as_slice()).unwrap();
			MLEDirectAdapter::from(mle)
		})
		.collect::<Vec<_>>();
	let sums = compositions
		.iter()
		.map(|composition| compute_composite_sum(&multilins, composition))
		.collect::<Vec<_>>();

	let claim = SumcheckClaim::new(
		n_vars,
		n_multilins,
		iter::zip(compositions, sums)
			.map(|(composition, sum)| CompositeSumClaim { composition, sum })
			.collect(),
	)
	.unwrap();

	let mut host_mem =
		hal.host_alloc(<BivariateSumcheckProver<F, Hal>>::required_host_memory(&claim));

	let host_alloc = HostBumpAllocator::new(host_mem.as_mut());
	let dev_alloc = BumpAllocator::new(dev_mem);

	let dev_multilins = evals
		.iter()
		.map(|evals_i| {
			let mut dev_multilin = dev_alloc.alloc(evals_i.len()).unwrap();
			hal.copy_h2d(evals_i, &mut dev_multilin).unwrap();
			dev_multilin
		})
		.collect::<Vec<_>>();
	// TODO: into_const would be useful here
	let dev_multilins = dev_multilins
		.iter()
		.map(Hal::DevMem::as_const)
		.collect::<Vec<_>>();

	assert!(
		dev_alloc.capacity() >= <BivariateSumcheckProver<F, Hal>>::required_device_memory(&claim)
	);

	let prover =
		BivariateSumcheckProver::new(hal, &dev_alloc, &host_alloc, &claim, dev_multilins).unwrap();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let batch_prover = BatchProver::new(vec![prover], &mut transcript).unwrap();
	let _batch_prover_output = batch_prover.run(&mut transcript).unwrap();

	let mut transcript = transcript.into_verifier();
	let verifier = BatchVerifier::new(slice::from_ref(&claim), &mut transcript).unwrap();

	let BatchSumcheckOutput {
		mut challenges,
		mut multilinear_evals,
	} = verifier.run(&mut transcript).unwrap();

	assert_eq!(multilinear_evals.len(), 1);
	let multilinear_evals = multilinear_evals.pop().unwrap();

	challenges.reverse(); // Reverse challenges because of high-to-low variable binding
	let query = MultilinearQuery::expand(&challenges);
	for (multilin_i, eval) in iter::zip(multilins, multilinear_evals) {
		assert_eq!(multilin_i.evaluate(query.to_ref()).unwrap(), eval);
	}
}

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
