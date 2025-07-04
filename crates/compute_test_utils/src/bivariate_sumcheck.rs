// Copyright 2025 Irreducible Inc.

use std::{
	iter::{self, repeat_with},
	slice,
};

use binius_compute::{
	ComputeData, ComputeHolder, ComputeLayer, ComputeMemory,
	alloc::{ComputeAllocator, HostBumpAllocator},
	ops::eq_ind_partial_eval,
};
use binius_core::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::HasherChallenger,
	polynomial::MultilinearComposite,
	protocols::sumcheck::{
		self, BatchSumcheckOutput, CompositeSumClaim, EqIndSumcheckClaim, SumcheckClaim,
		eq_ind::reduce_to_regular_sumchecks,
		front_loaded::BatchVerifier,
		immediate_switchover_heuristic,
		prove::{RegularSumcheckProver, SumcheckProver, front_loaded::BatchProver},
		v3::{
			bivariate_mlecheck::BivariateMLEcheckProver,
			bivariate_product::{BivariateSumcheckProver, calculate_round_evals},
		},
	},
	transcript::ProverTranscript,
};
use binius_field::{
	BinaryField1b, BinaryField8b, BinaryField128b, ExtensionField, Field, PackedBinaryField1x128b,
	PackedExtension, PackedField, TowerField,
	util::{inner_product_unchecked, powers},
};
use binius_hal::{SumcheckMultilinear, make_portable_backend};
use binius_hash::groestl::Groestl256;
use binius_math::{
	CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomain, EvaluationOrder,
	InterpolationDomain, MLEDirectAdapter, MultilinearExtension, MultilinearPoly, MultilinearQuery,
};
use bytemuck::must_cast_slice;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub fn generic_test_calculate_round_evals<Hal: ComputeLayer<BinaryField128b>>(
	mut compute_holder: impl ComputeHolder<BinaryField128b, Hal>,
	n_vars: usize,
) {
	type F = BinaryField128b;

	let ComputeData {
		hal,
		dev_alloc,
		host_alloc,
		..
	} = compute_holder.to_data();

	let mut rng = StdRng::seed_from_u64(0);
	let evals_1 = host_alloc.alloc(1 << n_vars).unwrap();
	let evals_2 = host_alloc.alloc(1 << n_vars).unwrap();
	let evals_3 = host_alloc.alloc(1 << n_vars).unwrap();

	evals_1.fill_with(|| <F as Field>::random(&mut rng));
	evals_2.fill_with(|| <F as Field>::random(&mut rng));
	evals_3.fill_with(|| <F as Field>::random(&mut rng));

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
		.map(|composition| compute_composite_sum(&multilins, n_vars, composition))
		.collect::<Vec<_>>();
	let sum = inner_product_unchecked(powers(batch_coeff), sums.iter().copied());

	let evals = calculate_round_evals(
		hal,
		n_vars,
		batch_coeff,
		&[
			Hal::DevMem::as_const(&evals_1_dev),
			Hal::DevMem::as_const(&evals_2_dev),
			Hal::DevMem::as_const(&evals_3_dev),
		],
		&indexed_compositions,
	)
	.unwrap();
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

pub fn generic_test_bivariate_sumcheck_prove_verify<F, Hal, ComputeHolderType>(
	mut compute_holder: ComputeHolderType,
	n_vars: usize,
	n_multilins: usize,
	n_compositions: usize,
) where
	F: TowerField,
	Hal: ComputeLayer<F>,
	ComputeHolderType: ComputeHolder<F, Hal>,
{
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
		let idx0 = rng.random_range(0..n_multilins);
		let idx1 = (idx0 + rng.random_range(0..n_multilins - 1)) % n_multilins;
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
		.map(|composition| compute_composite_sum(&multilins, n_vars, composition))
		.collect::<Vec<_>>();

	let claim = SumcheckClaim::new(
		n_vars,
		n_multilins,
		iter::zip(compositions, sums)
			.map(|(composition, sum)| CompositeSumClaim { composition, sum })
			.collect(),
	)
	.unwrap();

	let compute_data = compute_holder.to_data();
	let ComputeData {
		hal,
		mut dev_alloc,
		mut host_alloc,
		..
	} = compute_data;
	let host_alloc = host_alloc.subscope_allocator();
	let dev_alloc = dev_alloc.subscope_allocator();

	let claim_req_mem = <BivariateSumcheckProver<
		F,
		Hal,
		ComputeHolderType::DeviceComputeAllocator<'_>,
		ComputeHolderType::HostComputeAllocator<'_>,
	>>::required_host_memory(&claim);
	let max_eval_len = evals.iter().map(|elem| elem.len()).max().unwrap();
	let host_mem_size = usize::max(claim_req_mem, max_eval_len);
	let host_mem = host_alloc.alloc(host_mem_size).unwrap();

	let dev_multilins = evals
		.iter()
		.map(|evals_i| {
			let mut dev_multilin = dev_alloc.alloc(evals_i.len()).unwrap();
			let host_alloc = HostBumpAllocator::new(host_mem);
			let host_evals_slice = host_alloc.alloc(evals_i.len()).unwrap();
			host_evals_slice.as_mut().copy_from_slice(evals_i);
			hal.copy_h2d(host_evals_slice, &mut dev_multilin).unwrap();
			dev_multilin
		})
		.collect::<Vec<_>>();
	// TODO: to_const would be useful here
	let dev_multilins = dev_multilins
		.iter()
		.map(Hal::DevMem::as_const)
		.collect::<Vec<_>>();

	assert!(
		dev_alloc.capacity()
			>= <BivariateSumcheckProver<
				F,
				Hal,
				ComputeHolderType::DeviceComputeAllocator<'_>,
				ComputeHolderType::HostComputeAllocator<'_>,
			>>::required_device_memory(&claim)
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
	n_vars: usize,
	composition: Composition,
) -> P::Scalar
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Composition: CompositionPoly<P>,
{
	for multilinear in multilinears {
		assert_eq!(multilinear.n_vars(), n_vars);
	}

	let multilinears = multilinears.iter().collect::<Vec<_>>();
	let witness = MultilinearComposite::new(n_vars, composition, multilinears).unwrap();
	(0..(1 << n_vars))
		.map(|j| witness.evaluate_on_hypercube(j).unwrap())
		.sum()
}

fn evaluate_composite_at_point<F, M, Composition>(
	multilinears: &[M],
	n_vars: usize,
	composition: Composition,
	eval_point: &[F],
) -> F
where
	F: Field,
	M: MultilinearPoly<F> + Send + Sync,
	Composition: CompositionPoly<F>,
{
	for multilinear in multilinears {
		assert_eq!(multilinear.n_vars(), n_vars);
	}

	assert_eq!(eval_point.len(), n_vars);

	let multilinears = multilinears.iter().collect::<Vec<_>>();

	let eq_ind = MultilinearQuery::<F, _>::expand(eval_point);

	let multilinears = multilinears.iter().collect::<Vec<_>>();
	let witness = MultilinearComposite::new(n_vars, composition, multilinears.clone()).unwrap();
	(0..(1 << n_vars))
		.map(|j| witness.evaluate_on_hypercube(j).unwrap() * eq_ind.expansion()[j])
		.sum()
}

pub fn generic_test_bivariate_mlecheck_prove_verify<
	F,
	Hal,
	ComputeHolderType: ComputeHolder<F, Hal>,
>(
	mut compute_data: ComputeHolderType,
	n_vars: usize,
	n_multilins: usize,
	n_compositions: usize,
) where
	F: TowerField
		+ PackedField<Scalar = F>
		+ ExtensionField<BinaryField8b>
		+ PackedExtension<BinaryField8b>,
	Hal: ComputeLayer<F>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let evals = repeat_with(|| {
		repeat_with(|| <F as Field>::random(&mut rng))
			.take(1 << n_vars)
			.collect::<Vec<_>>()
	})
	.take(n_multilins)
	.collect::<Vec<_>>();

	let compositions = repeat_with(|| {
		// Choose 2 distinct indices at random
		let idx0 = rng.random_range(0..n_multilins);
		let idx1 = (idx0 + rng.random_range(0..n_multilins - 1)) % n_multilins;
		IndexComposition::new(n_multilins, [idx0, idx1], BivariateProduct::default()).unwrap()
	})
	.take(n_compositions)
	.collect::<Vec<_>>();

	let eq_ind_challenges = repeat_with(|| <F as Field>::random(&mut rng))
		.take(n_vars)
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
		.map(|composition| {
			evaluate_composite_at_point(&multilins, n_vars, composition, &eq_ind_challenges)
		})
		.collect::<Vec<_>>();

	let claim = EqIndSumcheckClaim::new(
		n_vars,
		n_multilins,
		iter::zip(compositions, sums)
			.map(|(composition, sum)| CompositeSumClaim { composition, sum })
			.collect(),
	)
	.unwrap();

	let sumcheck_multilinears = multilins
		.iter()
		.cloned()
		.map(|multilin| SumcheckMultilinear::transparent(multilin, &immediate_switchover_heuristic))
		.collect::<Vec<_>>();

	sumcheck::prove::eq_ind::validate_witness(
		n_vars,
		&sumcheck_multilinears,
		&eq_ind_challenges,
		claim.eq_ind_composite_sums().to_vec(),
	)
	.unwrap();

	let ComputeData {
		hal,
		dev_alloc,
		mut host_alloc,
		..
	} = compute_data.to_data();
	let host_alloc = host_alloc.subscope_allocator();

	let dev_multilins = evals
		.iter()
		.map(|evals_i| {
			let mut dev_multilin = dev_alloc.alloc(evals_i.len()).unwrap();
			hal.copy_h2d(evals_i, &mut dev_multilin).unwrap();
			Hal::DevMem::to_const(dev_multilin)
		})
		.collect::<Vec<_>>();

	assert!(
		dev_alloc.capacity()
			>= <BivariateMLEcheckProver<
				F,
				Hal,
				ComputeHolderType::HostComputeAllocator<'_>,
				ComputeHolderType::DeviceComputeAllocator<'_>,
			>>::required_device_memory(&claim, false)
	);

	let eq_ind_partial_evals =
		eq_ind_partial_eval(hal, &dev_alloc, &eq_ind_challenges[..n_vars.saturating_sub(1)])
			.unwrap();

	let prover = BivariateMLEcheckProver::new(
		hal,
		&dev_alloc,
		&host_alloc,
		&claim,
		dev_multilins,
		Hal::DevMem::as_const(&eq_ind_partial_evals),
		eq_ind_challenges,
	)
	.unwrap();

	let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();

	let batch_prover = BatchProver::new(vec![prover], &mut transcript).unwrap();
	let _batch_prover_output = batch_prover.run(&mut transcript).unwrap();

	let mut transcript = transcript.into_verifier();

	let verifier = BatchVerifier::new(
		&reduce_to_regular_sumchecks(slice::from_ref(&claim)).unwrap(),
		&mut transcript,
	)
	.unwrap();

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
