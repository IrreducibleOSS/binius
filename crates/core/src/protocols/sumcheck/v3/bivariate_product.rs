// Copyright 2025 Irreducible Inc.

use std::{iter, slice};

use binius_compute::{
	ComputeLayer, ComputeMemory, FSlice, KernelBuffer, KernelMemMap, SizedSlice,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
};
use binius_field::{Field, TowerField, util::powers};
use binius_math::{ArithExpr, CompositionPoly, EvaluationOrder, evaluate_univariate};
use binius_utils::bail;

use crate::{
	composition::{BivariateProduct, IndexComposition},
	protocols::sumcheck::{
		CompositeSumClaim, Error, RoundCoeffs, SumcheckClaim, prove::SumcheckProver,
	},
};

/// Sumcheck prover implementation for the special case of bivariate product compositions over
/// large-field multilinears.
///
/// This implements the [`SumcheckProver`] interface. The implementation uses a [`ComputeLayer`]
/// instance for expensive operations and the input multilinears are provided as device memory
/// slices.
pub struct BivariateSumcheckProver<'a, 'alloc, F: Field, Hal: ComputeLayer<F>> {
	hal: &'a Hal,
	dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
	host_alloc: &'a HostBumpAllocator<'a, F>,
	n_vars_initial: usize,
	n_vars_remaining: usize,
	multilins: Vec<SumcheckMultilinear<'a, F, Hal::DevMem>>,
	compositions: Vec<IndexComposition<BivariateProduct, 2>>,
	last_coeffs_or_sums: PhaseState<F>,
}

impl<'a, 'alloc, F, Hal> BivariateSumcheckProver<'a, 'alloc, F, Hal>
where
	F: TowerField,
	Hal: ComputeLayer<F>,
{
	pub fn new(
		hal: &'a Hal,
		dev_alloc: &'a BumpAllocator<'alloc, F, Hal::DevMem>,
		host_alloc: &'a HostBumpAllocator<'a, F>,
		claim: &SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
		multilins: Vec<FSlice<'a, F, Hal>>,
	) -> Result<Self, Error> {
		if Hal::DevMem::MIN_SLICE_LEN != 1 {
			todo!("support non-trivial minimum slice lengths");
		}

		let n_vars = claim.n_vars();

		// Check shape of multilinear witness inputs.
		assert_eq!(claim.n_multilinears(), multilins.len());
		for multilin in &multilins {
			if multilin.len() != 1 << n_vars {
				bail!(Error::NumberOfVariablesMismatch);
			}
		}

		// Wrap multilinear witness inputs as SumcheckMultilinears.
		let multilins = multilins
			.into_iter()
			.map(SumcheckMultilinear::PreFold)
			.collect();

		let (compositions, sums) = claim
			.composite_sums()
			.iter()
			.map(|CompositeSumClaim { composition, sum }| (composition.clone(), *sum))
			.unzip();

		Ok(Self {
			hal,
			dev_alloc,
			host_alloc,
			n_vars_initial: n_vars,
			n_vars_remaining: n_vars,
			multilins,
			compositions,
			last_coeffs_or_sums: PhaseState::InitialSums(sums),
		})
	}

	/// Returns the amount of host memory this sumcheck requires.
	pub fn required_host_memory(
		claim: &SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
	) -> usize {
		// In `finish()`, prover allocates a temporary host buffer for each of the fully folded
		// multilinear evaluations.
		claim.n_multilinears()
	}

	/// Returns the amount of device memory this sumcheck requires.
	pub fn required_device_memory(
		claim: &SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
	) -> usize {
		// In `fold()`, prover allocates device buffers for each of the folded multilinears. They
		// are each half of the size of the original multilinears.
		claim.n_multilinears() * (1 << (claim.n_vars() - 1))
	}
}

impl<'alloc, F, Hal> SumcheckProver<F> for BivariateSumcheckProver<'_, 'alloc, F, Hal>
where
	F: TowerField,
	Hal: ComputeLayer<F>,
{
	fn n_vars(&self) -> usize {
		self.n_vars_initial
	}

	fn evaluation_order(&self) -> EvaluationOrder {
		EvaluationOrder::HighToLow
	}

	fn execute(&mut self, batch_coeff: F) -> Result<RoundCoeffs<F>, Error> {
		let multilins = self
			.multilins
			.iter()
			.map(|multilin| multilin.const_slice())
			.collect::<Vec<_>>();
		let round_evals = calculate_round_evals(
			self.hal,
			self.n_vars_remaining,
			batch_coeff,
			&multilins,
			&self.compositions,
		)?;

		let batched_sum = match self.last_coeffs_or_sums {
			PhaseState::Coeffs(_) => {
				bail!(Error::ExpectedFold);
			}
			PhaseState::InitialSums(ref sums) => evaluate_univariate(sums, batch_coeff),
			PhaseState::BatchedSum(sum) => sum,
		};
		let round_coeffs = calculate_round_coeffs_from_evals(batched_sum, round_evals);
		self.last_coeffs_or_sums = PhaseState::Coeffs(round_coeffs.clone());
		println!("{:?}",round_coeffs);
		Ok(round_coeffs)
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		if self.n_vars_remaining == 0 {
			bail!(Error::ExpectedFinish);
		}

		// Update the stored multilinear sums.
		match self.last_coeffs_or_sums {
			PhaseState::Coeffs(ref coeffs) => {
				let new_sum = evaluate_univariate(&coeffs.0, challenge);
				self.last_coeffs_or_sums = PhaseState::BatchedSum(new_sum);
			}
			PhaseState::InitialSums(_) | PhaseState::BatchedSum(_) => {
				bail!(Error::ExpectedExecution);
			}
		}

		// Fold the multilinears
		let _ = self.hal.execute(|exec| {
			self.multilins = self
				.hal
				.map(exec, self.multilins.drain(..), |exec, multilin| {
					let folded_evals = match multilin {
						SumcheckMultilinear::PreFold(evals) => {
							debug_assert_eq!(evals.len(), 1 << self.n_vars_remaining);
							let (evals_0, evals_1) =
								Hal::DevMem::split_at(evals, 1 << (self.n_vars_remaining - 1));

							// Allocate new buffer for the folded evaluations and copy in evals_0.
							let mut folded_evals =
								self.dev_alloc.alloc(1 << (self.n_vars_remaining - 1))?;
							// This is kind of sketchy to do a copy without an execution context.
							self.hal.copy_d2d(evals_0, &mut folded_evals)?;

							self.hal.extrapolate_line(
								exec,
								&mut folded_evals,
								evals_1,
								challenge,
							)?;
							folded_evals
						}
						SumcheckMultilinear::PostFold(evals) => {
							debug_assert_eq!(evals.len(), 1 << self.n_vars_remaining);
							let (mut evals_0, evals_1) =
								Hal::DevMem::split_at_mut(evals, 1 << (self.n_vars_remaining - 1));
							self.hal.extrapolate_line(
								exec,
								&mut evals_0,
								Hal::DevMem::as_const(&evals_1),
								challenge,
							)?;
							evals_0
						}
					};
					Ok(SumcheckMultilinear::<F, Hal::DevMem>::PostFold(folded_evals))
				})?;

			Ok(Vec::new())
		})?;

		self.n_vars_remaining -= 1;
		Ok(())
	}

	fn finish(self: Box<Self>) -> Result<Vec<F>, Error> {
		match self.last_coeffs_or_sums {
			PhaseState::Coeffs(_) => {
				bail!(Error::ExpectedFold);
			}
			_ => match self.n_vars_remaining {
				0 => {}
				_ => bail!(Error::ExpectedExecution),
			},
		};

		// Copy the fully folded multilinear evaluations to the host.
		let buffer = self.host_alloc.alloc(self.multilins.len())?;
		for (multilin, dst_i) in iter::zip(self.multilins, &mut *buffer) {
			let vals = multilin.const_slice();
			debug_assert_eq!(vals.len(), 1);
			self.hal.copy_d2h(vals, slice::from_mut(dst_i))?;
		}
		Ok(buffer.to_vec())
	}
}

/// A multilinear polynomial that is being processed by a sumcheck prover.
#[derive(Debug, Clone)]
enum SumcheckMultilinear<'a, F, Mem: ComputeMemory<F>> {
	PreFold(Mem::FSlice<'a>),
	PostFold(Mem::FSliceMut<'a>),
}

impl<'a, F, Mem: ComputeMemory<F>> SumcheckMultilinear<'a, F, Mem> {
	fn const_slice(&self) -> Mem::FSlice<'_> {
		match self {
			Self::PreFold(slice) => Mem::narrow(slice),
			Self::PostFold(slice) => Mem::as_const(slice),
		}
	}
}

/// Calculates the evaluations of the products of pairs of partially specialized multilinear
/// polynomials for sumcheck.
///
/// This performs round evaluation for a special case sumcheck prover for a sumcheck over bivariate
/// products of multilinear polynomials, defined over the same field as the sumcheck challenges,
/// using high-to-low variable binding order.
///
/// In more detail, this function takes a slice of multilinear polynomials over a large field `F`
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
/// z_1 = \sum_{i=0}^{k-1} \alpha^i \sum_{v \in B_{m-1}} P_{t_{0,i}}(v || 1) P_{t_{1,i}}(v || 1)
/// \\\\
/// z_\infty = \sum_{i=0}^{k-1} \alpha^i \sum_{v \in B_{m-1}} P_{t_{0,i}}(v || \infty)
/// P_{t_{1,i}}(v || \infty)
/// $$
///
/// ## Returns
///
/// Returns the batched, summed evaluations at 1 and infinity.
pub fn calculate_round_evals<'a, F: TowerField, HAL: ComputeLayer<F>>(
	hal: &HAL,
	n_vars: usize,
	batch_coeff: F,
	multilins: &[FSlice<'a, F, HAL>],
	compositions: &[IndexComposition<BivariateProduct, 2>],
) -> Result<[F; 2], Error> {
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

fn calculate_round_coeffs_from_evals<F: Field>(sum: F, evals: [F; 2]) -> RoundCoeffs<F> {
	let [y_1, y_inf] = evals;
	let y_0 = sum - y_1;

	// P(X) = c_2 x² + c_1 x + c_0
	//
	// P(0) =                  c_0
	// P(1) = c_2    + c_1   + c_0
	// P(∞) = c_2

	let c_0 = y_0;
	let c_2 = y_inf;
	let c_1 = y_1 - c_0 - c_2;
	RoundCoeffs(vec![c_0, c_1, c_2])
}

#[derive(Debug)]
enum PhaseState<F: Field> {
	Coeffs(RoundCoeffs<F>),
	InitialSums(Vec<F>),
	BatchedSum(F),
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

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
	use binius_hash::groestl::Groestl256;
	use binius_math::{
		CompositionPoly, DefaultEvaluationDomainFactory, EvaluationDomain, EvaluationOrder,
		InterpolationDomain, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
		MultilinearQuery,
	};
	use bytemuck::{must_cast_slice, zeroed_vec};
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		composition::BivariateProduct,
		fiat_shamir::HasherChallenger,
		polynomial::MultilinearComposite,
		protocols::sumcheck::{
			BatchSumcheckOutput, CompositeSumClaim,
			front_loaded::BatchVerifier,
			immediate_switchover_heuristic,
			prove::{RegularSumcheckProver, SumcheckProver, front_loaded::BatchProver},
		},
		transcript::ProverTranscript,
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

	fn generic_test_bivariate_sumcheck_prove_verify<F, Hal>(hal: &Hal, dev_mem: FSliceMut<F, Hal>)
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
			dev_alloc.capacity()
				>= <BivariateSumcheckProver<F, Hal>>::required_device_memory(&claim)
		);

		let prover =
			BivariateSumcheckProver::new(hal, &dev_alloc, &host_alloc, &claim, dev_multilins)
				.unwrap();

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

	#[test]
	fn test_bivariate_sumcheck_prove_verify() {
		let hal = <CpuLayer<CanonicalTowerFamily>>::default();
		let mut dev_mem = zeroed_vec(1 << 12);
		generic_test_bivariate_sumcheck_prove_verify(&hal, &mut dev_mem)
	}
}
