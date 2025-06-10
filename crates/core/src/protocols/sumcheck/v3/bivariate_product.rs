// Copyright 2025 Irreducible Inc.

use std::{iter, slice};

use binius_compute::{
	ComputeLayer, ComputeLayerExecutor, ComputeMemory, FSlice, KernelBuffer, KernelExecutor,
	KernelMemMap, SizedSlice, SlicesBatch,
	alloc::{BumpAllocator, ComputeAllocator, HostBumpAllocator},
};
use binius_field::{Field, TowerField, util::powers};
use binius_math::{CompositionPoly, EvaluationOrder, evaluate_univariate};
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

		// This is because the batched verifier reads a batched polynomial from the transcript with
		// max degree from all compositions being proven. If our compoisition is empty, we add a
		// degree 0 polynomial to whatever is already being written to the transcript by the batch
		// prover.
		if self.compositions.is_empty() {
			Ok(RoundCoeffs(vec![]))
		} else {
			Ok(round_coeffs)
		}
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
			self.multilins = exec.map(self.multilins.drain(..), |exec, multilin| {
				let folded_evals = match multilin {
					SumcheckMultilinear::PreFold(evals) => {
						debug_assert_eq!(evals.len(), 1 << self.n_vars_remaining);

						let (evals_0, evals_1) = Hal::DevMem::split_half(evals);
						// Allocate new buffer for the folded evaluations and copy in evals_0.
						let mut folded_evals =
							self.dev_alloc.alloc(1 << (self.n_vars_remaining - 1))?;
						// This is kind of sketchy to do a copy without an execution context.
						self.hal.copy_d2d(evals_0, &mut folded_evals)?;

						exec.extrapolate_line(&mut folded_evals, evals_1, challenge)?;
						folded_evals
					}
					SumcheckMultilinear::PostFold(evals) => {
						debug_assert_eq!(evals.len(), 1 << self.n_vars_remaining);
						let (mut evals_0, evals_1) = Hal::DevMem::split_half_mut(evals);
						exec.extrapolate_line(
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
pub enum SumcheckMultilinear<'a, F, Mem: ComputeMemory<F>> {
	PreFold(Mem::FSlice<'a>),
	PostFold(Mem::FSliceMut<'a>),
}

impl<'a, F, Mem: ComputeMemory<F>> SumcheckMultilinear<'a, F, Mem> {
	pub fn const_slice(&self) -> Mem::FSlice<'_> {
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
		.map(|composition| hal.compile_expr(&CompositionPoly::<F>::expression(&composition)))
		.collect::<Result<Vec<_>, _>>()?;

	// n_vars - 1 is the number of variables in the halves of the split multilinear.
	let split_n_vars = n_vars - 1;
	let kernel_mappings = multilins
		.iter()
		.copied()
		.flat_map(|multilin| {
			let (lo_half, hi_half) = HAL::DevMem::split_half(multilin);
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
		exec.accumulate_kernels(
			|local_exec, log_chunks, mut buffers| {
				let log_chunk_size = split_n_vars - log_chunks;

				// Compute the composite evaluations at the point ONE.
				let mut acc_1 = local_exec.decl_value(F::ZERO)?;
				{
					let eval_1s = SlicesBatch::new(
						(0..multilins.len())
							.map(|i| buffers[i * 3 + 1].to_ref())
							.collect(),
						1 << log_chunk_size,
					);
					for (&batch_coeff, evaluator) in iter::zip(&batch_coeffs, &prod_evaluators) {
						local_exec.sum_composition_evals(
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
					local_exec.add(log_chunk_size, *evals_0, *evals_1, evals_inf)?;
				}

				// Compute the composite evaluations at the point Infinity.
				let mut acc_inf = local_exec.decl_value(F::ZERO)?;
				let eval_infs = SlicesBatch::new(
					(0..multilins.len())
						.map(|i| buffers[i * 3 + 2].to_ref())
						.collect(),
					1 << log_chunk_size,
				);
				for (&batch_coeff, evaluator) in iter::zip(&batch_coeffs, &prod_evaluators) {
					local_exec.sum_composition_evals(
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
pub enum PhaseState<F: Field> {
	Coeffs(RoundCoeffs<F>),
	InitialSums(Vec<F>),
	BatchedSum(F),
}

#[cfg(test)]
mod tests {
	use binius_compute::cpu::{CpuLayer, alloc::CpuComputeAllocator};
	use binius_compute_test_utils::bivariate_sumcheck::{
		generic_test_bivariate_sumcheck_prove_verify, generic_test_calculate_round_evals,
	};
	use binius_fast_compute::layer::FastCpuLayer;
	use binius_field::{
		BinaryField128b, PackedField, arch::OptimalUnderlier, as_packed_field::PackedType,
		tower::CanonicalTowerFamily,
	};
	use binius_math::B128;
	use bytemuck::zeroed_vec;

	use super::*;

	#[test]
	fn test_calculate_round_evals() {
		type Hal = CpuLayer<B128>;

		let hal = Hal::default();
		let mut dev_mem = vec![B128::ZERO; 1 << 10];
		let mut host_allocator = CpuComputeAllocator::new(dev_mem.len() * 2);
		let n_vars = 8;
		generic_test_calculate_round_evals(&hal, &mut dev_mem, &host_allocator.into_inner(), n_vars)
	}

	#[test]
	fn test_calculate_round_evals_fast_cpu() {
		type F = BinaryField128b;
		type Packed = PackedType<OptimalUnderlier, F>;
		type Hal = FastCpuLayer<CanonicalTowerFamily, Packed>;

		let hal = Hal::default();
		let mut dev_mem = vec![Packed::zero(); 1 << (10 - Packed::LOG_WIDTH)];
		let dev_mem = <<Hal as ComputeLayer<F>>::DevMem as ComputeMemory<F>>::FSliceMut::new_slice(
			&mut dev_mem,
		);
		let n_vars = 8;
		let mut host_allocator = CpuComputeAllocator::new(dev_mem.len() * 2);
		generic_test_calculate_round_evals(&hal, dev_mem, &host_allocator.into_inner(), n_vars)
	}

	#[test]
	fn test_bivariate_sumcheck_prove_verify() {
		let hal = <CpuLayer<B128>>::default();
		let mut dev_mem = zeroed_vec(1 << 12);
		let n_vars = 8;
		let n_multilins = 8;
		let n_compositions = 8;
		let mut host_allocator = CpuComputeAllocator::new(dev_mem.len() * 2);
		generic_test_bivariate_sumcheck_prove_verify(
			&hal,
			&mut dev_mem,
			&host_allocator.into_inner(),
			n_vars,
			n_multilins,
			n_compositions,
		)
	}

	#[test]
	fn test_bivariate_sumcheck_prove_verify_fast() {
		type F = BinaryField128b;
		type Packed = PackedType<OptimalUnderlier, F>;
		type Hal = FastCpuLayer<CanonicalTowerFamily, Packed>;

		let hal = Hal::default();
		let mut dev_mem = zeroed_vec(1 << (12 - Packed::LOG_WIDTH));
		let dev_mem = <<Hal as ComputeLayer<F>>::DevMem as ComputeMemory<F>>::FSliceMut::new_slice(
			&mut dev_mem,
		);
		let mut host_allocator = CpuComputeAllocator::new(dev_mem.len() * 2);
		let n_vars = 8;
		let n_multilins = 8;
		let n_compositions = 8;
		generic_test_bivariate_sumcheck_prove_verify(
			&hal,
			dev_mem,
			&host_allocator.into_inner(),
			n_vars,
			n_multilins,
			n_compositions,
		)
	}
}
