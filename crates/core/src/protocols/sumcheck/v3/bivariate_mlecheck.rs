// Copyright 2025 Irreducible Inc.

use std::{iter, mem, slice};

use binius_compute::{
	ComputeLayer, ComputeLayerExecutor, ComputeMemory, FSlice, KernelBuffer, KernelExecutor,
	KernelMemMap, SizedSlice, SlicesBatch, alloc::ComputeAllocator, cpu::CpuMemory,
};
use binius_field::{
	Field, TowerField,
	util::{eq, powers},
};
use binius_math::{ArithCircuit, CompositionPoly, EvaluationOrder, evaluate_univariate};
use binius_utils::bail;
use itertools::Itertools;

use super::bivariate_product::{PhaseState, SumcheckMultilinear};
use crate::{
	composition::{BivariateProduct, IndexComposition},
	protocols::sumcheck::{
		CompositeSumClaim, EqIndSumcheckClaim, Error, RoundCoeffs, prove::SumcheckProver,
	},
};

/// MLEcheck prover implementation for the special case of bivariate product compositions over
/// large-field multilinears.
///
/// This implements the [`SumcheckProver`] interface. The implementation uses a [`ComputeLayer`]
/// instance for expensive operations and the input multilinears are provided as device memory
/// slices.
pub struct BivariateMLEcheckProver<
	'a,
	F: Field,
	Hal: ComputeLayer<F>,
	HostAllocatorType,
	DeviceAllocatorType,
> where
	HostAllocatorType: ComputeAllocator<F, CpuMemory>,
	DeviceAllocatorType: ComputeAllocator<F, Hal::DevMem>,
{
	hal: &'a Hal,
	dev_alloc: &'a DeviceAllocatorType,
	host_alloc: &'a HostAllocatorType,
	n_vars_initial: usize,
	n_vars_remaining: usize,
	multilins: Vec<SumcheckMultilinear<'a, F, Hal::DevMem>>,
	compositions: Vec<IndexComposition<BivariateProduct, 2>>,
	last_coeffs_or_sums: PhaseState<F>,
	eq_ind_prefix_eval: F,
	// Wrapping it in Option is a temporary workaround for the lifetime problem
	eq_ind_partial_evals: Option<SumcheckMultilinear<'a, F, Hal::DevMem>>,
	eq_ind_challenges: Vec<F>,
}

impl<'a, F, Hal, HostAllocatorType, DeviceAllocatorType>
	BivariateMLEcheckProver<'a, F, Hal, HostAllocatorType, DeviceAllocatorType>
where
	F: TowerField,
	Hal: ComputeLayer<F>,
	HostAllocatorType: ComputeAllocator<F, CpuMemory>,
	DeviceAllocatorType: ComputeAllocator<F, Hal::DevMem>,
{
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		hal: &'a Hal,
		dev_alloc: &'a DeviceAllocatorType,
		host_alloc: &'a HostAllocatorType,
		claim: &EqIndSumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
		multilins: Vec<FSlice<'a, F, Hal>>,
		// Specify an existing tensor expansion for `eq_ind_challenges`. Avoids
		// duplicate work.
		eq_ind_partial_evals: FSlice<'a, F, Hal>,
		eq_ind_challenges: Vec<F>,
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
			.eq_ind_composite_sums()
			.iter()
			.map(|CompositeSumClaim { composition, sum }| (composition.clone(), *sum))
			.unzip();

		// Only one value of the expanded equality indicator is used per each
		// 1-variable subcube, thus it should be twice smaller.
		if eq_ind_partial_evals.len() != 1 << n_vars.saturating_sub(1) {
			bail!(Error::IncorrectEqIndPartialEvalsSize);
		}

		let eq_ind_partial_evals_buffer = SumcheckMultilinear::PreFold(eq_ind_partial_evals);

		Ok(Self {
			hal,
			dev_alloc,
			host_alloc,
			n_vars_initial: n_vars,
			n_vars_remaining: n_vars,
			multilins,
			compositions,
			last_coeffs_or_sums: PhaseState::InitialSums(sums),
			eq_ind_prefix_eval: F::ONE,
			eq_ind_partial_evals: Some(eq_ind_partial_evals_buffer),
			eq_ind_challenges,
		})
	}

	fn update_eq_ind_prefix_eval(&mut self, challenge: F) {
		// Update the running eq ind evaluation.
		self.eq_ind_prefix_eval *= eq(self.eq_ind_challenges[self.n_vars_remaining - 1], challenge);
	}

	/// Returns the amount of host memory this sumcheck requires.
	pub fn required_host_memory(
		claim: &EqIndSumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
	) -> usize {
		// In `finish()`, the prover allocates a temporary host buffer for each of the fully folded
		// multilinear evaluations, plus `eq_ind` evaluations, which will be appended at the end.
		claim.n_multilinears() + 1
	}

	/// Returns the amount of device memory this sumcheck requires.
	pub fn required_device_memory(
		claim: &EqIndSumcheckClaim<F, IndexComposition<BivariateProduct, 2>>,
		with_eq_ind_partial_evals: bool,
	) -> usize {
		// In `fold()`, the prover allocates device buffers for each of the folded multilinears,
		// plus for `eq_ind`. Each of them is half the size of the original multilinears.
		let n_multilinears = claim.n_multilinears() + if with_eq_ind_partial_evals { 0 } else { 1 };
		n_multilinears * (1 << (claim.n_vars() - 1))
	}

	pub fn fold_multilinears(&mut self, challenge: F) -> Result<(), Error> {
		use binius_compute::{FSlice, FSliceMut};

		type PreparedExtrapolateLineArgs<'a, F, Hal> = (FSliceMut<'a, F, Hal>, FSlice<'a, F, Hal>);

		let prepared_extrapolate_line_ops = self
			.multilins
			.drain(..)
			.map(
				|multilin| -> Result<
					PreparedExtrapolateLineArgs<'a, F, Hal>,
					binius_compute::Error,
				> {
					match multilin {
						SumcheckMultilinear::PreFold(evals) => {
							debug_assert_eq!(evals.len(), 1 << self.n_vars_remaining);
							let (evals_0, evals_1) =
								Hal::DevMem::split_half(evals);

							// Allocate new buffer for the folded evaluations and copy in evals_0.
							let mut folded_evals =
								self.dev_alloc.alloc(1 << (self.n_vars_remaining - 1))?;
							self.hal.copy_d2d(evals_0, &mut folded_evals)?;
							Ok((folded_evals, evals_1))
						}
						SumcheckMultilinear::PostFold(evals) => {
							debug_assert_eq!(evals.len(), 1 << self.n_vars_remaining);
							let (evals_0, evals_1) =
								Hal::DevMem::split_half_mut(evals);
							Ok((evals_0, Hal::DevMem::to_const(evals_1)))
						}
					}
				},
			)
			.collect_vec();
		self.hal.execute(|exec| {
			self.multilins = exec.map(
				prepared_extrapolate_line_ops.into_iter(),
				|exec, extrapolate_line_args| {
					let (mut evals_0, evals_1) = extrapolate_line_args?;
					exec.extrapolate_line(&mut evals_0, evals_1, challenge)?;
					Ok(SumcheckMultilinear::<F, Hal::DevMem>::PostFold(evals_0))
				},
			)?;

			Ok(Vec::new())
		})?;
		Ok(())
	}

	pub fn fold_eq_ind(&mut self) -> Result<(), Error> {
		let eq_ind_partial_evals = mem::take(&mut self.eq_ind_partial_evals).expect("exist");

		let split_n_vars = self.n_vars_remaining - 2;

		let (mut evals_0, evals_1) = match eq_ind_partial_evals {
			SumcheckMultilinear::PreFold(evals) => {
				let (evals_0, evals_1) = Hal::DevMem::split_half(evals);

				let mut buffer = self.dev_alloc.alloc(evals_0.len())?;
				self.hal.copy_d2d(evals_0, &mut buffer)?;

				(buffer, evals_1)
			}
			SumcheckMultilinear::PostFold(evals) => {
				let (evals_0, evals_1) = Hal::DevMem::split_half_mut(evals);

				let evals_1 = Hal::DevMem::to_const(evals_1);

				(evals_0, evals_1)
			}
		};

		let kernel_mappings = vec![
			KernelMemMap::ChunkedMut {
				data: Hal::DevMem::to_owned_mut(&mut evals_0),
				log_min_chunk_size: 0,
			},
			KernelMemMap::Chunked {
				data: Hal::DevMem::narrow(&evals_1),
				log_min_chunk_size: 0,
			},
		];

		let _ = self.hal.execute(|exec| {
			exec.map_kernels(
				|local_exec, log_chunks, mut buffers| {
					let log_chunk_size = split_n_vars - log_chunks;

					let Ok([KernelBuffer::Mut(evals_0), KernelBuffer::Ref(evals_1)]) =
						TryInto::<&mut [_; 2]>::try_into(buffers.as_mut_slice())
					else {
						panic!(
							"exec_kernels did not create the mapped buffers struct according to the mapping"
						);
					};
					local_exec.add_assign(log_chunk_size, *evals_1, evals_0)?;

					Ok(())
				},
				kernel_mappings,
			)?;

			Ok(Vec::new())
		})?;

		self.eq_ind_partial_evals = Some(SumcheckMultilinear::PostFold(evals_0));

		Ok(())
	}
}

impl<F, Hal, HostAllocatorType, DeviceAllocatorType> SumcheckProver<F>
	for BivariateMLEcheckProver<'_, F, Hal, HostAllocatorType, DeviceAllocatorType>
where
	F: TowerField,
	Hal: ComputeLayer<F>,
	HostAllocatorType: ComputeAllocator<F, CpuMemory>,
	DeviceAllocatorType: ComputeAllocator<F, Hal::DevMem>,
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
			self.eq_ind_partial_evals
				.as_ref()
				.expect("eq_ind_partial_evals not None")
				.const_slice(),
			&self.compositions,
		)?;

		let batched_sum = match self.last_coeffs_or_sums {
			PhaseState::Coeffs(_) => {
				bail!(Error::ExpectedFold);
			}
			PhaseState::InitialSums(ref sums) => evaluate_univariate(sums, batch_coeff),
			PhaseState::BatchedSum(sum) => sum,
		};

		let alpha = self.eq_ind_challenges[self.n_vars_remaining - 1];

		let prime_coeffs = calculate_round_coeffs_from_evals(batched_sum, round_evals, alpha);

		self.last_coeffs_or_sums = PhaseState::Coeffs(prime_coeffs.clone());

		// Convert v' polynomial into v polynomial
		// eq(X, α) = (1 − α) + (2 α − 1) X
		let prime_coeffs_scaled_by_constant_term = prime_coeffs.clone() * (F::ONE - alpha);

		let mut prime_coeffs_scaled_by_linear_term = prime_coeffs * (alpha.double() - F::ONE);

		prime_coeffs_scaled_by_linear_term.0.insert(0, F::ZERO); // Multiply prime polynomial by X

		let coeffs = (prime_coeffs_scaled_by_constant_term + &prime_coeffs_scaled_by_linear_term)
			* self.eq_ind_prefix_eval;

		Ok(coeffs)
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

		self.update_eq_ind_prefix_eval(challenge);

		self.fold_multilinears(challenge)?;

		if self.n_vars_remaining - 1 != 0 {
			self.fold_eq_ind()?;
		}

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

		let mut res = buffer.to_vec();

		res.push(self.eq_ind_prefix_eval);

		Ok(res)
	}
}

fn calculate_round_coeffs_from_evals<F: Field>(sum: F, evals: [F; 2], alpha: F) -> RoundCoeffs<F> {
	let [y_1, y_inf] = evals;

	let y_0 = (sum - y_1 * alpha) * (F::ONE - alpha).invert_or_zero();

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

fn calculate_round_evals<'a, F: TowerField, Hal: ComputeLayer<F>>(
	hal: &Hal,
	n_vars: usize,
	batch_coeff: F,
	multilins: &[FSlice<'a, F, Hal>],
	eq_ind_partial_evals: FSlice<'a, F, Hal>,
	compositions: &[IndexComposition<BivariateProduct, 2>],
) -> Result<[F; 2], Error> {
	let prod_evaluators = compositions
		.iter()
		.map(|composition| {
			let mut prod_expr = CompositionPoly::<F>::expression(&composition);
			// add eq_ind
			prod_expr *= ArithCircuit::var(multilins.len());

			hal.compile_expr(&prod_expr)
		})
		.collect::<Result<Vec<_>, _>>()?;

	// n_vars - 1 is the number of variables in the halves of the split multilinear.
	let split_n_vars = n_vars - 1;
	let mut kernel_mappings = multilins
		.iter()
		.copied()
		.flat_map(|multilin| {
			let (lo_half, hi_half) = Hal::DevMem::split_half(multilin);
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
		.collect::<Vec<_>>();

	kernel_mappings.push(KernelMemMap::Chunked {
		data: eq_ind_partial_evals,
		log_min_chunk_size: 0,
	});

	let batch_coeffs = powers(batch_coeff)
		.take(compositions.len())
		.collect::<Vec<_>>();

	let evals = hal.execute(|exec| {
		exec.accumulate_kernels(
			|local_exec, log_chunks, mut buffers| {
				let log_chunk_size = split_n_vars - log_chunks;

				let eq_ind = buffers.pop().expect(
					"The presence of eq_ind in the buffer is due to it being added earlier in the code.",
				);

				// Compute the composite evaluations at the point ONE.
				let mut acc_1 = local_exec.decl_value(F::ZERO)?;
				{
					let mut eval_1s_with_eq_ind = (0..multilins.len())
						.map(|i| buffers[i * 3 + 1].to_ref())
						.collect::<Vec<_>>();

					eval_1s_with_eq_ind.push(eq_ind.to_ref());

					let eval_1s_with_eq_ind =
						SlicesBatch::new(eval_1s_with_eq_ind, 1 << log_chunk_size);

					for (&batch_coeff, evaluator) in iter::zip(&batch_coeffs, &prod_evaluators) {
						local_exec.sum_composition_evals(
							&eval_1s_with_eq_ind,
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
				let mut eval_infs_with_eq_ind = (0..multilins.len())
					.map(|i| buffers[i * 3 + 2].to_ref())
					.collect::<Vec<_>>();

				eval_infs_with_eq_ind.push(eq_ind.to_ref());

				let eval_infs_with_eq_ind =
					SlicesBatch::new(eval_infs_with_eq_ind, 1 << log_chunk_size);

				for (&batch_coeff, evaluator) in iter::zip(&batch_coeffs, &prod_evaluators) {
					local_exec.sum_composition_evals(
						&eval_infs_with_eq_ind,
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
	use binius_compute::cpu::layer::CpuLayerHolder;
	use binius_core_test_utils::bivariate_sumcheck::generic_test_bivariate_mlecheck_prove_verify;
	use binius_fast_compute::layer::FastCpuLayerHolder;
	use binius_field::{
		arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
	};
	use binius_math::B128;

	#[test]
	fn test_bivariate_mlecheck_prove_verify() {
		let compute_holder = CpuLayerHolder::<B128>::new(1 << 13, 1 << 12);
		let n_vars = 8;
		let n_multilins = 8;
		let n_compositions = 8;
		generic_test_bivariate_mlecheck_prove_verify(
			compute_holder,
			n_vars,
			n_multilins,
			n_compositions,
		);
	}

	#[test]
	fn test_bivariate_mlecheck_prove_verify_fast() {
		type F = B128;
		type P = PackedType<OptimalUnderlier, F>;

		let compute_holder = FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << 13, 1 << 12);
		let n_vars = 8;
		let n_multilins = 8;
		let n_compositions = 8;
		generic_test_bivariate_mlecheck_prove_verify(
			compute_holder,
			n_vars,
			n_multilins,
			n_compositions,
		);
	}
}
