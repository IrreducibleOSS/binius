// Copyright 2024-2025 Irreducible Inc.

//! Functions that calculate the sumcheck round evaluations.
//!
//! This is one of the core computational tasks in the sumcheck proving algorithm.

use std::iter;

use binius_field::{Field, PackedExtension, PackedField, PackedSubfield};
use binius_math::{
	extrapolate_lines, CompositionPoly, EvaluationOrder, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly, MultilinearQuery, MultilinearQueryRef,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use itertools::{izip, Either, Itertools};
use stackalloc::stackalloc_with_iter;

use crate::{Error, RoundEvals, SumcheckEvaluator, SumcheckMultilinear};

trait SumcheckMultilinearAccess<P: PackedField> {
	/// The size of `Vec<P>` scratchspace used by [`subcube_evaluations`], if any.
	fn scratch_space_len(&self, subcube_vars: usize) -> Option<usize>;

	/// A way to obtain multilinear evaluations during sumcheck.
	///
	/// Sumcheck is conducted over boolean hypercubes represented in little endian form
	/// (faster strides correspond to lower variable indexes). Evaluation order may proceed both
	/// from lowest variable to highest, as well as in reverse. The $n$-dimensional evaluation
	/// hypercube can be split into subcubes of `subcube_vars` by substituting higher variables
	/// for the little endian binary representation of some index.
	///
	/// Assume $r$ round challenges have been sampled already. A sumcheck multilinear then
	/// is either an $n + r$-variate transparent where $r$ of its variables (lowest or highest)
	/// are to be projected onto round challenges, or $n$-variate folded multilinear where this
	/// projection has already taken place. It is further split into two $n-1$ variate subcubes
	/// by substituting 0 and 1 for its lowest or highest variable, depending on evaluation order.
	///
	/// Assuming `subcube_vars + index_vars = n-1` holds, we substitute the binary representation
	/// of `subcube_index` into higher indexed variables. Note that this sub-subcube ordering
	/// _does not_ depend on evaluation order.
	///
	/// Indexed subcube evaluations are written into `evals_0` and `evals_1` slices, where scalar
	/// order corresponds to the lower `P::LOG_WIDTH` variables of the `subcube_vars`-variate hypercube.
	///
	/// The method can potentially require a `&mut [P]` scratch space, whose length is given by a query
	/// to [`scratch_space_len`] and should be uniquely determined by `subcube_vars`.
	///
	/// ## Arguments
	///
	/// * `subcube_vars`  - the number of variables in the sub-subcube to evaluate over
	/// * `subcube_index` - the index of the subcube within the $n-1$-variate hypercube
	/// * `index_vars`    - number of bits in the `subcube_index`
	/// * `scratch_space` - optional scratch space
	/// * `evals_0`       - `subcube_vars`-variate hypercube with current variables substituted for 0
	/// * `evals_1`       - `subcube_vars`-variate hypercube with current variables substituted for 1
	#[allow(clippy::too_many_arguments)]
	fn subcube_evaluations<M: MultilinearPoly<P>>(
		&self,
		multilinear: &SumcheckMultilinear<P, M>,
		subcube_vars: usize,
		subcube_index: usize,
		index_vars: usize,
		scratch_space: Option<&mut [P]>,
		evals_0: &mut [P],
		evals_1: &mut [P],
	) -> Result<(), Error>;
}

/// Calculate the accumulated evaluations for an arbitrary sumcheck round.
///
/// See [`calculate_first_round_evals`] for an optimized version of this method
/// that works over small fields in the first round.
pub(crate) fn calculate_round_evals<FDomain, F, P, M, Evaluator, Composition>(
	evaluation_order: EvaluationOrder,
	n_vars: usize,
	tensor_query: Option<MultilinearQueryRef<P>>,
	multilinears: &[SumcheckMultilinear<P, M>],
	evaluators: &[Evaluator],
	finite_evaluation_points: &[FDomain],
) -> Result<Vec<RoundEvals<F>>, Error>
where
	FDomain: Field,
	F: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Sync,
	Evaluator: SumcheckEvaluator<P, Composition> + Sync,
	Composition: CompositionPoly<P>,
{
	let empty_query = MultilinearQuery::with_capacity(0);
	let tensor_query = tensor_query.unwrap_or_else(|| empty_query.to_ref());

	match evaluation_order {
		EvaluationOrder::LowToHigh => calculate_round_evals_with_access(
			n_vars,
			&LowToHighAccess { tensor_query },
			multilinears,
			evaluators,
			finite_evaluation_points,
		),
		EvaluationOrder::HighToLow => calculate_round_evals_with_access(
			n_vars,
			&HighToLowAccess { tensor_query },
			multilinears,
			evaluators,
			finite_evaluation_points,
		),
	}
}

fn calculate_round_evals_with_access<FDomain, F, P, M, Evaluator, Access, Composition>(
	n_vars: usize,
	access: &Access,
	multilinears: &[SumcheckMultilinear<P, M>],
	evaluators: &[Evaluator],
	nontrivial_evaluation_points: &[FDomain],
) -> Result<Vec<RoundEvals<F>>, Error>
where
	FDomain: Field,
	F: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Sync,
	Evaluator: SumcheckEvaluator<P, Composition> + Sync,
	Access: SumcheckMultilinearAccess<P> + Sync,
	Composition: CompositionPoly<P>,
{
	assert!(n_vars > 0, "Computing round evaluations requires at least a single variable.");

	let n_multilinears = multilinears.len();
	let n_round_evals = evaluators
		.iter()
		.map(|evaluator| evaluator.eval_point_indices().len());

	/// Process batches of vertices in parallel, accumulating the round evaluations.
	const MAX_SUBCUBE_VARS: usize = 6;
	let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars - 1);

	// Compute the union of all evaluation point index ranges.
	let eval_point_indices = evaluators
		.iter()
		.map(|evaluator| evaluator.eval_point_indices())
		.reduce(|range1, range2| range1.start.min(range2.start)..range1.end.max(range2.end))
		.unwrap_or(0..0);

	// Check that finite evaluation points  are of correct length (accounted for 0, 1 & infinity point).
	if nontrivial_evaluation_points.len() != eval_point_indices.end.saturating_sub(3) {
		bail!(Error::IncorrectNontrivialEvalPointsLength);
	}

	let index_vars = n_vars - 1 - subcube_vars;
	let packed_accumulators = (0..1 << index_vars)
		.into_par_iter()
		.try_fold(
			|| ParFoldStates::new(access, n_multilinears, n_round_evals.clone(), subcube_vars),
			|mut par_fold_states, subcube_index| {
				let ParFoldStates {
					multilinear_evals,
					scratch_space,
					round_evals,
				} = &mut par_fold_states;

				for (multilinear, evals) in izip!(multilinears, multilinear_evals.iter_mut()) {
					access.subcube_evaluations(
						multilinear,
						subcube_vars,
						subcube_index,
						index_vars,
						scratch_space.as_deref_mut(),
						&mut evals.evals_0,
						&mut evals.evals_1,
					)?;
				}

				// Proceed by evaluation point first to share interpolation work between evaluators.
				for eval_point_index in eval_point_indices.clone() {
					// Infinity point requires special evaluation rules
					let is_infinity_point = eval_point_index == 2;

					// Multilinears are evaluated at a point t via linear interpolation:
					//   f(z, xs) = f(0, xs) + z * (f(1, xs) - f(0, xs))
					// The first three points are treated specially:
					//   index 0 - z = 0   => f(z, xs) = f(0, xs)
					//   index 1 - z = 1   => f(z, xs) = f(1, xs)
					//   index 2 = z = inf => f(inf, xs) = high (f(0, xs) + z * (f(1, xs) - f(0, xs))) =
					//                                   = f(1, xs) - f(0, xs)
					//   index 3 and above - remaining finite evaluation points
					let evals_z_iter =
						multilinear_evals
							.iter_mut()
							.map(|evals| match eval_point_index {
								0 => evals.evals_0.as_slice(),
								1 => evals.evals_1.as_slice(),
								2 => {
									// infinity point
									izip!(&mut evals.evals_z, &evals.evals_0, &evals.evals_1)
										.for_each(|(eval_z, &eval_0, &eval_1)| {
											*eval_z = eval_1 - eval_0;
										});

									evals.evals_z.as_slice()
								}
								3.. => {
									// Account for the gap occupied by the 0, 1 & infinity point
									let eval_point =
										nontrivial_evaluation_points[eval_point_index - 3];
									let eval_point_broadcast =
										<PackedSubfield<P, FDomain>>::broadcast(eval_point);

									izip!(&mut evals.evals_z, &evals.evals_0, &evals.evals_1)
										.for_each(|(eval_z, &eval_0, &eval_1)| {
											// This is logically the same as calling
											// `binius_math::univariate::extrapolate_line`, except that we do
											// not repeat the broadcast of the subfield element to a packed
											// subfield.
											*eval_z = P::cast_ext(extrapolate_lines(
												P::cast_base(eval_0),
												P::cast_base(eval_1),
												eval_point_broadcast,
											));
										});

									evals.evals_z.as_slice()
								}
							});

					stackalloc_with_iter(n_multilinears, evals_z_iter, |evals_z| {
						for (evaluator, round_evals) in
							iter::zip(evaluators, round_evals.iter_mut())
						{
							let eval_point_indices = evaluator.eval_point_indices();
							if !eval_point_indices.contains(&eval_point_index) {
								continue;
							}

							round_evals[eval_point_index - eval_point_indices.start] += evaluator
								.process_subcube_at_eval_point(
									subcube_vars,
									subcube_index,
									is_infinity_point,
									evals_z,
								);
						}
					});
				}

				Ok(par_fold_states)
			},
		)
		.map(|states: Result<ParFoldStates<P>, Error>| -> Result<_, Error> {
			Ok(states?.round_evals)
		})
		// Simply sum up the fold partitions.
		.try_reduce(
			|| {
				evaluators
					.iter()
					.map(|evaluator| vec![P::zero(); evaluator.eval_point_indices().len()])
					.collect()
			},
			|lhs, rhs| {
				let sum = iter::zip(lhs, rhs)
					.map(|(mut lhs_vals, rhs_vals)| {
						for (lhs_val, rhs_val) in lhs_vals.iter_mut().zip(rhs_vals) {
							*lhs_val += rhs_val;
						}
						lhs_vals
					})
					.collect();
				Ok(sum)
			},
		)?;

	let evals = packed_accumulators
		.into_iter()
		.map(|vals| {
			RoundEvals(
				vals.into_iter()
					// Truncate subcubes smaller than packing width.
					.map(|packed_val| packed_val.iter().take(1 << subcube_vars).sum())
					.collect(),
			)
		})
		.collect();

	Ok(evals)
}

// Evals of a single multilinear over a subcube, at 0/1 and some interpolated point.
#[derive(Debug)]
struct MultilinearEvals<P: PackedField> {
	evals_0: Vec<P>,
	evals_1: Vec<P>,
	evals_z: Vec<P>,
}

impl<P: PackedField> MultilinearEvals<P> {
	fn new(subcube_vars: usize) -> Self {
		let len = 1 << subcube_vars.saturating_sub(P::LOG_WIDTH);
		Self {
			evals_0: zeroed_vec(len),
			evals_1: zeroed_vec(len),
			evals_z: zeroed_vec(len),
		}
	}
}

/// Parallel fold state, consisting of scratch area and result accumulator.
#[derive(Debug)]
struct ParFoldStates<P: PackedField> {
	// Evaluations at 0, 1 and domain points, per MLE. Scratch space.
	multilinear_evals: Vec<MultilinearEvals<P>>,

	// Additional scratch space.
	scratch_space: Option<Vec<P>>,

	// Accumulated sums of evaluations over univariate domain.
	//
	// Each element of the outer vector corresponds to one composite polynomial. Each element of
	// an inner vector contains the evaluations at different points.
	round_evals: Vec<Vec<P>>,
}

impl<P: PackedField> ParFoldStates<P> {
	fn new(
		access: &impl SumcheckMultilinearAccess<P>,
		n_multilinears: usize,
		n_round_evals: impl Iterator<Item = usize>,
		subcube_vars: usize,
	) -> Self {
		Self {
			multilinear_evals: (0..n_multilinears)
				.map(|_| MultilinearEvals::new(subcube_vars))
				.collect(),
			scratch_space: access
				.scratch_space_len(subcube_vars)
				.map(|len| zeroed_vec(len)),
			round_evals: n_round_evals
				.map(|n_round_evals| zeroed_vec(n_round_evals))
				.collect(),
		}
	}
}

#[derive(Debug)]
struct LowToHighAccess<'a, P: PackedField> {
	tensor_query: MultilinearQueryRef<'a, P>,
}

impl<P: PackedField> SumcheckMultilinearAccess<P> for LowToHighAccess<'_, P> {
	fn scratch_space_len(&self, subcube_vars: usize) -> Option<usize> {
		// We need to sample evaluations at both 0 & 1 prior to deinterleaving, thus +1.
		Some(1 << (subcube_vars + 1).saturating_sub(P::LOG_WIDTH))
	}

	fn subcube_evaluations<M: MultilinearPoly<P>>(
		&self,
		multilinear: &SumcheckMultilinear<P, M>,
		subcube_vars: usize,
		subcube_index: usize,
		_index_vars: usize,
		scratch_space: Option<&mut [P]>,
		evals_0: &mut [P],
		evals_1: &mut [P],
	) -> Result<(), Error> {
		let Some(scratch_space) = scratch_space else {
			bail!(Error::NoScratchSpace);
		};

		if scratch_space.len() != 1 << (subcube_vars + 1).saturating_sub(P::LOG_WIDTH)
			|| evals_0.len() != 1 << subcube_vars.saturating_sub(P::LOG_WIDTH)
			|| evals_1.len() != 1 << subcube_vars.saturating_sub(P::LOG_WIDTH)
		{
			bail!(Error::IncorrectDestSliceLengths);
		}

		match multilinear {
			SumcheckMultilinear::Transparent { multilinear, .. } => {
				if self.tensor_query.n_vars() == 0 {
					multilinear.subcube_evals(subcube_vars + 1, subcube_index, 0, scratch_space)?
				} else {
					multilinear.subcube_partial_low_evals(
						self.tensor_query,
						subcube_vars + 1,
						subcube_index,
						scratch_space,
					)?
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_evals,
			} => {
				let multilinear =
					MultilinearExtension::from_values_generic(large_field_folded_evals.as_slice())?;

				MLEDirectAdapter::from(multilinear).subcube_evals(
					subcube_vars + 1,
					subcube_index,
					0,
					scratch_space,
				)?
			}
		}

		// Evaluations at 0 & 1 are interleaved (the substituted variable is the lowest one), need
		// to deinterleave them first. This requires scratch space to enable simple linear time algorithm.
		let zeros = P::default();
		let interleaved_tuples = if scratch_space.len() == 1 {
			Either::Left(iter::once((scratch_space.first().expect("len==1"), &zeros)))
		} else {
			Either::Right(scratch_space.iter().tuples())
		};

		for ((&interleaved_0, &interleaved_1), evals_0, evals_1) in
			izip!(interleaved_tuples, evals_0, evals_1)
		{
			let (deinterleaved_0, deinterleaved_1) = if P::LOG_WIDTH > 0 {
				P::unzip(interleaved_0, interleaved_1, 0)
			} else {
				(interleaved_0, interleaved_1)
			};

			*evals_0 = deinterleaved_0;
			*evals_1 = deinterleaved_1;
		}

		Ok(())
	}
}

#[derive(Debug)]
struct HighToLowAccess<'a, P: PackedField> {
	tensor_query: MultilinearQueryRef<'a, P>,
}

impl<P: PackedField> SumcheckMultilinearAccess<P> for HighToLowAccess<'_, P> {
	fn scratch_space_len(&self, _subcube_vars: usize) -> Option<usize> {
		None
	}

	fn subcube_evaluations<M: MultilinearPoly<P>>(
		&self,
		multilinear: &SumcheckMultilinear<P, M>,
		subcube_vars: usize,
		subcube_index: usize,
		index_vars: usize,
		_scratch_space: Option<&mut [P]>,
		evals_0: &mut [P],
		evals_1: &mut [P],
	) -> Result<(), Error> {
		if evals_0.len() != 1 << subcube_vars.saturating_sub(P::LOG_WIDTH)
			|| evals_1.len() != 1 << subcube_vars.saturating_sub(P::LOG_WIDTH)
		{
			bail!(Error::IncorrectDestSliceLengths);
		}

		match multilinear {
			SumcheckMultilinear::Transparent { multilinear, .. } => {
				if self.tensor_query.n_vars() == 0 {
					multilinear.subcube_evals(subcube_vars, subcube_index, 0, evals_0)?;
					multilinear.subcube_evals(
						subcube_vars,
						subcube_index | 1 << index_vars,
						0,
						evals_1,
					)?;
				} else {
					multilinear.subcube_partial_high_evals(
						self.tensor_query,
						subcube_vars,
						subcube_index,
						evals_0,
					)?;
					multilinear.subcube_partial_high_evals(
						self.tensor_query,
						subcube_vars,
						subcube_index | 1 << index_vars,
						evals_1,
					)?;
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_evals,
			} => {
				let multilinear =
					MultilinearExtension::from_values_generic(large_field_folded_evals.as_slice())?;
				let adapter = MLEDirectAdapter::from(multilinear);
				adapter.subcube_evals(subcube_vars, subcube_index, 0, evals_0)?;
				adapter.subcube_evals(subcube_vars, subcube_index | 1 << index_vars, 0, evals_1)?;
			}
		}

		Ok(())
	}
}
