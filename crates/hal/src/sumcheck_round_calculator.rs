// Copyright 2024 Irreducible Inc.

//! Functions that calculate the sumcheck round evaluations.
//!
//! This is one of the core computational tasks in the sumcheck proving algorithm.

use crate::{Error, RoundEvals, SumcheckEvaluator, SumcheckMultilinear};
use binius_field::{ExtensionField, Field, PackedExtension, PackedField, RepackedExtension};
use binius_math::{
	deinterleave, extrapolate_line, CompositionPoly, MultilinearPoly, MultilinearQuery,
	MultilinearQueryRef,
};
use bytemuck::zeroed_vec;
use itertools::izip;
use rayon::prelude::*;
use stackalloc::stackalloc_with_iter;
use std::{iter, marker::PhantomData};

trait SumcheckMultilinearAccess<P: PackedField> {
	/// Calculate the evaluations of a sumcheck multilinear over a subcube.
	///
	/// A sumcheck multilinear with $n$ variables is either expressed as folded $n$-variate
	/// multilinear extension, given by $2^n$ evaluations, or as a transparent $n + r$-variate
	/// multilinear, with its low $r$ variables projected onto $r$ round challenges.
	///
	/// This method computes the evaluations over a subcube with the given number of variables and
	/// index.
	///
	/// ## Arguments
	///
	/// * `subcube_vars` - the number of variables in the subcube to evaluate over
	/// * `subcube_index` - the index of the subcube within the hypercube domain of the multilinear
	/// * `evals` - the output buffer to write the evaluations into
	fn subcube_evaluations(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		evals: &mut [P],
	) -> Result<(), Error>;
}

/// Calculate the accumulated evaluations for the first sumcheck round.
pub(crate) fn calculate_first_round_evals<FDomain, FBase, F, PBase, P, M, Evaluator, Composition>(
	n_vars: usize,
	multilinears: &[SumcheckMultilinear<P, M>],
	evaluators: &[Evaluator],
	evaluation_points: &[FDomain],
) -> Result<Vec<RoundEvals<F>>, Error>
where
	FDomain: Field,
	FBase: ExtensionField<FDomain>,
	F: Field + ExtensionField<FDomain> + ExtensionField<FBase>,
	PBase: PackedField<Scalar = FBase> + PackedExtension<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain> + RepackedExtension<PBase>,
	M: MultilinearPoly<P> + Send + Sync,
	Evaluator: SumcheckEvaluator<PBase, P, Composition> + Sync,
	Composition: CompositionPoly<P>,
{
	let accesses = multilinears
		.iter()
		.map(FirstRoundAccess::new)
		.collect::<Vec<_>>();
	calculate_round_evals(n_vars, &accesses, evaluators, evaluation_points)
}

/// Calculate the accumulated evaluations for an arbitrary sumcheck round.
///
/// See [`calculate_first_round_evals`] for an optimized version of this method
/// that works over small fields in the first round.
pub(crate) fn calculate_later_round_evals<FDomain, F, P, M, Evaluator, Composition>(
	n_vars: usize,
	tensor_query: Option<MultilinearQueryRef<P>>,
	multilinears: &[SumcheckMultilinear<P, M>],
	evaluators: &[Evaluator],
	evaluation_points: &[FDomain],
) -> Result<Vec<RoundEvals<F>>, Error>
where
	FDomain: Field,
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Send + Sync,
	Evaluator: SumcheckEvaluator<P, P, Composition> + Sync,
	Composition: CompositionPoly<P>,
{
	let empty_query = MultilinearQuery::with_capacity(0);
	let query = tensor_query.unwrap_or(empty_query.to_ref());

	let accesses = multilinears
		.iter()
		.map(|multilinear| LaterRoundAccess {
			multilinear,
			tensor_query: query,
		})
		.collect::<Vec<_>>();
	calculate_round_evals(n_vars, &accesses, evaluators, evaluation_points)
}

fn calculate_round_evals<FDomain, FBase, F, PBase, P, Evaluator, Access, Composition>(
	n_vars: usize,
	multilinears: &[Access],
	evaluators: &[Evaluator],
	evaluation_points: &[FDomain],
) -> Result<Vec<RoundEvals<F>>, Error>
where
	FDomain: Field,
	FBase: ExtensionField<FDomain>,
	F: Field + ExtensionField<FDomain> + ExtensionField<FBase>,
	PBase: PackedField<Scalar = FBase> + PackedExtension<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	Evaluator: SumcheckEvaluator<PBase, P, Composition> + Sync,
	Access: SumcheckMultilinearAccess<PBase> + Sync,
	Composition: CompositionPoly<P>,
{
	let n_multilinears = multilinears.len();
	let n_round_evals = evaluators
		.iter()
		.map(|evaluator| evaluator.eval_point_indices().len());

	/// Process batches of vertices in parallel, accumulating the round evaluations.
	const MAX_SUBCUBE_VARS: usize = 5;
	let subcube_vars = MAX_SUBCUBE_VARS.min(n_vars) - 1;

	// Compute the union of all evaluation point indice ranges.
	let eval_point_indices = evaluators
		.iter()
		.map(|evaluator| evaluator.eval_point_indices())
		.reduce(|range1, range2| range1.start.min(range2.start)..range1.end.max(range2.end))
		.unwrap_or(0..0);

	let packed_accumulators = (0..(1 << (n_vars - 1 - subcube_vars)))
		.into_par_iter()
		.fold(
			|| ParFoldStates::new(n_multilinears, n_round_evals.clone(), subcube_vars),
			|mut par_fold_states, subcube_index| {
				let ParFoldStates {
					multilinear_evals,
					interleaved_evals,
					round_evals,
				} = &mut par_fold_states;

				for (multilinear, evals) in iter::zip(multilinears, multilinear_evals.iter_mut()) {
					multilinear
						.subcube_evaluations(
							subcube_vars + 1,
							subcube_index,
							interleaved_evals.as_mut_slice(),
						)
						.expect("indices are in range");

					// Returned slice has interleaved 0/1 evals due to round variable
					// being the lowermost one. Deinterleave into two slices.
					deinterleave(subcube_vars, interleaved_evals.as_slice()).for_each(
						|(i, even, odd)| {
							evals.evals_0[i] = even;
							evals.evals_1[i] = odd;
						},
					);
				}

				// Proceed by evaluation point first to share interpolation work between evaluators.
				for eval_point_index in eval_point_indices.clone() {
					let eval_point = evaluation_points[eval_point_index];

					// Only points with indices two and above need to be interpolated.
					if eval_point_index >= 2 {
						for evals in multilinear_evals.iter_mut() {
							for (&eval_0, &eval_1, eval_z) in izip!(
								evals.evals_0.as_slice(),
								evals.evals_1.as_slice(),
								evals.evals_z.as_mut_slice(),
							) {
								*eval_z = extrapolate_line(eval_0, eval_1, eval_point);
							}
						}
					}

					let evals_z_iter =
						multilinear_evals
							.iter()
							.map(|evals| match eval_point_index {
								0 => evals.evals_0.as_slice(),
								1 => evals.evals_1.as_slice(),
								_ => evals.evals_z.as_slice(),
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
									evals_z,
								);
						}
					});
				}

				par_fold_states
			},
		)
		.map(|states| states.round_evals)
		// Simply sum up the fold partitions.
		.reduce(
			|| {
				evaluators
					.iter()
					.map(|evaluator| vec![P::zero(); evaluator.eval_point_indices().len()])
					.collect()
			},
			|lhs, rhs| {
				iter::zip(lhs, rhs)
					.map(|(mut lhs_vals, rhs_vals)| {
						for (lhs_val, rhs_val) in lhs_vals.iter_mut().zip(rhs_vals) {
							*lhs_val += rhs_val;
						}
						lhs_vals
					})
					.collect()
			},
		);

	let evals = packed_accumulators
		.into_iter()
		.map(|vals| {
			RoundEvals(
				vals.into_iter()
					.map(|packed_val| packed_val.iter().sum())
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
	fn new(n_vars: usize) -> Self {
		let len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
		Self {
			evals_0: zeroed_vec(len),
			evals_1: zeroed_vec(len),
			evals_z: zeroed_vec(len),
		}
	}
}

/// Parallel fold state, consisting of scratch area and result accumulator.
#[derive(Debug)]
struct ParFoldStates<PBase: PackedField, P: PackedField> {
	// Evaluations at 0, 1 and domain points, per MLE. Scratch space.
	multilinear_evals: Vec<MultilinearEvals<PBase>>,

	// Interleaved subcube evals/inner products as returned by eval01.
	// Double size compared to query (due to having 0 & 1 evals interleaved).
	// Scratch space.
	interleaved_evals: Vec<PBase>,

	// Accumulated sums of evaluations over univariate domain.
	//
	// Each element of the outer vector corresponds to one composite polynomial. Each element of
	// an inner vector contains the evaluations at different points.
	round_evals: Vec<Vec<P>>,
}

impl<PBase: PackedField, P: PackedField> ParFoldStates<PBase, P> {
	fn new(
		n_multilinears: usize,
		n_round_evals: impl Iterator<Item = usize>,
		subcube_vars: usize,
	) -> Self {
		Self {
			multilinear_evals: (0..n_multilinears)
				.map(|_| MultilinearEvals::new(subcube_vars))
				.collect(),
			interleaved_evals: vec![
				PBase::default();
				1 << (subcube_vars + 1).saturating_sub(PBase::LOG_WIDTH)
			],
			round_evals: n_round_evals
				.map(|n_round_evals| zeroed_vec(n_round_evals))
				.collect(),
		}
	}
}

#[derive(Debug)]
struct FirstRoundAccess<'a, PBase, P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	multilinear: &'a SumcheckMultilinear<P, M>,
	_marker: PhantomData<PBase>,
}

impl<'a, PBase, P, M> FirstRoundAccess<'a, PBase, P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	fn new(multilinear: &'a SumcheckMultilinear<P, M>) -> Self {
		Self {
			multilinear,
			_marker: PhantomData,
		}
	}
}

impl<'a, PBase, P, M> SumcheckMultilinearAccess<PBase> for FirstRoundAccess<'a, PBase, P, M>
where
	PBase: PackedField,
	P: RepackedExtension<PBase>,
	P::Scalar: ExtensionField<PBase::Scalar>,
	M: MultilinearPoly<P> + Send + Sync,
{
	fn subcube_evaluations(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		evals: &mut [PBase],
	) -> Result<(), Error> {
		if let SumcheckMultilinear::Transparent { multilinear, .. } = self.multilinear {
			let evals = <P as PackedExtension<PBase::Scalar>>::cast_exts_mut(evals);
			Ok(multilinear.subcube_evals(
				subcube_vars,
				subcube_index,
				<P::Scalar as ExtensionField<PBase::Scalar>>::LOG_DEGREE,
				evals,
			)?)
		} else {
			panic!("precondition: no folded multilinears in the first round");
		}
	}
}

#[derive(Debug)]
struct LaterRoundAccess<'a, P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	multilinear: &'a SumcheckMultilinear<P, M>,
	tensor_query: MultilinearQueryRef<'a, P>,
}

impl<'a, P, M> SumcheckMultilinearAccess<P> for LaterRoundAccess<'a, P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	fn subcube_evaluations(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		evals: &mut [P],
	) -> Result<(), Error> {
		match self.multilinear {
			SumcheckMultilinear::Transparent { multilinear, .. } => {
				// TODO: Stop using LaterRoundAccess for first round in RegularSumcheckProver and
				// GPASumcheckProver, then remove this conditional.
				if self.tensor_query.n_vars() == 0 {
					Ok(multilinear.subcube_evals(subcube_vars, subcube_index, 0, evals)?)
				} else {
					Ok(multilinear.subcube_inner_products(
						self.tensor_query,
						subcube_vars,
						subcube_index,
						evals,
					)?)
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_multilinear,
			} => Ok(large_field_folded_multilinear.subcube_evals(
				subcube_vars,
				subcube_index,
				0,
				evals,
			)?),
		}
	}
}
