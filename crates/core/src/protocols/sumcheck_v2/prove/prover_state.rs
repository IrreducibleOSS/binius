// Copyright 2024 Ulvetanna Inc.

use crate::{
	polynomial::{
		Error as PolynomialError, MultilinearExtensionSpecialized, MultilinearPoly,
		MultilinearQuery, MultilinearQueryRef,
	},
	protocols::{
		sumcheck_v2::{common::RoundCoeffs, error::Error},
		utils::deinterleave,
	},
};
use binius_field::{
	util::powers, ExtensionField, Field, PackedExtension, PackedField, RepackedExtension,
};
use binius_hal::ComputationBackend;
use binius_math::{evaluate_univariate, extrapolate_line};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use getset::CopyGetters;
use itertools::izip;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use stackalloc::stackalloc_with_iter;
use std::{iter, ops::Range};

/// An individual multilinear polynomial stored by the [`ProverState`].
#[derive(Debug, Clone)]
pub(super) enum SumcheckMultilinear<P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	/// Small field polynomial - to be folded into large field in `switchover` rounds.
	/// The switchover round decrements each round the multilinear is folded.
	Transparent {
		multilinear: M,
		switchover_round: usize,
	},
	/// Large field polynomial - halved in size each round
	Folded {
		large_field_folded_multilinear: MultilinearExtensionSpecialized<P, P>,
	},
}

pub trait SumcheckEvaluator<PBase: PackedField, P: PackedField> {
	/// The range of eval point indices over which composition evaluation and summation should happen.
	/// Returned range must equal the result of `n_round_evals()` in length.
	fn eval_point_indices(&self) -> Range<usize>;

	/// Compute composition evals over a subcube.
	///
	/// `sparse_batch_query` should contain multilinears evals over a subcube represented
	/// by `subcube_vars` and `subcube_index`.
	///
	/// Returns a packed sum (which may be spread across scalars).
	fn process_subcube_at_eval_point(
		&self,
		subcube_vars: usize,
		subcube_index: usize,
		sparse_batch_query: &[&[PBase]],
	) -> P;

	/// Given evaluations of the round polynomial, interpolate and return monomial coefficients
	///
	/// ## Arguments
	///
	/// * `round_evals`: the computed evaluations of the round polynomial
	fn round_evals_to_coeffs(
		&self,
		last_sum: P::Scalar,
		round_evals: Vec<P::Scalar>,
	) -> Result<Vec<P::Scalar>, PolynomialError>;
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
enum ProverStateCoeffsOrSums<F: Field> {
	Coeffs(Vec<RoundCoeffs<F>>),
	Sums(Vec<F>),
}

/// The stored state of a sumcheck prover, which encapsulates common implementation logic.
///
/// We expect that CPU sumcheck provers will internally maintain a [`ProverState`] instance and
/// customize the sumcheck logic through different [`SumcheckEvaluator`] implementations passed to
/// the common state object.
#[derive(Debug, CopyGetters)]
pub struct ProverState<FDomain, P, M, Backend>
where
	FDomain: Field,
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	/// The number of variables in the folded multilinears. This value decrements each round the
	/// state is folded.
	#[getset(get_copy = "pub")]
	n_vars: usize,
	multilinears: Vec<SumcheckMultilinear<P, M>>,
	evaluation_points: Vec<FDomain>,
	tensor_query: Option<MultilinearQuery<P, Backend>>,
	last_coeffs_or_sums: ProverStateCoeffsOrSums<P::Scalar>,
	backend: Backend,
}

impl<FDomain, F, P, M, Backend> ProverState<FDomain, P, M, Backend>
where
	FDomain: Field,
	F: Field + ExtensionField<FDomain>,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	pub fn new(
		multilinears: Vec<M>,
		claimed_sums: Vec<F>,
		evaluation_points: Vec<FDomain>,
		switchover_fn: impl Fn(usize) -> usize,
		backend: Backend,
	) -> Result<Self, Error> {
		let n_vars = multilinears
			.first()
			.map(|multilin| multilin.n_vars())
			.unwrap_or(0);
		for multilinear in multilinears.iter() {
			if multilinear.n_vars() != n_vars {
				bail!(Error::NumberOfVariablesMismatch);
			}
		}

		let switchover_rounds = multilinears
			.iter()
			.map(|multilinear| switchover_fn(multilinear.extension_degree()))
			.collect::<Vec<_>>();
		let max_switchover_round = switchover_rounds.iter().copied().max().unwrap_or_default();

		let multilinears = iter::zip(multilinears, switchover_rounds)
			.map(|(multilinear, switchover_round)| SumcheckMultilinear::Transparent {
				multilinear,
				switchover_round,
			})
			.collect();

		let tensor_query = MultilinearQuery::<_, Backend>::new(max_switchover_round)?;

		Ok(Self {
			n_vars,
			multilinears,
			evaluation_points,
			tensor_query: Some(tensor_query),
			last_coeffs_or_sums: ProverStateCoeffsOrSums::Sums(claimed_sums),
			backend,
		})
	}

	pub fn fold(&mut self, challenge: F) -> Result<(), Error> {
		if self.n_vars == 0 {
			bail!(Error::ExpectedFinish);
		}

		// Update the stored multilinear sums.
		match self.last_coeffs_or_sums {
			ProverStateCoeffsOrSums::Coeffs(ref round_coeffs) => {
				let new_sums = round_coeffs
					.iter()
					.map(|coeffs| evaluate_univariate(&coeffs.0, challenge))
					.collect();
				self.last_coeffs_or_sums = ProverStateCoeffsOrSums::Sums(new_sums);
			}
			ProverStateCoeffsOrSums::Sums(_) => {
				bail!(Error::ExpectedExecution);
			}
		}

		// Update the tensor query.
		if let Some(tensor_query) = self.tensor_query.take() {
			self.tensor_query = Some(tensor_query.update(&[challenge])?);
		}

		// Partial query for folding
		let single_variable_partial_query =
			MultilinearQuery::with_full_query(&[challenge], self.backend.clone())?;
		let single_variable_partial_query =
			MultilinearQueryRef::new(&single_variable_partial_query);

		let mut any_transparent_left = false;
		for multilinear in self.multilinears.iter_mut() {
			match multilinear {
				SumcheckMultilinear::Transparent {
					multilinear: inner_multilinear,
					ref mut switchover_round,
				} => {
					let tensor_query = self.tensor_query.as_ref()
						.expect(
							"tensor_query is guaranteed to be Some while there is still a transparent multilinear"
						);

					// TODO: would be nicer if switchover_round 0 meant to fold after the first round
					*switchover_round -= 1;
					if *switchover_round == 0 {
						// At switchover, perform inner products in large field and save them in a
						// newly created MLE.
						let large_field_folded_multilinear =
							inner_multilinear.evaluate_partial_low(tensor_query.to_ref())?;

						*multilinear = SumcheckMultilinear::Folded {
							large_field_folded_multilinear,
						};
					} else {
						any_transparent_left = true;
					}
				}
				SumcheckMultilinear::Folded {
					ref mut large_field_folded_multilinear,
				} => {
					// Post-switchover, simply halve large field MLE.
					*large_field_folded_multilinear = large_field_folded_multilinear
						.evaluate_partial_low(single_variable_partial_query.clone())?;
				}
			}
		}

		if !any_transparent_left {
			self.tensor_query = None;
		}

		self.n_vars -= 1;
		Ok(())
	}

	pub fn finish(self) -> Result<Vec<F>, Error> {
		match self.last_coeffs_or_sums {
			ProverStateCoeffsOrSums::Coeffs(_) => {
				bail!(Error::ExpectedFold);
			}
			ProverStateCoeffsOrSums::Sums(_) => match self.n_vars {
				0 => {}
				_ => bail!(Error::ExpectedExecution),
			},
		};

		let empty_query =
			MultilinearQuery::<_, Backend>::new(0).expect("constructing an empty query");
		self.multilinears
			.into_iter()
			.map(|multilinear| {
				let result = match multilinear {
					SumcheckMultilinear::Transparent {
						multilinear: inner_multilinear,
						..
					} => {
						let tensor_query = self.tensor_query.as_ref()
							.expect(
								"tensor_query is guaranteed to be Some while there is still a transparent multilinear"
							);
						inner_multilinear.evaluate(tensor_query.to_ref())
					}
					SumcheckMultilinear::Folded {
						large_field_folded_multilinear,
					} => large_field_folded_multilinear.evaluate(empty_query.to_ref()),
				};
				result.map_err(Error::Polynomial)
			})
			.collect()
	}

	pub fn calculate_round_coeffs<Evaluator: SumcheckEvaluator<P, P> + Sync>(
		&mut self,
		evaluators: &[Evaluator],
		batch_coeff: F,
	) -> Result<RoundCoeffs<F>, Error> {
		self.calculate_round_coeffs_with_eval01::<P, _>(Self::eval01, evaluators, batch_coeff)
	}

	pub(super) fn calculate_round_coeffs_with_eval01<PBase, Evaluator>(
		&mut self,
		eval01: impl Fn(&SumcheckMultilinear<P, M>, MultilinearQueryRef<P>, usize, usize, &mut [PBase])
			+ Sync,
		evaluators: &[Evaluator],
		batch_coeff: F,
	) -> Result<RoundCoeffs<F>, Error>
	where
		PBase: PackedField<Scalar: ExtensionField<FDomain>> + PackedExtension<FDomain>,
		Evaluator: SumcheckEvaluator<PBase, P> + Sync,
		F: ExtensionField<P::Scalar>,
	{
		let evals = self.calculate_round_evals(eval01, evaluators)?;

		let coeffs = match self.last_coeffs_or_sums {
			ProverStateCoeffsOrSums::Coeffs(_) => {
				bail!(Error::ExpectedFold);
			}
			ProverStateCoeffsOrSums::Sums(ref sums) => {
				if evaluators.len() != sums.len() {
					bail!(Error::IncorrectNumberOfEvaluators {
						expected: sums.len(),
					});
				}

				let coeffs = izip!(evaluators, sums, evals)
					.map(|(evaluator, &sum, RoundCoeffs(evals))| {
						let coeffs = evaluator.round_evals_to_coeffs(sum, evals)?;
						Ok::<_, Error>(RoundCoeffs(coeffs))
					})
					.collect::<Result<Vec<_>, _>>()?;
				self.last_coeffs_or_sums = ProverStateCoeffsOrSums::Coeffs(coeffs.clone());
				coeffs
			}
		};

		let batched_coeffs = coeffs
			.into_iter()
			.zip(powers(batch_coeff))
			.map(|(coeffs, scalar)| coeffs * scalar)
			.fold(RoundCoeffs::default(), |accum, coeffs| accum + &coeffs);

		Ok(batched_coeffs)
	}

	fn calculate_round_evals<PBase, Evaluator>(
		&self,
		eval01: impl Fn(&SumcheckMultilinear<P, M>, MultilinearQueryRef<P>, usize, usize, &mut [PBase])
			+ Sync,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundCoeffs<F>>, Error>
	where
		PBase: PackedField<Scalar: ExtensionField<FDomain>> + PackedExtension<FDomain>,
		Evaluator: SumcheckEvaluator<PBase, P> + Sync,
		F: ExtensionField<P::Scalar>,
	{
		let n_multilinears = self.multilinears.len();
		let n_round_evals = evaluators
			.iter()
			.map(|evaluator| evaluator.eval_point_indices().len());

		let empty_query =
			MultilinearQuery::<_, Backend>::new(0).expect("constructing an empty query");
		let query = self.tensor_query.as_ref().unwrap_or(&empty_query);

		/// Process batches of vertices in parallel, accumulating the round evaluations.
		const MAX_SUBCUBE_VARS: usize = 5;
		let subcube_vars = MAX_SUBCUBE_VARS.min(self.n_vars) - 1;

		// Compute the union of all evaluation point indice ranges.
		let eval_point_indices = evaluators
			.iter()
			.map(|evaluator| evaluator.eval_point_indices())
			.reduce(|range1, range2| range1.start.min(range2.start)..range1.end.max(range2.end))
			.unwrap_or(0..0);

		let packed_accumulators = (0..(1 << (self.n_vars - 1 - subcube_vars)))
			.into_par_iter()
			.fold(
				|| ParFoldStates::new(n_multilinears, n_round_evals.clone(), subcube_vars),
				|mut par_fold_states, subcube_index| {
					let ParFoldStates {
						multilinear_evals,
						interleaved_evals,
						round_evals,
					} = &mut par_fold_states;

					for (multilinear, evals) in
						iter::zip(&self.multilinears, multilinear_evals.iter_mut())
					{
						eval01(
							multilinear,
							query.to_ref(),
							subcube_vars + 1,
							subcube_index,
							interleaved_evals.as_mut_slice(),
						);

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
						let eval_point = self.evaluation_points[eval_point_index];

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

								round_evals[eval_point_index - eval_point_indices.start] +=
									evaluator.process_subcube_at_eval_point(
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
				RoundCoeffs(
					vals.into_iter()
						.map(|packed_val| packed_val.iter().sum())
						.collect(),
				)
			})
			.collect();

		Ok(evals)
	}

	fn eval01(
		multilinear: &SumcheckMultilinear<P, M>,
		query: MultilinearQueryRef<P>,
		subcube_vars: usize,
		subcube_index: usize,
		evals: &mut [P],
	) {
		let result = match multilinear {
			SumcheckMultilinear::Transparent { multilinear, .. } => {
				if query.n_vars() == 0 {
					multilinear.subcube_evals(subcube_vars, subcube_index, 0, evals)
				} else {
					multilinear.subcube_inner_products(query, subcube_vars, subcube_index, evals)
				}
			}

			SumcheckMultilinear::Folded {
				large_field_folded_multilinear,
			} => large_field_folded_multilinear.subcube_evals(subcube_vars, subcube_index, 0, evals),
		};

		result.expect("correct indices");
	}
}

pub(super) fn eval01_first_round<PBase, P, M>(
	multilinear: &SumcheckMultilinear<P, M>,
	_query: MultilinearQueryRef<P>,
	subcube_vars: usize,
	subcube_index: usize,
	evals: &mut [PBase],
) where
	PBase: PackedField,
	P: PackedField<Scalar: ExtensionField<PBase::Scalar>> + RepackedExtension<PBase>,
	M: MultilinearPoly<P> + Send + Sync,
{
	let result = if let SumcheckMultilinear::Transparent { multilinear, .. } = multilinear {
		let evals = <P as PackedExtension<PBase::Scalar>>::cast_exts_mut(evals);
		multilinear.subcube_evals(
			subcube_vars,
			subcube_index,
			log2_strict_usize(<P::Scalar as ExtensionField<PBase::Scalar>>::DEGREE),
			evals,
		)
	} else {
		panic!("no folded multilinears in the first round");
	};

	result.expect("correct indices")
}
