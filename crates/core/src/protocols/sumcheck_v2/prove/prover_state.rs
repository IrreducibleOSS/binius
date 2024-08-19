// Copyright 2024 Ulvetanna Inc.

use crate::{
	polynomial::{
		evaluate_univariate, Error as PolynomialError, MultilinearExtensionSpecialized,
		MultilinearPoly, MultilinearQuery,
	},
	protocols::{
		sumcheck_v2::{common::RoundCoeffs, error::Error},
		utils::packed_from_fn_with_offset,
	},
};
use binius_field::{util::powers, Field, PackedField};
use binius_utils::array_2d::Array2D;
use bytemuck::zeroed_vec;
use getset::CopyGetters;
use itertools::izip;
use rayon::prelude::*;
use std::{iter, ops::Range};

/// An individual multilinear polynomial stored by the [`ProverState`].
#[derive(Debug, Clone)]
enum SumcheckMultilinear<P, M>
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

pub trait SumcheckEvaluator<P: PackedField> {
	/// The number of points to evaluate at.
	fn n_round_evals(&self) -> usize;

	/// Process and update the round evaluations with the evaluations at a hypercube vertex.
	///
	/// ## Arguments
	///
	/// * `i`: index of the hypercube vertex under processing
	/// * `evals_0`: the n multilinear polynomial evaluations at 0
	/// * `evals_1`: the n multilinear polynomial evaluations at 1
	/// * `evals_z`: a scratch buffer of size n for storing multilinear polynomial evaluations at a
	///              point z
	/// * `round_evals`: the accumulated evaluations for the round
	fn process_vertex(
		&self,
		i: usize,
		evals_0: &[P],
		evals_1: &[P],
		evals_z: &mut [P],
		round_evals: &mut [P],
	);

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

/// Parallel fold state, consisting of scratch area and result accumulator.
#[derive(Debug)]
struct ParFoldStates<P: PackedField> {
	// Evaluations at 0, 1 and domain points, per MLE. Scratch space.
	evals_0: Array2D<P>,
	evals_1: Array2D<P>,
	evals_z: Array2D<P>,

	/// Accumulated sums of evaluations over univariate domain.
	///
	/// Each element of the outer vector corresponds to one composite polynomial. Each element of
	/// an inner vector contains the evaluations at different points.
	round_evals: Vec<Vec<P>>,
}

impl<P: PackedField> ParFoldStates<P> {
	fn new(
		n_multilinears: usize,
		n_round_evals: impl Iterator<Item = usize>,
		n_states: usize,
	) -> Self {
		Self {
			evals_0: Array2D::zeroes(n_states, n_multilinears),
			evals_1: Array2D::zeroes(n_states, n_multilinears),
			evals_z: Array2D::zeroes(n_states, n_multilinears),
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
pub struct ProverState<P, M>
where
	P: PackedField,
	M: MultilinearPoly<P> + Send + Sync,
{
	/// The number of variables in the folded multilinears. This value decrements each round the
	/// state is folded.
	#[getset(get_copy = "pub")]
	n_vars: usize,
	multilinears: Vec<SumcheckMultilinear<P, M>>,
	tensor_query: Option<MultilinearQuery<P>>,
	last_coeffs_or_sums: ProverStateCoeffsOrSums<P::Scalar>,
}

impl<F, P, M> ProverState<P, M>
where
	F: Field,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
{
	pub fn new(
		multilinears: Vec<M>,
		claimed_sums: Vec<F>,
		switchover_fn: impl Fn(usize) -> usize,
	) -> Result<Self, Error> {
		let n_vars = multilinears
			.first()
			.map(|multilin| multilin.n_vars())
			.unwrap_or(0);
		for multilinear in multilinears.iter() {
			if multilinear.n_vars() != n_vars {
				return Err(Error::NumberOfVariablesMismatch);
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

		let tensor_query = MultilinearQuery::new(max_switchover_round)?;

		Ok(Self {
			n_vars,
			multilinears,
			tensor_query: Some(tensor_query),
			last_coeffs_or_sums: ProverStateCoeffsOrSums::Sums(claimed_sums),
		})
	}

	pub fn fold(&mut self, challenge: F) -> Result<(), Error> {
		if self.n_vars == 0 {
			return Err(Error::ExpectedFinish);
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
				return Err(Error::ExpectedExecution);
			}
		}

		// Update the tensor query.
		if let Some(tensor_query) = self.tensor_query.take() {
			self.tensor_query = Some(tensor_query.update(&[challenge])?);
		}

		// Partial query for folding
		let single_variable_partial_query = MultilinearQuery::with_full_query(&[challenge])?;

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
							inner_multilinear.evaluate_partial_low(tensor_query)?;

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
						.evaluate_partial_low(&single_variable_partial_query)?;
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
				return Err(Error::ExpectedFold);
			}
			ProverStateCoeffsOrSums::Sums(_) => match self.n_vars {
				0 => {}
				_ => return Err(Error::ExpectedExecution),
			},
		};

		let empty_query = MultilinearQuery::new(0).expect("constructing an empty query");
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
						inner_multilinear.evaluate(tensor_query)
					}
					SumcheckMultilinear::Folded {
						large_field_folded_multilinear,
					} => large_field_folded_multilinear.evaluate(&empty_query),
				};
				result.map_err(Error::Polynomial)
			})
			.collect()
	}

	pub fn calculate_round_coeffs<Evaluator: SumcheckEvaluator<P> + Sync>(
		&mut self,
		evaluators: &[Evaluator],
		batch_coeff: F,
	) -> Result<RoundCoeffs<F>, Error> {
		let evals = self.calculate_round_evals(evaluators)?;

		let coeffs = match self.last_coeffs_or_sums {
			ProverStateCoeffsOrSums::Coeffs(_) => {
				return Err(Error::ExpectedFold);
			}
			ProverStateCoeffsOrSums::Sums(ref sums) => {
				if evaluators.len() != sums.len() {
					return Err(Error::IncorrectNumberOfEvaluators {
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

	fn calculate_round_evals<Evaluator: SumcheckEvaluator<P> + Sync>(
		&self,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let n_multilinears = self.multilinears.len();
		let n_round_evals = evaluators.iter().map(|evaluator| evaluator.n_round_evals());

		let empty_query = MultilinearQuery::new(0).expect("constructing an empty query");
		let query = self.tensor_query.as_ref().unwrap_or(&empty_query);

		/// Process batches of vertices in parallel, accumulating the round evaluations.
		const MAX_LOG_BATCH_SIZE: usize = 6;
		let log_batch_size = (self.n_vars - 1).min(MAX_LOG_BATCH_SIZE);
		let batch_size = 1 << log_batch_size;

		let packed_accumulators = (0..(1 << (self.n_vars - 1 - log_batch_size)))
			.into_par_iter()
			.fold(
				|| ParFoldStates::new(n_multilinears, n_round_evals.clone(), batch_size),
				|mut par_fold_states, vertex| {
					let begin = vertex << log_batch_size;
					let end = begin + batch_size;
					for (j, multilinear) in self.multilinears.iter().enumerate() {
						Self::eval01(
							query,
							multilinear,
							begin..end,
							&mut par_fold_states.evals_0,
							&mut par_fold_states.evals_1,
							j,
						);
					}

					for (evaluator, round_evals) in
						iter::zip(evaluators.iter(), par_fold_states.round_evals.iter_mut())
					{
						for k in 0..batch_size {
							evaluator.process_vertex(
								begin + k,
								par_fold_states.evals_0.get_row(k),
								par_fold_states.evals_1.get_row(k),
								par_fold_states.evals_z.get_row_mut(k),
								round_evals,
							);
						}
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
						.map(|evaluator| vec![P::zero(); evaluator.n_round_evals()])
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

	// Note the generic parameter - this method samples small field in first round and
	// large field post-switchover.
	#[inline]
	fn direct_sample<MD>(
		multilin: MD,
		indices: Range<usize>,
		evals_0: &mut Array2D<P>,
		evals_1: &mut Array2D<P>,
		col_index: usize,
	) where
		MD: MultilinearPoly<P>,
	{
		// TODO: Make a method on MultilinearPoly that does this for entire Array2Ds, like evaluate_subcube.
		for (k, i) in indices.enumerate() {
			evals_0[(k, col_index)] = packed_from_fn_with_offset(i, |idx| {
				multilin.evaluate_on_hypercube(idx << 1).unwrap_or_default()
			});
			evals_1[(k, col_index)] = packed_from_fn_with_offset(i, |idx| {
				multilin
					.evaluate_on_hypercube((idx << 1) + 1)
					.unwrap_or_default()
			});
		}
	}

	#[inline]
	fn subcube_inner_product(
		query: &MultilinearQuery<P>,
		multilin: &M,
		indices: Range<usize>,
		evals_0: &mut Array2D<P>,
		evals_1: &mut Array2D<P>,
		col_index: usize,
	) {
		multilin
			.evaluate_subcube(indices, query, evals_0, evals_1, col_index)
			.expect("indices within range");
	}

	fn eval01(
		query: &MultilinearQuery<P>,
		multilin: &SumcheckMultilinear<P, M>,
		indices: Range<usize>,
		evals_0: &mut Array2D<P>,
		evals_1: &mut Array2D<P>,
		col_index: usize,
	) {
		match multilin {
			SumcheckMultilinear::Transparent { multilinear, .. } => Self::subcube_inner_product(
				query,
				multilinear,
				indices,
				evals_0,
				evals_1,
				col_index,
			),

			SumcheckMultilinear::Folded {
				large_field_folded_multilinear,
			} => Self::direct_sample(
				large_field_folded_multilinear,
				indices,
				evals_0,
				evals_1,
				col_index,
			),
		}
	}
}
