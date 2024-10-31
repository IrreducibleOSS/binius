// Copyright 2024 Irreducible Inc.

use crate::{
	polynomial::Error as PolynomialError,
	protocols::sumcheck::{
		common::{determine_switchovers, equal_n_vars_check, RoundCoeffs},
		error::Error,
	},
};
use binius_field::{
	util::powers, ExtensionField, Field, PackedExtension, PackedField, RepackedExtension,
};
use binius_hal::{
	ComputationBackend, ComputationBackendExt, RoundEvals, SumcheckEvaluator, SumcheckMultilinear,
};
use binius_math::{
	evaluate_univariate, CompositionPoly, MLEDirectAdapter, MultilinearPoly, MultilinearQuery,
};
use binius_utils::bail;
use getset::CopyGetters;
use itertools::izip;
use rayon::prelude::*;
use std::{
	iter,
	sync::atomic::{AtomicBool, Ordering},
};
use tracing::instrument;

pub trait SumcheckInterpolator<F: Field> {
	/// Given evaluations of the round polynomial, interpolate and return monomial coefficients
	///
	/// ## Arguments
	///
	/// * `round_evals`: the computed evaluations of the round polynomial
	fn round_evals_to_coeffs(
		&self,
		last_sum: F,
		round_evals: Vec<F>,
	) -> Result<Vec<F>, PolynomialError>;
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
pub struct ProverState<'a, FDomain, P, M, Backend>
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
	tensor_query: Option<MultilinearQuery<P>>,
	last_coeffs_or_sums: ProverStateCoeffsOrSums<P::Scalar>,
	backend: &'a Backend,
}

impl<'a, FDomain, F, P, M, Backend> ProverState<'a, FDomain, P, M, Backend>
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
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let switchover_rounds = determine_switchovers(&multilinears, switchover_fn);
		Self::new_with_switchover_rounds(
			multilinears,
			&switchover_rounds,
			claimed_sums,
			evaluation_points,
			backend,
		)
	}

	pub fn new_with_switchover_rounds(
		multilinears: Vec<M>,
		switchover_rounds: &[usize],
		claimed_sums: Vec<F>,
		evaluation_points: Vec<FDomain>,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		let n_vars = equal_n_vars_check(&multilinears)?;

		if multilinears.len() != switchover_rounds.len() {
			bail!(Error::MultilinearSwitchoverSizeMismatch);
		}

		let max_switchover_round = switchover_rounds.iter().copied().max().unwrap_or_default();

		let multilinears = iter::zip(multilinears, switchover_rounds)
			.map(|(multilinear, &switchover_round)| SumcheckMultilinear::Transparent {
				multilinear,
				switchover_round,
			})
			.collect();

		let tensor_query = MultilinearQuery::with_capacity(max_switchover_round + 1);
		Ok(Self {
			n_vars,
			multilinears,
			evaluation_points,
			tensor_query: Some(tensor_query),
			last_coeffs_or_sums: ProverStateCoeffsOrSums::Sums(claimed_sums),
			backend,
		})
	}

	#[instrument(skip_all, name = "ProverState::fold", level = "debug")]
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
		let single_variable_partial_query = self.backend.multilinear_query(&[challenge])?;
		// Use Relaxed ordering for writes and the read, because:
		// * all writes can only update this value in the same direction of false->true
		// * the barrier at the end of rayon "parallel for" is a big enough synchronization point to be Relaxed about memory ordering of accesses to this Atomic.
		let any_transparent_left = AtomicBool::new(false);
		self.multilinears
			.par_iter_mut()
			.try_for_each(|multilinear| {
				match multilinear {
					SumcheckMultilinear::Transparent {
						multilinear: inner_multilinear,
						ref mut switchover_round,
					} => {
						if *switchover_round == 0 {
							let tensor_query = self.tensor_query.as_ref()
							.expect(
								"tensor_query is guaranteed to be Some while there is still a transparent multilinear"
							);

							// At switchover, perform inner products in large field and save them in a
							// newly created MLE.
							let large_field_folded_multilinear = MLEDirectAdapter::from(
								inner_multilinear.evaluate_partial_low(tensor_query.to_ref())?,
							);

							*multilinear = SumcheckMultilinear::Folded {
								large_field_folded_multilinear,
							};
						} else {
							*switchover_round -= 1;
							any_transparent_left.store(true, Ordering::Relaxed);
						}
					}
					SumcheckMultilinear::Folded {
						ref mut large_field_folded_multilinear,
					} => {
						// Post-switchover, simply halve large field MLE.
						*large_field_folded_multilinear = MLEDirectAdapter::from(
							large_field_folded_multilinear
								.evaluate_partial_low(single_variable_partial_query.to_ref())?,
						);
					}
				};
				Ok::<(), Error>(())
			})?;

		if !any_transparent_left.load(Ordering::Relaxed) {
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

		let empty_query = MultilinearQuery::with_capacity(0);
		self.multilinears
			.into_iter()
			.map(|multilinear| {
				match multilinear {
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
				}
				.map_err(Error::MathError)
			})
			.collect()
	}

	/// Calculate the accumulated evaluations for the first sumcheck round.
	#[instrument(skip_all, level = "debug")]
	pub fn calculate_first_round_evals<PBase, Evaluator, Composition>(
		&self,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundEvals<F>>, Error>
	where
		PBase: PackedField<Scalar: ExtensionField<FDomain>> + PackedExtension<FDomain>,
		P: PackedField<Scalar: ExtensionField<PBase::Scalar>> + RepackedExtension<PBase>,
		Evaluator: SumcheckEvaluator<PBase, P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		Ok(self.backend.sumcheck_compute_first_round_evals(
			self.n_vars,
			&self.multilinears,
			evaluators,
			&self.evaluation_points,
		)?)
	}

	/// Calculate the accumulated evaluations for an arbitrary sumcheck round.
	///
	/// See [`Self::calculate_first_round_evals`] for an optimized version of this method that
	/// operates over small fields in the first round.
	#[instrument(skip_all, level = "debug")]
	pub fn calculate_later_round_evals<Evaluator, Composition>(
		&self,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundEvals<F>>, Error>
	where
		Evaluator: SumcheckEvaluator<P, P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		Ok(self.backend.sumcheck_compute_later_round_evals(
			self.n_vars,
			self.tensor_query.as_ref().map(Into::into),
			&self.multilinears,
			evaluators,
			&self.evaluation_points,
		)?)
	}

	/// Calculate the batched round coefficients from the domain evaluations.
	///
	/// This both performs the polynomial interpolation over the evaluations and the mixing with
	/// the batching coefficient.
	pub fn calculate_round_coeffs_from_evals<Interpolator: SumcheckInterpolator<F>>(
		&mut self,
		interpolators: &[Interpolator],
		batch_coeff: F,
		evals: Vec<RoundEvals<F>>,
	) -> Result<RoundCoeffs<F>, Error> {
		let coeffs = match self.last_coeffs_or_sums {
			ProverStateCoeffsOrSums::Coeffs(_) => {
				bail!(Error::ExpectedFold);
			}
			ProverStateCoeffsOrSums::Sums(ref sums) => {
				if interpolators.len() != sums.len() {
					bail!(Error::IncorrectNumberOfEvaluators {
						expected: sums.len(),
					});
				}

				let coeffs = izip!(interpolators, sums, evals)
					.map(|(evaluator, &sum, RoundEvals(evals))| {
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
}
