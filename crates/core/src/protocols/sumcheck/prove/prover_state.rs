// Copyright 2024-2025 Irreducible Inc.

use binius_field::{util::powers, Field, PackedExtension, PackedField};
use binius_hal::{ComputationBackend, RoundEvals, SumcheckEvaluator, SumcheckMultilinear};
use binius_math::{
	evaluate_univariate, CompositionPoly, EvaluationOrder, MultilinearPoly, MultilinearQuery,
};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use getset::CopyGetters;
use itertools::izip;
use tracing::instrument;

use crate::{
	polynomial::Error as PolynomialError,
	protocols::sumcheck::{common::RoundCoeffs, error::Error},
};

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
	#[getset(get_copy = "pub")]
	evaluation_order: EvaluationOrder,
	// Make new ProverState with FSlice_Multilinears.
	multilinears: Vec<SumcheckMultilinear<P, M>>,
	nontrivial_evaluation_points: Vec<FDomain>,
	challenges: Vec<P::Scalar>,
	tensor_query: Option<MultilinearQuery<P>>,
	last_coeffs_or_sums: ProverStateCoeffsOrSums<P::Scalar>,
	backend: &'a Backend,
}

impl<'a, FDomain, F, P, M, Backend> ProverState<'a, FDomain, P, M, Backend>
where
	FDomain: Field,
	F: Field,
	P: PackedField<Scalar = F> + PackedExtension<FDomain>,
	M: MultilinearPoly<P> + Send + Sync,
	Backend: ComputationBackend,
{
	#[instrument(skip_all, level = "debug", name = "ProverState::new")]
	pub fn new(
		evaluation_order: EvaluationOrder,
		n_vars: usize,
		multilinears: Vec<SumcheckMultilinear<P, M>>,
		claimed_sums: Vec<F>,
		nontrivial_evaluation_points: Vec<FDomain>,
		backend: &'a Backend,
	) -> Result<Self, Error> {
		for multilinear in &multilinears {
			match *multilinear {
				SumcheckMultilinear::Transparent {
					ref multilinear,
					const_suffix: (_, suffix_len),
					..
				} => {
					if multilinear.n_vars() != n_vars {
						bail!(Error::NumberOfVariablesMismatch);
					}

					if suffix_len > 1 << n_vars {
						bail!(Error::IncorrectConstSuffixes);
					}
				}

				SumcheckMultilinear::Folded {
					large_field_folded_evals: ref evals,
					..
				} => {
					if evals.len() > 1 << n_vars.saturating_sub(P::LOG_WIDTH) {
						bail!(Error::IncorrectConstSuffixes);
					}
				}
			}
		}

		let tensor_query = multilinears
			.iter()
			.filter_map(|multilinear| match multilinear {
				SumcheckMultilinear::Transparent {
					switchover_round, ..
				} => Some(switchover_round),
				_ => None,
			})
			.max()
			.map(|&max_switchover_round| {
				MultilinearQuery::with_capacity(max_switchover_round.min(n_vars) + 1)
			});

		Ok(Self {
			n_vars,
			evaluation_order,
			multilinears,
			nontrivial_evaluation_points,
			tensor_query,
			challenges: Vec::new(),
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
					.par_iter()
					.map(|coeffs| evaluate_univariate(&coeffs.0, challenge))
					.collect();
				self.last_coeffs_or_sums = ProverStateCoeffsOrSums::Sums(new_sums);
			}
			ProverStateCoeffsOrSums::Sums(_) => {
				bail!(Error::ExpectedExecution);
			}
		}

		// Update the tensor query.
		match self.evaluation_order {
			EvaluationOrder::LowToHigh => self.challenges.push(challenge),
			EvaluationOrder::HighToLow => self.challenges.insert(0, challenge),
		}

		if let Some(tensor_query) = self.tensor_query.take() {
			self.tensor_query = match self.evaluation_order {
				EvaluationOrder::LowToHigh => Some(tensor_query.update(&[challenge])?),
				// REVIEW: not spending effort to come up with an inplace update method here, as the
				//         future of switchover is somewhat unclear in light of univariate skip, and
				//         switchover tensors are small-ish anyway.
				EvaluationOrder::HighToLow => Some(MultilinearQuery::expand(&self.challenges)),
			}
		}

		let any_transparent_left = self.backend.sumcheck_fold_multilinears(
			self.evaluation_order,
			self.n_vars,
			&mut self.multilinears,
			challenge,
			self.tensor_query.as_ref().map(Into::into),
		)?;

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
						large_field_folded_evals,
						suffix_eval,
					} => Ok(large_field_folded_evals
						.first()
						.map_or(suffix_eval, |packed| packed.get(0))
						.get(0)),
				}
				.map_err(Error::MathError)
			})
			.collect()
	}

	/// Calculate the accumulated evaluations for an arbitrary sumcheck round.
	#[instrument(skip_all, level = "debug")]
	pub fn calculate_round_evals<Evaluator, Composition>(
		&self,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundEvals<F>>, Error>
	where
		Evaluator: SumcheckEvaluator<P, Composition> + Sync,
		Composition: CompositionPoly<P>,
	{
		Ok(self.backend.sumcheck_compute_round_evals(
			self.evaluation_order,
			self.n_vars,
			self.tensor_query.as_ref().map(Into::into),
			&self.multilinears,
			evaluators,
			&self.nontrivial_evaluation_points,
		)?)
	}

	/// Calculate the batched round coefficients from the domain evaluations.
	///
	/// This both performs the polynomial interpolation over the evaluations and the mixing with
	/// the batching coefficient.
	#[instrument(skip_all, level = "debug")]
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
