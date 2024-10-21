// Copyright 2024 Ulvetanna Inc.

use super::round_calculator::{
	calculate_first_round_evals, calculate_later_round_evals, SumcheckEvaluator,
	SumcheckMultilinear,
};
use crate::{
	polynomial::{
		Error as PolynomialError, MultilinearPoly, MultilinearQuery, MultilinearQueryRef,
	},
	protocols::sumcheck_v2::{common::RoundCoeffs, error::Error},
};
use binius_field::{
	util::powers, ExtensionField, Field, PackedExtension, PackedField, RepackedExtension,
};
use binius_hal::ComputationBackend;
use binius_math::evaluate_univariate;
use binius_utils::bail;
use getset::CopyGetters;
use itertools::izip;
use std::iter;

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
			.map(|multilinear| switchover_fn(1 << multilinear.log_extension_degree()))
			.collect::<Vec<_>>();
		let max_switchover_round = switchover_rounds.iter().copied().max().unwrap_or_default();

		let multilinears = iter::zip(multilinears, switchover_rounds)
			.map(|(multilinear, switchover_round)| SumcheckMultilinear::Transparent {
				multilinear,
				switchover_round,
			})
			.collect();

		let tensor_query = MultilinearQuery::<_, Backend>::new(max_switchover_round + 1)?;

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
					if *switchover_round == 0 {
						let tensor_query = self.tensor_query.as_ref()
							.expect(
								"tensor_query is guaranteed to be Some while there is still a transparent multilinear"
							);

						// At switchover, perform inner products in large field and save them in a
						// newly created MLE.
						let large_field_folded_multilinear =
							inner_multilinear.evaluate_partial_low(tensor_query.to_ref())?;

						*multilinear = SumcheckMultilinear::Folded {
							large_field_folded_multilinear,
						};
					} else {
						*switchover_round -= 1;
						any_transparent_left = true;
					}
				}
				SumcheckMultilinear::Folded {
					ref mut large_field_folded_multilinear,
				} => {
					// Post-switchover, simply halve large field MLE.
					*large_field_folded_multilinear = large_field_folded_multilinear
						.evaluate_partial_low(single_variable_partial_query)?;
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

	/// Calculate the accumulated evaluations for the first sumcheck round.
	pub fn calculate_first_round_evals<PBase, Evaluator>(
		&self,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundCoeffs<F>>, Error>
	where
		PBase: PackedField<Scalar: ExtensionField<FDomain>> + PackedExtension<FDomain>,
		P: PackedField<Scalar: ExtensionField<PBase::Scalar>> + RepackedExtension<PBase>,
		Evaluator: SumcheckEvaluator<PBase, P> + Sync,
	{
		calculate_first_round_evals(
			self.n_vars,
			&self.multilinears,
			evaluators,
			&self.evaluation_points,
		)
	}

	/// Calculate the accumulated evaluations for an arbitrary sumcheck round.
	///
	/// See [`Self::calculate_first_round_evals`] for an optimized version of this method that
	/// operates over small fields in the first round.
	pub fn calculate_later_round_evals<Evaluator: SumcheckEvaluator<P, P> + Sync>(
		&self,
		evaluators: &[Evaluator],
	) -> Result<Vec<RoundCoeffs<F>>, Error> {
		calculate_later_round_evals(
			self.n_vars,
			self.tensor_query.as_ref().map(Into::into),
			&self.multilinears,
			evaluators,
			&self.evaluation_points,
		)
	}

	/// Calculate the batched round coefficients from the domain evaluations.
	///
	/// This both performs the polynomial interpolation over the evaluations and the mixing with
	/// the batching coefficient.
	pub fn calculate_round_coeffs_from_evals<Interpolator: SumcheckInterpolator<F>>(
		&mut self,
		interpolators: &[Interpolator],
		batch_coeff: F,
		evals: Vec<RoundCoeffs<F>>,
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
}
