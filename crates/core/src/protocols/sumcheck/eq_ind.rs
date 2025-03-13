// Copyright 2025 Irreducible Inc.

use binius_field::{util::eq, Field, PackedExtension, PackedField};
use binius_math::{ArithExpr, CompositionPoly};
use binius_utils::{bail, sorting::is_sorted_ascending};
use getset::{CopyGetters, Getters};

use super::{
	common::{CompositeSumClaim, SumcheckClaim},
	error::{Error, VerificationError},
};
use crate::protocols::sumcheck::BatchSumcheckOutput;

/// TODO expand comment
#[derive(Debug, Clone, CopyGetters)]
pub struct EqIndSumcheckClaim<F: Field, Composition> {
	#[getset(get_copy = "pub")]
	n_vars: usize,
	#[getset(get_copy = "pub")]
	n_multilinears: usize,
	eq_ind_composite_sums: Vec<CompositeSumClaim<F, Composition>>,
}

impl<F: Field, Composition> EqIndSumcheckClaim<F, Composition>
where
	Composition: CompositionPoly<F>,
{
	/// Constructs a new equality indicator sumcheck claim.
	///
	/// ## Throws
	///
	/// * [`Error::InvalidComposition`] if any of the composition polynomials in the composite
	///   claims vector do not have their number of variables equal to `n_multilinears`
	pub fn new(
		n_vars: usize,
		n_multilinears: usize,
		eq_ind_composite_sums: Vec<CompositeSumClaim<F, Composition>>,
	) -> Result<Self, Error> {
		for CompositeSumClaim {
			ref composition, ..
		} in &eq_ind_composite_sums
		{
			if composition.n_vars() != n_multilinears {
				bail!(Error::InvalidComposition {
					actual: composition.n_vars(),
					expected: n_multilinears,
				});
			}
		}
		Ok(Self {
			n_vars,
			n_multilinears,
			eq_ind_composite_sums,
		})
	}

	/// Returns the maximum individual degree of all composite polynomials.
	pub fn max_individual_degree(&self) -> usize {
		self.eq_ind_composite_sums
			.iter()
			.map(|composite_sum| composite_sum.composition.degree())
			.max()
			.unwrap_or(0)
	}

	pub fn eq_ind_composite_sums(&self) -> &[CompositeSumClaim<F, Composition>] {
		&self.eq_ind_composite_sums
	}
}

/// TODO rewrite this comment
/// Requirement: zerocheck challenges have been sampled before this is called
pub fn reduce_to_regular_sumchecks<F: Field, Composition: CompositionPoly<F>>(
	claims: &[EqIndSumcheckClaim<F, Composition>],
) -> Result<Vec<SumcheckClaim<F, ExtraProduct<&Composition>>>, Error> {
	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	claims
		.iter()
		.map(|eq_ind_sumcheck_claim| {
			let EqIndSumcheckClaim {
				n_vars,
				n_multilinears,
				eq_ind_composite_sums,
				..
			} = eq_ind_sumcheck_claim;
			SumcheckClaim::new(
				*n_vars,
				*n_multilinears + 1,
				eq_ind_composite_sums
					.iter()
					.map(|composite_sum| CompositeSumClaim {
						composition: ExtraProduct {
							inner: &composite_sum.composition,
						},
						sum: composite_sum.sum,
					})
					.collect(),
			)
		})
		.collect()
}

/// TODO update this comment
/// Verify the validity of the sumcheck outputs for a reduced zerocheck.
///
/// This takes in the output of the reduced sumcheck protocol and returns the output for the
/// zerocheck instance. This simply strips off the multilinear evaluation of the eq indicator
/// polynomial and verifies that the value is correct.
///
/// Note that due to univariatization of some rounds the number of challenges may be less than
/// the maximum number of variables among claims.
pub fn verify_sumcheck_outputs<F: Field, Composition: CompositionPoly<F>>(
	claims: &[EqIndSumcheckClaim<F, Composition>],
	eq_ind_challenges: &[F],
	sumcheck_output: BatchSumcheckOutput<F>,
) -> Result<BatchSumcheckOutput<F>, Error> {
	let BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		mut multilinear_evals,
	} = sumcheck_output;

	assert_eq!(multilinear_evals.len(), claims.len());

	// Check that the claims are in descending order by n_vars
	if !is_sorted_ascending(claims.iter().map(|claim| claim.n_vars()).rev()) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_n_vars = claims
		.first()
		.map(|claim| claim.n_vars())
		.unwrap_or_default();

	//	assert!(sumcheck_challenges.len() <= max_n_vars);
	assert_eq!(eq_ind_challenges.len(), sumcheck_challenges.len());

	let mut eq_ind_eval = F::ONE;
	let mut last_n_vars = 0;
	for (claim, multilinear_evals) in claims.iter().zip(multilinear_evals.iter_mut()).rev() {
		assert_eq!(claim.n_multilinears() + 1, multilinear_evals.len());

		while last_n_vars < claim.n_vars() && last_n_vars < sumcheck_challenges.len() {
			let sumcheck_challenge =
				sumcheck_challenges[sumcheck_challenges.len() - 1 - last_n_vars];
			let eq_ind_challenge = eq_ind_challenges[eq_ind_challenges.len() - 1 - last_n_vars];
			eq_ind_eval *= eq(sumcheck_challenge, eq_ind_challenge);
			last_n_vars += 1;
		}

		let multilinear_evals_last = multilinear_evals
			.pop()
			.expect("checked above that multilinear_evals length is at least 1");
		if eq_ind_eval != multilinear_evals_last {
			return Err(VerificationError::IncorrectEqIndEvaluation.into());
		}
	}

	Ok(BatchSumcheckOutput {
		challenges: sumcheck_challenges,
		multilinear_evals,
	})
}

#[derive(Debug)]
pub struct ExtraProduct<Composition> {
	pub inner: Composition,
}

impl<P, Composition> CompositionPoly<P> for ExtraProduct<Composition>
where
	P: PackedField,
	Composition: CompositionPoly<P>,
{
	fn n_vars(&self) -> usize {
		self.inner.n_vars() + 1
	}

	fn degree(&self) -> usize {
		self.inner.degree() + 1
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		self.inner.expression() * ArithExpr::Var(self.inner.n_vars())
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		let n_vars = self.n_vars();
		if query.len() != n_vars {
			bail!(binius_math::Error::IncorrectQuerySize { expected: n_vars });
		}

		let inner_eval = self.inner.evaluate(&query[..n_vars - 1])?;
		Ok(inner_eval * query[n_vars - 1])
	}

	fn binary_tower_level(&self) -> usize {
		self.inner.binary_tower_level()
	}
}
