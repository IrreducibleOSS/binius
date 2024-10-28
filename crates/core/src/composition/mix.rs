// Copyright 2024 Irreducible Inc.

use crate::polynomial::Error;
use binius_field::{Field, PackedField, TowerField};
use binius_math::CompositionPoly;
use binius_utils::bail;
use std::fmt::Debug;

/// A composition polynomial that securely batches several underlying multivariate polynomials.
///
/// Given several sumcheck instances over different multivariate polynomials, it is a useful
/// optimization to batch them together by taking a linear combination. Since sumcheck verifies the
/// sum of the values of a multivariate's evaluations, the summation over the hypercube commutes
/// with a linear combination of the polynomials.
///
/// The `MixComposition` chooses the multiplicative coefficients so that a batched sumcheck on the
/// composed polynomial succeeds if and only if all of the underlying sumcheck statements would
/// also succeed (with high probability). The current implementation uses powers of an interactively
/// verifier-sampled challenge as mixing coefficients, and soundness of the batching technique holds
/// following the Schwartz-Zippel lemma.
#[derive(Clone, Debug)]
pub struct MixComposition<P: PackedField, IC> {
	n_vars: usize,
	max_individual_degree: usize,

	challenge: P::Scalar,

	inner_compositions: IC,
}

pub trait HornerCompositions<P: PackedField> {
	fn evaluate(&self, challenge: P::Scalar, query: &[P]) -> Result<Option<P>, Error>;

	fn evaluate_with_inner_evals(
		&self,
		challenge: P::Scalar,
		inner_evals: &[P::Scalar],
	) -> Result<P::Scalar, Error>;
}

impl<P: PackedField> HornerCompositions<P> for () {
	fn evaluate(&self, _challenge: P::Scalar, _query: &[P]) -> Result<Option<P>, Error> {
		Ok(None)
	}

	fn evaluate_with_inner_evals(
		&self,
		_challenge: P::Scalar,
		inner_evals: &[P::Scalar],
	) -> Result<P::Scalar, Error> {
		if !inner_evals.is_empty() {
			bail!(Error::IncorrectInnerEvalsLength);
		}

		Ok(P::Scalar::ZERO)
	}
}

impl<P: PackedField, C, IC> HornerCompositions<P> for (Vec<C>, IC)
where
	C: CompositionPoly<P>,
	IC: HornerCompositions<P>,
{
	fn evaluate(&self, challenge: P::Scalar, query: &[P]) -> Result<Option<P>, Error> {
		let mut acc = self.1.evaluate(challenge, query)?;

		for inner_poly in &self.0 {
			acc =
				Some(inner_poly.evaluate(query)? + acc.map_or(P::zero(), |tail| tail * challenge));
		}

		Ok(acc)
	}

	fn evaluate_with_inner_evals(
		&self,
		challenge: P::Scalar,
		inner_evals: &[P::Scalar],
	) -> Result<P::Scalar, Error> {
		if self.0.len() > inner_evals.len() {
			bail!(Error::IncorrectInnerEvalsLength);
		}

		let (tail, head) = inner_evals.split_at(inner_evals.len() - self.0.len());

		let mut acc = self.1.evaluate_with_inner_evals(challenge, tail)?;

		for &inner_eval in head {
			acc = inner_eval + acc * challenge;
		}

		Ok(acc)
	}
}

impl<P, IC> CompositionPoly<P> for MixComposition<P, IC>
where
	P: PackedField<Scalar: TowerField>,
	IC: HornerCompositions<P> + Clone + Debug + Send + Sync,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.max_individual_degree
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		Ok(self
			.inner_compositions
			.evaluate(self.challenge, query)
			.map_err(|err| binius_math::Error::PolynomialError(Box::new(err)))?
			.unwrap_or(P::zero()))
	}

	fn binary_tower_level(&self) -> usize {
		P::Scalar::TOWER_LEVEL
	}
}

pub fn empty_mix_composition<P: PackedField>(
	n_vars: usize,
	challenge: P::Scalar,
) -> MixComposition<P, ()> {
	MixComposition {
		n_vars,
		max_individual_degree: 0,

		challenge,
		inner_compositions: (),
	}
}

impl<P: PackedField, IC> MixComposition<P, IC> {
	#[allow(clippy::type_complexity)]
	pub fn include<Q>(self, compositions: Q) -> Result<MixComposition<P, (Vec<Q::Item>, IC)>, Error>
	where
		Q: IntoIterator,
		Q::Item: CompositionPoly<P>,
	{
		let compositions_vec = compositions.into_iter().collect::<Vec<_>>();

		let same_nvars = compositions_vec
			.iter()
			.all(|poly| poly.n_vars() == self.n_vars);

		if !same_nvars {
			bail!(Error::IncorrectArityInMixedComposition {
				expected: self.n_vars,
			});
		}

		let max_individual_degrees_new = compositions_vec.iter().map(|poly| poly.degree()).max();
		let max_individual_degree = max_individual_degrees_new
			.unwrap_or(0)
			.max(self.max_individual_degree);

		Ok(MixComposition {
			n_vars: self.n_vars,
			max_individual_degree,

			challenge: self.challenge,
			inner_compositions: (compositions_vec, self.inner_compositions),
		})
	}
}

impl<P: PackedField, IC: HornerCompositions<P>> MixComposition<P, IC> {
	pub fn evaluate_with_inner_evals(&self, inner_evals: &[P::Scalar]) -> Result<P::Scalar, Error> {
		self.inner_compositions
			.evaluate_with_inner_evals(self.challenge, inner_evals)
	}
}
