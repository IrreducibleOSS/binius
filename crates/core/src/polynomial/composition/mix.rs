// Copyright 2024 Ulvetanna Inc.

use crate::polynomial::{CompositionPoly, Error};
use binius_field::{Field, PackedField, TowerField};
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
pub struct MixComposition<F: Field, IC> {
	n_vars: usize,
	max_individual_degree: usize,

	challenge: F,

	inner_compositions: IC,
}

pub trait HornerCompositions<F: Field> {
	fn evaluate<P: PackedField<Scalar = F>>(
		&self,
		challenge: F,
		query: &[P],
	) -> Result<Option<P>, Error>;
	fn evaluate_with_inner_evals(&self, challenge: F, inner_evals: &[F]) -> Result<F, Error>;
}

impl<F: Field> HornerCompositions<F> for () {
	fn evaluate<P: PackedField<Scalar = F>>(
		&self,
		_challenge: F,
		_query: &[P],
	) -> Result<Option<P>, Error> {
		Ok(None)
	}

	fn evaluate_with_inner_evals(&self, _challenge: F, inner_evals: &[F]) -> Result<F, Error> {
		if !inner_evals.is_empty() {
			return Err(Error::IncorrectInnerEvalsLength);
		}

		Ok(F::ZERO)
	}
}

impl<F: Field, C, IC> HornerCompositions<F> for (Vec<C>, IC)
where
	C: CompositionPoly<F>,
	IC: HornerCompositions<F>,
{
	fn evaluate<P: PackedField<Scalar = F>>(
		&self,
		challenge: F,
		query: &[P],
	) -> Result<Option<P>, Error> {
		let mut acc = self.1.evaluate(challenge, query)?;

		for inner_poly in &self.0 {
			acc = Some(
				inner_poly.evaluate::<P>(query)? + acc.map_or(P::zero(), |tail| tail * challenge),
			);
		}

		Ok(acc)
	}

	fn evaluate_with_inner_evals(&self, challenge: F, inner_evals: &[F]) -> Result<F, Error> {
		if self.0.len() > inner_evals.len() {
			return Err(Error::IncorrectInnerEvalsLength);
		}

		let (tail, head) = inner_evals.split_at(inner_evals.len() - self.0.len());

		let mut acc = self.1.evaluate_with_inner_evals(challenge, tail)?;

		for &inner_eval in head {
			acc = inner_eval + acc * challenge;
		}

		Ok(acc)
	}
}

impl<F, IC> CompositionPoly<F> for MixComposition<F, IC>
where
	F: TowerField,
	IC: HornerCompositions<F> + Clone + Debug + Send + Sync,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.max_individual_degree
	}

	fn evaluate<P: PackedField<Scalar = F>>(&self, query: &[P]) -> Result<P, Error> {
		Ok(self
			.inner_compositions
			.evaluate(self.challenge, query)?
			.unwrap_or(P::zero()))
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

pub fn empty_mix_composition<F: Field>(n_vars: usize, challenge: F) -> MixComposition<F, ()> {
	MixComposition {
		n_vars,
		max_individual_degree: 0,

		challenge,
		inner_compositions: (),
	}
}

impl<F: Field, IC> MixComposition<F, IC> {
	#[allow(clippy::type_complexity)]
	pub fn include<Q>(self, compositions: Q) -> Result<MixComposition<F, (Vec<Q::Item>, IC)>, Error>
	where
		Q: IntoIterator,
		Q::Item: CompositionPoly<F>,
	{
		let compositions_vec = compositions.into_iter().collect::<Vec<_>>();

		let same_nvars = compositions_vec
			.iter()
			.all(|poly| poly.n_vars() == self.n_vars);

		if !same_nvars {
			return Err(Error::IncorrectArityInMixedComposition {
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

impl<F: Field, IC: HornerCompositions<F>> MixComposition<F, IC> {
	pub fn evaluate_with_inner_evals(&self, inner_evals: &[F]) -> Result<F, Error> {
		self.inner_compositions
			.evaluate_with_inner_evals(self.challenge, inner_evals)
	}
}
