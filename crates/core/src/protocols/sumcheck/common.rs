// Copyright 2024 Irreducible Inc.

use super::error::Error;
use binius_field::{
	util::{inner_product_unchecked, powers},
	ExtensionField, Field, PackedField,
};
use binius_math::{CompositionPoly, MultilinearPoly};
use binius_utils::bail;
use getset::{CopyGetters, Getters};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// A claim about the sum of the values of a multilinear composite polynomial over the boolean
/// hypercube.
///
/// This struct contains a composition polynomial and a claimed sum and implicitly refers to a
/// sequence of multilinears that are composed. This is typically embedded within a
/// [`SumcheckClaim`], which contains more metadata about the multilinears (eg. the number of
/// variables they are defined over).
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct CompositeSumClaim<F: Field, Composition> {
	pub composition: Composition,
	pub sum: F,
}

/// A group of claims about the sum of the values of multilinear composite polynomials over the
/// boolean hypercube.
///
/// All polynomials in the group of claims are compositions of the same sequence of multilinear
/// polynomials. By defining [`SumcheckClaim`] in this way, the sumcheck protocol can implement
/// efficient batch proving and verification and reduce to a set of multilinear evaluations of the
/// same polynomials. In other words, this grouping deduplicates prover work and proof data that
/// would be redundant in a more naive implementation.
#[derive(Debug, CopyGetters)]
pub struct SumcheckClaim<F: Field, C> {
	#[getset(get_copy = "pub")]
	n_vars: usize,
	#[getset(get_copy = "pub")]
	n_multilinears: usize,
	composite_sums: Vec<CompositeSumClaim<F, C>>,
}

impl<F: Field, Composition> SumcheckClaim<F, Composition>
where
	Composition: CompositionPoly<F>,
{
	/// Constructs a new sumcheck claim.
	///
	/// ## Throws
	///
	/// * [`Error::InvalidComposition`] if any of the composition polynomials in the composite
	///   claims vector do not have their number of variables equal to `n_multilinears`
	pub fn new(
		n_vars: usize,
		n_multilinears: usize,
		composite_sums: Vec<CompositeSumClaim<F, Composition>>,
	) -> Result<Self, Error> {
		for CompositeSumClaim {
			ref composition, ..
		} in composite_sums.iter()
		{
			if composition.n_vars() != n_multilinears {
				bail!(Error::InvalidComposition {
					expected_n_vars: n_multilinears,
				});
			}
		}
		Ok(Self {
			n_vars,
			n_multilinears,
			composite_sums,
		})
	}

	/// Returns the maximum individual degree of all composite polynomials.
	pub fn max_individual_degree(&self) -> usize {
		self.composite_sums
			.iter()
			.map(|composite_sum| composite_sum.composition.degree())
			.max()
			.unwrap_or(0)
	}

	pub fn composite_sums(&self) -> &[CompositeSumClaim<F, Composition>] {
		&self.composite_sums
	}
}

/// A univariate polynomial in monomial basis.
///
/// The coefficient at position `i` in the inner vector corresponds to the term $X^i$.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoundCoeffs<F: Field>(pub Vec<F>);

impl<F: Field> RoundCoeffs<F> {
	/// Representation in an isomorphic field
	pub fn isomorphic<FI: Field + From<F>>(self) -> RoundCoeffs<FI> {
		RoundCoeffs(self.0.into_iter().map(Into::into).collect())
	}

	/// Truncate one coefficient from the polynomial to a more compact round proof.
	pub fn truncate(mut self) -> RoundProof<F> {
		let new_len = self.0.len().saturating_sub(1);
		self.0.truncate(new_len);
		RoundProof(self)
	}
}

impl<F: Field> Add<&Self> for RoundCoeffs<F> {
	type Output = RoundCoeffs<F>;

	fn add(mut self, rhs: &Self) -> Self::Output {
		self += rhs;
		self
	}
}

impl<F: Field> AddAssign<&Self> for RoundCoeffs<F> {
	fn add_assign(&mut self, rhs: &Self) {
		if self.0.len() < rhs.0.len() {
			self.0.resize(rhs.0.len(), F::ZERO);
		}

		for (lhs_i, &rhs_i) in self.0.iter_mut().zip(rhs.0.iter()) {
			*lhs_i += rhs_i;
		}
	}
}

impl<F: Field> Mul<F> for RoundCoeffs<F> {
	type Output = RoundCoeffs<F>;

	fn mul(mut self, rhs: F) -> Self::Output {
		self *= rhs;
		self
	}
}

impl<F: Field> MulAssign<F> for RoundCoeffs<F> {
	fn mul_assign(&mut self, rhs: F) {
		for coeff in self.0.iter_mut() {
			*coeff *= rhs;
		}
	}
}

/// A sumcheck round proof is a univariate polynomial in monomial basis with the coefficient of the
/// highest-degree term truncated off.
///
/// Since the verifier knows the claimed sum of the polynomial values at the points 0 and 1, the
/// high-degree term coefficient can be easily recovered. Truncating the coefficient off saves a
/// small amount of proof data.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoundProof<F: Field>(pub RoundCoeffs<F>);

impl<F: Field> RoundProof<F> {
	/// Recovers all univariate polynomial coefficients from the compressed round proof.
	///
	/// The prover has sent coefficients for the purported ith round polynomial
	/// $r_i(X) = \sum_{j=0}^d a_j * X^j$.
	/// However, the prover has not sent the highest degree coefficient $a_d$.
	/// The verifier will need to recover this missing coefficient.
	///
	/// Let $s$ denote the current round's claimed sum.
	/// The verifier expects the round polynomial $r_i$ to satisfy the identity
	/// $s = r_i(0) + r_i(1)$.
	/// Using
	///     $r_i(0) = a_0$
	///     $r_i(1) = \sum_{j=0}^d a_j$
	/// There is a unique $a_d$ that allows $r_i$ to satisfy the above identity.
	/// Specifically
	///     $a_d = s - a_0 - \sum_{j=0}^{d-1} a_j$
	///
	/// Not sending the whole round polynomial is an optimization.
	/// In the unoptimized version of the protocol, the verifier will halt and reject
	/// if given a round polynomial that does not satisfy the above identity.
	pub fn recover(self, sum: F) -> RoundCoeffs<F> {
		let RoundProof(RoundCoeffs(mut coeffs)) = self;
		let first_coeff = coeffs.first().copied().unwrap_or(F::ZERO);
		let last_coeff = sum - first_coeff - coeffs.iter().sum::<F>();
		coeffs.push(last_coeff);
		RoundCoeffs(coeffs)
	}

	/// The truncated polynomial coefficients.
	pub fn coeffs(&self) -> &[F] {
		&self.0 .0
	}

	/// Representation in an isomorphic field
	pub fn isomorphic<FI: Field + From<F>>(self) -> RoundProof<FI> {
		RoundProof(self.0.isomorphic())
	}
}

/// A sumcheck batch proof.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Proof<F: Field> {
	/// The round proofs for each round.
	pub rounds: Vec<RoundProof<F>>,
	/// The claimed evaluations of all multilinears at the point defined by the sumcheck verifier
	/// challenges.
	///
	/// The structure is a vector of vectors of field elements. Each entry of the outer vector
	/// corresponds to one [`SumcheckClaim`] in a batch. Each inner vector contains the evaluations
	/// of the multilinears referenced by that claim.
	pub multilinear_evals: Vec<Vec<F>>,
}

impl<F: Field> Proof<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> Proof<FI> {
		Proof {
			rounds: self
				.rounds
				.into_iter()
				.map(|round| round.isomorphic())
				.collect(),
			multilinear_evals: self
				.multilinear_evals
				.into_iter()
				.map(|prover_evals| prover_evals.into_iter().map(Into::into).collect())
				.collect(),
		}
	}
}

#[derive(Debug, PartialEq, Eq)]
pub struct BatchSumcheckOutput<F: Field> {
	pub challenges: Vec<F>,
	pub multilinear_evals: Vec<Vec<F>>,
}

impl<F: Field> BatchSumcheckOutput<F> {
	pub fn isomorphic<FI: Field + From<F>>(self) -> BatchSumcheckOutput<FI> {
		BatchSumcheckOutput {
			challenges: self.challenges.into_iter().map(Into::into).collect(),
			multilinear_evals: self
				.multilinear_evals
				.into_iter()
				.map(|prover_evals| prover_evals.into_iter().map(Into::into).collect())
				.collect(),
		}
	}
}

/// Constructs a switchover function thaw returns the round number where folded multilinear is at
/// least 2^k times smaller (in bytes) than the original, or 1 when not applicable.
pub fn standard_switchover_heuristic(k: isize) -> impl Fn(usize) -> usize + Copy {
	move |extension_degree: usize| {
		((extension_degree.ilog2() as isize + k).max(0) as usize).saturating_sub(1)
	}
}

/// Sumcheck switchover heuristic that begins folding immediately in the first round.
pub fn immediate_switchover_heuristic(_extension_degree: usize) -> usize {
	0
}

/// Determine switchover rounds for a slice of multilinears.
pub fn determine_switchovers<P, M>(
	multilinears: &[M],
	switchover_fn: impl Fn(usize) -> usize,
) -> Vec<usize>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	multilinears
		.iter()
		.map(|multilinear| switchover_fn(1 << multilinear.log_extension_degree()))
		.collect()
}

/// Check that all multilinears in a slice are of the same size.
pub fn equal_n_vars_check<P, M>(multilinears: &[M]) -> Result<usize, Error>
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	let n_vars = multilinears
		.first()
		.map(|multilinear| multilinear.n_vars())
		.unwrap_or_default();
	for multilinear in multilinears {
		if multilinear.n_vars() != n_vars {
			bail!(Error::NumberOfVariablesMismatch);
		}
	}
	Ok(n_vars)
}

/// Check that evaluations of all multilinears can actually be embedded in the scalar
/// type of small field `PBase`.
///
/// Returns binary logarithm of the embedding degree.
pub fn small_field_embedding_degree_check<PBase, P, M>(multilinears: &[M]) -> Result<(), Error>
where
	PBase: PackedField,
	P: PackedField<Scalar: ExtensionField<PBase::Scalar>>,
	M: MultilinearPoly<P>,
{
	let log_embedding_degree = <P::Scalar as ExtensionField<PBase::Scalar>>::LOG_DEGREE;

	for multilinear in multilinears {
		if multilinear.log_extension_degree() < log_embedding_degree {
			bail!(Error::MultilinearEvalsCannotBeEmbeddedInBaseField);
		}
	}

	Ok(())
}

/// Multiply a sequence of field elements by the consecutive powers of `batch_coeff`
pub fn batch_weighted_value<F: Field>(batch_coeff: F, values: impl Iterator<Item = F>) -> F {
	// Multiplying by batch_coeff is important for security!
	batch_coeff * inner_product_unchecked(powers(batch_coeff), values)
}
