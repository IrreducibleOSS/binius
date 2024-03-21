// Copyright 2024 Ulvetanna Inc.

use super::{
	error::Error,
	sumcheck::{SumcheckClaim, SumcheckWitness},
};
use crate::{
	field::{Field, TowerField},
	oracle::{CompositePolyOracle, MultilinearPolyOracle, MultivariatePolyOracle},
	polynomial::{
		CompositionPoly, Error as PolynomialError, MultilinearComposite, MultilinearPoly,
	},
};
use p3_challenger::CanSample;
use std::{borrow::Borrow, cell::RefCell, sync::Arc};
use thread_local::ThreadLocal;

/// Securely mix several sumcheck claims into a single claim.
///
/// The single claim is valid if and only if all of the mixed claims are valid (with high
/// probability). See [`MixComposition`] for details.
pub fn mix_claims<'a, F: TowerField, CH>(
	n_vars: usize,
	multilinears: Vec<MultilinearPolyOracle<F>>,
	claims: impl Iterator<Item = &'a SumcheckClaim<F>>,
	challenger: CH,
) -> Result<SumcheckClaim<F>, Error>
where
	CH: CanSample<F>,
{
	let claims = claims.collect::<Vec<_>>();

	let composition =
		MixComposition::new(&multilinears, claims.iter().map(|claim| &claim.poly), challenger)?;
	let inner_sums = claims.iter().map(|claim| claim.sum).collect::<Vec<_>>();
	let mixed_sum = composition.evaluate_with_inner_evals(&inner_sums)?;
	Ok(SumcheckClaim {
		poly: MultivariatePolyOracle::Composite(CompositePolyOracle::new(
			n_vars,
			multilinears,
			Arc::new(composition),
		)?),
		sum: mixed_sum,
	})
}

/// Construct the sumcheck witness for a mixed sumcheck claim.
pub fn mix_witnesses<F, M, BM>(
	claim: SumcheckClaim<F>,
	multilinears: Vec<BM>,
) -> Result<SumcheckWitness<F, M, BM>, Error>
where
	F: TowerField,
	M: MultilinearPoly<F> + ?Sized,
	BM: Borrow<M>,
{
	let composite = claim.poly.into_composite();
	let witness =
		MultilinearComposite::new(composite.n_vars(), composite.composition(), multilinears)?;
	Ok(witness)
}

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
#[derive(Debug)]
pub struct MixComposition<F: Field> {
	n_vars: usize,
	inner_polys: Vec<MixedPolynomial<F>>,
	inner_query_buffer: ThreadLocal<RefCell<Vec<F>>>,
}

impl<F: Field> Clone for MixComposition<F> {
	fn clone(&self) -> Self {
		Self {
			n_vars: self.n_vars,
			inner_polys: self.inner_polys.clone(),
			inner_query_buffer: ThreadLocal::new(),
		}
	}
}

impl<F: TowerField> MixComposition<F> {
	pub fn new<'a, CH>(
		multilinears: &[MultilinearPolyOracle<F>],
		polys: impl Iterator<Item = &'a MultivariatePolyOracle<F>>,
		mut challenger: CH,
	) -> Result<Self, Error>
	where
		CH: CanSample<F>,
	{
		let challenge = challenger.sample();

		let mut next_coefficient = None;
		let inner_polys = polys
			.map(|constraint| -> Result<_, Error> {
				let composite = constraint.clone().into_composite();
				let indices = composite
					.inner_polys()
					.iter()
					.map(|composite_multilin| {
						multilinears
							.iter()
							.position(|multilin| multilin == composite_multilin)
							.ok_or(Error::MixedMultilinearNotFound)
					})
					.collect::<Result<Vec<_>, _>>()?;
				let coefficient = next_coefficient;
				next_coefficient = Some(next_coefficient.unwrap_or(F::ONE) * challenge);
				Ok(MixedPolynomial {
					composition: composite.composition(),
					coefficient,
					indices,
				})
			})
			.collect::<Result<Vec<_>, _>>()?;

		Ok(MixComposition {
			n_vars: multilinears.len(),
			inner_polys,
			inner_query_buffer: ThreadLocal::new(),
		})
	}

	pub fn evaluate_with_inner_evals(&self, inner_evals: &[F]) -> Result<F, PolynomialError> {
		if inner_evals.len() != self.inner_polys.len() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.inner_polys.len(),
			});
		}

		let coefficients = self.inner_polys.iter().map(|poly| poly.coefficient);
		let eval = inner_evals
			.iter()
			.zip(coefficients)
			.map(|(&inner_eval, coefficient)| match coefficient {
				Some(coeff) => coeff * inner_eval,
				None => inner_eval,
			})
			.sum();
		Ok(eval)
	}
}

impl<F: TowerField> CompositionPoly<F> for MixComposition<F> {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		self.inner_polys
			.iter()
			.map(|poly| poly.composition.degree())
			.max()
			.unwrap_or(0)
	}

	fn evaluate(&self, query: &[F]) -> Result<F, PolynomialError> {
		self.evaluate_packed(query)
	}

	fn evaluate_packed(&self, query: &[F]) -> Result<F, PolynomialError> {
		let mut inner_query = self.inner_query_buffer.get_or_default().borrow_mut();
		inner_query.reserve(query.len());
		let eval = self
			.inner_polys
			.iter()
			.map(|poly| {
				inner_query.clear();
				for idx in poly.indices.iter().copied() {
					inner_query.push(query[idx]);
				}
				let inner_eval = poly.composition.evaluate_packed(&inner_query)?;
				let scaled_eval = match poly.coefficient {
					Some(coeff) => coeff * inner_eval,
					None => inner_eval,
				};
				Ok::<_, PolynomialError>(scaled_eval)
			})
			.sum::<Result<_, _>>()?;
		Ok(eval)
	}

	fn binary_tower_level(&self) -> usize {
		F::TOWER_LEVEL
	}
}

#[derive(Debug, Clone)]
struct MixedPolynomial<F: Field> {
	composition: Arc<dyn CompositionPoly<F>>,
	coefficient: Option<F>,
	indices: Vec<usize>,
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		field::BinaryField128b,
		oracle::{CommittedBatchSpec, CommittedId, MultilinearOracleSet, ShiftVariant},
	};

	struct TestConstantChallenger<F: Field> {
		challenge: F,
	}

	impl<F: Field> CanSample<F> for TestConstantChallenger<F> {
		fn sample(&mut self) -> F {
			self.challenge
		}
	}

	#[derive(Debug)]
	struct AddConstraint;

	impl CompositionPoly<BinaryField128b> for AddConstraint {
		fn n_vars(&self) -> usize {
			4
		}

		fn degree(&self) -> usize {
			1
		}

		fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
			self.evaluate_packed(query)
		}

		fn evaluate_packed(
			&self,
			query: &[BinaryField128b],
		) -> Result<BinaryField128b, PolynomialError> {
			if query.len() != self.n_vars() {
				return Err(PolynomialError::IncorrectQuerySize {
					expected: self.n_vars(),
				});
			}
			let x = query[0];
			let y = query[1];
			let z = query[2];
			let c_in = query[3];
			Ok(x + y + c_in - z)
		}

		fn binary_tower_level(&self) -> usize {
			0
		}
	}

	#[derive(Debug)]
	struct CarryConstraint;

	impl CompositionPoly<BinaryField128b> for CarryConstraint {
		fn n_vars(&self) -> usize {
			4
		}

		fn degree(&self) -> usize {
			2
		}

		fn evaluate(&self, query: &[BinaryField128b]) -> Result<BinaryField128b, PolynomialError> {
			self.evaluate_packed(query)
		}

		fn evaluate_packed(
			&self,
			query: &[BinaryField128b],
		) -> Result<BinaryField128b, PolynomialError> {
			if query.len() != self.n_vars() {
				return Err(PolynomialError::IncorrectQuerySize {
					expected: self.n_vars(),
				});
			}
			let x = query[0];
			let y = query[1];
			let c_in = query[2];
			let c_out = query[3];
			Ok(x * y + x * c_in + y * c_in - c_out)
		}

		fn binary_tower_level(&self) -> usize {
			0
		}
	}

	#[test]
	fn test_mix_addition_polys_with_deterministic_challenge() {
		let mut challenger = TestConstantChallenger {
			challenge: BinaryField128b::new(0x10),
		};

		let n_vars = 12;

		let mut oracles = MultilinearOracleSet::new();
		let batch_id = oracles.add_committed_batch(CommittedBatchSpec {
			round_id: 1,
			n_vars,
			n_polys: 4,
			tower_level: 0,
		});

		let x_oracle_id = oracles.committed_oracle_id(CommittedId { batch_id, index: 0 });
		let y_oracle_id = oracles.committed_oracle_id(CommittedId { batch_id, index: 1 });
		let z_oracle_id = oracles.committed_oracle_id(CommittedId { batch_id, index: 2 });
		let c_out_oracle_id = oracles.committed_oracle_id(CommittedId { batch_id, index: 3 });
		let c_in_oracle_id = oracles
			.add_shifted(c_out_oracle_id, 1, 5, ShiftVariant::LogicalRight)
			.unwrap();

		let x_oracle = oracles.oracle(x_oracle_id);
		let y_oracle = oracles.oracle(y_oracle_id);
		let z_oracle = oracles.oracle(z_oracle_id);
		let c_out_oracle = oracles.oracle(c_out_oracle_id);
		let c_in_oracle = oracles.oracle(c_in_oracle_id);

		let add_constraint = CompositePolyOracle::new(
			n_vars,
			vec![
				x_oracle.clone(),
				y_oracle.clone(),
				z_oracle.clone(),
				c_in_oracle.clone(),
			],
			Arc::new(AddConstraint),
		)
		.unwrap();

		let carry_constraint = CompositePolyOracle::new(
			n_vars,
			vec![
				x_oracle.clone(),
				y_oracle.clone(),
				c_in_oracle.clone(),
				c_out_oracle.clone(),
			],
			Arc::new(CarryConstraint),
		)
		.expect("Failed to create CarryConstraint PolyOracle");

		let mix_composition = MixComposition::new(
			&[
				x_oracle.clone(),
				y_oracle.clone(),
				z_oracle.clone(),
				c_out_oracle.clone(),
				c_in_oracle.clone(),
			],
			[
				MultivariatePolyOracle::Composite(add_constraint.clone()),
				MultivariatePolyOracle::Composite(carry_constraint.clone()),
			]
			.iter(),
			&mut challenger,
		)
		.unwrap();

		assert_eq!(mix_composition.n_vars(), 5);
		assert_eq!(mix_composition.degree(), 2);
		assert_eq!(mix_composition.binary_tower_level(), 7);

		let evals = [0x11, 0x12, 0x13, 0x14, 0x15].map(BinaryField128b::new);
		let mix_eval = mix_composition.evaluate(&evals).unwrap();
		let add_eval = add_constraint
			.composition()
			.evaluate(&[0x11, 0x12, 0x13, 0x15].map(BinaryField128b::new))
			.unwrap();
		let carry_eval = carry_constraint
			.composition()
			.evaluate(&[0x11, 0x12, 0x15, 0x14].map(BinaryField128b::new))
			.unwrap();
		assert_eq!(mix_eval, add_eval + challenger.challenge * carry_eval);

		let mix_eval_with_inner = mix_composition
			.evaluate_with_inner_evals(&[add_eval, carry_eval])
			.unwrap();
		assert_eq!(mix_eval_with_inner, mix_eval);
	}
}
