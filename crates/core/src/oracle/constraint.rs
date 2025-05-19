// Copyright 2024-2025 Irreducible Inc.

use core::iter::IntoIterator;
use std::sync::Arc;

use binius_field::{Field, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::{ArithCircuit, CompositionPoly};
use binius_utils::bail;
use itertools::Itertools;

use super::{Error, MultilinearOracleSet, MultilinearPolyVariant, OracleId};

/// Composition trait object that can be used to create lists of compositions of differing
/// concrete types.
pub type TypeErasedComposition<P> = Arc<dyn CompositionPoly<P>>;

/// Constraint is a type erased composition along with a predicate on its values on the boolean
/// hypercube
#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct Constraint<F: Field> {
	pub name: String,
	pub composition: ArithCircuit<F>,
	pub predicate: ConstraintPredicate<F>,
}

/// Predicate can either be a sum of values of a composition on the hypercube (sumcheck) or equality
/// to zero on the hypercube (zerocheck)
#[derive(Clone, Debug, SerializeBytes, DeserializeBytes)]
pub enum ConstraintPredicate<F: Field> {
	Sum(F),
	Zero,
}

/// Constraint set is a group of constraints that operate over the same set of oracle-identified
/// multilinears
#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct ConstraintSet<F: Field> {
	pub n_vars: usize,
	pub oracle_ids: Vec<OracleId>,
	pub constraints: Vec<Constraint<F>>,
}

// A deferred constraint constructor that instantiates index composition after the superset of
// oracles is known
#[allow(clippy::type_complexity)]
struct UngroupedConstraint<F: Field> {
	name: String,
	oracle_ids: Vec<OracleId>,
	composition: ArithCircuit<F>,
	predicate: ConstraintPredicate<F>,
}

/// A builder struct that turns individual compositions over oraclized multilinears into a set of
/// type erased `IndexComposition` instances operating over a superset of oracles of all
/// constraints.
#[derive(Default)]
pub struct ConstraintSetBuilder<F: Field> {
	constraints: Vec<UngroupedConstraint<F>>,
}

impl<F: Field> ConstraintSetBuilder<F> {
	pub const fn new() -> Self {
		Self {
			constraints: Vec::new(),
		}
	}

	pub fn add_sumcheck(
		&mut self,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		composition: ArithCircuit<F>,
		sum: F,
	) {
		self.constraints.push(UngroupedConstraint {
			name: "sumcheck".into(),
			oracle_ids: oracle_ids.into_iter().collect(),
			composition,
			predicate: ConstraintPredicate::Sum(sum),
		});
	}

	pub fn add_zerocheck(
		&mut self,
		name: impl ToString,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		composition: ArithCircuit<F>,
	) {
		self.constraints.push(UngroupedConstraint {
			name: name.to_string(),
			oracle_ids: oracle_ids.into_iter().collect(),
			composition,
			predicate: ConstraintPredicate::Zero,
		});
	}

	/// Build a single constraint set, requiring that all included oracle n_vars are the same
	pub fn build_one(
		self,
		oracles: &MultilinearOracleSet<impl TowerField>,
	) -> Result<ConstraintSet<F>, Error> {
		let mut oracle_ids = self
			.constraints
			.iter()
			.flat_map(|constraint| constraint.oracle_ids.clone())
			.collect::<Vec<_>>();
		if oracle_ids.is_empty() {
			// Do not bail!, this error is handled in evalcheck.
			return Err(Error::EmptyConstraintSet);
		}
		for id in &oracle_ids {
			if !oracles.is_valid_oracle_id(*id) {
				bail!(Error::InvalidOracleId(*id));
			}
		}
		oracle_ids.sort();
		oracle_ids.dedup();

		let n_vars = oracle_ids
			.first()
			.map(|id| oracles.n_vars(*id))
			.unwrap_or_default();

		for id in &oracle_ids {
			if oracles.n_vars(*id) != n_vars {
				bail!(Error::ConstraintSetNvarsMismatch {
					expected: n_vars,
					got: oracles.n_vars(*id)
				});
			}
		}

		// at this point the superset of oracles is known and index compositions
		// may be finally instantiated
		let constraints =
			self.constraints
				.into_iter()
				.map(|constraint| Constraint {
					name: constraint.name,
					composition: constraint
						.composition
						.remap_vars(&positions(&constraint.oracle_ids, &oracle_ids).expect(
							"precondition: oracle_ids is a superset of constraint.oracle_ids",
						))
						.expect("Infallible by ConstraintSetBuilder invariants."),
					predicate: constraint.predicate,
				})
				.collect();

		Ok(ConstraintSet {
			n_vars,
			oracle_ids,
			constraints,
		})
	}

	/// Create one ConstraintSet for every unique n_vars used.
	///
	/// Note that you can't mix oracles with different n_vars in a single constraint.
	pub fn build(
		self,
		oracles: &MultilinearOracleSet<impl TowerField>,
	) -> Result<Vec<ConstraintSet<F>>, Error> {
		let connected_oracle_chunks = self
			.constraints
			.iter()
			.map(|constraint| constraint.oracle_ids.clone())
			.chain(oracles.polys().filter_map(|oracle| match &oracle.variant {
				MultilinearPolyVariant::Shifted(shifted) => Some(vec![oracle.id(), shifted.id()]),
				MultilinearPolyVariant::LinearCombination(linear_combination) => {
					Some(linear_combination.polys().chain([oracle.id()]).collect())
				}
				_ => None,
			}))
			.collect::<Vec<_>>();

		let connected_oracle_chunks = connected_oracle_chunks
			.iter()
			.map(|x| x.iter().map(|y| y.index()).collect::<Vec<usize>>())
			.collect::<Vec<Vec<usize>>>();

		let groups = binius_utils::graph::connected_components(&connected_oracle_chunks);

		let n_vars_and_constraints = self
			.constraints
			.into_iter()
			.map(|constraint| {
				if constraint.oracle_ids.is_empty() {
					bail!(Error::EmptyConstraintSet);
				}
				for id in &constraint.oracle_ids {
					if !oracles.is_valid_oracle_id(*id) {
						bail!(Error::InvalidOracleId(*id));
					}
				}
				let n_vars = constraint
					.oracle_ids
					.first()
					.map(|id| oracles.n_vars(*id))
					.unwrap();

				for id in &constraint.oracle_ids {
					if oracles.n_vars(*id) != n_vars {
						bail!(Error::ConstraintSetNvarsMismatch {
							expected: n_vars,
							got: oracles.n_vars(*id)
						});
					}
				}
				Ok::<_, Error>((n_vars, constraint))
			})
			.collect::<Result<Vec<_>, _>>()?;

		let grouped_constraints = n_vars_and_constraints
			.into_iter()
			.sorted_by_key(|(_, constraint)| groups[constraint.oracle_ids[0].index()])
			.chunk_by(|(_, constraint)| groups[constraint.oracle_ids[0].index()]);

		let constraint_sets = grouped_constraints
			.into_iter()
			.map(|(_, grouped_constraints)| {
				let mut constraints = vec![];
				let mut oracle_ids = vec![];

				let grouped_constraints = grouped_constraints.into_iter().collect::<Vec<_>>();
				let (n_vars, _) = grouped_constraints[0];

				for (_, constraint) in grouped_constraints {
					oracle_ids.extend(&constraint.oracle_ids);
					constraints.push(constraint);
				}
				oracle_ids.sort();
				oracle_ids.dedup();

				let constraints = constraints
					.into_iter()
					.map(|constraint| Constraint {
						name: constraint.name,
						composition: constraint
							.composition
							.remap_vars(&positions(&constraint.oracle_ids, &oracle_ids).expect(
								"precondition: oracle_ids is a superset of constraint.oracle_ids",
							))
							.expect("Infallible by ConstraintSetBuilder invariants."),
						predicate: constraint.predicate,
					})
					.collect();

				ConstraintSet {
					constraints,
					oracle_ids,
					n_vars,
				}
			})
			.collect();

		Ok(constraint_sets)
	}
}

/// Find index of every subset element within the superset.
/// If the superset contains duplicate elements the index of the first match is used
///
/// Returns None if the subset contains elements that don't exist in the superset
fn positions<T: Eq>(subset: &[T], superset: &[T]) -> Option<Vec<usize>> {
	subset
		.iter()
		.map(|subset_item| {
			superset
				.iter()
				.position(|superset_item| superset_item == subset_item)
		})
		.collect()
}
