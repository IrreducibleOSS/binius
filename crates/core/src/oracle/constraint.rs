// Copyright 2024-2025 Irreducible Inc.

use core::iter::IntoIterator;
use std::sync::Arc;

use binius_field::{Field, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::{ArithCircuit, CompositionPoly};
use binius_utils::bail;

use super::{Error, MultilinearOracleSet, OracleId};
use crate::constraint_system::TableId;

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
/// multilinears.
///
/// The difference from the [`ConstraintSet`] is that the latter is for the public API and this
/// one should is supposed to be used within the core only.
#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct SizedConstraintSet<F: Field> {
	pub n_vars: usize,
	pub oracle_ids: Vec<OracleId>,
	pub constraints: Vec<Constraint<F>>,
}

impl<F: Field> SizedConstraintSet<F> {
	pub fn new(n_vars: usize, u: ConstraintSet<F>) -> Self {
		let oracle_ids = u.oracle_ids;
		let constraints = u.constraints;

		Self {
			n_vars,
			oracle_ids,
			constraints,
		}
	}
}

/// Constraint set is a group of constraints that operate over the same set of oracle-identified
/// multilinears. The multilinears are expected to be of the same size.
#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
pub struct ConstraintSet<F: Field> {
	pub table_id: TableId,
	pub log_values_per_row: usize,
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

	/// Build a single constraint set, requiring that all included oracle n_vars are the same
	pub fn build_one(
		self,
		oracles: &MultilinearOracleSet<impl TowerField>,
	) -> Result<SizedConstraintSet<F>, Error> {
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

		Ok(SizedConstraintSet {
			n_vars,
			oracle_ids,
			constraints,
		})
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
