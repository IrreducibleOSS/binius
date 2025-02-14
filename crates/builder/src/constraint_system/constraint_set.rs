// Copyright 2024-2025 Irreducible Inc.

use core::iter::IntoIterator;
use std::sync::Arc;

use binius_core::oracle::{Constraint, ConstraintPredicate};
use binius_field::{Field, TowerField};
use binius_math::{ArithExpr, CompositionPolyOS};
use binius_utils::bail;
use itertools::Itertools;

use super::{OracleId, B128};
use crate::error::Error;

#[derive(Debug, Clone)]
pub struct ConstraintSet {
	pub oracle_ids: Vec<OracleId>,
	pub constraints: Vec<Constraint<B128>>,
}

// A deferred constraint constructor that instantiates index composition after the superset of oracles is known
#[allow(clippy::type_complexity)]
struct UngroupedConstraint {
	name: Arc<str>,
	oracle_ids: Vec<OracleId>,
	composition: ArithExpr<B128>,
	predicate: ConstraintPredicate<B128>,
}

/// A builder struct that turns individual compositions over oraclized multilinears into a set of
/// type erased `IndexComposition` instances operating over a superset of oracles of all constraints.
#[derive(Default)]
pub struct ConstraintSetBuilder {
	constraints: Vec<UngroupedConstraint>,
}

impl ConstraintSetBuilder {
	pub const fn new() -> Self {
		Self {
			constraints: Vec::new(),
		}
	}

	pub fn add_sumcheck(
		&mut self,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		composition: ArithExpr<B128>,
		sum: B128,
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
		composition: ArithExpr<B128>,
	) {
		self.constraints.push(UngroupedConstraint {
			name: name.to_string().into(),
			oracle_ids: oracle_ids.into_iter().collect(),
			composition,
			predicate: ConstraintPredicate::Zero,
		});
	}

	/// Build a single constraint set, requiring that all included oracle n_vars are the same
	pub fn build(
		self,
		// oracles: &MultilinearOracleSet<impl TowerField>,
	) -> Result<ConstraintSet, Error> {
		let mut oracle_ids = self
			.constraints
			.iter()
			.flat_map(|constraint| constraint.oracle_ids.clone())
			.collect::<Vec<_>>();
		// if oracle_ids.is_empty() {
		// 	// Do not bail!, this error is handled in evalcheck.
		// 	return Err(Error::EmptyConstraintSet);
		// }
		// for id in &oracle_ids {
		// 	if !oracles.is_valid_oracle_id(*id) {
		// 		bail!(Error::InvalidOracleId(*id));
		// 	}
		// }
		oracle_ids.sort();
		oracle_ids.dedup();

		// let n_vars = oracle_ids
		// 	.first()
		// 	.map(|id| oracles.n_vars(*id))
		// 	.unwrap_or_default();

		// for id in &oracle_ids {
		// 	if oracles.n_vars(*id) != n_vars {
		// 		bail!(Error::ConstraintSetNvarsMismatch {
		// 			expected: n_vars,
		// 			got: oracles.n_vars(*id)
		// 		});
		// 	}
		// }

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
