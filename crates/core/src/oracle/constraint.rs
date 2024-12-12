// Copyright 2024 Irreducible Inc.

use super::{Error, MultilinearOracleSet, OracleId};
use binius_field::{Field, TowerField};
use binius_math::{ArithExpr, CompositionPolyOS};
use binius_utils::bail;
use itertools::Itertools;
use std::sync::Arc;

/// Composition trait object that can be used to create lists of compositions of differing
/// concrete types.
pub type TypeErasedComposition<P> = Arc<dyn CompositionPolyOS<P>>;

/// Constraint is a type erased composition along with a predicate on its values on the boolean hypercube
#[derive(Debug, Clone)]
pub struct Constraint<F: Field> {
	pub composition: ArithExpr<F>,
	pub predicate: ConstraintPredicate<F>,
}

/// Predicate can either be a sum of values of a composition on the hypercube (sumcheck) or equality to zero
/// on the hypercube (zerocheck)
#[derive(Clone, Debug)]
pub enum ConstraintPredicate<F: Field> {
	Sum(F),
	Zero,
}

/// Constraint set is a group of constraints that operate over the same set of oracle-identified multilinears
#[derive(Debug, Clone)]
pub struct ConstraintSet<F: Field> {
	pub n_vars: usize,
	pub oracle_ids: Vec<OracleId>,
	pub constraints: Vec<Constraint<F>>,
}

// A deferred constraint constructor that instantiates index composition after the superset of oracles is known
#[allow(clippy::type_complexity)]
struct ConstraintThunk<F: Field> {
	oracle_ids: Vec<OracleId>,
	composition_thunk: Box<dyn FnOnce(&[OracleId]) -> ArithExpr<F>>,
	predicate: ConstraintPredicate<F>,
}

/// A builder struct that turns individual compositions over oraclized multilinears into a set of
/// type erased `IndexComposition` instances operating over a superset of oracles of all constraints.
#[derive(Default)]
pub struct ConstraintSetBuilder<F: Field> {
	constraint_thunks: Vec<ConstraintThunk<F>>,
}

impl<F: Field> ConstraintSetBuilder<F> {
	pub fn new() -> Self {
		Self {
			constraint_thunks: Vec::new(),
		}
	}

	pub fn add_sumcheck<const N: usize>(
		&mut self,
		oracle_ids: [OracleId; N],
		composition: ArithExpr<F>,
		sum: F,
	) {
		self.constraint_thunks.push(ConstraintThunk {
			oracle_ids: oracle_ids.into(),
			composition_thunk: thunk(oracle_ids, composition),
			predicate: ConstraintPredicate::Sum(sum),
		});
	}

	pub fn add_zerocheck<const N: usize>(
		&mut self,
		oracle_ids: [OracleId; N],
		composition: ArithExpr<F>,
	) {
		self.constraint_thunks.push(ConstraintThunk {
			oracle_ids: oracle_ids.into(),
			composition_thunk: thunk(oracle_ids, composition),
			predicate: ConstraintPredicate::Zero,
		});
	}

	/// Build a single constraint set, requiring that all included oracle n_vars are the same
	pub fn build_one(
		self,
		oracles: &MultilinearOracleSet<impl TowerField>,
	) -> Result<ConstraintSet<F>, Error> {
		let mut oracle_ids = self
			.constraint_thunks
			.iter()
			.flat_map(|thunk| thunk.oracle_ids.clone())
			.collect::<Vec<_>>();
		if oracle_ids.is_empty() {
			// Do not bail!, this error is handled in evalcheck.
			return Err(Error::EmptyConstraintSet);
		}
		for id in oracle_ids.iter() {
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

		for id in oracle_ids.iter() {
			if oracles.n_vars(*id) != n_vars {
				bail!(Error::ConstraintSetNvarsMismatch {
					expected: n_vars,
					got: oracles.n_vars(*id)
				});
			}
		}

		// at this point the superset of oracles is known and index compositions
		// may be finally instantiated
		let constraints = self
			.constraint_thunks
			.into_iter()
			.map(|constraint_thunk| {
				let composition = (constraint_thunk.composition_thunk)(&oracle_ids);
				Constraint {
					composition,
					predicate: constraint_thunk.predicate,
				}
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
		let groups = binius_utils::graph::connected_components(
			&self
				.constraint_thunks
				.iter()
				.map(|thunk| thunk.oracle_ids.as_slice())
				.collect::<Vec<_>>(),
		);

		let n_vars_and_constraints = self
			.constraint_thunks
			.into_iter()
			.map(|thunk| {
				if thunk.oracle_ids.is_empty() {
					bail!(Error::EmptyConstraintSet);
				}
				for id in thunk.oracle_ids.iter() {
					if !oracles.is_valid_oracle_id(*id) {
						bail!(Error::InvalidOracleId(*id));
					}
				}
				let n_vars = thunk
					.oracle_ids
					.first()
					.map(|id| oracles.n_vars(*id))
					.unwrap();

				for id in thunk.oracle_ids.iter() {
					if oracles.n_vars(*id) != n_vars {
						bail!(Error::ConstraintSetNvarsMismatch {
							expected: n_vars,
							got: oracles.n_vars(*id)
						});
					}
				}
				Ok::<_, Error>((n_vars, thunk))
			})
			.collect::<Result<Vec<_>, _>>()?;

		let grouped_constraints = n_vars_and_constraints
			.into_iter()
			.sorted_by_key(|(_, thunk)| groups[thunk.oracle_ids[0]])
			.chunk_by(|(_, thunk)| groups[thunk.oracle_ids[0]]);

		let constraint_sets = grouped_constraints
			.into_iter()
			.map(|(_, grouped_thunks)| {
				let mut thunks = vec![];
				let mut oracle_ids = vec![];

				let grouped_thunks = grouped_thunks.into_iter().collect::<Vec<_>>();
				let (n_vars, _) = grouped_thunks[0];

				for (_, thunk) in grouped_thunks {
					oracle_ids.extend(&thunk.oracle_ids);
					thunks.push(thunk);
				}
				oracle_ids.sort();
				oracle_ids.dedup();
				let constraints = thunks
					.into_iter()
					.map(|thunk| Constraint {
						composition: (thunk.composition_thunk)(&oracle_ids),
						predicate: thunk.predicate,
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

// Oracle superset is unknown until the very creation of the constraint set. We
// type erase the index composition constructor to be able to store them in the builder.
//
// Wikipedia: thunk is a subroutine used to inject a calculation into another subroutine.
// Thunks are primarily used to delay a calculation until its result is needed, or to insert
// operations at the beginning or end of the other subroutine.
#[allow(clippy::type_complexity)]
fn thunk<F: Field, const N: usize>(
	oracle_ids: [OracleId; N],
	composition: ArithExpr<F>,
) -> Box<dyn FnOnce(&[OracleId]) -> ArithExpr<F>> {
	Box::new(move |all_oracle_ids| {
		let indices = oracle_ids.map(|subset_item| {
			all_oracle_ids
				.iter()
				.position(|superset_item| superset_item == &subset_item)
				.expect("precondition: all_oracle_ids is a superset of oracle_ids")
		});

		composition
			.remap_vars(&indices)
			.expect("Infallible by ConstraintSetBuilder invariants.")
	})
}
