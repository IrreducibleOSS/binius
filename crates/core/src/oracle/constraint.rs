// Copyright 2024 Ulvetanna Inc.

use super::OracleId;
use crate::{composition::index_composition, polynomial::CompositionPoly};
use binius_field::{Field, PackedField};
use std::sync::Arc;

/// Composition trait object that can be used to create lists of compositions of differing
/// concrete types.
pub type TypeErasedComposition<P> = Arc<dyn CompositionPoly<P>>;

/// Constraint is a type erased composition along with a predicate on its values on the boolean hypercube
#[derive(Clone)]
pub struct Constraint<P: PackedField> {
	pub composition: TypeErasedComposition<P>,
	pub predicate: ConstraintPredicate<P::Scalar>,
}

/// Predicate can either be a sum of values of a composition on the hypercube (sumcheck) or equality to zero
/// on the hypercube (zerocheck)
#[derive(Clone, Debug)]
pub enum ConstraintPredicate<F: Field> {
	Sum(F),
	Zero,
}

impl<F: Field> ConstraintPredicate<F> {
	/// Representation in an isomorphic field
	pub fn isomorphic<FI: Field + From<F>>(self) -> ConstraintPredicate<FI> {
		match self {
			ConstraintPredicate::Sum(sum) => ConstraintPredicate::Sum(sum.into()),
			ConstraintPredicate::Zero => ConstraintPredicate::Zero,
		}
	}
}

/// Constraint set is a group of constraints that operate over the same set of oracle-identified multilinears
#[derive(Clone)]
pub struct ConstraintSet<P: PackedField> {
	pub oracle_ids: Vec<OracleId>,
	pub constraints: Vec<Constraint<P>>,
}

// A deferred constraint constructor that instantiates index composition after the superset of oracles is known
#[allow(clippy::type_complexity)]
struct ConstraintThunk<P: PackedField> {
	composition_thunk: Box<dyn FnOnce(&[OracleId]) -> TypeErasedComposition<P>>,
	predicate: ConstraintPredicate<P::Scalar>,
}

/// A builder struct that turns individual compositions over oraclized multilinears into a set of
/// type erased `IndexComposition` instances operating over a superset of oracles of all constraints.
pub struct ConstraintSetBuilder<P: PackedField> {
	oracle_ids: Vec<OracleId>,
	constraint_thunks: Vec<ConstraintThunk<P>>,
}

impl<P: PackedField> ConstraintSetBuilder<P> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			oracle_ids: Vec::new(),
			constraint_thunks: Vec::new(),
		}
	}

	pub fn add_sumcheck<Composition, const N: usize>(
		&mut self,
		oracle_ids: [OracleId; N],
		composition: Composition,
		sum: P::Scalar,
	) where
		Composition: CompositionPoly<P> + 'static,
	{
		self.oracle_ids.extend(&oracle_ids);
		self.constraint_thunks.push(ConstraintThunk {
			composition_thunk: thunk(oracle_ids, composition),
			predicate: ConstraintPredicate::Sum(sum),
		});
	}

	pub fn add_zerocheck<Composition, const N: usize>(
		&mut self,
		oracle_ids: [OracleId; N],
		composition: Composition,
	) where
		Composition: CompositionPoly<P> + 'static,
	{
		self.oracle_ids.extend(&oracle_ids);
		self.constraint_thunks.push(ConstraintThunk {
			composition_thunk: thunk(oracle_ids, composition),
			predicate: ConstraintPredicate::Zero,
		});
	}

	pub fn build(self) -> ConstraintSet<P> {
		let mut oracle_ids = self.oracle_ids;

		oracle_ids.sort();
		oracle_ids.dedup();

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

		ConstraintSet {
			oracle_ids,
			constraints,
		}
	}
}

// Oracle superset is unknown until the very creation of the constraint set. We
// type erase the index composition constructor to be able to store them in the builder.
//
// Wikipedia: thunk is a subroutine used to inject a calculation into another subroutine.
// Thunks are primarily used to delay a calculation until its result is needed, or to insert
// operations at the beginning or end of the other subroutine.
#[allow(clippy::type_complexity)]
fn thunk<P, Composition, const N: usize>(
	oracle_ids: [OracleId; N],
	composition: Composition,
) -> Box<dyn FnOnce(&[OracleId]) -> TypeErasedComposition<P>>
where
	P: PackedField,
	Composition: CompositionPoly<P> + 'static,
{
	Box::new(move |all_oracle_ids| {
		let indexed = index_composition(all_oracle_ids, oracle_ids, composition)
			.expect("Infallible by ConstraintSetBuilder invariants.");

		Arc::new(indexed)
	})
}
