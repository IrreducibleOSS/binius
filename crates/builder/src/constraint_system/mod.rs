// Copyright 2024-2025 Irreducible Inc.

pub mod builder;
pub mod constraint_set;
pub mod table_builder;

pub type U = binius_field::arch::OptimalUnderlier;
pub type TableId = usize;
pub type OracleId = usize;
pub type ChannelId = usize;
pub type B128 = binius_field::BinaryField128b;

use std::{array, cell::RefCell, fmt::Debug, rc::Rc, sync::Arc};

use binius_core::{constraint_system::channel::Flush, oracle::Constraint};
use binius_field::{arch::OptimalUnderlier, Field, TowerField};
use binius_utils::bail;
pub use builder::ConstraintSystemBuilder;
use constraint_set::ConstraintSet;
// pub use constraint_system::{ConstraintSystem, Filler, Oracle, OracleInfo, OracleVariant, Table};
use getset::{CopyGetters, Getters};
pub use table_builder::TableBuilder;

use crate::error::Error;

// pub type Filler = Box<dyn Fn(&[&[U]], &mut [U])>;

pub struct Filler(Box<dyn Fn(&[&[U]], &mut [U])>);
impl Filler {
	pub fn new(filler: impl Fn(&[&[U]], &mut [U]) + 'static) -> Self {
		Self(Box::new(filler))
	}
}

use std::fmt;
impl<'a> fmt::Debug for Filler {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// Provide a custom message since there’s no useful generic representation.
		f.write_str("Filler { <closure> }")
	}
}
// fn(&[&[U]], &mut [U]);

#[derive(Debug)]
pub struct OracleInfo {
	pub oracle: Oracle,
	pub filler: Option<Filler>,
}

#[derive(Debug, Default)]
pub struct ConstraintSystem {
	pub oracle_infos: Vec<OracleInfo>,
	pub tables: Vec<Table>,
	pub oracles_to_tables: Vec<TableId>,
	pub tables_to_oracles: Option<Vec<Vec<OracleId>>>,
	pub channel_count: usize,
}

#[derive(Debug, Clone)]
pub struct Table {
	pub id: TableId,
	pub name: String,
	pub non_zero_oracle_ids: Vec<OracleId>,
	pub flushes: Vec<Flush>,
	pub constraint_set: ConstraintSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Oracle {
	pub id: OracleId,
	pub name: String,
	pub n_vars: Option<usize>,
	pub tower_level: usize,
	pub variant: OracleVariant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleVariant {
	Original,
	Derived(Derived),
	// Transparent(TransparentPolyOracle<F>),
	Repeated(Repeated),
	Projected(Projected),
	Shifted(Shifted),
	Packed(Packed),
	LinearCombination(LinearCombination),
	ZeroPadded(OracleId),
}

// impl these things

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derived {
	pub dependencies: Vec<OracleId>,
}

impl Oracle {
	pub fn id(&self) -> OracleId {
		self.id
	}

	pub fn new(
		id: OracleId,
		name: String,
		n_vars: Option<usize>,
		tower_level: usize,
		variant: OracleVariant,
	) -> Self {
		Self {
			id,
			name,
			n_vars,
			tower_level,
			variant,
		}
	}

	// pub fn label(&self) -> String {
	// 	format!("{}: {}", self.type_str(), self.name)
	// }

	// fn type_str(&self) -> &str {
	// 	use OracleVariant::*;
	// 	match self.variant {
	// 		Transparent { .. } => "Transparent",
	// 		Committed { .. } => "Committed",
	// 		Repeated { .. } => "Repeating",
	// 		Projected { .. } => "Projected",
	// 		Shifted { .. } => "Shifted",
	// 		Packed { .. } => "Packed",
	// 		LinearCombination { .. } => "LinearCombination",
	// 		ZeroPadded { .. } => "ZeroPadded",
	// 	}
	// }

	pub fn set_n_vars(&mut self, n_vars: usize) {
		self.n_vars = Some(n_vars);
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	// pub fn set_n_var

	// safety: only call when constraint system is intialized
	pub fn n_vars(&self) -> usize {
		self.n_vars.unwrap()
		// match self.n_vars {
		// 	Some(n_vars) => Ok(n_vars),
		// 	None => Err(Error::NVarsWhenUninitialized),
		// }
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn binary_tower_level(&self) -> usize {
		self.tower_level
	}

	// pub fn into_composite(self) -> CompositePolyOracle<F> {
	// 	let composite =
	// 		CompositePolyOracle::new(self.n_vars(), vec![self], IdentityCompositionPoly);
	// 	composite.expect("Can always apply the identity composition to one variable")
	// }
}

//

// TRANSPARENT

// /// A transparent multilinear polynomial oracle.
// ///
// /// See the [`MultilinearPolyOracle`] documentation for context.
// #[derive(Debug, Clone, Getters, CopyGetters)]
// pub struct TransparentPolyOracle<F: Field> {
// 	#[get = "pub"]
// 	poly: Arc<dyn MultivariatePoly<F>>,
// }

// impl<F: TowerField> TransparentPolyOracle<F> {
// 	fn new(poly: Arc<dyn MultivariatePoly<F>>) -> Result<Self, Error> {
// 		if poly.binary_tower_level() > F::TOWER_LEVEL {
// 			bail!(Error::TowerLevelTooHigh {
// 				tower_level: poly.binary_tower_level(),
// 			});
// 		}
// 		Ok(TransparentPolyOracle { poly })
// 	}
// }

// impl<F: Field> TransparentPolyOracle<F> {
// 	/// Maximum tower level of the oracle's values over the boolean hypercube.
// 	pub fn binary_tower_level(&self) -> usize {
// 		self.poly.binary_tower_level()
// 	}
// }

// impl<F: Field> PartialEq for TransparentPolyOracle<F> {
// 	fn eq(&self, other: &Self) -> bool {
// 		Arc::ptr_eq(&self.poly, &other.poly)
// 	}
// }

// impl<F: Field> Eq for TransparentPolyOracle<F> {}

// REPEATED

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]

pub struct Repeated {
	#[get_copy = "pub"]
	id: OracleId,
	#[get_copy = "pub"]
	log_count: Option<usize>,
}
impl Repeated {
	pub fn new(id: OracleId, log_count: Option<usize>) -> Self {
		Self { id, log_count }
	}
	pub fn set_log_count(&mut self, log_count: usize) {
		self.log_count = Some(log_count)
	}
}

// PROJECTED

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProjectionVariant {
	FirstVars,
	LastVars,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Projected {
	#[get_copy = "pub"]
	id: OracleId,
	// for now left constant, independent of height
	#[get = "pub"]
	values: Vec<B128>,
	#[get_copy = "pub"]
	projection_variant: ProjectionVariant,
}

impl Projected {
	fn new(
		// oracles: &MultilinearOracleSet<F>,
		id: OracleId,
		values: Vec<B128>,
		projection_variant: ProjectionVariant,
	) -> Result<Self, Error> {
		// TODO. this is also checked in the caller i think
		// let n_vars = oracles.n_vars(id);
		// let values_len = values.len();
		// if values_len > n_vars {
		// 	bail!(Error::InvalidProjection { n_vars, values_len });
		// }
		Ok(Self {
			id,
			values,
			projection_variant,
		})
	}
}

// SHIFTED

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShiftVariant {
	CircularLeft,
	LogicalLeft,
	LogicalRight,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Shifted {
	#[get_copy = "pub"]
	id: OracleId,
	#[get_copy = "pub"]
	shift_offset: usize,
	// #[get_copy = "pub"]
	block_size: Option<usize>,
	#[get_copy = "pub"]
	shift_variant: ShiftVariant,
}

impl Shifted {
	fn new<F: TowerField>(
		inner_id: OracleId,
		shift_offset: usize,
		block_size: Option<usize>,
		shift_variant: ShiftVariant,
	) -> Result<Self, Error> {
		// TODO
		// if block_size > oracles.n_vars(id) {
		// 	bail!(PolynomialError::InvalidBlockSize {
		// 		n_vars: oracles.n_vars(id),
		// 	});
		// }
		// TODO
		// if shift_offset == 0 || shift_offset >= 1 << block_size {
		// 	bail!(PolynomialError::InvalidShiftOffset {
		// 		max_shift_offset: (1 << block_size) - 1,
		// 		shift_offset,
		// 	});
		// }

		Ok(Self {
			id: inner_id,
			shift_offset,
			block_size,
			shift_variant,
		})
	}

	// for now...
	pub fn block_size(&self) -> usize {
		self.block_size.unwrap()
	}

	pub fn block_size_checked(&self) -> Option<usize> {
		self.block_size
	}

	pub fn set_block_size(&mut self, block_size: usize) {
		// check it...
		self.block_size = Some(block_size)
	}
}

// PACKED, replaceable with an inline struct?

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Packed {
	#[get_copy = "pub"]
	id: OracleId,
	/// The number of tower levels increased by the packing operation.
	///
	/// This is the base 2 logarithm of the field extension, and is called $\kappa$ in [DP23],
	/// Section 4.3.
	///
	/// [DP23]: https://eprint.iacr.org/2023/1784
	#[get_copy = "pub"]
	log_degree: usize,
}

// LINEAR COMBINATION

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct LinearCombination {
	// n_vars: Option<usize>,
	#[get_copy = "pub"]
	offset: B128,
	inner: Vec<(OracleId, B128)>,
}

impl LinearCombination {
	fn new(
		// oracles: &MultilinearOracleSet<F>,
		offset: B128,
		inner: impl IntoIterator<Item = (OracleId, B128)>,
	) -> Result<Self, Error> {
		let inner = inner.into_iter().collect::<Vec<_>>();

		// TODO
		// if !inner.iter().all(|(id, _)| oracles.n_vars(*id) == n_vars) {
		// 	return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
		// }
		Ok(Self {
			// n_vars: None,
			offset,
			inner,
		})
	}

	// pub fn n_vars(&self) -> usize {
	// 	self.n_vars.expect("")
	// }

	pub fn n_polys(&self) -> usize {
		self.inner.len()
	}

	pub fn polys(&self) -> impl Iterator<Item = OracleId> + '_ {
		self.inner.iter().map(|(id, _)| *id)
	}

	pub fn coefficients(&self) -> impl Iterator<Item = B128> + '_ {
		self.inner.iter().map(|(_, coeff)| *coeff)
	}
}

// #[cfg(test)]
// mod tests {
// 	use binius_field::{BinaryField128b, BinaryField1b, Field, TowerField};

// 	use super::{MultilinearOracleSet, ProjectionVariant};

// 	#[test]
// 	fn add_projection_with_all_vars() {
// 		type F = BinaryField128b;
// 		let mut oracles = MultilinearOracleSet::<F>::new();
// 		let data = oracles.add_committed(5, BinaryField1b::TOWER_LEVEL);
// 		let projected = oracles
// 			.add_projected(
// 				data,
// 				vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ONE],
// 				ProjectionVariant::FirstVars,
// 			)
// 			.unwrap();
// 		let _ = oracles.oracle(projected);
// 	}
// }
