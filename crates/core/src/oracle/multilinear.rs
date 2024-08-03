// Copyright 2024 Ulvetanna Inc.

use crate::{
	oracle::{BatchId, CommittedBatch, CommittedId, CompositePolyOracle, Error},
	polynomial::{Error as PolynomialError, IdentityCompositionPoly, MultivariatePoly},
};
use binius_field::{Field, TowerField};
use getset::{CopyGetters, Getters};
use std::{fmt::Debug, sync::Arc};

/// Identifier for a multilinear oracle in a [`MultilinearOracleSet`].
pub type OracleId = usize;

/// Metadata about a batch of committed multilinear polynomials.
///
/// This is kept internal to `MultilinearOracleVec`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommittedBatchMeta {
	oracle_ids: Vec<OracleId>,
	n_vars: usize,
	tower_level: usize,
}

/// Metadata about multilinear oracles.
///
/// This is kept internal to [`MultilinearOracleSet`].
#[derive(Debug, Clone)]
enum MultilinearOracleMeta<F: TowerField> {
	Transparent(Arc<dyn MultivariatePoly<F>>),
	Committed(CommittedId),
	Repeating {
		inner_id: OracleId,
		log_count: usize,
	},
	Interleaved(OracleId, OracleId),
	Merged(OracleId, OracleId),
	Shifted {
		inner_id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	},
	Packed {
		inner_id: OracleId,
		log_degree: usize,
	},
	Projected {
		inner_id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	},
	LinearCombination {
		n_vars: usize,
		offset: F,
		inner: Vec<(OracleId, F)>,
	},
	ZeroPadded {
		inner_id: OracleId,
		n_vars: usize,
	},
}

/// An ordered set of multilinear polynomial oracles.
///
/// The multilinear polynomial oracles form a directed acyclic graph, where each multilinear oracle
/// is either transparent, committed, or derived from one or more others. Each oracle is assigned a
/// unique `OracleId`.
///
/// The oracle set also tracks the committed polynomial in batches where each batch is committed
/// together with a polynomial commitment scheme.
#[derive(Debug, Clone)]
pub struct MultilinearOracleSet<F: TowerField> {
	batches: Vec<CommittedBatchMeta>,
	oracles: Vec<MultilinearOracleMeta<F>>,
}

impl<F: TowerField> MultilinearOracleSet<F> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			batches: Vec::new(),
			oracles: Vec::new(),
		}
	}

	fn add(&mut self, oracle: MultilinearOracleMeta<F>) -> OracleId {
		let id = self.oracles.len();
		self.oracles.push(oracle);
		id
	}

	pub fn add_transparent(
		&mut self,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, Error> {
		if poly.binary_tower_level() > F::TOWER_LEVEL {
			return Err(Error::TowerLevelTooHigh {
				tower_level: poly.binary_tower_level(),
			});
		}
		let id = self.add(MultilinearOracleMeta::Transparent(Arc::new(poly)));
		Ok(id)
	}

	pub fn add_committed_batch(&mut self, n_vars: usize, tower_level: usize) -> BatchId {
		self.batches.push(CommittedBatchMeta {
			oracle_ids: vec![],
			n_vars,
			tower_level,
		});
		self.batches.len() - 1
	}

	pub fn add_committed(&mut self, batch_id: BatchId) -> OracleId {
		let oracle_id = self.oracles.len();
		let index = self.batches[batch_id].oracle_ids.len();
		self.batches[batch_id].oracle_ids.push(oracle_id);
		self.oracles
			.push(MultilinearOracleMeta::Committed(CommittedId { batch_id, index }));
		oracle_id
	}

	pub fn add_committed_multiple<const N: usize>(&mut self, batch_id: BatchId) -> [OracleId; N] {
		[0; N].map(|_| self.add_committed(batch_id))
	}

	pub fn add_repeating(&mut self, id: OracleId, log_count: usize) -> Result<OracleId, Error> {
		if id >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id));
		}
		let id = self.add(MultilinearOracleMeta::Repeating {
			inner_id: id,
			log_count,
		});
		Ok(id)
	}

	pub fn add_interleaved(&mut self, id0: OracleId, id1: OracleId) -> Result<OracleId, Error> {
		if id0 >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id0));
		}
		if id1 >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id1));
		}

		let n_vars_0 = self.n_vars(id0);
		let n_vars_1 = self.n_vars(id1);
		if n_vars_0 != n_vars_1 {
			return Err(Error::NumberOfVariablesMismatch);
		}

		let id = self.add(MultilinearOracleMeta::Interleaved(id0, id1));
		Ok(id)
	}

	pub fn add_merged(&mut self, id0: OracleId, id1: OracleId) -> Result<OracleId, Error> {
		if id0 >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id0));
		}
		if id1 >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id1));
		}

		let n_vars_0 = self.n_vars(id0);
		let n_vars_1 = self.n_vars(id1);
		if n_vars_0 != n_vars_1 {
			return Err(Error::NumberOfVariablesMismatch);
		}

		let id = self.add(MultilinearOracleMeta::Merged(id0, id1));
		Ok(id)
	}

	pub fn add_shifted(
		&mut self,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, Error> {
		if id >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id));
		}

		let inner_n_vars = self.n_vars(id);
		if block_bits > inner_n_vars {
			return Err(PolynomialError::InvalidBlockSize {
				n_vars: inner_n_vars,
			}
			.into());
		}

		if offset == 0 || offset >= 1 << block_bits {
			return Err(PolynomialError::InvalidShiftOffset {
				max_shift_offset: (1 << block_bits) - 1,
				shift_offset: offset,
			}
			.into());
		}

		let id = self.add(MultilinearOracleMeta::Shifted {
			inner_id: id,
			offset,
			block_bits,
			variant,
		});
		Ok(id)
	}

	pub fn add_packed(&mut self, id: OracleId, log_degree: usize) -> Result<OracleId, Error> {
		if id >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id));
		}

		let inner_n_vars = self.n_vars(id);
		if log_degree > inner_n_vars {
			return Err(Error::NotEnoughVarsForPacking {
				n_vars: inner_n_vars,
				log_degree,
			});
		}

		let id = self.add(MultilinearOracleMeta::Packed {
			inner_id: id,
			log_degree,
		});

		Ok(id)
	}

	pub fn add_projected(
		&mut self,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<OracleId, Error> {
		let inner_n_vars = self.n_vars(id);
		let values_len = values.len();
		if values_len >= inner_n_vars {
			return Err(Error::InvalidProjection {
				n_vars: inner_n_vars,
				values_len,
			});
		}
		let id = self.add(MultilinearOracleMeta::Projected {
			inner_id: id,
			values,
			variant,
		});
		Ok(id)
	}

	pub fn add_linear_combination(
		&mut self,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		self.add_linear_combination_with_offset(n_vars, F::ZERO, inner)
	}

	pub fn add_linear_combination_with_offset(
		&mut self,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		let inner = inner
			.into_iter()
			.map(|(inner_id, coeff)| {
				if inner_id >= self.oracles.len() {
					return Err(Error::InvalidOracleId(inner_id));
				}
				if self.n_vars(inner_id) != n_vars {
					return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
				}
				Ok((inner_id, coeff))
			})
			.collect::<Result<_, _>>()?;

		let id = self.add(MultilinearOracleMeta::LinearCombination {
			n_vars,
			offset,
			inner,
		});
		Ok(id)
	}

	pub fn add_zero_padded(&mut self, id: OracleId, n_vars: usize) -> Result<OracleId, Error> {
		if id >= self.oracles.len() {
			return Err(Error::InvalidOracleId(id));
		}

		if self.n_vars(id) > n_vars {
			return Err(Error::IncorrectNumberOfVariables {
				expected: self.n_vars(id),
			});
		};

		let id = self.add(MultilinearOracleMeta::ZeroPadded {
			inner_id: id,
			n_vars,
		});
		Ok(id)
	}

	pub fn committed_batch(&self, id: BatchId) -> CommittedBatch {
		let batch = &self.batches[id];
		CommittedBatch {
			id,
			n_vars: batch.n_vars,
			n_polys: batch.oracle_ids.len(),
			tower_level: batch.tower_level,
		}
	}

	pub fn committed_batches(&self) -> Vec<CommittedBatch> {
		self.batches
			.iter()
			.enumerate()
			.map(|(id, batch)| CommittedBatch {
				id,
				n_vars: batch.n_vars,
				n_polys: batch.oracle_ids.len(),
				tower_level: batch.tower_level,
			})
			.collect()
	}

	pub fn committed_oracle_id(&self, id: CommittedId) -> OracleId {
		let CommittedId { batch_id, index } = id;
		self.batches[batch_id].oracle_ids[index]
	}

	pub fn committed_oracle_ids(&self, batch_id: BatchId) -> impl Iterator<Item = OracleId> {
		self.batches[batch_id].clone().oracle_ids.into_iter()
	}

	pub fn committed_oracle(&self, id: CommittedId) -> MultilinearPolyOracle<F> {
		self.oracle(self.committed_oracle_id(id))
	}

	pub fn oracle(&self, id: OracleId) -> MultilinearPolyOracle<F> {
		match &self.oracles[id] {
			MultilinearOracleMeta::Transparent(poly) => MultilinearPolyOracle::Transparent(
				id,
				TransparentPolyOracle::new(poly.clone())
					.expect("polynomial validated by add_transparent"),
			),
			MultilinearOracleMeta::Committed(CommittedId { batch_id, index }) => {
				let batch = &self.batches[*batch_id];
				MultilinearPolyOracle::Committed {
					id: CommittedId {
						batch_id: *batch_id,
						index: *index,
					},
					oracle_id: id,
					n_vars: batch.n_vars,
					tower_level: batch.tower_level,
				}
			}
			MultilinearOracleMeta::Repeating {
				inner_id,
				log_count,
			} => MultilinearPolyOracle::Repeating {
				id,
				inner: Box::new(self.oracle(*inner_id)),
				log_count: *log_count,
			},
			MultilinearOracleMeta::Interleaved(inner_id_0, inner_id_1) => {
				MultilinearPolyOracle::Interleaved(
					id,
					Box::new(self.oracle(*inner_id_0)),
					Box::new(self.oracle(*inner_id_1)),
				)
			}
			MultilinearOracleMeta::Merged(inner_id_0, inner_id_1) => MultilinearPolyOracle::Merged(
				id,
				Box::new(self.oracle(*inner_id_0)),
				Box::new(self.oracle(*inner_id_1)),
			),
			MultilinearOracleMeta::Shifted {
				inner_id,
				offset,
				block_bits,
				variant,
			} => MultilinearPolyOracle::Shifted(
				id,
				Shifted::new(self.oracle(*inner_id), *offset, *block_bits, *variant)
					.expect("shift parameters validated by add_shifted"),
			),
			MultilinearOracleMeta::Packed {
				inner_id,
				log_degree,
			} => MultilinearPolyOracle::Packed(
				id,
				Packed {
					inner: Box::new(self.oracle(*inner_id)),
					log_degree: *log_degree,
				},
			),
			MultilinearOracleMeta::Projected {
				inner_id,
				values,
				variant,
			} => MultilinearPolyOracle::Projected(
				id,
				Projected::new(self.oracle(*inner_id), values.clone(), *variant)
					.expect("projection parameters validated by add_projected"),
			),
			MultilinearOracleMeta::LinearCombination {
				n_vars,
				offset,
				inner,
			} => MultilinearPolyOracle::LinearCombination(
				id,
				LinearCombination::new(
					*n_vars,
					*offset,
					inner
						.iter()
						.map(|(inner_id, coeff)| (self.oracle(*inner_id), *coeff)),
				)
				.expect("linear combination parameters validated by add_linear_combination"),
			),
			MultilinearOracleMeta::ZeroPadded { inner_id, n_vars } => {
				MultilinearPolyOracle::ZeroPadded {
					id,
					inner: Box::new(self.oracle(*inner_id)),
					n_vars: *n_vars,
				}
			}
		}
	}

	pub fn n_vars(&self, id: OracleId) -> usize {
		use MultilinearOracleMeta::*;
		match &self.oracles[id] {
			Transparent(poly) => poly.n_vars(),
			Committed(CommittedId { batch_id, .. }) => self.batches[*batch_id].n_vars,
			Repeating {
				inner_id,
				log_count,
			} => self.n_vars(*inner_id) + log_count,
			Interleaved(inner_id_0, _) => self.n_vars(*inner_id_0) + 1,
			Merged(inner_id_0, _) => self.n_vars(*inner_id_0) + 1,
			Shifted { inner_id, .. } => self.n_vars(*inner_id),
			Packed {
				inner_id,
				log_degree,
			} => self.n_vars(*inner_id) - log_degree,
			Projected {
				inner_id, values, ..
			} => self.n_vars(*inner_id) - values.len(),
			LinearCombination { n_vars, .. } => *n_vars,
			ZeroPadded { n_vars, .. } => *n_vars,
		}
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn tower_level(&self, id: OracleId) -> usize {
		use MultilinearOracleMeta::*;
		match &self.oracles[id] {
			Transparent(poly) => poly.binary_tower_level(),
			Committed(CommittedId { batch_id, .. }) => self.batches[*batch_id].tower_level,
			Repeating { inner_id, .. } => self.tower_level(*inner_id),
			Interleaved(inner_id_0, inner_id_1) => self
				.tower_level(*inner_id_0)
				.max(self.tower_level(*inner_id_1)),
			Merged(inner_id_0, inner_id_1) => self
				.tower_level(*inner_id_0)
				.max(self.tower_level(*inner_id_1)),
			Shifted { inner_id, .. } => self.tower_level(*inner_id),
			Packed {
				inner_id,
				log_degree,
			} => self.tower_level(*inner_id) + log_degree,
			Projected { .. } => F::TOWER_LEVEL,
			// TODO: We can derive this more tightly by inspecting the coefficients and inner
			// polynomials.
			LinearCombination { .. } => F::TOWER_LEVEL,
			ZeroPadded { .. } => F::TOWER_LEVEL,
		}
	}
}

/// A multilinear polynomial oracle in the polynomial IOP model.
///
/// In the multilinear polynomial IOP model, a prover sends multilinear polynomials to an oracle,
/// and the verifier may at the end of the protocol query their evaluations at chosen points. An
/// oracle is a verifier and prover's shared view of a polynomial that can be queried for
/// evaluations by the verifier.
///
/// There are three fundamental categories of oracles:
///
/// 1. *Transparent oracles*. These are multilinear polynomials with a succinct description and
///    evaluation algorithm that are known to the verifier. When the verifier queries a transparent
///    oracle, it evaluates the polynomial itself.
/// 2. *Committed oracles*. These are polynomials actually sent by the prover. When the polynomial
///    IOP is compiled to an interactive protocol, these polynomial are committed with a polynomial
///    commitment scheme.
/// 3. *Virtual oracles*. A virtual multilinear oracle is not actually sent by the prover, but
///    instead admits an interactive reduction for evaluation queries to evaluation queries to
///    other oracles. This is formalized in [DP23] Section 4.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultilinearPolyOracle<F: Field> {
	Transparent(OracleId, TransparentPolyOracle<F>),
	Committed {
		id: CommittedId,
		oracle_id: OracleId,
		n_vars: usize,
		tower_level: usize,
	},
	Repeating {
		id: OracleId,
		inner: Box<MultilinearPolyOracle<F>>,
		log_count: usize,
	},
	Interleaved(OracleId, Box<MultilinearPolyOracle<F>>, Box<MultilinearPolyOracle<F>>),
	Merged(OracleId, Box<MultilinearPolyOracle<F>>, Box<MultilinearPolyOracle<F>>),
	Projected(OracleId, Projected<F>),
	Shifted(OracleId, Shifted<F>),
	Packed(OracleId, Packed<F>),
	LinearCombination(OracleId, LinearCombination<F>),
	ZeroPadded {
		id: OracleId,
		inner: Box<MultilinearPolyOracle<F>>,
		n_vars: usize,
	},
}

/// A transparent multilinear polynomial oracle.
///
/// See the [`MultilinearPolyOracle`] documentation for context.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct TransparentPolyOracle<F: Field> {
	#[get = "pub"]
	poly: Arc<dyn MultivariatePoly<F>>,
}

impl<F: TowerField> TransparentPolyOracle<F> {
	fn new(poly: Arc<dyn MultivariatePoly<F>>) -> Result<Self, Error> {
		if poly.binary_tower_level() > F::TOWER_LEVEL {
			return Err(Error::TowerLevelTooHigh {
				tower_level: poly.binary_tower_level(),
			});
		}
		Ok(TransparentPolyOracle { poly })
	}
}

impl<F: Field> TransparentPolyOracle<F> {
	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn binary_tower_level(&self) -> usize {
		self.poly.binary_tower_level()
	}
}

impl<F: Field> PartialEq for TransparentPolyOracle<F> {
	fn eq(&self, other: &Self) -> bool {
		Arc::ptr_eq(&self.poly, &other.poly)
	}
}

impl<F: Field> Eq for TransparentPolyOracle<F> {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProjectionVariant {
	FirstVars,
	LastVars,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Projected<F: Field> {
	#[get = "pub"]
	inner: Box<MultilinearPolyOracle<F>>,
	#[get = "pub"]
	values: Vec<F>,
	#[get_copy = "pub"]
	projection_variant: ProjectionVariant,
}

impl<F: Field> Projected<F> {
	fn new(
		inner: MultilinearPolyOracle<F>,
		values: Vec<F>,
		projection_variant: ProjectionVariant,
	) -> Result<Self, Error> {
		let n_vars = inner.n_vars();
		let values_len = values.len();
		if values_len >= n_vars {
			return Err(Error::InvalidProjection { n_vars, values_len });
		}
		Ok(Self {
			inner: inner.into(),
			values,
			projection_variant,
		})
	}

	fn n_vars(&self) -> usize {
		self.inner.n_vars() - self.values.len()
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShiftVariant {
	CircularLeft,
	LogicalLeft,
	LogicalRight,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Shifted<F: Field> {
	inner: Box<MultilinearPolyOracle<F>>,
	#[get_copy = "pub"]
	shift_offset: usize,
	#[get_copy = "pub"]
	block_size: usize,
	#[get_copy = "pub"]
	shift_variant: ShiftVariant,
}

impl<F: Field> Shifted<F> {
	fn new(
		inner: MultilinearPolyOracle<F>,
		shift_offset: usize,
		block_size: usize,
		shift_variant: ShiftVariant,
	) -> Result<Self, Error> {
		if block_size > inner.n_vars() {
			return Err(crate::polynomial::error::Error::InvalidBlockSize {
				n_vars: inner.n_vars(),
			}
			.into());
		}

		if shift_offset == 0 || shift_offset >= 1 << block_size {
			return Err(crate::polynomial::error::Error::InvalidShiftOffset {
				max_shift_offset: (1 << block_size) - 1,
				shift_offset,
			}
			.into());
		}

		Ok(Self {
			inner: inner.into(),
			shift_offset,
			block_size,
			shift_variant,
		})
	}

	pub fn inner(&self) -> &MultilinearPolyOracle<F> {
		&self.inner
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct Packed<F: Field> {
	#[get = "pub"]
	inner: Box<MultilinearPolyOracle<F>>,
	/// The number of tower levels increased by the packing operation.
	///
	/// This is the base 2 logarithm of the field extension, and is called $\kappa$ in [DP23],
	/// Section 4.3.
	///
	/// [DP23]: https://eprint.iacr.org/2023/1784
	#[get_copy = "pub"]
	log_degree: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters)]
pub struct LinearCombination<F: Field> {
	#[get_copy = "pub"]
	n_vars: usize,
	#[get_copy = "pub"]
	offset: F,
	inner: Vec<(Box<MultilinearPolyOracle<F>>, F)>,
}

impl<F: Field> LinearCombination<F> {
	fn new(
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (MultilinearPolyOracle<F>, F)>,
	) -> Result<Self, Error> {
		let inner = inner
			.into_iter()
			.map(|(poly, coeff)| {
				if poly.n_vars() != n_vars {
					return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
				}
				Ok((Box::new(poly), coeff))
			})
			.collect::<Result<_, _>>()?;

		Ok(Self {
			n_vars,
			offset,
			inner,
		})
	}

	pub fn n_polys(&self) -> usize {
		self.inner.len()
	}

	pub fn polys(&self) -> impl Iterator<Item = &MultilinearPolyOracle<F>> {
		self.inner.iter().map(|(poly, _)| poly.as_ref())
	}

	pub fn coefficients(&self) -> impl Iterator<Item = F> + '_ {
		self.inner.iter().map(|(_, coeff)| *coeff)
	}
}

impl<F: Field> MultilinearPolyOracle<F> {
	pub fn id(&self) -> OracleId {
		use MultilinearPolyOracle::*;
		match self {
			Transparent(id, _) => *id,
			Committed { oracle_id, .. } => *oracle_id,
			Repeating { id, .. } => *id,
			Interleaved(id, ..) => *id,
			Merged(id, ..) => *id,
			Projected(id, _) => *id,
			Shifted(id, _) => *id,
			Packed(id, _) => *id,
			LinearCombination(id, _) => *id,
			ZeroPadded { id, .. } => *id,
		}
	}

	pub fn n_vars(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent(_, transparent) => transparent.poly().n_vars(),
			Committed { n_vars, .. } => *n_vars,
			Repeating {
				inner, log_count, ..
			} => inner.n_vars() + log_count,
			Interleaved(_, poly0, ..) => 1 + poly0.n_vars(),
			Merged(_, poly0, ..) => 1 + poly0.n_vars(),
			Projected(_, projected) => projected.n_vars(),
			Shifted(_, shifted) => shifted.inner().n_vars(),
			Packed(_, packed) => packed.inner().n_vars() - packed.log_degree(),
			LinearCombination(_, lin_com) => lin_com.n_vars,
			ZeroPadded { n_vars, .. } => *n_vars,
		}
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn binary_tower_level(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent(_, transparent) => transparent.binary_tower_level(),
			Committed { tower_level, .. } => *tower_level,
			Repeating { inner, .. } => inner.binary_tower_level(),
			Interleaved(_, poly0, poly1) => {
				poly0.binary_tower_level().max(poly1.binary_tower_level())
			}
			Merged(_, poly0, poly1) => poly0.binary_tower_level().max(poly1.binary_tower_level()),
			// TODO: This is wrong, should be F::TOWER_LEVEL
			Projected(_, projected) => projected.inner().binary_tower_level(),
			Shifted(_, shifted) => shifted.inner().binary_tower_level(),
			Packed(_, packed) => packed.log_degree + packed.inner().binary_tower_level(),
			LinearCombination(_, lin_com) => lin_com
				.inner
				.iter()
				.map(|(poly, _)| poly.binary_tower_level())
				.max()
				.unwrap_or(0),
			ZeroPadded { inner, .. } => inner.binary_tower_level(),
		}
	}

	pub fn into_composite(self) -> CompositePolyOracle<F> {
		let composite =
			CompositePolyOracle::new(self.n_vars(), vec![self], IdentityCompositionPoly);
		composite.expect("Can always apply the identity composition to one variable")
	}
}
