// Copyright 2024 Irreducible Inc.

use crate::{
	oracle::{BatchId, CommittedBatch, CommittedId, CompositePolyOracle, Error},
	polynomial::{Error as PolynomialError, IdentityCompositionPoly, MultivariatePoly},
};
use binius_field::{Field, TowerField};
use binius_utils::bail;
use getset::{CopyGetters, Getters};
use std::{array, fmt::Debug, sync::Arc};

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

/// Meta struct that lets you add optional `name` for the Multilinear before adding to the
/// [`MultilinearOracleSet`]
pub struct MultilinearOracleSetAddition<'a, F: TowerField> {
	name: Option<String>,
	mut_ref: &'a mut MultilinearOracleSet<F>,
}

impl<'a, F: TowerField> MultilinearOracleSetAddition<'a, F> {
	pub fn transparent(self, poly: impl MultivariatePoly<F> + 'static) -> Result<OracleId, Error> {
		if poly.binary_tower_level() > F::TOWER_LEVEL {
			bail!(Error::TowerLevelTooHigh {
				tower_level: poly.binary_tower_level(),
			});
		}
		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Transparent {
			poly: Arc::new(poly),
			name: self.name,
		});
		Ok(id)
	}

	pub fn committed(mut self, batch_id: BatchId) -> OracleId {
		let name = self.name.take();
		self.add_committed_with_name(batch_id, name)
	}

	pub fn committed_multiple<const N: usize>(mut self, batch_id: BatchId) -> [OracleId; N] {
		match &self.name.take() {
			None => [0; N].map(|_| self.add_committed_with_name(batch_id, None)),
			Some(s) => {
				let x: [usize; N] = array::from_fn(|i| i);
				x.map(|i| self.add_committed_with_name(batch_id, Some(format!("{}_{}", s, i))))
			}
		}
	}

	pub fn repeating(self, id: OracleId, log_count: usize) -> Result<OracleId, Error> {
		if id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id));
		}
		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Repeating {
			inner_id: id,
			log_count,
			name: self.name,
		});
		Ok(id)
	}

	pub fn interleaved(self, id0: OracleId, id1: OracleId) -> Result<OracleId, Error> {
		if id0 >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id0));
		}
		if id1 >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id1));
		}

		let n_vars_0 = self.mut_ref.n_vars(id0);
		let n_vars_1 = self.mut_ref.n_vars(id1);
		if n_vars_0 != n_vars_1 {
			bail!(Error::NumberOfVariablesMismatch);
		}

		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Interleaved {
			poly0: id0,
			poly1: id1,
			name: self.name,
		});
		Ok(id)
	}

	pub fn merged(self, id0: OracleId, id1: OracleId) -> Result<OracleId, Error> {
		if id0 >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id0));
		}
		if id1 >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id1));
		}

		let n_vars_0 = self.mut_ref.n_vars(id0);
		let n_vars_1 = self.mut_ref.n_vars(id1);
		if n_vars_0 != n_vars_1 {
			bail!(Error::NumberOfVariablesMismatch);
		}

		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Merged {
			poly0: id0,
			poly1: id1,
			name: self.name,
		});
		Ok(id)
	}

	pub fn shifted(
		self,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, Error> {
		if id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id));
		}

		let inner_n_vars = self.mut_ref.n_vars(id);
		if block_bits > inner_n_vars {
			bail!(PolynomialError::InvalidBlockSize {
				n_vars: inner_n_vars,
			});
		}

		if offset == 0 || offset >= 1 << block_bits {
			bail!(PolynomialError::InvalidShiftOffset {
				max_shift_offset: (1 << block_bits) - 1,
				shift_offset: offset,
			});
		}

		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Shifted {
			inner_id: id,
			offset,
			block_bits,
			variant,
			name: self.name,
		});
		Ok(id)
	}

	pub fn packed(self, id: OracleId, log_degree: usize) -> Result<OracleId, Error> {
		if id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id));
		}

		let inner_n_vars = self.mut_ref.n_vars(id);
		if log_degree > inner_n_vars {
			bail!(Error::NotEnoughVarsForPacking {
				n_vars: inner_n_vars,
				log_degree,
			});
		}

		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Packed {
			inner_id: id,
			log_degree,
			name: self.name,
		});

		Ok(id)
	}

	pub fn projected(
		self,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<OracleId, Error> {
		let inner_n_vars = self.mut_ref.n_vars(id);
		let values_len = values.len();
		if values_len >= inner_n_vars {
			bail!(Error::InvalidProjection {
				n_vars: inner_n_vars,
				values_len,
			});
		}
		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::Projected {
			inner_id: id,
			values,
			variant,
			name: self.name,
		});
		Ok(id)
	}

	pub fn linear_combination(
		self,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		self.linear_combination_with_offset(n_vars, F::ZERO, inner)
	}

	pub fn linear_combination_with_offset(
		self,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		let inner = inner
			.into_iter()
			.map(|(inner_id, coeff)| {
				if inner_id >= self.mut_ref.oracles.len() {
					return Err(Error::InvalidOracleId(inner_id));
				}
				if self.mut_ref.n_vars(inner_id) != n_vars {
					return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
				}
				Ok((inner_id, coeff))
			})
			.collect::<Result<_, _>>()?;

		let id = self
			.mut_ref
			.add_to_set(MultilinearOracleMeta::LinearCombination {
				n_vars,
				offset,
				inner,
				name: self.name,
			});
		Ok(id)
	}

	pub fn zero_padded(self, id: OracleId, n_vars: usize) -> Result<OracleId, Error> {
		if id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(id));
		}

		if self.mut_ref.n_vars(id) > n_vars {
			bail!(Error::IncorrectNumberOfVariables {
				expected: self.mut_ref.n_vars(id),
			});
		};

		let id = self.mut_ref.add_to_set(MultilinearOracleMeta::ZeroPadded {
			inner_id: id,
			n_vars,
			name: self.name,
		});
		Ok(id)
	}

	fn add_committed_with_name(&mut self, batch_id: BatchId, name: Option<String>) -> OracleId {
		let oracle_id = self.mut_ref.oracles.len();
		let index = self.mut_ref.batches[batch_id].oracle_ids.len();
		self.mut_ref.batches[batch_id].oracle_ids.push(oracle_id);
		self.mut_ref.oracles.push(MultilinearOracleMeta::Committed {
			committed_id: CommittedId { batch_id, index },
			name,
		});
		oracle_id
	}
}

/// Metadata about multilinear oracles.
///
/// This is kept internal to [`MultilinearOracleSet`].
#[derive(Debug, Clone)]
enum MultilinearOracleMeta<F: TowerField> {
	Transparent {
		poly: Arc<dyn MultivariatePoly<F>>,
		name: Option<String>,
	},
	Committed {
		committed_id: CommittedId,
		name: Option<String>,
	},
	Repeating {
		inner_id: OracleId,
		log_count: usize,
		name: Option<String>,
	},
	Interleaved {
		poly0: OracleId,
		poly1: OracleId,
		name: Option<String>,
	},
	Merged {
		poly0: OracleId,
		poly1: OracleId,
		name: Option<String>,
	},
	Shifted {
		inner_id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
		name: Option<String>,
	},
	Packed {
		inner_id: OracleId,
		log_degree: usize,
		name: Option<String>,
	},
	Projected {
		inner_id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
		name: Option<String>,
	},
	LinearCombination {
		n_vars: usize,
		offset: F,
		inner: Vec<(OracleId, F)>,
		name: Option<String>,
	},
	ZeroPadded {
		inner_id: OracleId,
		n_vars: usize,
		name: Option<String>,
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
#[derive(Default, Debug, Clone)]
pub struct MultilinearOracleSet<F: TowerField> {
	batches: Vec<CommittedBatchMeta>,
	oracles: Vec<MultilinearOracleMeta<F>>,
}

impl<F: TowerField> MultilinearOracleSet<F> {
	pub fn new() -> Self {
		Self {
			batches: Vec::new(),
			oracles: Vec::new(),
		}
	}

	pub fn n_batches(&self) -> usize {
		self.batches.len()
	}

	pub fn iter(&self) -> impl Iterator<Item = MultilinearPolyOracle<F>> + '_ {
		(0..self.oracles.len()).map(|id| self.oracle(id))
	}

	pub fn add(&mut self) -> MultilinearOracleSetAddition<F> {
		MultilinearOracleSetAddition {
			name: None,
			mut_ref: self,
		}
	}

	pub fn add_named<S: ToString>(&mut self, s: S) -> MultilinearOracleSetAddition<F> {
		MultilinearOracleSetAddition {
			name: Some(s.to_string()),
			mut_ref: self,
		}
	}

	pub fn is_valid_oracle_id(&self, id: OracleId) -> bool {
		id < self.oracles.len()
	}

	fn add_to_set(&mut self, oracle: MultilinearOracleMeta<F>) -> OracleId {
		let id = self.oracles.len();
		self.oracles.push(oracle);
		id
	}

	pub fn add_transparent(
		&mut self,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, Error> {
		self.add().transparent(poly)
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
		self.add().committed(batch_id)
	}

	pub fn add_committed_multiple<const N: usize>(&mut self, batch_id: BatchId) -> [OracleId; N] {
		self.add().committed_multiple(batch_id)
	}

	pub fn add_repeating(&mut self, id: OracleId, log_count: usize) -> Result<OracleId, Error> {
		self.add().repeating(id, log_count)
	}

	pub fn add_interleaved(&mut self, id0: OracleId, id1: OracleId) -> Result<OracleId, Error> {
		self.add().interleaved(id0, id1)
	}

	pub fn add_merged(&mut self, id0: OracleId, id1: OracleId) -> Result<OracleId, Error> {
		self.add().merged(id0, id1)
	}

	pub fn add_shifted(
		&mut self,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, Error> {
		self.add().shifted(id, offset, block_bits, variant)
	}

	pub fn add_packed(&mut self, id: OracleId, log_degree: usize) -> Result<OracleId, Error> {
		self.add().packed(id, log_degree)
	}

	pub fn add_projected(
		&mut self,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<OracleId, Error> {
		self.add().projected(id, values, variant)
	}

	pub fn add_linear_combination(
		&mut self,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		self.add().linear_combination(n_vars, inner)
	}

	pub fn add_linear_combination_with_offset(
		&mut self,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		self.add()
			.linear_combination_with_offset(n_vars, offset, inner)
	}

	pub fn add_zero_padded(&mut self, id: OracleId, n_vars: usize) -> Result<OracleId, Error> {
		self.add().zero_padded(id, n_vars)
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
			MultilinearOracleMeta::Transparent { poly, name } => {
				MultilinearPolyOracle::Transparent {
					id,
					inner: TransparentPolyOracle::new(poly.clone())
						.expect("polynomial validated by add_transparent"),
					name: name.clone(),
				}
			}
			MultilinearOracleMeta::Committed { committed_id, name } => {
				let batch = &self.batches[committed_id.batch_id];
				MultilinearPolyOracle::Committed {
					id: *committed_id,
					oracle_id: id,
					n_vars: batch.n_vars,
					tower_level: batch.tower_level,
					name: name.clone(),
				}
			}
			MultilinearOracleMeta::Repeating {
				inner_id,
				log_count,
				name,
			} => MultilinearPolyOracle::Repeating {
				id,
				inner: Box::new(self.oracle(*inner_id)),
				log_count: *log_count,
				name: name.clone(),
			},
			MultilinearOracleMeta::Interleaved { poly0, poly1, name } => {
				MultilinearPolyOracle::Interleaved {
					id,
					poly0: Box::new(self.oracle(*poly0)),
					poly1: Box::new(self.oracle(*poly1)),
					name: name.clone(),
				}
			}
			MultilinearOracleMeta::Merged { poly0, poly1, name } => MultilinearPolyOracle::Merged {
				id,
				poly0: Box::new(self.oracle(*poly0)),
				poly1: Box::new(self.oracle(*poly1)),
				name: name.clone(),
			},
			MultilinearOracleMeta::Shifted {
				inner_id,
				offset,
				block_bits,
				variant,
				name,
			} => MultilinearPolyOracle::Shifted {
				id,
				shifted: Shifted::new(self.oracle(*inner_id), *offset, *block_bits, *variant)
					.expect("shift parameters validated by add_shifted"),
				name: name.clone(),
			},
			MultilinearOracleMeta::Packed {
				inner_id,
				log_degree,
				name,
			} => MultilinearPolyOracle::Packed {
				id,
				packed: Packed {
					inner: Box::new(self.oracle(*inner_id)),
					log_degree: *log_degree,
				},
				name: name.clone(),
			},
			MultilinearOracleMeta::Projected {
				inner_id,
				values,
				variant,
				name,
			} => MultilinearPolyOracle::Projected {
				id,
				projected: Projected::new(self.oracle(*inner_id), values.clone(), *variant)
					.expect("projection parameters validated by add_projected"),
				name: name.clone(),
			},
			MultilinearOracleMeta::LinearCombination {
				n_vars,
				offset,
				inner,
				name,
			} => MultilinearPolyOracle::LinearCombination {
				id,
				linear_combination: LinearCombination::new(
					*n_vars,
					*offset,
					inner
						.iter()
						.map(|(inner_id, coeff)| (self.oracle(*inner_id), *coeff)),
				)
				.expect("linear combination parameters validated by add_linear_combination"),
				name: name.clone(),
			},
			MultilinearOracleMeta::ZeroPadded {
				inner_id,
				n_vars,
				name,
			} => MultilinearPolyOracle::ZeroPadded {
				id,
				inner: Box::new(self.oracle(*inner_id)),
				n_vars: *n_vars,
				name: name.clone(),
			},
		}
	}

	pub fn n_vars(&self, id: OracleId) -> usize {
		use MultilinearOracleMeta::*;
		match &self.oracles[id] {
			Transparent { poly, .. } => poly.n_vars(),
			Committed { committed_id, .. } => self.batches[committed_id.batch_id].n_vars,
			Repeating {
				inner_id,
				log_count,
				..
			} => self.n_vars(*inner_id) + log_count,
			Interleaved { poly0, .. } => self.n_vars(*poly0) + 1,
			Merged { poly0, .. } => self.n_vars(*poly0) + 1,
			Shifted { inner_id, .. } => self.n_vars(*inner_id),
			Packed {
				inner_id,
				log_degree,
				..
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
			Transparent { poly, .. } => poly.binary_tower_level(),
			Committed { committed_id, .. } => self.batches[committed_id.batch_id].tower_level,
			Repeating { inner_id, .. } => self.tower_level(*inner_id),
			Interleaved { poly0, poly1, .. } => {
				self.tower_level(*poly0).max(self.tower_level(*poly1))
			}
			Merged { poly0, poly1, .. } => self.tower_level(*poly0).max(self.tower_level(*poly1)),
			Shifted { inner_id, .. } => self.tower_level(*inner_id),
			Packed {
				inner_id,
				log_degree,
				..
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
	Transparent {
		id: OracleId,
		inner: TransparentPolyOracle<F>,
		name: Option<String>,
	},
	Committed {
		id: CommittedId,
		oracle_id: OracleId,
		n_vars: usize,
		tower_level: usize,
		name: Option<String>,
	},
	Repeating {
		id: OracleId,
		inner: Box<MultilinearPolyOracle<F>>,
		log_count: usize,
		name: Option<String>,
	},
	Interleaved {
		id: OracleId,
		poly0: Box<MultilinearPolyOracle<F>>,
		poly1: Box<MultilinearPolyOracle<F>>,
		name: Option<String>,
	},
	Merged {
		id: OracleId,
		poly0: Box<MultilinearPolyOracle<F>>,
		poly1: Box<MultilinearPolyOracle<F>>,
		name: Option<String>,
	},
	Projected {
		id: OracleId,
		projected: Projected<F>,
		name: Option<String>,
	},
	Shifted {
		id: OracleId,
		shifted: Shifted<F>,
		name: Option<String>,
	},
	Packed {
		id: OracleId,
		packed: Packed<F>,
		name: Option<String>,
	},
	LinearCombination {
		id: OracleId,
		linear_combination: LinearCombination<F>,
		name: Option<String>,
	},
	ZeroPadded {
		id: OracleId,
		inner: Box<MultilinearPolyOracle<F>>,
		n_vars: usize,
		name: Option<String>,
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
			bail!(Error::TowerLevelTooHigh {
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
			bail!(Error::InvalidProjection { n_vars, values_len });
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
			bail!(crate::polynomial::error::Error::InvalidBlockSize {
				n_vars: inner.n_vars(),
			});
		}

		if shift_offset == 0 || shift_offset >= 1 << block_size {
			bail!(crate::polynomial::error::Error::InvalidShiftOffset {
				max_shift_offset: (1 << block_size) - 1,
				shift_offset,
			});
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
			Transparent { id, .. } => *id,
			Committed { oracle_id, .. } => *oracle_id,
			Repeating { id, .. } => *id,
			Interleaved { id, .. } => *id,
			Merged { id, .. } => *id,
			Projected { id, .. } => *id,
			Shifted { id, .. } => *id,
			Packed { id, .. } => *id,
			LinearCombination { id, .. } => *id,
			ZeroPadded { id, .. } => *id,
		}
	}

	pub fn label(&self) -> String {
		match self.name() {
			Some(name) => format!("{}: {}", self.type_str(), name),
			None => format!("{}: id={}", self.type_str(), self.id()),
		}
	}

	pub fn name(&self) -> Option<&str> {
		use MultilinearPolyOracle::*;
		match self {
			Transparent { name, .. } => name.as_deref(),
			Committed { name, .. } => name.as_deref(),
			Repeating { name, .. } => name.as_deref(),
			Interleaved { name, .. } => name.as_deref(),
			Merged { name, .. } => name.as_deref(),
			Projected { name, .. } => name.as_deref(),
			Shifted { name, .. } => name.as_deref(),
			Packed { name, .. } => name.as_deref(),
			LinearCombination { name, .. } => name.as_deref(),
			ZeroPadded { name, .. } => name.as_deref(),
		}
	}

	fn type_str(&self) -> &str {
		use MultilinearPolyOracle::*;
		match self {
			Transparent { .. } => "Transparent",
			Committed { .. } => "Committed",
			Repeating { .. } => "Repeating",
			Interleaved { .. } => "Interleaved",
			Merged { .. } => "Merged",
			Projected { .. } => "Projected",
			Shifted { .. } => "Shifted",
			Packed { .. } => "Packed",
			LinearCombination { .. } => "LinearCombination",
			ZeroPadded { .. } => "ZeroPadded",
		}
	}

	pub fn n_vars(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent { inner, .. } => inner.poly().n_vars(),
			Committed { n_vars, .. } => *n_vars,
			Repeating {
				inner, log_count, ..
			} => inner.n_vars() + log_count,
			Interleaved { poly0, .. } => 1 + poly0.n_vars(),
			Merged { poly0, .. } => 1 + poly0.n_vars(),
			Projected { projected, .. } => projected.n_vars(),
			Shifted { shifted, .. } => shifted.inner().n_vars(),
			Packed { packed, .. } => packed.inner().n_vars() - packed.log_degree(),
			LinearCombination {
				linear_combination, ..
			} => linear_combination.n_vars,
			ZeroPadded { n_vars, .. } => *n_vars,
		}
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn binary_tower_level(&self) -> usize {
		use MultilinearPolyOracle::*;
		match self {
			Transparent { inner, .. } => inner.binary_tower_level(),
			Committed { tower_level, .. } => *tower_level,
			Repeating { inner, .. } => inner.binary_tower_level(),
			Interleaved { poly0, poly1, .. } => {
				poly0.binary_tower_level().max(poly1.binary_tower_level())
			}
			Merged { poly0, poly1, .. } => {
				poly0.binary_tower_level().max(poly1.binary_tower_level())
			}
			// TODO: This is wrong, should be F::TOWER_LEVEL
			Projected { projected, .. } => projected.inner().binary_tower_level(),
			Shifted { shifted, .. } => shifted.inner().binary_tower_level(),
			Packed { packed, .. } => packed.log_degree + packed.inner().binary_tower_level(),
			LinearCombination {
				linear_combination, ..
			} => linear_combination
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
