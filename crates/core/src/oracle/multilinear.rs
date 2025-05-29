// Copyright 2024-2025 Irreducible Inc.

use std::{array, sync::Arc};

use binius_fast_compute::arith_circuit::ArithCircuitPoly;
use binius_field::{BinaryField128b, Field, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::ArithCircuit;
use binius_utils::{
	DeserializeBytes, SerializationError, SerializationMode, SerializeBytes, bail, ensure,
};
use getset::{CopyGetters, Getters};

use crate::{
	oracle::{CompositePolyOracle, Error, OracleId},
	polynomial::{Error as PolynomialError, IdentityCompositionPoly, MultivariatePoly},
};

/// Meta struct that lets you add optional `name` for the Multilinear before adding to the
/// [`MultilinearOracleSet`]
pub struct MultilinearOracleSetAddition<'a, F: TowerField> {
	name: Option<String>,
	mut_ref: &'a mut MultilinearOracleSet<F>,
}

impl<F: TowerField> MultilinearOracleSetAddition<'_, F> {
	pub fn transparent(self, poly: impl MultivariatePoly<F> + 'static) -> Result<OracleId, Error> {
		if poly.binary_tower_level() > F::TOWER_LEVEL {
			bail!(Error::TowerLevelTooHigh {
				tower_level: poly.binary_tower_level(),
			});
		}

		let inner = TransparentPolyOracle::new(Arc::new(poly))?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner.poly.n_vars(),
			tower_level: inner.poly.binary_tower_level(),
			name: self.name,
			variant: MultilinearPolyVariant::Transparent(inner),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn structured(self, n_vars: usize, expr: ArithCircuit<F>) -> Result<OracleId, Error> {
		if expr.binary_tower_level() > F::TOWER_LEVEL {
			bail!(Error::TowerLevelTooHigh {
				tower_level: expr.binary_tower_level(),
			});
		}

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars,
			tower_level: expr.binary_tower_level(),
			name: self.name,
			variant: MultilinearPolyVariant::Structured(expr),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn committed(mut self, n_vars: usize, tower_level: usize) -> OracleId {
		let name = self.name.take();
		self.add_committed_with_name(n_vars, tower_level, name)
	}

	pub fn committed_multiple<const N: usize>(
		mut self,
		n_vars: usize,
		tower_level: usize,
	) -> [OracleId; N] {
		match &self.name.take() {
			None => [0; N].map(|_| self.add_committed_with_name(n_vars, tower_level, None)),
			Some(s) => {
				let x: [usize; N] = array::from_fn(|i| i);
				x.map(|i| {
					self.add_committed_with_name(n_vars, tower_level, Some(format!("{s}_{i}")))
				})
			}
		}
	}

	pub fn repeating(self, inner_id: OracleId, log_count: usize) -> Result<OracleId, Error> {
		ensure!(self.mut_ref.is_valid_oracle_id(inner_id), Error::InvalidOracleId(inner_id));

		let inner = &self.mut_ref[inner_id];

		let tower_level = inner.tower_level;
		let n_vars = inner.n_vars + log_count;
		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::Repeating {
				id: inner_id,
				log_count,
			},
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn shifted(
		self,
		inner_id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, Error> {
		ensure!(self.mut_ref.is_valid_oracle_id(inner_id), Error::InvalidOracleId(inner_id));

		let inner = &self.mut_ref[inner_id];
		if block_bits > inner.n_vars {
			bail!(PolynomialError::InvalidBlockSize {
				n_vars: inner.n_vars,
			});
		}

		if offset == 0 || offset >= 1 << block_bits {
			bail!(PolynomialError::InvalidShiftOffset {
				max_shift_offset: (1 << block_bits) - 1,
				shift_offset: offset,
			});
		}

		let shifted = Shifted::new(inner, offset, block_bits, variant)?;

		let tower_level = inner.tower_level;
		let n_vars = inner.n_vars;
		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::Shifted(shifted),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn packed(self, inner_id: OracleId, log_degree: usize) -> Result<OracleId, Error> {
		ensure!(self.mut_ref.is_valid_oracle_id(inner_id), Error::InvalidOracleId(inner_id));

		let inner_n_vars = self.mut_ref.n_vars(inner_id);
		if log_degree > inner_n_vars {
			bail!(Error::NotEnoughVarsForPacking {
				n_vars: inner_n_vars,
				log_degree,
			});
		}

		let inner_tower_level = self.mut_ref.tower_level(inner_id);

		let packed = Packed {
			id: inner_id,
			log_degree,
		};

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner_n_vars - log_degree,
			tower_level: inner_tower_level + log_degree,
			name: self.name,
			variant: MultilinearPolyVariant::Packed(packed),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn projected(
		self,
		inner_id: OracleId,
		values: Vec<F>,
		start_index: usize,
	) -> Result<OracleId, Error> {
		ensure!(self.mut_ref.is_valid_oracle_id(inner_id), Error::InvalidOracleId(inner_id));

		let inner_n_vars = self.mut_ref.n_vars(inner_id);
		let values_len = values.len();
		if values_len > inner_n_vars {
			bail!(Error::InvalidProjection {
				n_vars: inner_n_vars,
				values_len,
			});
		}

		let inner = &self.mut_ref[inner_id];
		// TODO: This is wrong, should be F::TOWER_LEVEL
		let tower_level = inner.binary_tower_level();
		let projected = Projected::new(inner, values, start_index)?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner_n_vars - values_len,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::Projected(projected),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn projected_last_vars(
		self,
		inner_id: OracleId,
		values: Vec<F>,
	) -> Result<OracleId, Error> {
		ensure!(self.mut_ref.is_valid_oracle_id(inner_id), Error::InvalidOracleId(inner_id));

		let inner_n_vars = self.mut_ref.n_vars(inner_id);
		let start_index = inner_n_vars - values.len();
		let values_len = values.len();
		if values_len > inner_n_vars {
			bail!(Error::InvalidProjection {
				n_vars: inner_n_vars,
				values_len,
			});
		}

		let inner = &self.mut_ref[inner_id];
		// TODO: This is wrong, should be F::TOWER_LEVEL
		let tower_level = inner.binary_tower_level();
		let projected = Projected::new(inner, values, start_index)?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner_n_vars - values_len,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::Projected(projected),
		};

		Ok(self.mut_ref.add_to_set(oracle))
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
				ensure!(
					self.mut_ref.is_valid_oracle_id(inner_id),
					Error::InvalidOracleId(inner_id)
				);
				if self.mut_ref.n_vars(inner_id) != n_vars {
					return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
				}
				Ok((self.mut_ref[inner_id].clone(), coeff))
			})
			.collect::<Result<Vec<_>, _>>()?;

		let tower_level = inner
			.iter()
			.map(|(oracle, coeff)| oracle.binary_tower_level().max(coeff.min_tower_level()))
			.max()
			.unwrap_or(0)
			.max(offset.min_tower_level());

		let linear_combination = LinearCombination::new(n_vars, offset, inner)?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::LinearCombination(linear_combination),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn composite_mle(
		self,
		n_vars: usize,
		inner: impl IntoIterator<Item = OracleId>,
		comp: ArithCircuit<F>,
	) -> Result<OracleId, Error> {
		let inner = inner
			.into_iter()
			.map(|inner_id| {
				ensure!(
					self.mut_ref.is_valid_oracle_id(inner_id),
					Error::InvalidOracleId(inner_id)
				);
				if self.mut_ref.n_vars(inner_id) != n_vars {
					return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
				}
				Ok(self.mut_ref[inner_id].clone())
			})
			.collect::<Result<Vec<_>, _>>()?;

		let tower_level = inner
			.iter()
			.map(|oracle| oracle.binary_tower_level())
			.max()
			.unwrap_or(0);

		let composite_mle = CompositeMLE::new(n_vars, inner, comp)?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::Composite(composite_mle),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn zero_padded(
		self,
		inner_id: OracleId,
		n_pad_vars: usize,
		nonzero_index: usize,
		start_index: usize,
	) -> Result<OracleId, Error> {
		let inner_n_vars = self.mut_ref.n_vars(inner_id);
		if start_index > inner_n_vars {
			bail!(Error::InvalidStartIndex {
				expected: inner_n_vars
			});
		}

		let inner = &self.mut_ref[inner_id];
		let tower_level = inner.binary_tower_level();
		let padded = ZeroPadded::new(inner, n_pad_vars, nonzero_index, start_index)?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner_n_vars + n_pad_vars,
			tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::ZeroPadded(padded),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	fn add_committed_with_name(
		&mut self,
		n_vars: usize,
		tower_level: usize,
		name: Option<String>,
	) -> OracleId {
		let oracle = |oracle_id: OracleId| MultilinearPolyOracle {
			id: oracle_id,
			n_vars,
			tower_level,
			name: name.clone(),
			variant: MultilinearPolyVariant::Committed,
		};

		self.mut_ref.add_to_set(oracle)
	}
}

/// An ordered set of multilinear polynomial oracles.
///
/// The multilinear polynomial oracles form a directed acyclic graph, where each multilinear oracle
/// is either transparent, committed, or derived from one or more others. Each oracle is assigned a
/// unique `OracleId`.
///
/// The oracle set also tracks the committed polynomial in batches where each batch is committed
/// together with a polynomial commitment scheme.
#[derive(Default, Debug, Clone, SerializeBytes, DeserializeBytes)]
#[deserialize_bytes(eval_generics(F = BinaryField128b))]
pub struct MultilinearOracleSet<F: TowerField> {
	oracles: Vec<MultilinearPolyOracle<F>>,
}

impl<F: TowerField> MultilinearOracleSet<F> {
	pub const fn new() -> Self {
		Self {
			oracles: Vec::new(),
		}
	}

	pub fn size(&self) -> usize {
		self.oracles.len()
	}

	pub fn polys(&self) -> impl Iterator<Item = &MultilinearPolyOracle<F>> + '_ {
		(0..self.oracles.len()).map(|index| &self[OracleId::from_index(index)])
	}

	pub fn ids(&self) -> impl Iterator<Item = OracleId> {
		(0..self.oracles.len()).map(OracleId::from_index)
	}

	pub fn iter(&self) -> impl Iterator<Item = (OracleId, &MultilinearPolyOracle<F>)> + '_ {
		(0..self.oracles.len()).map(|index| {
			let oracle_id = OracleId::from_index(index);
			(oracle_id, &self[oracle_id])
		})
	}

	pub const fn add(&mut self) -> MultilinearOracleSetAddition<F> {
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
		id.index() < self.oracles.len()
	}

	fn add_to_set(
		&mut self,
		oracle: impl FnOnce(OracleId) -> MultilinearPolyOracle<F>,
	) -> OracleId {
		let id = OracleId::from_index(self.oracles.len());
		self.oracles.push(oracle(id));
		id
	}

	pub fn add_transparent(
		&mut self,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, Error> {
		self.add().transparent(poly)
	}

	pub fn add_committed(&mut self, n_vars: usize, tower_level: usize) -> OracleId {
		self.add().committed(n_vars, tower_level)
	}

	pub fn add_committed_multiple<const N: usize>(
		&mut self,
		n_vars: usize,
		tower_level: usize,
	) -> [OracleId; N] {
		self.add().committed_multiple(n_vars, tower_level)
	}

	pub fn add_repeating(&mut self, id: OracleId, log_count: usize) -> Result<OracleId, Error> {
		self.add().repeating(id, log_count)
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

	/// Adds a projection to the variables starting at `start_index`.
	pub fn add_projected(
		&mut self,
		id: OracleId,
		values: Vec<F>,
		start_index: usize,
	) -> Result<OracleId, Error> {
		self.add().projected(id, values, start_index)
	}

	/// Adds a projection to the last variables.
	pub fn add_projected_last_vars(
		&mut self,
		id: OracleId,
		values: Vec<F>,
	) -> Result<OracleId, Error> {
		self.add().projected_last_vars(id, values)
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

	pub fn add_zero_padded(
		&mut self,
		id: OracleId,
		n_pad_vars: usize,
		nonzero_index: usize,
		start_index: usize,
	) -> Result<OracleId, Error> {
		self.add()
			.zero_padded(id, n_pad_vars, nonzero_index, start_index)
	}

	pub fn add_composite_mle(
		&mut self,
		n_vars: usize,
		inner: impl IntoIterator<Item = OracleId>,
		comp: ArithCircuit<F>,
	) -> Result<OracleId, Error> {
		self.add().composite_mle(n_vars, inner, comp)
	}

	pub fn n_vars(&self, id: OracleId) -> usize {
		self[id].n_vars()
	}

	pub fn label(&self, id: OracleId) -> String {
		self[id].label()
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn tower_level(&self, id: OracleId) -> usize {
		self[id].binary_tower_level()
	}
}

impl<F: TowerField> std::ops::Index<OracleId> for MultilinearOracleSet<F> {
	type Output = MultilinearPolyOracle<F>;

	fn index(&self, id: OracleId) -> &Self::Output {
		&self.oracles[id.index()]
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
///    instead admits an interactive reduction for evaluation queries to evaluation queries to other
///    oracles. This is formalized in [DP23] Section 4.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
#[derive(Debug, Clone, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
#[deserialize_bytes(eval_generics(F = BinaryField128b))]
pub struct MultilinearPolyOracle<F: TowerField> {
	pub id: OracleId,
	pub name: Option<String>,
	pub n_vars: usize,
	pub tower_level: usize,
	pub variant: MultilinearPolyVariant<F>,
}

#[derive(Debug, Clone, PartialEq, Eq, SerializeBytes)]
pub enum MultilinearPolyVariant<F: TowerField> {
	Committed,
	Transparent(TransparentPolyOracle<F>),
	/// A structured virtual polynomial is one that can be evaluated succinctly by a verifier.
	///
	/// These are referred to as "MLE-structured" tables in [Lasso]. The evaluation algorithm is
	/// expressed as an arithmetic circuit, of polynomial size in the number of variables.
	///
	/// [Lasso]: <https://eprint.iacr.org/2023/1216>
	Structured(ArithCircuit<F>),
	Repeating {
		id: OracleId,
		log_count: usize,
	},
	Projected(Projected<F>),
	Shifted(Shifted),
	Packed(Packed),
	LinearCombination(LinearCombination<F>),
	ZeroPadded(ZeroPadded),
	Composite(CompositeMLE<F>),
}

impl<F: TowerField> MultilinearPolyVariant<F> {
	/// Returns true if this multilinear polynomial variant is committed, as opposed to virtual.
	pub fn is_committed(&self) -> bool {
		matches!(self, Self::Committed)
	}
}

impl DeserializeBytes for MultilinearPolyVariant<BinaryField128b> {
	fn deserialize(
		mut buf: impl bytes::Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(match u8::deserialize(&mut buf, mode)? {
			0 => Self::Committed,
			1 => Self::Transparent(DeserializeBytes::deserialize(buf, mode)?),
			2 => Self::Structured(DeserializeBytes::deserialize(buf, mode)?),
			3 => Self::Repeating {
				id: DeserializeBytes::deserialize(&mut buf, mode)?,
				log_count: DeserializeBytes::deserialize(buf, mode)?,
			},
			4 => Self::Projected(DeserializeBytes::deserialize(buf, mode)?),
			5 => Self::Shifted(DeserializeBytes::deserialize(buf, mode)?),
			6 => Self::Packed(DeserializeBytes::deserialize(buf, mode)?),
			7 => Self::LinearCombination(DeserializeBytes::deserialize(buf, mode)?),
			8 => Self::ZeroPadded(DeserializeBytes::deserialize(buf, mode)?),
			variant_index => {
				return Err(SerializationError::UnknownEnumVariant {
					name: "MultilinearPolyVariant",
					index: variant_index,
				});
			}
		})
	}
}

/// A transparent multilinear polynomial oracle.
///
/// See the [`MultilinearPolyOracle`] documentation for context.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct TransparentPolyOracle<F: Field> {
	#[get = "pub"]
	poly: Arc<dyn MultivariatePoly<F>>,
}

impl<F: TowerField> SerializeBytes for TransparentPolyOracle<F> {
	fn serialize(
		&self,
		mut write_buf: impl bytes::BufMut,
		mode: SerializationMode,
	) -> Result<(), SerializationError> {
		self.poly.erased_serialize(&mut write_buf, mode)
	}
}

impl DeserializeBytes for TransparentPolyOracle<BinaryField128b> {
	fn deserialize(
		read_buf: impl bytes::Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self {
			poly: Box::<dyn MultivariatePoly<BinaryField128b>>::deserialize(read_buf, mode)?.into(),
		})
	}
}

impl<F: TowerField> TransparentPolyOracle<F> {
	fn new(poly: Arc<dyn MultivariatePoly<F>>) -> Result<Self, Error> {
		if poly.binary_tower_level() > F::TOWER_LEVEL {
			bail!(Error::TowerLevelTooHigh {
				tower_level: poly.binary_tower_level(),
			});
		}
		Ok(Self { poly })
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

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeBytes, DeserializeBytes)]
pub struct Projected<F: TowerField> {
	#[get_copy = "pub"]
	id: OracleId,
	#[get = "pub"]
	values: Vec<F>,
	#[get_copy = "pub"]
	start_index: usize,
}

impl<F: TowerField> Projected<F> {
	fn new(
		oracle: &MultilinearPolyOracle<F>,
		values: Vec<F>,
		start_index: usize,
	) -> Result<Self, Error> {
		if values.len() + start_index > oracle.n_vars() {
			bail!(Error::InvalidProjection {
				n_vars: oracle.n_vars(),
				values_len: values.len()
			});
		}
		Ok(Self {
			id: oracle.id(),
			values,
			start_index,
		})
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeBytes, DeserializeBytes)]
pub struct ZeroPadded {
	#[get_copy = "pub"]
	id: OracleId,
	#[get_copy = "pub"]
	n_pad_vars: usize,
	#[get_copy = "pub"]
	nonzero_index: usize,
	#[get_copy = "pub"]
	start_index: usize,
}

impl ZeroPadded {
	fn new<F: TowerField>(
		oracle: &MultilinearPolyOracle<F>,
		n_pad_vars: usize,
		nonzero_index: usize,
		start_index: usize,
	) -> Result<Self, Error> {
		if start_index > oracle.n_vars() {
			bail!(Error::InvalidStartIndex {
				expected: oracle.n_vars(),
			});
		}

		if nonzero_index > 1 << n_pad_vars {
			bail!(Error::InvalidNonzeroIndex {
				expected: 1 << n_pad_vars,
			});
		}

		Ok(Self {
			id: oracle.id(),
			n_pad_vars,
			nonzero_index,
			start_index,
		})
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
pub enum ShiftVariant {
	CircularLeft,
	LogicalLeft,
	LogicalRight,
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeBytes, DeserializeBytes)]
pub struct Shifted {
	#[get_copy = "pub"]
	id: OracleId,
	#[get_copy = "pub"]
	shift_offset: usize,
	#[get_copy = "pub"]
	block_size: usize,
	#[get_copy = "pub"]
	shift_variant: ShiftVariant,
}

impl Shifted {
	fn new<F: TowerField>(
		oracle: &MultilinearPolyOracle<F>,
		shift_offset: usize,
		block_size: usize,
		shift_variant: ShiftVariant,
	) -> Result<Self, Error> {
		if block_size > oracle.n_vars() {
			bail!(PolynomialError::InvalidBlockSize {
				n_vars: oracle.n_vars(),
			});
		}

		if shift_offset == 0 || shift_offset >= 1 << block_size {
			bail!(PolynomialError::InvalidShiftOffset {
				max_shift_offset: (1 << block_size) - 1,
				shift_offset,
			});
		}

		Ok(Self {
			id: oracle.id(),
			shift_offset,
			block_size,
			shift_variant,
		})
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeBytes, DeserializeBytes)]
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

#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeBytes, DeserializeBytes)]
pub struct LinearCombination<F: TowerField> {
	#[get_copy = "pub"]
	n_vars: usize,
	#[get_copy = "pub"]
	offset: F,
	inner: Vec<(OracleId, F)>,
}

impl<F: TowerField> LinearCombination<F> {
	fn new(
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (MultilinearPolyOracle<F>, F)>,
	) -> Result<Self, Error> {
		let inner = inner
			.into_iter()
			.map(|(oracle, value)| {
				if oracle.n_vars() == n_vars {
					Ok((oracle.id(), value))
				} else {
					Err(Error::IncorrectNumberOfVariables { expected: n_vars })
				}
			})
			.collect::<Result<Vec<_>, _>>()?;
		Ok(Self {
			n_vars,
			offset,
			inner,
		})
	}

	pub fn n_polys(&self) -> usize {
		self.inner.len()
	}

	pub fn polys(&self) -> impl Iterator<Item = OracleId> + '_ {
		self.inner.iter().map(|(id, _)| *id)
	}

	pub fn coefficients(&self) -> impl Iterator<Item = F> + '_ {
		self.inner.iter().map(|(_, coeff)| *coeff)
	}
}

/// MLE of a multivariate polynomial evaluated on multilinear oracles.
///
/// i.e. the MLE of the evaluations of $C(M_1, M_2, \ldots, M_n)$ on $\{0, 1\}^\mu$ where:
/// - $C$ is a arbitrary polynomial in $n$ variables
/// - $M_1, M_2, \ldots, M_n$ are multilinear oracles in μ variables
///
/// ($C$ should be sufficiently lightweight to be evaluated by the verifier)
#[derive(Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeBytes)]
pub struct CompositeMLE<F: TowerField> {
	/// $\mu$
	#[get_copy = "pub"]
	n_vars: usize,
	/// $M_1, M_2, \ldots, M_n$
	#[getset(get = "pub")]
	inner: Vec<OracleId>,
	/// $C$
	#[getset(get = "pub")]
	c: ArithCircuitPoly<F>,
}

impl<F: TowerField> CompositeMLE<F> {
	pub fn new(
		n_vars: usize,
		inner: impl IntoIterator<Item = MultilinearPolyOracle<F>>,
		c: ArithCircuit<F>,
	) -> Result<Self, Error> {
		let inner = inner
			.into_iter()
			.map(|oracle| {
				if oracle.n_vars() == n_vars {
					Ok(oracle.id())
				} else {
					Err(Error::IncorrectNumberOfVariables { expected: n_vars })
				}
			})
			.collect::<Result<Vec<_>, _>>()?;
		let c = ArithCircuitPoly::with_n_vars(inner.len(), c)
			.map_err(|_| Error::CompositionMismatch)?; // occurs if `c` has more variables than `inner.len()`
		Ok(Self { n_vars, inner, c })
	}

	pub fn polys(&self) -> impl Iterator<Item = OracleId> + '_ {
		self.inner.iter().copied()
	}

	pub fn n_polys(&self) -> usize {
		self.inner.len()
	}
}

impl<F: TowerField> MultilinearPolyOracle<F> {
	pub const fn id(&self) -> OracleId {
		self.id
	}

	pub fn label(&self) -> String {
		match self.name() {
			Some(name) => format!("{}: {}", self.type_str(), name),
			None => format!("{}: id={}", self.type_str(), self.id()),
		}
	}

	pub fn name(&self) -> Option<&str> {
		self.name.as_deref()
	}

	const fn type_str(&self) -> &str {
		match self.variant {
			MultilinearPolyVariant::Transparent(_) => "Transparent",
			MultilinearPolyVariant::Structured(_) => "Structured",
			MultilinearPolyVariant::Committed => "Committed",
			MultilinearPolyVariant::Repeating { .. } => "Repeating",
			MultilinearPolyVariant::Projected(_) => "Projected",
			MultilinearPolyVariant::Shifted(_) => "Shifted",
			MultilinearPolyVariant::Packed(_) => "Packed",
			MultilinearPolyVariant::LinearCombination(_) => "LinearCombination",
			MultilinearPolyVariant::ZeroPadded(_) => "ZeroPadded",
			MultilinearPolyVariant::Composite(_) => "CompositeMLE",
		}
	}

	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub const fn binary_tower_level(&self) -> usize {
		self.tower_level
	}

	pub fn into_composite(self) -> CompositePolyOracle<F> {
		let composite =
			CompositePolyOracle::new(self.n_vars(), vec![self], IdentityCompositionPoly);
		composite.expect("Can always apply the identity composition to one variable")
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField1b, BinaryField128b, Field, TowerField};

	use super::MultilinearOracleSet;

	#[test]
	fn add_projection_with_all_vars() {
		type F = BinaryField128b;
		let mut oracles = MultilinearOracleSet::<F>::new();
		let data = oracles.add_committed(5, BinaryField1b::TOWER_LEVEL);
		let start_index = 0;
		let projected = oracles
			.add_projected(data, vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ONE], start_index)
			.unwrap();
		let _ = &oracles[projected];
	}
}
