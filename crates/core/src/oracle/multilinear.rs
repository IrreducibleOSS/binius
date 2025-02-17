// Copyright 2024-2025 Irreducible Inc.

use std::{array, fmt::Debug, sync::Arc};

use binius_field::{
	serialization, BinaryField128b, DeserializeCanonical, Field, SerializeCanonical, TowerField,
};
use binius_macros::{DeserializeCanonical, SerializeCanonical};
use binius_utils::bail;
use getset::{CopyGetters, Getters};

use crate::{
	oracle::{CompositePolyOracle, Error},
	polynomial::{Error as PolynomialError, IdentityCompositionPoly, MultivariatePoly},
};

/// Identifier for a multilinear oracle in a [`MultilinearOracleSet`].
pub type OracleId = usize;

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
					self.add_committed_with_name(n_vars, tower_level, Some(format!("{}_{}", s, i)))
				})
			}
		}
	}

	pub fn repeating(self, inner_id: OracleId, log_count: usize) -> Result<OracleId, Error> {
		if inner_id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(inner_id));
		}

		let inner = self.mut_ref.get_from_set(inner_id);

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner.n_vars + log_count,
			tower_level: inner.tower_level,
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
		if inner_id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(inner_id));
		}

		let inner = self.mut_ref.get_from_set(inner_id);
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

		let shifted = Shifted::new(&inner, offset, block_bits, variant)?;

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars: inner.n_vars,
			tower_level: inner.tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::Shifted(shifted),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn packed(self, inner_id: OracleId, log_degree: usize) -> Result<OracleId, Error> {
		if inner_id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(inner_id));
		}

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
		variant: ProjectionVariant,
	) -> Result<OracleId, Error> {
		let inner_n_vars = self.mut_ref.n_vars(inner_id);
		let values_len = values.len();
		if values_len > inner_n_vars {
			bail!(Error::InvalidProjection {
				n_vars: inner_n_vars,
				values_len,
			});
		}

		let inner = self.mut_ref.get_from_set(inner_id);
		// TODO: This is wrong, should be F::TOWER_LEVEL
		let tower_level = inner.binary_tower_level();
		let projected = Projected::new(&inner, values, variant)?;

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
				if inner_id >= self.mut_ref.oracles.len() {
					return Err(Error::InvalidOracleId(inner_id));
				}
				if self.mut_ref.n_vars(inner_id) != n_vars {
					return Err(Error::IncorrectNumberOfVariables { expected: n_vars });
				}
				Ok((self.mut_ref.get_from_set(inner_id), coeff))
			})
			.collect::<Result<Vec<_>, _>>()?;

		let tower_level = inner
			.iter()
			.map(|(oracle, _)| oracle.binary_tower_level())
			.max()
			.unwrap_or(0);

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

	pub fn zero_padded(self, inner_id: OracleId, n_vars: usize) -> Result<OracleId, Error> {
		if inner_id >= self.mut_ref.oracles.len() {
			bail!(Error::InvalidOracleId(inner_id));
		}

		if self.mut_ref.n_vars(inner_id) > n_vars {
			bail!(Error::IncorrectNumberOfVariables {
				expected: self.mut_ref.n_vars(inner_id),
			});
		};

		let inner = self.mut_ref.get_from_set(inner_id);

		let oracle = |id: OracleId| MultilinearPolyOracle {
			id,
			n_vars,
			tower_level: inner.tower_level,
			name: self.name,
			variant: MultilinearPolyVariant::ZeroPadded(inner_id),
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
#[derive(Default, Debug, Clone, SerializeCanonical)]
pub struct MultilinearOracleSet<F: TowerField> {
	oracles: Vec<MultilinearPolyOracle<F>>,
}

impl binius_field::DeserializeCanonical for MultilinearOracleSet<BinaryField128b> {
	fn deserialize_canonical(mut read_buf: impl bytes::Buf) -> Result<Self, serialization::Error>
	where
		Self: Sized,
	{
		Ok(Self {
			oracles: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
		})
	}
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

	fn add_to_set(
		&mut self,
		oracle: impl FnOnce(OracleId) -> MultilinearPolyOracle<F>,
	) -> OracleId {
		let id = self.oracles.len();
		self.oracles.push(oracle(id));
		id
	}

	fn get_from_set(&self, id: OracleId) -> MultilinearPolyOracle<F> {
		self.oracles[id].clone()
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

	pub fn oracle(&self, id: OracleId) -> MultilinearPolyOracle<F> {
		self.oracles[id].clone()
	}

	pub fn n_vars(&self, id: OracleId) -> usize {
		self.oracles[id].n_vars()
	}

	pub fn label(&self, id: OracleId) -> String {
		self.oracles[id].label()
	}

	/// Maximum tower level of the oracle's values over the boolean hypercube.
	pub fn tower_level(&self, id: OracleId) -> usize {
		self.oracles[id].binary_tower_level()
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
#[derive(Debug, Clone, PartialEq, Eq, SerializeCanonical)]
pub struct MultilinearPolyOracle<F: TowerField> {
	pub id: OracleId,
	pub name: Option<String>,
	pub n_vars: usize,
	pub tower_level: usize,
	pub variant: MultilinearPolyVariant<F>,
}

impl DeserializeCanonical for MultilinearPolyOracle<BinaryField128b> {
	fn deserialize_canonical(mut read_buf: impl bytes::Buf) -> Result<Self, serialization::Error>
	where
		Self: Sized,
	{
		Ok(Self {
			id: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			name: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			n_vars: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			tower_level: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			variant: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
		})
	}
}

#[derive(Debug, Clone, PartialEq, Eq, SerializeCanonical)]
pub enum MultilinearPolyVariant<F: TowerField> {
	Committed,
	Transparent(TransparentPolyOracle<F>),
	Repeating { id: usize, log_count: usize },
	Projected(Projected<F>),
	Shifted(Shifted),
	Packed(Packed),
	LinearCombination(LinearCombination<F>),
	ZeroPadded(OracleId),
}

impl DeserializeCanonical for MultilinearPolyVariant<BinaryField128b> {
	fn deserialize_canonical(mut buf: impl bytes::Buf) -> Result<Self, serialization::Error>
	where
		Self: Sized,
	{
		Ok(match u8::deserialize_canonical(&mut buf)? {
			0 => Self::Committed,
			1 => Self::Transparent(DeserializeCanonical::deserialize_canonical(&mut buf)?),
			2 => Self::Repeating {
				id: DeserializeCanonical::deserialize_canonical(&mut buf)?,
				log_count: DeserializeCanonical::deserialize_canonical(&mut buf)?,
			},
			3 => Self::Projected(DeserializeCanonical::deserialize_canonical(&mut buf)?),
			4 => Self::Shifted(DeserializeCanonical::deserialize_canonical(&mut buf)?),
			5 => Self::Packed(DeserializeCanonical::deserialize_canonical(&mut buf)?),
			6 => Self::LinearCombination(DeserializeCanonical::deserialize_canonical(&mut buf)?),
			7 => Self::ZeroPadded(DeserializeCanonical::deserialize_canonical(&mut buf)?),
			variant_index => {
				return Err(serialization::Error::UnknownEnumVariant {
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

impl<F: TowerField> SerializeCanonical for TransparentPolyOracle<F> {
	fn serialize_canonical(
		&self,
		mut write_buf: impl bytes::BufMut,
	) -> Result<(), binius_field::serialization::Error> {
		self.poly.erased_serialize_canonical(&mut write_buf)
	}
}

impl DeserializeCanonical for TransparentPolyOracle<BinaryField128b> {
	fn deserialize_canonical(
		mut read_buf: impl bytes::Buf,
	) -> Result<Self, binius_field::serialization::Error>
	where
		Self: Sized,
	{
		let poly: Box<dyn MultivariatePoly<BinaryField128b>> =
			DeserializeCanonical::deserialize_canonical(&mut read_buf)?;
		Ok(Self { poly: poly.into() })
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, SerializeCanonical, DeserializeCanonical)]
pub enum ProjectionVariant {
	FirstVars,
	LastVars,
}

#[derive(
	Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeCanonical, DeserializeCanonical,
)]
pub struct Projected<F: TowerField> {
	#[get_copy = "pub"]
	id: OracleId,
	#[get = "pub"]
	values: Vec<F>,
	#[get_copy = "pub"]
	projection_variant: ProjectionVariant,
}

impl<F: TowerField> Projected<F> {
	fn new(
		oracle: &MultilinearPolyOracle<F>,
		values: Vec<F>,
		projection_variant: ProjectionVariant,
	) -> Result<Self, Error> {
		if values.len() > oracle.n_vars() {
			bail!(Error::InvalidProjection {
				n_vars: oracle.n_vars(),
				values_len: values.len()
			});
		}
		Ok(Self {
			id: oracle.id(),
			values,
			projection_variant,
		})
	}
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, SerializeCanonical, DeserializeCanonical)]
pub enum ShiftVariant {
	CircularLeft,
	LogicalLeft,
	LogicalRight,
}

#[derive(
	Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeCanonical, DeserializeCanonical,
)]
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
	#[allow(clippy::missing_const_for_fn)]
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

#[derive(
	Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeCanonical, DeserializeCanonical,
)]
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

#[derive(
	Debug, Clone, PartialEq, Eq, Getters, CopyGetters, SerializeCanonical, DeserializeCanonical,
)]
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
			MultilinearPolyVariant::Transparent { .. } => "Transparent",
			MultilinearPolyVariant::Committed { .. } => "Committed",
			MultilinearPolyVariant::Repeating { .. } => "Repeating",
			MultilinearPolyVariant::Projected { .. } => "Projected",
			MultilinearPolyVariant::Shifted { .. } => "Shifted",
			MultilinearPolyVariant::Packed { .. } => "Packed",
			MultilinearPolyVariant::LinearCombination { .. } => "LinearCombination",
			MultilinearPolyVariant::ZeroPadded { .. } => "ZeroPadded",
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
	use binius_field::{BinaryField128b, BinaryField1b, Field, TowerField};

	use super::{MultilinearOracleSet, ProjectionVariant};

	#[test]
	fn add_projection_with_all_vars() {
		type F = BinaryField128b;
		let mut oracles = MultilinearOracleSet::<F>::new();
		let data = oracles.add_committed(5, BinaryField1b::TOWER_LEVEL);
		let projected = oracles
			.add_projected(
				data,
				vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ONE],
				ProjectionVariant::FirstVars,
			)
			.unwrap();
		let _ = oracles.oracle(projected);
	}
}
