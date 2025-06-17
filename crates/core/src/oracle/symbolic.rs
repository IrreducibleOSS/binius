// Copyright 2025 Irreducible Inc.

//! A symbolic representation of the multilinear oracle set. The main difference from the concrete
//! one is that it is sizeless, i.e. does not have any information about n_vars.

use std::{array, sync::Arc};

use binius_field::{BinaryField128b, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_math::ArithCircuit;
use binius_utils::{
	DeserializeBytes, SerializationError, SerializationMode, bail,
	checked_arithmetics::log2_ceil_usize,
};

use super::{
	CompositeMLE, LinearCombination, MultilinearOracleSet, MultilinearPolyOracle,
	MultilinearPolyVariant, Packed, Projected, ShiftVariant, Shifted, TransparentPolyOracle,
	ZeroPadded,
};
use crate::{
	constraint_system::TableId,
	oracle::{Error, OracleId},
	polynomial::{Error as PolynomialError, MultivariatePoly},
};

#[derive(Debug, Clone, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
#[deserialize_bytes(eval_generics(F = BinaryField128b))]
pub struct SymbolicMultilinearOracle<F: TowerField> {
	pub id: OracleId,
	pub name: Option<String>,
	pub table_id: TableId,
	pub log_values_per_row: usize,
	pub tower_level: usize,
	pub variant: SymbolicMultilinearPolyVariant<F>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
pub enum ProjectionVariant {
	/// Project values starting at the given index.
	Offset(usize),
	/// Add projection to the last variables.
	Last,
}

#[derive(Debug, Clone, PartialEq, Eq, SerializeBytes)]
pub enum SymbolicMultilinearPolyVariant<F: TowerField> {
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
	},
	Projected {
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	},
	Shifted {
		id: OracleId,
		shift_offset: usize,
		block_size: usize,
		shift_variant: ShiftVariant,
	},
	Packed {
		id: OracleId,
		/// The number of tower levels increased by the packing operation.
		///
		/// This is the base 2 logarithm of the field extension, and is called $\kappa$ in [DP23],
		/// Section 4.3.
		///
		/// [DP23]: https://eprint.iacr.org/2023/1784
		log_degree: usize,
	},
	LinearCombination {
		offset: F,
		inner: Vec<(OracleId, F)>,
	},
	ZeroPadded {
		id: OracleId,
		n_pad_vars: usize,
		nonzero_index: usize,
		start_index: usize,
	},
	Composite {
		inner: Vec<OracleId>,
		circuit: ArithCircuit<F>,
	},
}

impl DeserializeBytes for SymbolicMultilinearPolyVariant<BinaryField128b> {
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
			},
			4 => Self::Projected {
				id: DeserializeBytes::deserialize(&mut buf, mode)?,
				values: DeserializeBytes::deserialize(&mut buf, mode)?,
				variant: DeserializeBytes::deserialize(buf, mode)?,
			},
			5 => Self::Shifted {
				id: DeserializeBytes::deserialize(&mut buf, mode)?,
				shift_offset: DeserializeBytes::deserialize(&mut buf, mode)?,
				block_size: DeserializeBytes::deserialize(&mut buf, mode)?,
				shift_variant: DeserializeBytes::deserialize(buf, mode)?,
			},
			6 => Self::Packed {
				id: DeserializeBytes::deserialize(&mut buf, mode)?,
				log_degree: DeserializeBytes::deserialize(buf, mode)?,
			},
			7 => Self::LinearCombination {
				offset: DeserializeBytes::deserialize(&mut buf, mode)?,
				inner: DeserializeBytes::deserialize(buf, mode)?,
			},
			8 => Self::ZeroPadded {
				id: DeserializeBytes::deserialize(&mut buf, mode)?,
				n_pad_vars: DeserializeBytes::deserialize(&mut buf, mode)?,
				nonzero_index: DeserializeBytes::deserialize(&mut buf, mode)?,
				start_index: DeserializeBytes::deserialize(buf, mode)?,
			},
			9 => Self::Composite {
				inner: DeserializeBytes::deserialize(&mut buf, mode)?,
				circuit: DeserializeBytes::deserialize(buf, mode)?,
			},
			variant_index => {
				return Err(SerializationError::UnknownEnumVariant {
					name: "SymbolicMultilinearPolyVariant",
					index: variant_index,
				});
			}
		})
	}
}

#[derive(Default, Debug, Clone, SerializeBytes, DeserializeBytes)]
#[deserialize_bytes(eval_generics(F = BinaryField128b))]
pub struct SymbolicMultilinearOracleSet<F: TowerField> {
	oracles: Vec<SymbolicMultilinearOracle<F>>,
}

impl<F: TowerField> SymbolicMultilinearOracleSet<F> {
	pub fn new() -> Self {
		Self {
			oracles: Vec::new(),
		}
	}

	/// Instantiate the given symbolic multilinear oracle set into the concrete one.
	pub fn instantiate(&self, table_sizes: &[usize]) -> Result<MultilinearOracleSet<F>, Error> {
		let mut mos = MultilinearOracleSet::new();
		for oracle in &self.oracles {
			let table_size = table_sizes
				.get(oracle.table_id)
				.ok_or(Error::TableSizeMissing {
					table_id: oracle.table_id,
				})?;
			if *table_size == 0 {
				mos.add_skip();
				continue;
			}
			let log_capacity = log2_ceil_usize(*table_size);
			// Come up with the n_vars for the multilinear oracle being created. This is typically
			// `log_capacity + log_values_per_row`. However, there is an exception for `Transparent`
			// which has a n_vars for just a single row.
			let n_vars = match oracle.variant {
				SymbolicMultilinearPolyVariant::Transparent(ref transparent_poly_oracle) => {
					transparent_poly_oracle.poly().n_vars()
				}
				_ => log_capacity + oracle.log_values_per_row,
			};
			let tower_level = oracle.tower_level;
			let variant = instantiate_oracle_variant(&mos, oracle, n_vars)?;
			mos.add_to_set(|id: OracleId| MultilinearPolyOracle {
				id,
				name: oracle.name.clone(),
				n_vars,
				tower_level,
				variant,
			});
		}
		Ok(mos)
	}

	/// Adds a new oracle to the set.
	pub fn add_oracle<S: ToString>(
		&mut self,
		table_id: usize,
		log_values_per_row: usize,
		s: S,
	) -> Builder<'_, F> {
		Builder {
			mut_ref: self,
			name: Some(s.to_string()),
			table_id,
			log_values_per_row,
		}
	}

	fn add_to_set(
		&mut self,
		oracle: impl FnOnce(OracleId) -> SymbolicMultilinearOracle<F>,
	) -> OracleId {
		let id = OracleId::from_index(self.oracles.len());
		self.oracles.push(oracle(id));
		id
	}

	pub fn size(&self) -> usize {
		self.oracles.len()
	}

	pub fn polys(&self) -> impl Iterator<Item = &SymbolicMultilinearOracle<F>> + '_ {
		(0..self.oracles.len()).map(|index| &self[OracleId::from_index(index)])
	}

	pub fn ids(&self) -> impl Iterator<Item = OracleId> {
		(0..self.oracles.len()).map(OracleId::from_index)
	}

	pub fn iter(&self) -> impl Iterator<Item = (OracleId, &SymbolicMultilinearOracle<F>)> + '_ {
		(0..self.oracles.len()).map(|index| {
			let oracle_id = OracleId::from_index(index);
			(oracle_id, &self[oracle_id])
		})
	}

	pub fn label(&self, oracle_id: OracleId) -> Option<String> {
		self[oracle_id].name.clone()
	}
}

fn instantiate_oracle_variant<F: TowerField>(
	mos: &MultilinearOracleSet<F>,
	oracle: &SymbolicMultilinearOracle<F>,
	n_vars: usize,
) -> Result<MultilinearPolyVariant<F>, Error> {
	use self::{MultilinearPolyVariant as Sized, SymbolicMultilinearPolyVariant as Symbolic};

	let variant = match &oracle.variant {
		Symbolic::Committed => MultilinearPolyVariant::Committed,
		Symbolic::Transparent(transparent_poly_oracle) => {
			Sized::Transparent(transparent_poly_oracle.clone())
		}
		Symbolic::Structured(arith_circuit) => Sized::Structured(arith_circuit.clone()),
		Symbolic::Repeating { id } => {
			let log_count = n_vars - mos.n_vars(*id);
			Sized::Repeating { id: *id, log_count }
		}
		Symbolic::Projected {
			id,
			values,
			variant,
		} => {
			let start_index = match variant {
				ProjectionVariant::Offset(offset) => *offset,
				ProjectionVariant::Last => n_vars - values.len(),
			};
			let projected = Projected::new(mos, *id, values.clone(), start_index)?;
			Sized::Projected(projected)
		}
		Symbolic::Shifted {
			id,
			shift_offset,
			block_size,
			shift_variant,
		} => {
			let shifted = Shifted::new(mos, *id, *shift_offset, *block_size, *shift_variant)?;
			MultilinearPolyVariant::Shifted(shifted)
		}
		Symbolic::Packed { id, log_degree } => {
			let packed = Packed::new(*id, *log_degree);
			MultilinearPolyVariant::Packed(packed)
		}
		Symbolic::LinearCombination { offset, inner } => {
			let linear_combination = LinearCombination::new(mos, n_vars, *offset, inner.clone())?;
			MultilinearPolyVariant::LinearCombination(linear_combination)
		}
		Symbolic::ZeroPadded {
			id,
			n_pad_vars,
			nonzero_index,
			start_index,
		} => {
			let zero_padded = ZeroPadded::new(mos, *id, *n_pad_vars, *nonzero_index, *start_index)?;
			MultilinearPolyVariant::ZeroPadded(zero_padded)
		}
		Symbolic::Composite { inner, circuit } => {
			let composite_mle = CompositeMLE::new(mos, n_vars, inner.clone(), circuit.clone())?;
			MultilinearPolyVariant::Composite(composite_mle)
		}
	};
	Ok(variant)
}

impl<F: TowerField> std::ops::Index<OracleId> for SymbolicMultilinearOracleSet<F> {
	type Output = SymbolicMultilinearOracle<F>;

	fn index(&self, id: OracleId) -> &Self::Output {
		&self.oracles[id.index()]
	}
}

pub struct Builder<'a, F: TowerField> {
	mut_ref: &'a mut SymbolicMultilinearOracleSet<F>,
	name: Option<String>,
	table_id: usize,
	log_values_per_row: usize,
}

impl<'a, F: TowerField> Builder<'a, F> {
	pub fn transparent(self, poly: impl MultivariatePoly<F> + 'static) -> Result<OracleId, Error> {
		if poly.binary_tower_level() > F::TOWER_LEVEL {
			bail!(Error::TowerLevelTooHigh {
				tower_level: poly.binary_tower_level(),
			});
		}

		let inner = TransparentPolyOracle::new(Arc::new(poly))?;

		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level: inner.poly().binary_tower_level(),
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Transparent(inner),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn structured(self, expr: ArithCircuit<F>) -> Result<OracleId, Error> {
		if expr.binary_tower_level() > F::TOWER_LEVEL {
			bail!(Error::TowerLevelTooHigh {
				tower_level: expr.binary_tower_level(),
			});
		}

		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level: expr.binary_tower_level(),
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Structured(expr),
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn committed(mut self, tower_level: usize) -> OracleId {
		let name = self.name.take();
		self.add_committed_with_name(tower_level, name)
	}

	pub fn committed_multiple<const N: usize>(mut self, tower_level: usize) -> [OracleId; N] {
		match &self.name.take() {
			None => [0; N].map(|_| self.add_committed_with_name(tower_level, None)),
			Some(s) => {
				let x: [usize; N] = array::from_fn(|i| i);
				x.map(|i| self.add_committed_with_name(tower_level, Some(format!("{s}_{i}"))))
			}
		}
	}

	pub fn repeating(self, inner_id: OracleId) -> Result<OracleId, Error> {
		let inner = &self.mut_ref[inner_id];

		let tower_level = inner.tower_level;
		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Repeating { id: inner_id },
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
		if offset == 0 || offset >= 1 << block_bits {
			bail!(PolynomialError::InvalidShiftOffset {
				max_shift_offset: (1 << block_bits) - 1,
				shift_offset: offset,
			});
		}

		let tower_level = self.mut_ref[inner_id].tower_level;
		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Shifted {
				id: inner_id,
				shift_offset: offset,
				block_size: block_bits,
				shift_variant: variant,
			},
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn packed(self, inner_id: OracleId, log_degree: usize) -> Result<OracleId, Error> {
		let inner_tower_level = self.mut_ref[inner_id].tower_level;

		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level: inner_tower_level + log_degree,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Packed {
				id: inner_id,
				log_degree,
			},
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn projected(
		self,
		inner_id: OracleId,
		values: Vec<F>,
		start_index: usize,
	) -> Result<OracleId, Error> {
		// TODO: This is wrong, should be F::TOWER_LEVEL
		let tower_level = self.mut_ref[inner_id].tower_level;
		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Projected {
				id: inner_id,
				values,
				variant: ProjectionVariant::Offset(start_index),
			},
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn projected_last_vars(
		self,
		inner_id: OracleId,
		values: Vec<F>,
	) -> Result<OracleId, Error> {
		// TODO: This is wrong, should be F::TOWER_LEVEL
		let tower_level = self.mut_ref[inner_id].tower_level;
		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Projected {
				id: inner_id,
				values,
				variant: ProjectionVariant::Last,
			},
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn linear_combination(
		self,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		self.linear_combination_with_offset(F::ZERO, inner)
	}

	pub fn linear_combination_with_offset(
		self,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, Error> {
		let inner = inner.into_iter().collect::<Vec<_>>();
		let tower_level = inner
			.iter()
			.map(|(oracle_id, coeff)| {
				self.mut_ref[*oracle_id]
					.tower_level
					.max(coeff.min_tower_level())
			})
			.max()
			.unwrap_or(0)
			.max(offset.min_tower_level());

		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::LinearCombination { offset, inner },
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	pub fn composite_mle(
		self,
		inner: impl IntoIterator<Item = OracleId>,
		comp: ArithCircuit<F>,
	) -> Result<OracleId, Error> {
		let inner = inner.into_iter().collect::<Vec<_>>();
		let tower_level = inner
			.iter()
			.map(|oracle_id| self.mut_ref[*oracle_id].tower_level)
			.max()
			.unwrap_or(0);

		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::Composite {
				inner,
				circuit: comp,
			},
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
		let inner = &self.mut_ref[inner_id];
		let tower_level = inner.tower_level;
		let oracle = |id: OracleId| SymbolicMultilinearOracle {
			id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: self.name,
			variant: SymbolicMultilinearPolyVariant::ZeroPadded {
				id: inner_id,
				n_pad_vars,
				nonzero_index,
				start_index,
			},
		};

		Ok(self.mut_ref.add_to_set(oracle))
	}

	fn add_committed_with_name(&mut self, tower_level: usize, name: Option<String>) -> OracleId {
		let oracle = |oracle_id: OracleId| SymbolicMultilinearOracle {
			id: oracle_id,
			table_id: self.table_id,
			log_values_per_row: self.log_values_per_row,
			tower_level,
			name: name.clone(),
			variant: SymbolicMultilinearPolyVariant::Committed,
		};

		self.mut_ref.add_to_set(oracle)
	}
}
