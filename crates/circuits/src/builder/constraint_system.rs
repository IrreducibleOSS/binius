// Copyright 2024 Irreducible Inc.

use std::{cell::RefCell, rc::Rc};

use crate::builder::witness;
use anyhow::anyhow;
use binius_core::{
	constraint_system::{
		channel::{ChannelId, Flush, FlushDirection},
		ConstraintSystem,
	},
	oracle::{
		BatchId, ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet, OracleId,
		ProjectionVariant, ShiftVariant,
	},
	polynomial::MultivariatePoly,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	ExtensionField, TowerField,
};
use binius_math::CompositionPolyOS;

#[derive(Default)]
pub struct ConstraintSystemBuilder<'arena, U, F, FBase = F>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase>,
	F: TowerField + ExtensionField<FBase>,
	FBase: TowerField,
{
	oracles: Rc<RefCell<MultilinearOracleSet<F>>>,
	batch_ids: Vec<(usize, usize, BatchId)>,
	constraints: ConstraintSetBuilder<PackedType<U, F>>,
	constraints_base: ConstraintSetBuilder<PackedType<U, FBase>>,
	non_zero_oracle_ids: Vec<OracleId>,
	flushes: Vec<Flush>,
	witness: Option<witness::Builder<'arena, U, F>>,
	next_channel_id: ChannelId,
	namespace_path: Vec<String>,
}

impl<'arena, U, F, FBase> ConstraintSystemBuilder<'arena, U, F, FBase>
where
	U: UnderlierType + PackScalar<F> + PackScalar<FBase>,
	F: TowerField + ExtensionField<FBase>,
	FBase: TowerField,
{
	pub fn new() -> Self {
		Self::default()
	}

	pub fn new_with_witness(allocator: &'arena bumpalo::Bump) -> Self {
		let oracles = Rc::new(RefCell::new(MultilinearOracleSet::new()));
		Self {
			witness: Some(witness::Builder::new(allocator, oracles.clone())),
			oracles,
			..Default::default()
		}
	}

	#[allow(clippy::type_complexity)]
	pub fn build(
		self,
	) -> Result<ConstraintSystem<PackedType<U, F>, PackedType<U, FBase>>, anyhow::Error> {
		let table_constraints = self.constraints.build(&self.oracles.borrow())?;
		let table_constraints_base = self.constraints_base.build(&self.oracles.borrow())?;
		Ok(ConstraintSystem {
			max_channel_id: self
				.flushes
				.iter()
				.map(|flush| flush.channel_id)
				.max()
				.unwrap_or(0),
			table_constraints,
			table_constraints_base,
			non_zero_oracle_ids: self.non_zero_oracle_ids,
			oracles: Rc::into_inner(self.oracles)
				.ok_or(anyhow!(
					"Failed to build ConstraintSystem: references still exist to oracles"
				))?
				.into_inner(),
			flushes: self.flushes,
		})
	}

	pub fn witness(&mut self) -> Option<&mut witness::Builder<'arena, U, F>> {
		self.witness.as_mut()
	}

	pub fn take_witness(
		&mut self,
	) -> Result<MultilinearExtensionIndex<'arena, U, F>, anyhow::Error> {
		Option::take(&mut self.witness)
			.ok_or_else(|| {
				anyhow!("Witness is missing. Are you in verifier mode, or have you already extraced the witness?")
			})?
			.build()
	}

	pub fn send(&mut self, channel_id: ChannelId, oracle_ids: impl IntoIterator<Item = OracleId>) {
		self.flushes.push(Flush {
			channel_id,
			direction: FlushDirection::Push,
			oracles: oracle_ids.into_iter().collect(),
		});
	}

	pub fn receive(
		&mut self,
		channel_id: ChannelId,
		oracle_ids: impl IntoIterator<Item = OracleId>,
	) {
		self.flushes.push(Flush {
			channel_id,
			direction: FlushDirection::Pull,
			oracles: oracle_ids.into_iter().collect(),
		});
	}

	pub fn assert_zero<const N: usize>(
		&mut self,
		oracle_ids: [OracleId; N],
		composition: impl Clone
			+ CompositionPolyOS<PackedType<U, F>>
			+ CompositionPolyOS<PackedType<U, FBase>>
			+ 'static,
	) {
		self.constraints_base
			.add_zerocheck(oracle_ids, composition.clone());
		self.constraints.add_zerocheck(oracle_ids, composition);
	}

	pub fn assert_not_zero(&mut self, oracle_id: OracleId) {
		self.non_zero_oracle_ids.push(oracle_id);
	}

	pub fn add_channel(&mut self) -> ChannelId {
		let channel_id = self.next_channel_id;
		self.next_channel_id += 1;
		channel_id
	}

	pub fn add_committed(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		tower_level: usize,
	) -> OracleId {
		let batch_id = self.get_or_create_batch_id(n_vars, tower_level);
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.committed(batch_id)
	}

	pub fn add_committed_multiple<const N: usize>(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		tower_level: usize,
	) -> [OracleId; N] {
		let batch_id = self.get_or_create_batch_id(n_vars, tower_level);
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.committed_multiple(batch_id)
	}

	fn get_or_create_batch_id(&mut self, n_vars: usize, tower_level: usize) -> BatchId {
		if let Some((_, _, batch_id)) =
			self.batch_ids
				.iter()
				.copied()
				.find(|(prev_n_vars, prev_tower_level, _)| {
					*prev_n_vars == n_vars && *prev_tower_level == tower_level
				}) {
			batch_id
		} else {
			let batch_id = self
				.oracles
				.borrow_mut()
				.add_committed_batch(n_vars, tower_level);
			self.batch_ids.push((n_vars, tower_level, batch_id));
			batch_id
		}
	}

	pub fn add_linear_combination(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.linear_combination(n_vars, inner)
	}

	pub fn add_linear_combination_with_offset(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.linear_combination_with_offset(n_vars, offset, inner)
	}

	pub fn add_packed(
		&mut self,
		name: impl ToString,
		id: OracleId,
		log_degree: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.packed(id, log_degree)
	}

	pub fn add_projected(
		&mut self,
		name: impl ToString,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<usize, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.projected(id, values, variant)
	}

	pub fn add_repeating(
		&mut self,
		name: impl ToString,
		id: OracleId,
		log_count: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.repeating(id, log_count)
	}

	pub fn add_shifted(
		&mut self,
		name: impl ToString,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.shifted(id, offset, block_bits, variant)
	}

	pub fn add_transparent(
		&mut self,
		name: impl ToString,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.transparent(poly)
	}

	pub fn add_zero_padded(
		&mut self,
		name: impl ToString,
		id: OracleId,
		n_vars: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.zero_padded(id, n_vars)
	}

	fn scoped_name(&self, name: impl ToString) -> String {
		let name = name.to_string();
		if self.namespace_path.is_empty() {
			name
		} else {
			format!("{}::{name}", self.namespace_path.join("::"))
		}
	}

	/// Anything pushed to the namespace will become part of oracle name, which is useful for debugging.
	///
	/// Use `pop_namespace(&mut self)` to remove the latest name.
	///
	/// Example
	/// ```
	/// use binius_circuits::builder::ConstraintSystemBuilder;
	/// use binius_field::{TowerField, BinaryField128b, BinaryField1b, arch::OptimalUnderlier};
	///
	/// let log_size = 14;
	///
	/// let mut builder = ConstraintSystemBuilder::<OptimalUnderlier, BinaryField128b>::new();
	/// builder.push_namespace("a");
	/// let x = builder.add_committed("x", log_size, BinaryField1b::TOWER_LEVEL);
	/// builder.push_namespace("b");
	/// let y = builder.add_committed("y", log_size, BinaryField1b::TOWER_LEVEL);
	/// builder.pop_namespace();
	/// builder.pop_namespace();
	/// let z = builder.add_committed("z", log_size, BinaryField1b::TOWER_LEVEL);
	///
	/// let system = builder.build().unwrap();
	/// assert_eq!(system.oracles.oracle(x).name().unwrap(), "a::x");
	/// assert_eq!(system.oracles.oracle(y).name().unwrap(), "a::b::y");
	/// assert_eq!(system.oracles.oracle(z).name().unwrap(), "z");
	/// ```
	pub fn push_namespace(&mut self, name: impl ToString) {
		self.namespace_path.push(name.to_string());
	}

	pub fn pop_namespace(&mut self) {
		self.namespace_path.pop();
	}
}
