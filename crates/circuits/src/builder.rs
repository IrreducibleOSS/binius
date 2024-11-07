// Copyright 2024 Irreducible Inc.

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
	TowerField,
};
use binius_math::CompositionPoly;

#[derive(Default)]
pub struct ConstraintSystemBuilder<U, F>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	oracles: MultilinearOracleSet<F>,
	batch_ids: Vec<(usize, usize, BatchId)>,
	constraints: ConstraintSetBuilder<PackedType<U, F>>,
	non_zero_oracle_ids: Vec<OracleId>,
	flushes: Vec<Flush>,
	witness: Option<MultilinearExtensionIndex<'static, U, F>>,
	next_channel_id: ChannelId,
	namespace_path: Vec<String>,
}

impl<U, F> ConstraintSystemBuilder<U, F>
where
	U: UnderlierType + PackScalar<F>,
	F: TowerField,
{
	pub fn new() -> Self {
		Self::default()
	}

	pub fn new_with_witness() -> Self {
		Self {
			witness: Some(MultilinearExtensionIndex::new()),
			..Default::default()
		}
	}

	pub fn build(self) -> Result<ConstraintSystem<PackedType<U, F>>, anyhow::Error> {
		Ok(ConstraintSystem {
			max_channel_id: self
				.flushes
				.iter()
				.map(|flush| flush.channel_id)
				.max()
				.unwrap_or(0),
			table_constraints: self.constraints.build(&self.oracles)?,
			non_zero_oracle_ids: self.non_zero_oracle_ids,
			oracles: self.oracles,
			flushes: self.flushes,
		})
	}

	pub fn witness(&mut self) -> Option<&mut MultilinearExtensionIndex<'static, U, F>> {
		self.witness.as_mut()
	}

	pub fn take_witness(&mut self) -> Option<MultilinearExtensionIndex<'static, U, F>> {
		Option::take(&mut self.witness)
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
		composition: impl CompositionPoly<PackedType<U, F>> + 'static,
	) {
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
			let batch_id = self.oracles.add_committed_batch(n_vars, tower_level);
			self.batch_ids.push((n_vars, tower_level, batch_id));
			batch_id
		}
	}

	pub fn add_interleaved(
		&mut self,
		name: impl ToString,
		id0: OracleId,
		id1: OracleId,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.add_named(self.scoped_name(name))
			.interleaved(id0, id1)
	}

	pub fn add_merged(
		&mut self,
		name: impl ToString,
		id0: OracleId,
		id1: OracleId,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.add_named(self.scoped_name(name))
			.merged(id0, id1)
	}

	pub fn add_linear_combination(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
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
			.add_named(self.scoped_name(name))
			.shifted(id, offset, block_bits, variant)
	}

	pub fn add_transparent(
		&mut self,
		name: impl ToString,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, OracleError> {
		self.oracles
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
