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
	flushes: Vec<Flush>,
	witness: Option<MultilinearExtensionIndex<'static, U, F>>,
	next_channel_id: ChannelId,
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

	pub fn add_channel(&mut self) -> ChannelId {
		let channel_id = self.next_channel_id;
		self.next_channel_id += 1;
		channel_id
	}

	pub fn add_committed(&mut self, n_vars: usize, tower_level: usize) -> OracleId {
		let batch_id = self.get_or_create_batch_id(n_vars, tower_level);
		self.oracles.add_committed(batch_id)
	}

	pub fn add_committed_multiple<const N: usize>(
		&mut self,
		n_vars: usize,
		tower_level: usize,
	) -> [OracleId; N] {
		let batch_id = self.get_or_create_batch_id(n_vars, tower_level);
		self.oracles.add_committed_multiple(batch_id)
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
		id0: OracleId,
		id1: OracleId,
	) -> Result<OracleId, OracleError> {
		self.oracles.add_interleaved(id0, id1)
	}

	pub fn add_merged(&mut self, id0: OracleId, id1: OracleId) -> Result<OracleId, OracleError> {
		self.oracles.add_merged(id0, id1)
	}

	pub fn add_linear_combination(
		&mut self,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles.add_linear_combination(n_vars, inner)
	}

	pub fn add_linear_combination_with_offset(
		&mut self,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.add_linear_combination_with_offset(n_vars, offset, inner)
	}

	pub fn add_packed(&mut self, id: OracleId, log_degree: usize) -> Result<OracleId, OracleError> {
		self.oracles.add_packed(id, log_degree)
	}

	pub fn add_projected(
		&mut self,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<usize, OracleError> {
		self.oracles.add_projected(id, values, variant)
	}

	pub fn add_repeating(
		&mut self,
		id: OracleId,
		log_count: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles.add_repeating(id, log_count)
	}

	pub fn add_shifted(
		&mut self,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, OracleError> {
		self.oracles.add_shifted(id, offset, block_bits, variant)
	}

	pub fn add_transparent(
		&mut self,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, OracleError> {
		self.oracles.add_transparent(poly)
	}

	pub fn add_zero_padded(
		&mut self,
		id: OracleId,
		n_vars: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles.add_zero_padded(id, n_vars)
	}
}
