// Copyright 2024 Ulvetanna Inc.

pub mod channel;
pub mod error;
pub mod validate;

use binius_field::{PackedField, TowerField};
use channel::{ChannelId, Flush};

use crate::oracle::{ConstraintSet, MultilinearOracleSet};

pub struct ConstraintSystem<P: PackedField<Scalar: TowerField>> {
	pub oracles: MultilinearOracleSet<P::Scalar>,
	pub table_constraints: Vec<ConstraintSet<P>>,
	pub flushes: Vec<Flush>,
	pub max_channel_id: ChannelId,
}
