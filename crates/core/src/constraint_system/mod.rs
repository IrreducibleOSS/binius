// Copyright 2024 Ulvetanna Inc.

pub mod channel;
pub mod error;
pub mod validate;

use binius_field::{PackedField, TowerField};
use channel::{ChannelId, Flush};

use crate::oracle::{ConstraintSet, MultilinearOracleSet};

/// Contains the 3 things that place constraints on witness data in Binius
/// - virtual oracles
/// - polynomial constraints
/// - channel flushes
///
/// As a result, a ConstraintSystem allows us to validate all of these
/// constraints against a witness, as well as enabling generic prove/verify
pub struct ConstraintSystem<P: PackedField<Scalar: TowerField>> {
	pub oracles: MultilinearOracleSet<P::Scalar>,
	pub table_constraints: Vec<ConstraintSet<P>>,
	pub flushes: Vec<Flush>,
	pub max_channel_id: ChannelId,
}
