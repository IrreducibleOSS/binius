// Copyright 2024-2025 Irreducible Inc.

pub mod channel;
mod common;
pub mod error;
mod prove;
pub mod validate;
mod verify;

use binius_field::{serialization, BinaryField128b, DeserializeCanonical, TowerField};
use binius_macros::SerializeCanonical;
use channel::{ChannelId, Flush};
pub use prove::prove;
pub use verify::verify;

use crate::oracle::{ConstraintSet, MultilinearOracleSet, OracleId};

/// Contains the 3 things that place constraints on witness data in Binius
/// - virtual oracles
/// - polynomial constraints
/// - channel flushes
///
/// As a result, a ConstraintSystem allows us to validate all of these
/// constraints against a witness, as well as enabling generic prove/verify
#[derive(Debug, Clone, SerializeCanonical)]
pub struct ConstraintSystem<F: TowerField> {
	pub oracles: MultilinearOracleSet<F>,
	pub table_constraints: Vec<ConstraintSet<F>>,
	pub non_zero_oracle_ids: Vec<OracleId>,
	pub flushes: Vec<Flush>,
	pub max_channel_id: ChannelId,
}

impl DeserializeCanonical for ConstraintSystem<BinaryField128b> {
	fn deserialize_canonical(mut read_buf: impl bytes::Buf) -> Result<Self, serialization::Error>
	where
		Self: Sized,
	{
		Ok(Self {
			oracles: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			table_constraints: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			non_zero_oracle_ids: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			flushes: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
			max_channel_id: DeserializeCanonical::deserialize_canonical(&mut read_buf)?,
		})
	}
}

impl<F: TowerField> ConstraintSystem<F> {
	pub const fn no_base_constraints(self) -> Self {
		self
	}
}

/// Constraint system proof that has been serialized into bytes
#[derive(Debug, Clone)]
pub struct Proof {
	pub transcript: Vec<u8>,
}

impl Proof {
	pub fn get_proof_size(&self) -> usize {
		self.transcript.len()
	}
}
