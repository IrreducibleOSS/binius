// Copyright 2024-2025 Irreducible Inc.

pub mod channel;
mod common;
pub mod error;
pub mod exp;
mod prove;
pub mod validate;
mod verify;

use binius_field::{BinaryField128b, TowerField};
use binius_macros::SerializeBytes;
use binius_utils::{DeserializeBytes, SerializationError, SerializationMode};
use channel::{ChannelId, Flush};
use exp::Exp;
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
#[derive(Debug, Clone, SerializeBytes)]
pub struct ConstraintSystem<F: TowerField> {
	pub oracles: MultilinearOracleSet<F>,
	pub table_constraints: Vec<ConstraintSet<F>>,
	pub non_zero_oracle_ids: Vec<OracleId>,
	pub flushes: Vec<Flush<F>>,
	pub exponents: Vec<Exp<F>>,
	pub max_channel_id: ChannelId,
}

impl DeserializeBytes for ConstraintSystem<BinaryField128b> {
	fn deserialize(
		mut read_buf: impl bytes::Buf,
		mode: SerializationMode,
	) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self {
			oracles: DeserializeBytes::deserialize(&mut read_buf, mode)?,
			table_constraints: DeserializeBytes::deserialize(&mut read_buf, mode)?,
			non_zero_oracle_ids: DeserializeBytes::deserialize(&mut read_buf, mode)?,
			flushes: DeserializeBytes::deserialize(&mut read_buf, mode)?,
			exponents: DeserializeBytes::deserialize(&mut read_buf, mode)?,
			max_channel_id: DeserializeBytes::deserialize(&mut read_buf, mode)?,
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
