// Copyright 2024-2025 Irreducible Inc.

pub mod channel;
mod common;
pub mod error;
pub mod exp;
mod prove;
pub mod validate;
mod verify;

use binius_field::{BinaryField128b, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use channel::Flush;
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
#[derive(Debug, Clone, SerializeBytes, DeserializeBytes)]
#[deserialize_bytes(eval_generics(F = BinaryField128b))]
pub struct ConstraintSystem<F: TowerField> {
	pub oracles: MultilinearOracleSet<F>,
	pub table_constraints: Vec<ConstraintSet<F>>,
	pub non_zero_oracle_ids: Vec<OracleId>,
	pub flushes: Vec<Flush<F>>,
	pub exponents: Vec<Exp<F>>,
	pub channel_count: usize,
}

impl<F: TowerField> ConstraintSystem<F> {}

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
