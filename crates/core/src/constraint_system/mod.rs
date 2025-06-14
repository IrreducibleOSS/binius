// Copyright 2024-2025 Irreducible Inc.

pub mod channel;
mod common;
pub mod error;
pub mod exp;
mod prove;
pub mod validate;
mod verify;

#[cfg(test)]
mod tests;

use binius_field::{BinaryField128b, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};
use binius_utils::{SerializationMode, SerializeBytes};
use channel::Flush;
use digest::{Digest, Output};
use exp::Exp;
pub use prove::{prove, prove_with_params};
pub use verify::{verify, verify_with_params};

use crate::{
	oracle::{ConstraintSet, MultilinearOracleSet, OracleId},
	protocols::fri::FRIConfigParams,
};

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
	pub table_size_specs: Vec<TableSizeSpec>,
}

impl<F: TowerField> ConstraintSystem<F> {
	/// Returns the hash digest of this constraint system.
	///
	/// This assumes that the constraint system should be serializable.
	pub fn digest<Hash: Digest>(&self) -> Output<Hash> {
		let mut buf = Vec::new();
		self.serialize(&mut buf, SerializationMode::CanonicalTower)
			.expect("the constraint system should be serializable");
		Hash::digest(&buf)
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

pub type TableId = usize;

/// A category of the size specification of a table.
///
/// Tables can have size restrictions, where certain columns, specifically structured columns,
/// are only allowed for certain size specifications.
#[derive(Debug, Copy, Clone, SerializeBytes, DeserializeBytes)]
pub enum TableSizeSpec {
	/// The table size may be arbitrary.
	Arbitrary,
	/// The table size may be any power of two.
	PowerOfTwo,
	/// The table size must be a fixed power of two.
	Fixed { log_size: usize },
}
