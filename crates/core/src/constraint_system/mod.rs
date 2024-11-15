// Copyright 2024 Irreducible Inc.

pub mod channel;
mod common;
pub mod error;
mod prove;
pub mod validate;
mod verify;

use binius_field::{PackedField, TowerField};
use channel::{ChannelId, Flush};
pub use prove::prove;
pub use verify::verify;

use crate::{
	merkle_tree_vcs::BinaryMerkleTreeScheme,
	oracle::{ConstraintSet, MultilinearOracleSet, OracleId},
	poly_commit::{batch_pcs, fri_pcs},
	protocols::{
		gkr_gpa::GrandProductBatchProof, greedy_evalcheck::GreedyEvalcheckProof, sumcheck,
	},
};

/// Contains the 3 things that place constraints on witness data in Binius
/// - virtual oracles
/// - polynomial constraints
/// - channel flushes
///
/// As a result, a ConstraintSystem allows us to validate all of these
/// constraints against a witness, as well as enabling generic prove/verify
#[derive(Debug, Clone)]
pub struct ConstraintSystem<P: PackedField<Scalar: TowerField>, PBase: PackedField = P> {
	pub oracles: MultilinearOracleSet<P::Scalar>,
	pub table_constraints: Vec<ConstraintSet<P>>,
	pub table_constraints_base: Vec<ConstraintSet<PBase>>,
	pub non_zero_oracle_ids: Vec<OracleId>,
	pub flushes: Vec<Flush>,
	pub max_channel_id: ChannelId,
}

impl<P: PackedField<Scalar: TowerField>, PBase: PackedField> ConstraintSystem<P, PBase> {
	pub fn no_base_constraints(self) -> ConstraintSystem<P> {
		ConstraintSystem {
			oracles: self.oracles,
			table_constraints: self.table_constraints.clone(),
			table_constraints_base: self.table_constraints,
			non_zero_oracle_ids: self.non_zero_oracle_ids,
			flushes: self.flushes,
			max_channel_id: self.max_channel_id,
		}
	}
}

/// Constraint system proof with the standard PCS.
pub type Proof<F, Digest, Hash, Compress> = ProofGenericPCS<
	F,
	Digest,
	batch_pcs::Proof<fri_pcs::Proof<F, BinaryMerkleTreeScheme<Digest, Hash, Compress>>>,
>;

/// Constraint system proof with a generic [`crate::poly_commit::PolyCommitScheme`].
#[derive(Debug, Clone)]
pub struct ProofGenericPCS<F: TowerField, PCSComm, PCSProof> {
	pub commitments: Vec<PCSComm>,
	pub flush_products: Vec<F>,
	pub non_zero_products: Vec<F>,
	pub prodcheck_proof: GrandProductBatchProof<F>,
	pub zerocheck_univariate_proof: sumcheck::univariate_zerocheck::ZerocheckUnivariateProof<F>,
	pub zerocheck_proof: sumcheck::Proof<F>,
	pub univariatizing_proof: sumcheck::Proof<F>,
	pub greedy_evalcheck_proof: GreedyEvalcheckProof<F>,
	pub pcs_proofs: Vec<PCSProof>,
	pub transcript: Vec<u8>,
	pub advice: Vec<u8>,
}
