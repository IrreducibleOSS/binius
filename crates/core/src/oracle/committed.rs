// Copyright 2024 Ulvetanna Inc.

/// Committed polynomial batches are identified by their index.
pub type BatchId = usize;

// Round ID 0 is precommitment.
pub type RoundId = usize;

/// Metadata about a batch of committed multilinear polynomials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedBatchSpec {
	pub round_id: RoundId,
	pub n_vars: usize,
	pub n_polys: usize,
	pub tower_level: usize,
}

/// A batch of committed multilinear polynomials with a unique batch ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedBatch {
	pub id: BatchId,
	pub round_id: RoundId,
	pub n_vars: usize,
	pub n_polys: usize,
	pub tower_level: usize,
}

/// Committed polynomials are identified by a batch ID and an index in the batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::Display)]
#[display(fmt = "({}, {})", batch_id, index)]
pub struct CommittedId {
	pub batch_id: BatchId,
	pub index: usize,
}
