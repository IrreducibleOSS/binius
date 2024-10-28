// Copyright 2024 Irreducible Inc.

/// Committed polynomial batches are identified by their index.
pub type BatchId = usize;

/// A batch of committed multilinear polynomials with a unique batch ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedBatch {
	pub id: BatchId,
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
