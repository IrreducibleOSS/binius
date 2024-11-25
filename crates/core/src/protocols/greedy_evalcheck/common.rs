// Copyright 2024 Irreducible Inc.

use crate::{oracle::BatchId, protocols::evalcheck::SameQueryPcsClaim};
use binius_field::Field;

#[derive(Debug)]
pub struct GreedyEvalcheckProveOutput<F: Field> {
	pub same_query_claims: Vec<(BatchId, SameQueryPcsClaim<F>)>,
}
