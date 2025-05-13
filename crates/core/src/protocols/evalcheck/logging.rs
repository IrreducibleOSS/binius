// Copyright 2025 Irreducible Inc.

use binius_utils::impl_debug_with_json;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct MLEFoldHighDimensionsData {
	n_claims: usize,
}

impl MLEFoldHighDimensionsData {
	pub(super) const fn new(n_claims: usize) -> Self {
		Self { n_claims }
	}
}

impl_debug_with_json!(MLEFoldHighDimensionsData);
