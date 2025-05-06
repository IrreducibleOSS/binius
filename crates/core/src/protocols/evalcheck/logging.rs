// Copyright 2025 Irreducible Inc.

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct MLEFoldHighDimensionsData {
	n_claims: usize,
}

impl MLEFoldHighDimensionsData {
	pub(super) fn new(n_claims: usize) -> Self {
		Self { n_claims }
	}
}
