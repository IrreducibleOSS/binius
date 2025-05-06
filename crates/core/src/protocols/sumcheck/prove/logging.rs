// Copyright 2024-2025 Irreducible Inc.

use std::collections::HashMap;

use binius_field::{Field, PackedField};
use binius_utils::impl_debug_with_json;
use serde::Serialize;

use super::SumcheckProver;
use crate::protocols::sumcheck::prove::batch_zerocheck::ZerocheckProver;

#[derive(Serialize, PartialEq, Eq, Hash)]
struct ProverData {
	n_vars: usize,
	domain_size: Option<usize>,
}

#[derive(Serialize)]
pub struct FoldLowDimensionsData(HashMap<ProverData, usize>);

impl FoldLowDimensionsData {
	pub fn new<'a, 'b, P: PackedField, Prover: ZerocheckProver<'a, P> + 'b>(
		skip_rounds: usize,
		constraints: impl IntoIterator<Item = &'b Prover>,
	) -> Self {
		let mut claim_n_vars = HashMap::new();
		for constraint in constraints {
			*claim_n_vars
				.entry(ProverData {
					n_vars: constraint.n_vars(),
					domain_size: constraint.domain_size(skip_rounds),
				})
				.or_default() += 1;
		}

		Self(claim_n_vars)
	}
}

impl_debug_with_json!(FoldLowDimensionsData);

#[derive(Serialize)]
pub struct PIOPCompilerFoldData {
	n_vars: usize,
}

impl PIOPCompilerFoldData {
	pub fn new<F: Field>(prover: &impl SumcheckProver<F>) -> Self {
		Self {
			n_vars: prover.n_vars(),
		}
	}
}

impl_debug_with_json!(PIOPCompilerFoldData);

#[derive(Serialize)]
pub struct ExpandQueryData {
	log_n: usize,
}

impl ExpandQueryData {
	pub fn new<F: Field>(query: &[F]) -> Self {
		let log_n = if query.is_empty() {
			0
		} else {
			query.len().ilog2() as usize
		};
		Self { log_n }
	}
}

impl_debug_with_json!(ExpandQueryData);

#[derive(Serialize)]
pub struct UnivariateSkipCalculateCoeffsData {
	n_vars: usize,
	skip_vars: usize,
	n_multilinears: usize,
	log_batch: usize,
}

impl UnivariateSkipCalculateCoeffsData {
	pub fn new(n_vars: usize, skip_vars: usize, n_multilinears: usize, log_batch: usize) -> Self {
		Self {
			n_vars,
			skip_vars,
			n_multilinears,
			log_batch,
		}
	}
}

impl_debug_with_json!(UnivariateSkipCalculateCoeffsData);
