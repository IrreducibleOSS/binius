// Copyright 2025 Irreducible Inc.

use std::collections::HashMap;

use binius_field::Field;

use crate::oracle::ConstraintSet;

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct RegularSumcheckDimensionsData {
	claim_n_vars: HashMap<usize, usize>,
}

impl RegularSumcheckDimensionsData {
	pub(super) fn new<'a, F: Field>(
		constraints: impl IntoIterator<Item = &'a ConstraintSet<F>>,
	) -> Self {
		let mut claim_n_vars = HashMap::new();
		for constraint in constraints {
			*claim_n_vars.entry(constraint.n_vars).or_default() += 1;
		}

		Self { claim_n_vars }
	}
}
