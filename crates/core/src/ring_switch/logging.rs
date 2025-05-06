use std::collections::HashMap;

use binius_field::{PackedField, TowerField};
use binius_math::MultilinearPoly;

use super::common::EvalClaimSuffixDesc;

#[derive(Debug)]
#[allow(dead_code)]
pub struct MLEFoldHisgDimensionsData {
	witness_n_vars: HashMap<usize, usize>,
}

impl MLEFoldHisgDimensionsData {
	pub fn new<'a, P: PackedField, M: MultilinearPoly<P> + 'a>(
		multilinears: impl IntoIterator<Item = &'a M>,
	) -> Self {
		let mut witness_n_vars = HashMap::new();
		for multilinear in multilinears {
			*witness_n_vars.entry(multilinear.n_vars()).or_default() += 1;
		}

		Self { witness_n_vars }
	}
}

#[derive(Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub struct EvalClaimSuffixData {
	pub suffix_desc_kappa: usize,
	pub suffix_len: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CalculateRingSwitchEqIndData(HashMap<EvalClaimSuffixData, usize>);

impl CalculateRingSwitchEqIndData {
	pub fn new<'a, F: TowerField>(
		constraints: impl IntoIterator<Item = &'a EvalClaimSuffixDesc<F>>,
	) -> Self {
		let mut claim_n_vars = HashMap::new();
		for constraint in constraints {
			*claim_n_vars
				.entry(EvalClaimSuffixData {
					suffix_desc_kappa: constraint.kappa,
					suffix_len: constraint.suffix.len(),
				})
				.or_default() += 1;
		}

		Self(claim_n_vars)
	}
}
