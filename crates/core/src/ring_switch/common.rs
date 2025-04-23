// Copyright 2024-2025 Irreducible Inc.

use std::sync::Arc;

use binius_field::{Field, TowerField};
use binius_utils::sparse_index::SparseIndex;

use super::error::Error;
use crate::{
	oracle::{MultilinearOracleSet, MultilinearPolyOracle, MultilinearPolyVariant},
	piop::CommitMeta,
	protocols::evalcheck::EvalcheckMultilinearClaim,
};

/// A prefix of an evaluation claim query.
///
/// For an evaluation point $(z_0, ..., z_\ell)$, the prefix is $(z_0, ..., z_{\kappa-1})$, where
/// $\kappa$ is the binary logarithm of the embedding degree.
#[derive(Debug)]
pub struct EvalClaimPrefixDesc<F: Field> {
	pub prefix: Vec<F>,
}

impl<F: Field> EvalClaimPrefixDesc<F> {
	pub fn kappa(&self) -> usize {
		self.prefix.len()
	}
}

/// A suffix of an evaluation claim query.
///
/// For an evaluation point $(z_0, ..., z_\ell)$, the prefix is $(z_\kappa, ..., z_{\ell-1})$,
/// where $\kappa$ is the binary logarithm of the embedding degree.
#[derive(Debug)]
pub struct EvalClaimSuffixDesc<F: Field> {
	pub suffix: Arc<[F]>,
	pub kappa: usize,
}

/// An incomplete [`crate::piop::PIOPSumcheckClaim`] for which the sum has not been determined yet.
#[derive(Debug)]
pub struct PIOPSumcheckClaimDesc<'a, F: Field> {
	/// Index of the committed multilinear, referencing the commit metadata.
	pub committed_idx: usize,
	/// Index of the suffix descriptor, referencing the slice in an [`EvalClaimSystem`].
	pub suffix_desc_idx: usize,
	pub eval_claim: &'a EvalcheckMultilinearClaim<F>,
}

/// A system of relations required to verify multilinear evaluation claims using the batched
/// FRI-Binius protocol.
///
/// Fields are public because this is internal to the module.
#[derive(Debug)]
pub struct EvalClaimSystem<'a, F: Field> {
	pub commit_meta: &'a CommitMeta,
	pub prefix_descs: Vec<EvalClaimPrefixDesc<F>>,
	pub suffix_descs: Vec<EvalClaimSuffixDesc<F>>,
	pub sumcheck_claim_descs: Vec<PIOPSumcheckClaimDesc<'a, F>>,
	pub eval_claim_to_prefix_desc_index: Vec<usize>,
}

impl<'a, F: TowerField> EvalClaimSystem<'a, F> {
	/// Constructs a new system of claim relations.
	///
	/// ## Arguments
	///
	/// * `commit_meta` - metadata about the polynomial commitment.
	/// * `oracle_to_commit_index` - a sparse index mapping oracle IDs to IDs in the commit
	///   metadata.
	/// * `eval_claims` - the evaluation claims on committed multilinear polynomials.
	pub fn new(
		oracles: &MultilinearOracleSet<F>,
		commit_meta: &'a CommitMeta,
		oracle_to_commit_index: &SparseIndex<usize>,
		eval_claims: &'a [EvalcheckMultilinearClaim<F>],
	) -> Result<Self, Error> {
		// Sort evaluation claims in ascending order by number of packed variables. This must
		// happen before we do any further index mapping.
		let mut eval_claims = eval_claims.iter().collect::<Vec<_>>();
		eval_claims.sort_by_key(|claim| match oracles.oracle(claim.id) {
			// The number of packed variables is n_vars + tower_level - F::TOWER_LEVEL. Just use
			// n_vars + tower_level as the sort key because we haven't checked that the subtraction
			// wouldn't underflow yet.
			MultilinearPolyOracle {
				n_vars,
				tower_level,
				variant: MultilinearPolyVariant::Committed,
				..
			} => n_vars + tower_level,
			// Ignore any non-committed oracles for now, they'll be caught later in a context where
			// we can return an error.
			_ => 0,
		});

		let (
			prefix_descs,
			eval_claim_to_prefix_desc_index,
			suffix_descs,
			eval_claim_to_suffix_desc_index,
		) = group_claims_by_eval_point(oracles, &eval_claims)?;

		let sumcheck_claim_descs = eval_claims
			.into_iter()
			.enumerate()
			.map(|(i, eval_claim)| {
				let oracle = oracles.oracle(eval_claim.id);
				if !matches!(oracle.variant, MultilinearPolyVariant::Committed) {
					return Err(Error::EvalcheckClaimForDerivedPoly { id: eval_claim.id });
				}
				let committed_idx = oracle_to_commit_index
					.get(oracle.id())
					.copied()
					.ok_or_else(|| Error::OracleToCommitIndexMissingEntry { id: eval_claim.id })?;
				let suffix_desc_idx = eval_claim_to_suffix_desc_index[i];
				Ok(PIOPSumcheckClaimDesc {
					committed_idx,
					suffix_desc_idx,
					eval_claim,
				})
			})
			.collect::<Result<Vec<_>, _>>()?;

		Ok(Self {
			commit_meta,
			prefix_descs,
			suffix_descs,
			sumcheck_claim_descs,
			eval_claim_to_prefix_desc_index,
		})
	}

	pub fn max_claim_kappa(&self) -> usize {
		self.prefix_descs
			.iter()
			.map(|desc| desc.kappa())
			.max()
			.unwrap_or(0)
	}
}

#[allow(clippy::type_complexity)]
fn group_claims_by_eval_point<F: TowerField>(
	oracles: &MultilinearOracleSet<F>,
	claims: &[&EvalcheckMultilinearClaim<F>],
) -> Result<(Vec<EvalClaimPrefixDesc<F>>, Vec<usize>, Vec<EvalClaimSuffixDesc<F>>, Vec<usize>), Error>
{
	let mut prefix_descs = Vec::<EvalClaimPrefixDesc<F>>::new();
	let mut suffix_descs = Vec::<EvalClaimSuffixDesc<F>>::new();
	let mut claim_to_prefix_index = Vec::with_capacity(claims.len());
	let mut claim_to_suffix_index = Vec::with_capacity(claims.len());
	for claim in claims {
		let MultilinearPolyOracle {
			id,
			tower_level,
			variant: MultilinearPolyVariant::Committed,
			..
		} = oracles.oracle(claim.id)
		else {
			return Err(Error::EvalcheckClaimForDerivedPoly { id: claim.id });
		};
		let kappa = F::TOWER_LEVEL.checked_sub(tower_level).ok_or_else(|| {
			Error::OracleTowerLevelTooHigh {
				id,
				max: F::TOWER_LEVEL,
			}
		})?;

		let (prefix, suffix) = if claim.eval_point.len() < kappa {
			// If evaluation point is less than kappa, pad the evaluation point with 0s
			let mut prefix = Vec::with_capacity(kappa);
			prefix.extend_from_slice(&claim.eval_point);
			prefix.resize(kappa, F::ZERO);
			(prefix, &[][..])
		} else {
			let (prefix, suffix) = claim.eval_point.split_at(kappa);
			(prefix.to_vec(), suffix)
		};

		let prefix_id = prefix_descs
			.iter()
			.position(|desc| desc.prefix == prefix)
			.unwrap_or_else(|| {
				let index = prefix_descs.len();
				prefix_descs.push(EvalClaimPrefixDesc { prefix });
				index
			});
		claim_to_prefix_index.push(prefix_id);

		let suffix_id = suffix_descs
			.iter()
			.position(|desc| &*desc.suffix == suffix && desc.kappa == kappa)
			.unwrap_or_else(|| {
				let index = suffix_descs.len();
				suffix_descs.push(EvalClaimSuffixDesc {
					suffix: suffix.to_vec().into(),
					kappa,
				});
				index
			});
		claim_to_suffix_index.push(suffix_id);
	}

	Ok((prefix_descs, claim_to_prefix_index, suffix_descs, claim_to_suffix_index))
}
