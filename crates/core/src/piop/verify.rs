// Copyright 2024-2025 Irreducible Inc.

use std::{borrow::Borrow, cmp::Ordering, iter, ops::Range};

use binius_field::{BinaryField, ExtensionField, Field, TowerField};
use binius_math::evaluate_piecewise_multilinear;
use binius_utils::{DeserializeBytes, bail, checked_arithmetics::log2_ceil_usize};
use getset::CopyGetters;
use tracing::instrument;

use super::error::{Error, VerificationError};
use crate::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::MerkleTreeScheme,
	piop::util::ResizeableIndex,
	polynomial::MultivariatePoly,
	protocols::{
		fri::{self, FRIParams, FRIVerifier, estimate_optimal_arity},
		sumcheck::{
			CompositeSumClaim, SumcheckClaim, front_loaded::BatchVerifier as SumcheckBatchVerifier,
		},
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
	transcript::VerifierTranscript,
};

/// Metadata about a batch of committed multilinear polynomials.
///
/// In the multilinear polynomial IOP model, several multilinear polynomials can be sent to the
/// oracle by the prover in each round. These multilinears can be committed as a batch by
/// interpolating them into a piecewise multilinear whose evaluations are the concatenation of the
/// piecewise evaluations. This metadata captures the "shape" of the batch, meaning the number of
/// variables of all polynomials in the batch.
#[derive(Debug, CopyGetters)]
pub struct CommitMeta {
	n_multilins_by_vars: Vec<usize>,
	offsets_by_vars: Vec<usize>,
	/// The total number of variables of the interpolating multilinear.
	#[getset(get_copy = "pub")]
	total_vars: usize,
	/// The total number of multilinear pieces in the batch.
	#[getset(get_copy = "pub")]
	total_multilins: usize,
}

impl CommitMeta {
	/// Constructs a new [`CommitMeta`].
	///
	/// ## Arguments
	///
	/// * `n_multilins_by_vars` - a vector index mapping numbers of variables to the number of
	///   multilinears in the batch with that number of variables
	pub fn new(n_multilins_by_vars: Vec<usize>) -> Self {
		let (offsets_by_vars, total_multilins, total_elems) =
			n_multilins_by_vars.iter().enumerate().fold(
				(Vec::with_capacity(n_multilins_by_vars.len()), 0, 0),
				|(mut offsets, total_multilins, total_elems), (n_vars, &count)| {
					offsets.push(total_multilins);
					(offsets, total_multilins + count, total_elems + (count << n_vars))
				},
			);

		Self {
			offsets_by_vars,
			n_multilins_by_vars,
			total_vars: total_elems.next_power_of_two().ilog2() as usize,
			total_multilins,
		}
	}

	/// Constructs a new [`CommitMeta`] from a sequence of committed polynomials described by their
	/// number of variables.
	pub fn with_vars(n_varss: impl IntoIterator<Item = usize>) -> Self {
		let mut n_multilins_by_vars = ResizeableIndex::new();
		for n_vars in n_varss {
			*n_multilins_by_vars.get_mut(n_vars) += 1;
		}
		Self::new(n_multilins_by_vars.into_vec())
	}

	/// Returns the maximum number of variables of any individual multilinear.
	pub fn max_n_vars(&self) -> usize {
		self.n_multilins_by_vars.len().saturating_sub(1)
	}

	/// Returns a vector index mapping numbers of variables to the number of multilinears in the
	/// batch with that number of variables.
	pub fn n_multilins_by_vars(&self) -> &[usize] {
		&self.n_multilins_by_vars
	}

	/// Returns the range of indices into the structure that have the given number of variables.
	pub fn range_by_vars(&self, n_vars: usize) -> Range<usize> {
		let start = self.offsets_by_vars[n_vars];
		start..start + self.n_multilins_by_vars[n_vars]
	}
}

/// A sumcheck claim that can be processed by the PIOP compiler.
///
/// These are a specific form of sumcheck claims over products of a committed polynomial and a
/// transparent polynomial, referencing by index into external vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PIOPSumcheckClaim<F: Field> {
	/// Number of variables of the multivariate polynomial the sumcheck claim is about.
	pub n_vars: usize,
	/// Index of the committed multilinear.
	pub committed: usize,
	/// Index of the transparent multilinear.
	pub transparent: usize,
	/// Claimed sum of the inner product of the hypercube evaluations of the committed and
	/// transparent polynomials.
	pub sum: F,
}

fn make_commit_params_with_constant_arity<F, FEncode>(
	commit_meta: &CommitMeta,
	security_bits: usize,
	log_inv_rate: usize,
	arity: usize,
) -> Result<FRIParams<F, FEncode>, Error>
where
	F: BinaryField + ExtensionField<FEncode>,
	FEncode: BinaryField,
{
	assert!(arity > 0);

	let log_dim = commit_meta.total_vars.saturating_sub(arity);
	let log_batch_size = commit_meta.total_vars.min(arity);
	let rs_code = ReedSolomonCode::new(log_dim, log_inv_rate)?;
	let n_test_queries = fri::calculate_n_test_queries::<F, _>(security_bits, &rs_code)?;

	let cap_height = log2_ceil_usize(n_test_queries);
	let fold_arities = std::iter::repeat_n(
		arity,
		(commit_meta.total_vars).saturating_sub(cap_height.saturating_sub(log_inv_rate)) / arity,
	)
	.collect::<Vec<_>>();
	// here is the down-to-earth explanation of what we're doing: we want the terminal codeword's
	// log-length to be at least as large as the Merkle cap height. note that `total_vars +
	// log_inv_rate - sum(fold_arities)` is exactly the log-length of the terminal codeword; we want
	// this number to be ≥ cap height. so fold_arities will repeat `arity` the maximal number of
	// times possible, while maintaining that `total_vars + log_inv_rate - sum(fold_arities) ≥
	// cap_height` stays true. this arity-selection strategy can be characterized as: "terminate as
	// late as you can, while maintaining that no Merkle cap is strictly smaller than `cap_height`."
	// this strategy does attain that property: the Merkle path height of the last non-terminal
	// codeword will equal the log-length of the terminal codeword, which is ≥ cap height by fiat.
	// moreover, if we terminated later than we are above, then this would stop being true. imagine
	// what would happen if we took the above terminal codeword and continued folding.
	// in that case, we would Merklize this word, again with the coset-bundling trick; the
	// post-bundling path height would thus be `total_vars + log_inv_rate - sum(fold_arities) -
	// arity`. but we already agreed (by the maximality of the number of times we subtracted
	// `arity`) that the above number will be < cap_height. in other words, its Merkle cap will be
	// short. equivalently: this is the latest termination for which the `min` in
	// `optimal_verify_layer` will never trigger; i.e., we will have log2_ceil_usize(n_queries) ≤
	// tree_depth there. it can be shown that this strategy beats any strategy which terminates
	// later than it does (in other words, by doing this, we are NOT terminating TOO early!).
	// this doesn't mean that we should't terminate EVEN earlier (maybe we should). but this
	// approach is conservative and simple; and it's easy to show that you won't lose by doing this.

	// see https://github.com/IrreducibleOSS/binius/pull/300 for proof of this fact

	// how should we handle the case `fold_arities = []`, i.e. total_vars + log_inv_rate -
	// cap_height < arity? in that case, we would lose nothing by making the entire thing
	// interleaved, i.e., setting `log_batch_size := total_vars`, so `terminal_codeword` lives in
	// the interleaving of the repetition code (and so is itself a repetition codeword!). encoding
	// is trivial. but there's a circularity: whether `total_vars + log_inv_rate - cap_height <
	// arity` or not depends on `cap_height`, which depends on `n_test_queries`, which depends on
	// `log_dim`--- soundness depends on block length!---which finally itself depends on whether
	// we're using the repetition code or not. of course this circular dependency is artificial,
	// since in the case `log_batch_size = total_vars` and `log_dim = 0`, we're sending the entire
	// message anyway, so the FRI portion is essentially trivial / superfluous, and the security is
	// perfect. and in any case we could evade it simply by calculating `n_test_queries` and
	// `cap_height` using the provisional `log_dim := total_vars.saturating_sub(arity)`, proceeding
	// as above, and only then, if we find out post facto that `fold_arities = []`, overwriting
	// `log_batch_size := total_vars` and `log_dim = 0`---and even recalculating `n_test_queries` if
	// we wanted (though of course it doesn't matter---we could do 0 queries in that case, and we
	// would still get security---and in fact during the actual querying part we will skip querying
	// anyway). in any case, from a purely code-simplicity point of view, the simplest approach is
	// to bite the bullet and let `log_batch_size := min(total_vars, arity)` for good---and keep it
	// there, even if we post-facto find out that `fold_arities = []`. the cost of this is that the
	// prover has to do a nontrivial (though small!) interleaved encoding, as opposed to a trivial
	// one.
	let fri_params = FRIParams::new(rs_code, log_batch_size, fold_arities, n_test_queries)?;
	Ok(fri_params)
}

pub fn make_commit_params_with_optimal_arity<F, FEncode, MTScheme>(
	commit_meta: &CommitMeta,
	_merkle_scheme: &MTScheme,
	security_bits: usize,
	log_inv_rate: usize,
) -> Result<FRIParams<F, FEncode>, Error>
where
	F: BinaryField + ExtensionField<FEncode>,
	FEncode: BinaryField,
	MTScheme: MerkleTreeScheme<F>,
{
	let arity = estimate_optimal_arity(
		commit_meta.total_vars + log_inv_rate,
		size_of::<MTScheme::Digest>(),
		size_of::<F>(),
	);
	make_commit_params_with_constant_arity(commit_meta, security_bits, log_inv_rate, arity)
}

/// A description of a sumcheck claim arising from a FRI PCS sumcheck.
///
/// This is a description of a sumcheck claim with indices referencing into two slices of
/// multilinear polynomials: one slice of committed polynomials and one slice of transparent
/// polynomials. All referenced polynomials are supposed to have the same number of variables.
#[derive(Debug, Clone)]
pub struct SumcheckClaimDesc<F: Field> {
	pub committed_indices: Range<usize>,
	pub transparent_indices: Range<usize>,
	pub composite_sums: Vec<CompositeSumClaim<F, IndexComposition<BivariateProduct, 2>>>,
}

impl<F: Field> SumcheckClaimDesc<F> {
	pub fn n_committed(&self) -> usize {
		self.committed_indices.len()
	}

	pub fn n_transparent(&self) -> usize {
		self.transparent_indices.len()
	}
}

pub fn make_sumcheck_claim_descs<F: Field>(
	commit_meta: &CommitMeta,
	transparent_n_vars_iter: impl Iterator<Item = usize>,
	claims: &[PIOPSumcheckClaim<F>],
) -> Result<Vec<SumcheckClaimDesc<F>>, Error> {
	// Map of n_vars to sumcheck claim descriptions
	let mut sumcheck_claim_descs = vec![
		SumcheckClaimDesc {
			committed_indices: 0..0,
			transparent_indices: 0..0,
			composite_sums: vec![],
		};
		commit_meta.max_n_vars() + 1
	];

	// Set the n_committed and committed_offset fields on the sumcheck claim descriptions.
	let mut last_offset = 0;
	for (&n_multilins, claim_desc) in
		iter::zip(commit_meta.n_multilins_by_vars(), &mut sumcheck_claim_descs)
	{
		claim_desc.committed_indices.start = last_offset;
		last_offset += n_multilins;
		claim_desc.committed_indices.end = last_offset;
	}

	// Check that transparents are sorted by number of variables and set the n_transparent and
	// transparent_offset fields on the sumcheck claim descriptions.
	let mut current_n_vars = 0;
	for transparent_n_vars in transparent_n_vars_iter {
		match transparent_n_vars.cmp(&current_n_vars) {
			Ordering::Less => return Err(Error::TransparentsNotSorted),
			Ordering::Greater => {
				let current_desc = &sumcheck_claim_descs[current_n_vars];
				let offset = current_desc.transparent_indices.end;

				current_n_vars = transparent_n_vars;
				let next_desc = &mut sumcheck_claim_descs[current_n_vars];
				next_desc.transparent_indices = offset..offset;
			}
			_ => {}
		}

		sumcheck_claim_descs[current_n_vars].transparent_indices.end += 1;
	}

	// Convert the PCS sumcheck claims into the sumcheck claim descriptions. The main difference is
	// that we group the PCS sumcheck claims by number of multilinear variables and ultimately
	// create a `SumcheckClaim` for each.
	for (i, claim) in claims.iter().enumerate() {
		let claim_desc = &mut sumcheck_claim_descs[claim.n_vars];

		// Check that claim committed and transparent indices are in the valid range for the number
		// of variables.
		if !claim_desc.committed_indices.contains(&claim.committed) {
			bail!(Error::SumcheckClaimVariablesMismatch { index: i });
		}
		if !claim_desc.transparent_indices.contains(&claim.transparent) {
			bail!(Error::SumcheckClaimVariablesMismatch { index: i });
		}

		let composition = IndexComposition::new(
			claim_desc.committed_indices.len() + claim_desc.transparent_indices.len(),
			[
				claim.committed - claim_desc.committed_indices.start,
				claim_desc.committed_indices.len() + claim.transparent
					- claim_desc.transparent_indices.start,
			],
			BivariateProduct::default(),
		)
		.expect(
			"claim.committed and claim.transparent are checked to be in the correct ranges above",
		);
		claim_desc.composite_sums.push(CompositeSumClaim {
			sum: claim.sum,
			composition,
		});
	}

	Ok(sumcheck_claim_descs)
}

/// Verifies a batch of sumcheck claims that are products of committed polynomials from a committed
/// batch and transparent polynomials.
///
/// ## Arguments
///
/// * `commit_meta` - metadata about the committed batch of multilinears
/// * `merkle_scheme` - the Merkle tree commitment scheme used in FRI
/// * `fri_params` - the FRI parameters for the commitment opening protocol
/// * `transparents` - a slice of transparent polynomials in ascending order by number of variables
/// * `claims` - a batch of sumcheck claims referencing committed polynomials in the batch described
///   by `commit_meta` and the transparent polynomials in `transparents`
/// * `proof` - the proof reader
#[instrument("piop::verify", skip_all)]
pub fn verify<'a, F, FEncode, Challenger_, MTScheme>(
	commit_meta: &CommitMeta,
	merkle_scheme: &MTScheme,
	fri_params: &FRIParams<F, FEncode>,
	commitment: &MTScheme::Digest,
	transparents: &[impl Borrow<dyn MultivariatePoly<F> + 'a>],
	claims: &[PIOPSumcheckClaim<F>],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<(), Error>
where
	F: TowerField + ExtensionField<FEncode>,
	FEncode: BinaryField,
	Challenger_: Challenger,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
	// Map of n_vars to sumcheck claim descriptions
	let sumcheck_claim_descs = make_sumcheck_claim_descs(
		commit_meta,
		transparents.iter().map(|poly| poly.borrow().n_vars()),
		claims,
	)?;

	let non_empty_sumcheck_descs = sumcheck_claim_descs
		.iter()
		.enumerate()
		// Keep sumcheck claims with >0 committed multilinears, even with 0 composite claims. This
		// indicates unconstrained columns, but we still need the final evaluations from the
		// sumcheck prover in order to derive the final FRI value.
		.filter(|(_n_vars, desc)| !desc.committed_indices.is_empty());
	let sumcheck_claims = non_empty_sumcheck_descs
		.clone()
		.map(|(n_vars, desc)| {
			// Make a single sumcheck claim with compositions of the committed and transparent
			// polynomials with `n_vars` variables
			SumcheckClaim::new(
				n_vars,
				desc.committed_indices.len() + desc.transparent_indices.len(),
				desc.composite_sums.clone(),
			)
		})
		.collect::<Result<Vec<_>, _>>()?;

	// Interleaved front-loaded sumcheck
	let BatchInterleavedSumcheckFRIOutput {
		challenges,
		multilinear_evals,
		fri_final,
	} = verify_interleaved_fri_sumcheck(
		commit_meta.total_vars(),
		fri_params,
		merkle_scheme,
		&sumcheck_claims,
		commitment,
		transcript,
	)?;

	let mut piecewise_evals = verify_transparent_evals(
		commit_meta,
		non_empty_sumcheck_descs,
		multilinear_evals,
		transparents,
		&challenges,
	)?;

	// Verify the committed evals against the FRI final value.
	piecewise_evals.reverse();
	let n_pieces_by_vars = sumcheck_claim_descs
		.iter()
		.map(|desc| desc.n_committed())
		.collect::<Vec<_>>();
	let piecewise_eval =
		evaluate_piecewise_multilinear(&challenges, &n_pieces_by_vars, &mut piecewise_evals)?;
	if piecewise_eval != fri_final {
		return Err(VerificationError::IncorrectSumcheckEvaluation.into());
	}

	Ok(())
}

// Verify the transparent evals and collect the committed evals.
#[instrument(skip_all, level = "debug")]
fn verify_transparent_evals<'a, 'b, F: Field>(
	commit_meta: &CommitMeta,
	sumcheck_descs: impl Iterator<Item = (usize, &'a SumcheckClaimDesc<F>)>,
	multilinear_evals: Vec<Vec<F>>,
	transparents: &[impl Borrow<dyn MultivariatePoly<F> + 'b>],
	challenges: &[F],
) -> Result<Vec<F>, Error> {
	// Reverse the challenges to get the correct order for transparents. This is required because
	// the sumcheck is using high-to-low folding.
	let mut challenges_rev = challenges.to_vec();
	challenges_rev.reverse();
	let n_challenges = challenges.len();

	let mut piecewise_evals = Vec::with_capacity(commit_meta.total_multilins());
	for ((n_vars, desc), multilinear_evals) in iter::zip(sumcheck_descs, multilinear_evals) {
		let (committed_evals, transparent_evals) = multilinear_evals.split_at(desc.n_committed());
		piecewise_evals.extend_from_slice(committed_evals);

		assert_eq!(transparent_evals.len(), desc.n_transparent());
		for (i, (&claimed_eval, transparent)) in
			iter::zip(transparent_evals, &transparents[desc.transparent_indices.clone()])
				.enumerate()
		{
			let computed_eval = transparent
				.borrow()
				.evaluate(&challenges_rev[n_challenges - n_vars..])?;
			if claimed_eval != computed_eval {
				return Err(VerificationError::IncorrectTransparentEvaluation {
					index: desc.transparent_indices.start + i,
				}
				.into());
			}
		}
	}
	Ok(piecewise_evals)
}

#[derive(Debug)]
struct BatchInterleavedSumcheckFRIOutput<F> {
	challenges: Vec<F>,
	multilinear_evals: Vec<Vec<F>>,
	fri_final: F,
}

/// Runs the interleaved sumcheck & FRI invocation, reducing to committed and transparent
/// multilinear evaluation checks.
///
/// ## Preconditions
///
/// * `n_rounds` is greater than or equal to the maximum number of variables of any claim
/// * `claims` are sorted in ascending order by number of variables
#[instrument(skip_all)]
fn verify_interleaved_fri_sumcheck<F, FEncode, Challenger_, MTScheme>(
	n_rounds: usize,
	fri_params: &FRIParams<F, FEncode>,
	merkle_scheme: &MTScheme,
	claims: &[SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>],
	codeword_commitment: &MTScheme::Digest,
	proof: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchInterleavedSumcheckFRIOutput<F>, Error>
where
	F: TowerField + ExtensionField<FEncode>,
	FEncode: BinaryField,
	Challenger_: Challenger,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
	let mut arities_iter = fri_params.fold_arities().iter();
	let mut fri_commitments = Vec::with_capacity(fri_params.n_oracles());
	let mut next_commit_round = arities_iter.next().copied();

	let mut sumcheck_verifier = SumcheckBatchVerifier::new(claims, proof)?;
	let mut multilinear_evals = Vec::with_capacity(claims.len());
	let mut challenges = Vec::with_capacity(n_rounds);
	for round_no in 0..n_rounds {
		let mut reader = proof.message();
		while let Some(claim_multilinear_evals) = sumcheck_verifier.try_finish_claim(&mut reader)? {
			multilinear_evals.push(claim_multilinear_evals);
		}
		sumcheck_verifier.receive_round_proof(&mut reader)?;

		let challenge = proof.sample();
		challenges.push(challenge);

		sumcheck_verifier.finish_round(challenge)?;

		let observe_fri_comm = next_commit_round.is_some_and(|round| round == round_no + 1);
		if observe_fri_comm {
			let comm = proof
				.message()
				.read()
				.map_err(VerificationError::Transcript)?;
			fri_commitments.push(comm);
			next_commit_round = arities_iter.next().map(|arity| round_no + 1 + arity);
		}
	}

	let mut reader = proof.message();
	while let Some(claim_multilinear_evals) = sumcheck_verifier.try_finish_claim(&mut reader)? {
		multilinear_evals.push(claim_multilinear_evals);
	}
	sumcheck_verifier.finish()?;

	let verifier = FRIVerifier::new(
		fri_params,
		merkle_scheme,
		codeword_commitment,
		&fri_commitments,
		&challenges,
	)?;
	let fri_final = verifier.verify(proof)?;

	Ok(BatchInterleavedSumcheckFRIOutput {
		challenges,
		multilinear_evals,
		fri_final,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_commit_meta_new_empty() {
		let n_multilins_by_vars = vec![];
		let commit_meta = CommitMeta::new(n_multilins_by_vars);

		assert_eq!(commit_meta.total_vars, 0);
		assert_eq!(commit_meta.total_multilins, 0);
		assert!(commit_meta.n_multilins_by_vars.is_empty());
		assert!(commit_meta.offsets_by_vars.is_empty());
	}

	#[test]
	fn test_commit_meta_new_single_variable() {
		let n_multilins_by_vars = vec![4];
		let commit_meta = CommitMeta::new(n_multilins_by_vars.clone());

		assert_eq!(commit_meta.total_vars, 2);
		assert_eq!(commit_meta.total_multilins, 4);
		assert_eq!(commit_meta.n_multilins_by_vars, n_multilins_by_vars);
		assert_eq!(commit_meta.offsets_by_vars, vec![0]);
	}

	#[test]
	fn test_commit_meta_new_multiple_variables() {
		let n_multilins_by_vars = vec![3, 5, 2];

		let commit_meta = CommitMeta::new(n_multilins_by_vars.clone());

		// Sum is 3*2^0 + 5*2^1 + 2*2^2 = 21, next power of 2's log2 is 5
		assert_eq!(commit_meta.total_vars, 5);
		// 3 + 5 + 2
		assert_eq!(commit_meta.total_multilins, 10);
		assert_eq!(commit_meta.n_multilins_by_vars, n_multilins_by_vars);
		assert_eq!(commit_meta.offsets_by_vars, vec![0, 3, 8]);
	}

	#[test]
	#[allow(clippy::identity_op)]
	fn test_commit_meta_new_large_numbers() {
		let n_multilins_by_vars = vec![1_000_000, 2_000_000];
		let commit_meta = CommitMeta::new(n_multilins_by_vars.clone());

		let expected_total_elems = 1_000_000 * (1 << 0) + 2_000_000 * (1 << 1) as usize;
		let expected_total_vars = expected_total_elems.next_power_of_two().ilog2() as usize;

		assert_eq!(commit_meta.total_vars, expected_total_vars);
		assert_eq!(commit_meta.total_multilins, 3_000_000);
		assert_eq!(commit_meta.n_multilins_by_vars, n_multilins_by_vars);
		assert_eq!(commit_meta.offsets_by_vars, vec![0, 1_000_000]);
	}

	#[test]
	fn test_with_vars_empty() {
		let commit_meta = CommitMeta::with_vars(vec![]);

		assert_eq!(commit_meta.total_vars, 0);
		assert_eq!(commit_meta.total_multilins, 0);
		assert!(commit_meta.n_multilins_by_vars().is_empty());
		assert!(commit_meta.offsets_by_vars.is_empty());
	}

	#[test]
	fn test_with_vars_single_variable() {
		let commit_meta = CommitMeta::with_vars(vec![0, 0, 0, 0]);

		assert_eq!(commit_meta.total_vars, 2);
		assert_eq!(commit_meta.total_multilins, 4);
		assert_eq!(commit_meta.n_multilins_by_vars(), &[4]);
		assert_eq!(commit_meta.offsets_by_vars, vec![0]);
	}

	#[test]
	#[allow(clippy::identity_op)]
	fn test_with_vars_multiple_variables() {
		let commit_meta = CommitMeta::with_vars(vec![2, 3, 3, 4]);

		let expected_total_elems = 1 * (1 << 2) + 2 * (1 << 3) + 1 * (1 << 4) as usize;
		let expected_total_vars = expected_total_elems.next_power_of_two().ilog2() as usize;

		assert_eq!(commit_meta.total_vars, expected_total_vars);
		assert_eq!(commit_meta.total_multilins, 4);
		assert_eq!(commit_meta.n_multilins_by_vars(), &[0, 0, 1, 2, 1]);
		assert_eq!(commit_meta.offsets_by_vars, vec![0, 0, 0, 1, 3]);
	}

	#[test]
	fn test_with_vars_large_numbers() {
		// 1,000,000 polynomials with 0 variables
		let vars = vec![0; 1_000_000];
		let commit_meta = CommitMeta::with_vars(vars);

		// All polynomials with 0 variables
		let expected_total_elems = 1_000_000 * (1 << 0) as usize;
		let expected_total_vars = expected_total_elems.next_power_of_two().ilog2() as usize;

		assert_eq!(commit_meta.total_vars, expected_total_vars);
		assert_eq!(commit_meta.total_multilins, 1_000_000);
		assert_eq!(commit_meta.n_multilins_by_vars(), &[1_000_000]);
		assert_eq!(commit_meta.offsets_by_vars, vec![0]);
	}

	#[test]
	#[allow(clippy::identity_op)]
	fn test_with_vars_mixed_variables() {
		let vars = vec![0, 1, 1, 2, 2, 2, 3];
		let commit_meta = CommitMeta::with_vars(vars);

		// Sum of evaluations
		let expected_total_elems =
			1 * (1 << 0) + 2 * (1 << 1) + 3 * (1 << 2) + 1 * (1 << 3) as usize;
		let expected_total_vars = expected_total_elems.next_power_of_two().ilog2() as usize;

		assert_eq!(commit_meta.total_vars, expected_total_vars);
		assert_eq!(commit_meta.total_multilins, 7); // Total polynomials
		assert_eq!(commit_meta.n_multilins_by_vars(), &[1, 2, 3, 1]);
		assert_eq!(commit_meta.offsets_by_vars, vec![0, 1, 3, 6]);
	}
}
