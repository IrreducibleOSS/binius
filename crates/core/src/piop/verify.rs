// Copyright 2024 Irreducible, Inc

use std::{borrow::Borrow, cmp::Ordering, iter, ops::Range};

use binius_field::{BinaryField, ExtensionField, Field, TowerField};
use binius_math::evaluate_piecewise_multilinear;
use binius_ntt::NTTOptions;
use binius_utils::{bail, serialization::DeserializeBytes};
use getset::CopyGetters;
use tracing::instrument;

use super::error::{Error, VerificationError};
use crate::{
	composition::{BivariateProduct, IndexComposition},
	fiat_shamir::{CanSample, CanSampleBits},
	merkle_tree::MerkleTreeScheme,
	piop::util::ResizeableIndex,
	polynomial::MultivariatePoly,
	protocols::{
		fri::{self, estimate_optimal_arity, FRIParams, FRIVerifier},
		sumcheck::{
			front_loaded::BatchVerifier as SumcheckBatchVerifier, CompositeSumClaim, SumcheckClaim,
		},
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
	transcript::{CanRead, Proof},
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
	///     multilinears in the batch with that number of variables
	pub fn new(n_multilins_by_vars: Vec<usize>) -> Self {
		let offsets_by_vars = n_multilins_by_vars
			.iter()
			.copied()
			.scan(0, |offset, n_multilins| {
				let last_offset = *offset;
				*offset += n_multilins;
				Some(last_offset)
			})
			.collect();

		let total_elems = n_multilins_by_vars
			.iter()
			.enumerate()
			.map(|(n_vars, n_pieces)| n_pieces << n_vars)
			.sum::<usize>();
		let total_vars = total_elems.next_power_of_two().ilog2() as usize;
		let total_multilins = n_multilins_by_vars.iter().copied().sum();

		Self {
			offsets_by_vars,
			n_multilins_by_vars,
			total_vars,
			total_multilins,
		}
	}

	/// Constructs a new [`CommitMeta`] from a sequence of committed polynomials described by their
	/// number of variables.
	pub fn with_vars(n_varss: impl IntoIterator<Item = usize>) -> Self {
		let mut n_multilins_by_vars = ResizeableIndex::<usize>::new();
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

	let fold_arities = iter::repeat(arity)
		// The total arities must be strictly less than n_packed_vars, hence the -1
		.take(commit_meta.total_vars.saturating_sub(1) / arity)
		.collect::<Vec<_>>();

	// Choose the interleaved code batch size to align with the first fold arity, which is
	// optimal.
	let log_batch_size = fold_arities.first().copied().unwrap_or(0);
	let log_dim = commit_meta.total_vars - log_batch_size;

	let rs_code = ReedSolomonCode::new(log_dim, log_inv_rate, NTTOptions::default())?;
	let n_test_queries = fri::calculate_n_test_queries::<F, _>(security_bits, &rs_code)?;
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
/// * `claims` - a batch of sumcheck claims referencing committed polynomials in the batch
///     described by `commit_meta` and the transparent polynomials in `transparents`
/// * `proof` - the proof reader
#[instrument("piop::verify", skip_all)]
pub fn verify<'a, F, FEncode, Transcript, Advice, MTScheme>(
	commit_meta: &CommitMeta,
	merkle_scheme: &MTScheme,
	fri_params: &FRIParams<F, FEncode>,
	commitment: &MTScheme::Digest,
	transparents: &[impl Borrow<dyn MultivariatePoly<F> + 'a>],
	claims: &[PIOPSumcheckClaim<F>],
	proof: &mut Proof<Transcript, Advice>,
) -> Result<(), Error>
where
	F: TowerField + ExtensionField<FEncode>,
	FEncode: BinaryField,
	Transcript: CanSample<F> + CanRead + CanSampleBits<usize>,
	Advice: CanRead,
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
		.filter(|(_n_vars, desc)| !desc.composite_sums.is_empty());
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
		proof,
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
	let mut piecewise_evals = Vec::with_capacity(commit_meta.total_multilins());
	for ((n_vars, desc), multilinear_evals) in iter::zip(sumcheck_descs, multilinear_evals) {
		let (committed_evals, transparent_evals) = multilinear_evals.split_at(desc.n_committed());
		piecewise_evals.extend_from_slice(committed_evals);

		assert_eq!(transparent_evals.len(), desc.n_transparent());
		for (i, (&claimed_eval, transparent)) in
			iter::zip(transparent_evals, &transparents[desc.transparent_indices.clone()])
				.enumerate()
		{
			let computed_eval = transparent.borrow().evaluate(&challenges[..n_vars])?;
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
fn verify_interleaved_fri_sumcheck<F, FEncode, Transcript, Advice, MTScheme>(
	n_rounds: usize,
	fri_params: &FRIParams<F, FEncode>,
	merkle_scheme: &MTScheme,
	claims: &[SumcheckClaim<F, IndexComposition<BivariateProduct, 2>>],
	codeword_commitment: &MTScheme::Digest,
	proof: &mut Proof<Transcript, Advice>,
) -> Result<BatchInterleavedSumcheckFRIOutput<F>, Error>
where
	F: TowerField + ExtensionField<FEncode>,
	FEncode: BinaryField,
	Transcript: CanSample<F> + CanRead + CanSampleBits<usize>,
	Advice: CanRead,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
{
	let mut arities_iter = fri_params.fold_arities().iter();
	let mut fri_commitments = Vec::with_capacity(fri_params.n_oracles());
	let mut next_commit_round = arities_iter.next().copied();

	let mut sumcheck_verifier = SumcheckBatchVerifier::new(claims, &mut proof.transcript)?;
	let mut multilinear_evals = Vec::with_capacity(claims.len());
	let mut challenges = Vec::with_capacity(n_rounds);
	for round_no in 0..n_rounds {
		while let Some(claim_multilinear_evals) =
			sumcheck_verifier.try_finish_claim(&mut proof.transcript)?
		{
			multilinear_evals.push(claim_multilinear_evals);
		}
		sumcheck_verifier.receive_round_proof(&mut proof.transcript)?;

		let challenge = proof.transcript.sample();
		challenges.push(challenge);

		sumcheck_verifier.finish_round(challenge)?;

		let observe_fri_comm = next_commit_round.is_some_and(|round| round == round_no + 1);
		if observe_fri_comm {
			let comm = proof
				.transcript
				.read()
				.map_err(VerificationError::Transcript)?;
			fri_commitments.push(comm);
			next_commit_round = arities_iter.next().map(|arity| round_no + 1 + arity);
		}
	}

	while let Some(claim_multilinear_evals) =
		sumcheck_verifier.try_finish_claim(&mut proof.transcript)?
	{
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
	let fri_final = verifier.verify(&mut proof.advice, &mut proof.transcript)?;

	Ok(BatchInterleavedSumcheckFRIOutput {
		challenges,
		multilinear_evals,
		fri_final,
	})
}
