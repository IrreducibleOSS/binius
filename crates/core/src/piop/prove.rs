// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	packed::{get_packed_slice_unchecked, set_packed_slice, set_packed_slice_unchecked},
	scalars_collection::{CollectionSubrangeMut, PackedSliceMut, ScalarsCollectionMut},
	BinaryField, Field, PackedExtension, PackedField, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{
	EvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly,
};
use binius_maybe_rayon::{iter::IntoParallelIterator, prelude::*};
use binius_ntt::{NTTOptions, ThreadingSettings};
use binius_utils::{
	bail, checked_arithmetics::checked_log_2, sorting::is_sorted_ascending, SerializeBytes,
};
use either::Either;
use itertools::{chain, Itertools};

use super::{
	error::Error,
	verify::{make_sumcheck_claim_descs, PIOPSumcheckClaim},
};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::{MerkleTreeProver, MerkleTreeScheme},
	piop::CommitMeta,
	protocols::{
		fri,
		fri::{FRIFolder, FRIParams, FoldRoundOutput},
		sumcheck,
		sumcheck::{
			immediate_switchover_heuristic,
			prove::{
				front_loaded::BatchProver as SumcheckBatchProver, RegularSumcheckProver,
				SumcheckProver,
			},
		},
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
	transcript::ProverTranscript,
};

/// Reorders the scalars in a slice of packed field elements by reversing the bits of their indices.
/// TODO: investigate if we can optimize this.
fn reverse_index_bits<F>(collection: &mut impl ScalarsCollectionMut<F>) {
	let log_len = checked_log_2(collection.len());
	for i in 0..collection.len() {
		let bit_reversed_index = i
			.reverse_bits()
			.wrapping_shr((usize::BITS as usize - log_len) as _);
		if i < bit_reversed_index {
			// Safety: `i` and `j` are guaranteed to be in bounds of the slice
			unsafe {
				let tmp = collection.get_unchecked(i);
				collection.set_unchecked(i, collection.get_unchecked(bit_reversed_index));
				collection.set_unchecked(bit_reversed_index, tmp);
			}
		}
	}
}

// ## Preconditions
//
// * all multilinears in `multilins` have at least log_extension_degree packed variables
// * all multilinears in `multilins` have `packed_evals()` is Some
// * multilinears are sorted in ascending order by number of packed variables
// * `message_buffer` is initialized to all zeros
// * `message_buffer` is larger than the total number of scalars in the multilinears
fn merge_multilins<P, M>(multilins: &[M], message_buffer: &mut [P])
where
	P: PackedField,
	M: MultilinearPoly<P>,
{
	let mut mle_iter = multilins.iter().rev();

	// First copy all the polynomials where the number of elements is a multiple of the packing
	// width.
	let get_n_packed_vars = |mle: &M| mle.n_vars() - mle.log_extension_degree();
	let mut full_packed_mles = Vec::new(); // (evals, corresponding buffer where to copy)
	let mut remaining_buffer = message_buffer;
	for mle in mle_iter.peeking_take_while(|mle| get_n_packed_vars(mle) >= P::LOG_WIDTH) {
		let evals = mle
			.packed_evals()
			.expect("guaranteed by function precondition");
		let (chunk, rest) = remaining_buffer.split_at_mut(evals.len());
		full_packed_mles.push((evals, chunk));
		remaining_buffer = rest;
	}
	full_packed_mles
		.into_par_iter()
		.for_each(|(evals, mut chunk)| {
			chunk.copy_from_slice(evals);
			reverse_index_bits(&mut chunk);
		});

	// Now copy scalars from the remaining multilinears, which have too few elements to copy full
	// packed elements.
	let mut scalar_offset = 0;
	let mut remaining_buffer = PackedSliceMut::new(remaining_buffer);
	for mle in mle_iter {
		let evals = mle
			.packed_evals()
			.expect("guaranteed by function precondition");
		let packed_eval = evals[0];
		let len = 1 << get_n_packed_vars(mle);
		let mut packed_chunk =
			CollectionSubrangeMut::new(&mut remaining_buffer, scalar_offset, len);
		for i in 0..len {
			packed_chunk.set(i, packed_eval.get(i));
		}
		reverse_index_bits(&mut packed_chunk);

		scalar_offset += len;
	}
}

/// Commits a batch of multilinear polynomials.
///
/// The multilinears this function accepts as arguments may be defined over subfields of `F`. In
/// this case, we commit to these multilinears by instead committing to their "packed"
/// multilinears. These are the multilinear extensions of their packed coefficients over subcubes
/// of the size of the extension degree.
///
/// ## Arguments
///
/// * `fri_params` - the FRI parameters for the commitment opening protocol
/// * `merkle_prover` - the Merkle tree prover used in FRI
/// * `multilins` - a batch of multilinear polynomials to commit. The multilinears provided may be
///     defined over subfields of `F`. They must be in ascending order by the number of variables
///     in the packed multilinear (ie. number of variables minus log extension degree).
#[tracing::instrument("piop::commit", skip_all)]
pub fn commit<F, FEncode, P, M, MTScheme, MTProver>(
	fri_params: &FRIParams<F, FEncode>,
	merkle_prover: &MTProver,
	multilins: &[M],
) -> Result<fri::CommitOutput<P, MTScheme::Digest, MTProver::Committed>, Error>
where
	F: BinaryField,
	FEncode: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<FEncode>,
	M: MultilinearPoly<P>,
	MTScheme: MerkleTreeScheme<F>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
{
	for (i, multilin) in multilins.iter().enumerate() {
		if multilin.n_vars() < multilin.log_extension_degree() {
			return Err(Error::OracleTooSmall {
				// i is not an OracleId, but whatever, that's a problem for whoever has to debug
				// this
				id: i,
				n_vars: multilin.n_vars(),
				min_vars: multilin.log_extension_degree(),
			});
		}
		if multilin.packed_evals().is_none() {
			return Err(Error::CommittedPackedEvaluationsMissing { id: i });
		}
	}

	let n_packed_vars = multilins
		.iter()
		.map(|multilin| multilin.n_vars() - multilin.log_extension_degree());
	if !is_sorted_ascending(n_packed_vars) {
		return Err(Error::CommittedsNotSorted);
	}

	// TODO: this should be passed in to avoid recomputing twiddles
	let rs_code = ReedSolomonCode::new(
		fri_params.rs_code().log_dim(),
		fri_params.rs_code().log_inv_rate(),
		&NTTOptions {
			precompute_twiddles: true,
			thread_settings: ThreadingSettings::MultithreadedDefault,
		},
	)?;
	let output =
		fri::commit_interleaved_with(&rs_code, fri_params, merkle_prover, |message_buffer| {
			merge_multilins(multilins, message_buffer)
		})?;

	Ok(output)
}

/// Proves a batch of sumcheck claims that are products of committed polynomials from a committed
/// batch and transparent polynomials.
///
/// The arguments corresponding to the committed multilinears must be the output of [`commit`].
#[allow(clippy::too_many_arguments)]
#[tracing::instrument("piop::prove", skip_all)]
pub fn prove<F, FDomain, FEncode, P, M, DomainFactory, MTScheme, MTProver, Challenger_, Backend>(
	fri_params: &FRIParams<F, FEncode>,
	merkle_prover: &MTProver,
	domain_factory: DomainFactory,
	commit_meta: &CommitMeta,
	committed: MTProver::Committed,
	codeword: &[P],
	committed_multilins: &[M],
	transparent_multilins: &[M],
	claims: &[PIOPSumcheckClaim<F>],
	transcript: &mut ProverTranscript<Challenger_>,
	backend: &Backend,
) -> Result<(), Error>
where
	F: TowerField,
	FDomain: Field,
	FEncode: BinaryField,
	P: PackedField<Scalar = F>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode>,
	M: MultilinearPoly<P> + Send + Sync,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
	Challenger_: Challenger,
	Backend: ComputationBackend,
{
	// Map of n_vars to sumcheck claim descriptions
	let sumcheck_claim_descs = make_sumcheck_claim_descs(
		commit_meta,
		transparent_multilins.iter().map(|poly| poly.n_vars()),
		claims,
	)?;

	// The committed multilinears provided by argument are committed *small field* multilinears.
	// Create multilinears representing the packed polynomials here. Eventually, we would like to
	// refactor the calling code so that the PIOP only handles *big field* multilinear witnesses.
	let packed_committed_multilins = committed_multilins
		.iter()
		.enumerate()
		.map(|(i, committed_multilin)| {
			let packed_evals = committed_multilin
				.packed_evals()
				.ok_or(Error::CommittedPackedEvaluationsMissing { id: i })?;
			let packed_multilin = MultilinearExtension::from_values_slice(packed_evals)?;
			Ok::<_, Error>(MLEDirectAdapter::from(packed_multilin))
		})
		.collect::<Result<Vec<_>, _>>()?;

	let non_empty_sumcheck_descs = sumcheck_claim_descs
		.iter()
		.enumerate()
		.filter(|(_n_vars, desc)| !desc.composite_sums.is_empty());
	let sumcheck_provers = non_empty_sumcheck_descs
		.clone()
		.map(|(_n_vars, desc)| {
			let multilins = chain!(
				packed_committed_multilins[desc.committed_indices.clone()]
					.iter()
					.map(Either::Left),
				transparent_multilins[desc.transparent_indices.clone()]
					.iter()
					.map(Either::Right),
			)
			.collect::<Vec<_>>();
			RegularSumcheckProver::new(
				EvaluationOrder::HighToLow,
				multilins,
				desc.composite_sums.iter().cloned(),
				&domain_factory,
				immediate_switchover_heuristic,
				backend,
			)
		})
		.collect::<Result<Vec<_>, _>>()?;

	prove_interleaved_fri_sumcheck(
		commit_meta.total_vars(),
		fri_params,
		merkle_prover,
		sumcheck_provers,
		codeword,
		&committed,
		transcript,
	)?;

	Ok(())
}

fn prove_interleaved_fri_sumcheck<F, FEncode, P, MTScheme, MTProver, Challenger_>(
	n_rounds: usize,
	fri_params: &FRIParams<F, FEncode>,
	merkle_prover: &MTProver,
	sumcheck_provers: Vec<impl SumcheckProver<F>>,
	codeword: &[P],
	committed: &MTProver::Committed,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<(), Error>
where
	F: TowerField,
	FEncode: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<FEncode>,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
	Challenger_: Challenger,
{
	let mut fri_prover = FRIFolder::new(fri_params, merkle_prover, codeword, committed)?;

	let mut sumcheck_batch_prover = SumcheckBatchProver::new(sumcheck_provers, transcript)?;

	for _ in 0..n_rounds {
		sumcheck_batch_prover.send_round_proof(&mut transcript.message())?;
		let challenge = transcript.sample();
		sumcheck_batch_prover.receive_challenge(challenge)?;

		match fri_prover.execute_fold_round(challenge)? {
			FoldRoundOutput::NoCommitment => {}
			FoldRoundOutput::Commitment(round_commitment) => {
				transcript.message().write(&round_commitment);
			}
		}
	}

	sumcheck_batch_prover.finish(&mut transcript.message())?;
	fri_prover.finish_proof(transcript)?;
	Ok(())
}

pub fn validate_sumcheck_witness<F, P, M>(
	committed_multilins: &[M],
	transparent_multilins: &[M],
	claims: &[PIOPSumcheckClaim<F>],
) -> Result<(), Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P> + Send + Sync,
{
	let packed_committed = committed_multilins
		.iter()
		.enumerate()
		.map(|(i, unpacked_committed)| {
			let packed_evals = unpacked_committed
				.packed_evals()
				.ok_or(Error::CommittedPackedEvaluationsMissing { id: i })?;
			let packed_committed = MultilinearExtension::from_values_slice(packed_evals)?;
			Ok::<_, Error>(packed_committed)
		})
		.collect::<Result<Vec<_>, _>>()?;

	for (i, claim) in claims.iter().enumerate() {
		let committed = &packed_committed[claim.committed];
		if committed.n_vars() != claim.n_vars {
			bail!(sumcheck::Error::NumberOfVariablesMismatch);
		}

		let transparent = &transparent_multilins[claim.transparent];
		if transparent.n_vars() != claim.n_vars {
			bail!(sumcheck::Error::NumberOfVariablesMismatch);
		}

		let sum = (0..(1 << claim.n_vars))
			.into_par_iter()
			.map(|j| {
				let committed_eval = committed
					.evaluate_on_hypercube(j)
					.expect("j is less than 1 << n_vars; committed.n_vars is checked above");
				let transparent_eval = transparent
					.evaluate_on_hypercube(j)
					.expect("j is less than 1 << n_vars; transparent.n_vars is checked above");
				committed_eval * transparent_eval
			})
			.sum::<F>();

		if sum != claim.sum {
			bail!(sumcheck::Error::SumcheckNaiveValidationFailure {
				composition_index: i,
			});
		}
	}
	Ok(())
}
