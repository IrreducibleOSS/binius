// Copyright 2024-2025 Irreducible Inc.

use std::{borrow::Cow, ops::Deref};

use binius_field::{
	packed::PackedSliceMut, BinaryField, Field, PackedExtension, PackedField, TowerField,
};
use binius_hal::ComputationBackend;
use binius_math::{
	EvaluationDomainFactory, EvaluationOrder, MLEDirectAdapter, MultilinearExtension,
	MultilinearPoly,
};
use binius_maybe_rayon::{iter::IntoParallelIterator, prelude::*};
use binius_ntt::{NTTOptions, ThreadingSettings};
use binius_utils::{
	bail,
	checked_arithmetics::checked_log_2,
	random_access_sequence::{RandomAccessSequenceMut, SequenceSubrangeMut},
	sorting::is_sorted_ascending,
	SerializeBytes,
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

#[inline(always)]
fn reverse_bits(x: usize, log_len: usize) -> usize {
	x.reverse_bits()
		.wrapping_shr((usize::BITS as usize - log_len) as _)
}

/// Reorders the scalars in a slice of packed field elements by reversing the bits of their indices.
/// TODO: investigate if we can optimize this.
fn reverse_index_bits<T: Copy>(collection: &mut impl RandomAccessSequenceMut<T>) {
	let log_len = checked_log_2(collection.len());
	for i in 0..collection.len() {
		let bit_reversed_index = reverse_bits(i, log_len);
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
fn merge_multilins<F, P, Data>(
	multilins: &[MultilinearExtension<P, Data>],
	message_buffer: &mut [P],
) where
	F: TowerField,
	P: PackedField<Scalar = F>,
	Data: Deref<Target = [P]>,
{
	let mut mle_iter = multilins.iter().rev();

	// First copy all the polynomials where the number of elements is a multiple of the packing
	// width.
	let mut full_packed_mles = Vec::new(); // (evals, corresponding buffer where to copy)
	let mut remaining_buffer = message_buffer;
	for mle in mle_iter.peeking_take_while(|mle| mle.n_vars() >= P::LOG_WIDTH) {
		let evals = mle.evals();
		let (chunk, rest) = remaining_buffer.split_at_mut(evals.len());
		full_packed_mles.push((evals, chunk));
		remaining_buffer = rest;
	}
	full_packed_mles.into_par_iter().for_each(|(evals, chunk)| {
		chunk.copy_from_slice(evals);
		reverse_index_bits(&mut PackedSliceMut::new(chunk));
	});

	// Now copy scalars from the remaining multilinears, which have too few elements to copy full
	// packed elements.
	let mut scalar_offset = 0;
	let mut remaining_buffer = PackedSliceMut::new(remaining_buffer);
	for mle in mle_iter {
		let packed_eval = mle.evals()[0];
		let len = 1 << mle.n_vars();
		let mut packed_chunk = SequenceSubrangeMut::new(&mut remaining_buffer, scalar_offset, len);
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
///   defined over subfields of `F`. They must be in ascending order by the number of variables
///   in the packed multilinear (ie. number of variables minus log extension degree).
pub fn commit<F, FEncode, P, M, MTScheme, MTProver>(
	fri_params: &FRIParams<F, FEncode>,
	merkle_prover: &MTProver,
	multilins: &[M],
) -> Result<fri::CommitOutput<P, MTScheme::Digest, MTProver::Committed>, Error>
where
	F: TowerField,
	FEncode: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<FEncode>,
	M: MultilinearPoly<P>,
	MTScheme: MerkleTreeScheme<F>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
{
	let packed_multilins = multilins
		.iter()
		.enumerate()
		.map(|(i, unpacked_committed)| packed_committed(i, unpacked_committed))
		.collect::<Result<Vec<_>, _>>()?;
	if !is_sorted_ascending(packed_multilins.iter().map(|mle| mle.n_vars())) {
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
			merge_multilins(&packed_multilins, message_buffer)
		})?;

	Ok(output)
}

/// Proves a batch of sumcheck claims that are products of committed polynomials from a committed
/// batch and transparent polynomials.
///
/// The arguments corresponding to the committed multilinears must be the output of [`commit`].
#[allow(clippy::too_many_arguments)]
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
		.map(|(i, unpacked_committed)| {
			packed_committed(i, unpacked_committed).map(MLEDirectAdapter::from)
		})
		.collect::<Result<Vec<_>, _>>()?;

	let non_empty_sumcheck_descs = sumcheck_claim_descs
		.iter()
		.enumerate()
		// Keep sumcheck claims with >0 committed multilinears, even with 0 composite claims. This
		// indicates unconstrained columns, but we still need the final evaluations from the
		// sumcheck prover in order to derive the final FRI value.
		.filter(|(_n_vars, desc)| !desc.committed_indices.is_empty());
	let sumcheck_provers = non_empty_sumcheck_descs
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

	for round in 0..n_rounds {
		let _span = tracing::info_span!(
			"[phase] PIOP Compiler Round",
			phase = "piop_compiler",
			round = round
		)
		.entered();

		let bivariate_sumcheck_span = tracing::info_span!(
			"[phase] Bivariate Sumcheck",
			phase = "piop_compiler",
			round = round,
			perfetto_category = "phase.sub"
		)
		.entered();
		let bivariate_sumcheck_calculate_coeffs_span = tracing::info_span!(
			"[task] (PIOP Compiler) Calculate Coeffs",
			phase = "piop_compiler",
			round = round,
			perfetto_category = "task.main"
		)
		.entered();
		sumcheck_batch_prover.send_round_proof(&mut transcript.message())?;
		drop(bivariate_sumcheck_calculate_coeffs_span);
		let challenge = transcript.sample();
		let bivariate_sumcheck_all_folds_span = tracing::info_span!(
			"[task] (PIOP Compiler) Fold (All Rounds)",
			phase = "piop_compiler",
			round = round,
			perfetto_category = "task.main"
		)
		.entered();
		sumcheck_batch_prover.receive_challenge(challenge)?;
		drop(bivariate_sumcheck_all_folds_span);
		drop(bivariate_sumcheck_span);

		let fri_fold_rounds_span = tracing::info_span!(
			"[phase] FRI Fold Rounds",
			phase = "piop_compiler",
			round = round,
			perfetto_category = "phase.sub"
		)
		.entered();
		match fri_prover.execute_fold_round(challenge)? {
			FoldRoundOutput::NoCommitment => {}
			FoldRoundOutput::Commitment(round_commitment) => {
				transcript.message().write(&round_commitment);
			}
		}
		drop(fri_fold_rounds_span);
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
		.map(|(i, unpacked_committed)| packed_committed(i, unpacked_committed))
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

/// Creates a multilinear extension of the packed evalations of a small-field multilinear.
///
/// Given a multilinear $P \in T_{\iota}[X_0, \ldots, X_{n-1}]$, this creates the multilinear
/// extension $\hat{P} \in T_{\tau}[X_0, \ldots, X_{n - \kappa - 1}]$. In the case where
/// $n < \kappa$, which is when a polynomial is too full to have even a single packed evaluation,
/// the polynomial is extended by padding with more variables, which corresponds to repeating its
/// subcube evaluations.
fn packed_committed<F, P, M>(
	id: usize,
	unpacked_committed: &M,
) -> Result<MultilinearExtension<P, Cow<'_, [P]>>, Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P>,
{
	let unpacked_n_vars = unpacked_committed.n_vars();
	let packed_committed = if unpacked_n_vars < unpacked_committed.log_extension_degree() {
		let packed_eval = padded_packed_eval(unpacked_committed);
		MultilinearExtension::new(0, Cow::Owned(vec![P::set_single(packed_eval)]))
	} else {
		let packed_evals = unpacked_committed
			.packed_evals()
			.ok_or(Error::CommittedPackedEvaluationsMissing { id })?;
		MultilinearExtension::from_values_generic(Cow::Borrowed(packed_evals))
	}?;
	Ok(packed_committed)
}

#[inline]
fn padded_packed_eval<F, P, M>(multilin: &M) -> F
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	M: MultilinearPoly<P>,
{
	let n_vars = multilin.n_vars();
	let kappa = multilin.log_extension_degree();
	assert!(n_vars < kappa);

	(0..1 << kappa)
		.map(|i| {
			let iota = F::TOWER_LEVEL - kappa;
			let scalar = <F as TowerField>::basis(iota, i)
				.expect("i is in range 0..1 << log_extension_degree");
			multilin
				.evaluate_on_hypercube_and_scale(i % (1 << n_vars), scalar)
				.expect("i is in range 0..1 << n_vars")
		})
		.sum()
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::PackedBinaryField2x128b;
	use rand::{rngs::StdRng, SeedableRng};

	use super::*;

	#[test]
	fn test_merge_multilins() {
		let mut rng = StdRng::seed_from_u64(0);

		let multilins = (0usize..8)
			.map(|n_vars| {
				let data = repeat_with(|| PackedBinaryField2x128b::random(&mut rng))
					.take(1 << n_vars.saturating_sub(PackedBinaryField2x128b::LOG_WIDTH))
					.collect::<Vec<_>>();

				MultilinearExtension::new(n_vars, data).unwrap()
			})
			.collect::<Vec<_>>();
		let scalars = (0..8).map(|i| 1usize << i).sum::<usize>();
		let mut buffer =
			vec![PackedBinaryField2x128b::zero(); scalars.div_ceil(PackedBinaryField2x128b::WIDTH)];
		merge_multilins(&multilins, &mut buffer);

		let scalars = PackedField::iter_slice(&buffer).take(scalars).collect_vec();
		let mut offset = 0;
		for multilin in multilins.iter().rev() {
			let scalars = &scalars[offset..];
			for (i, v) in PackedField::iter_slice(multilin.evals())
				.take(1 << multilin.n_vars())
				.enumerate()
			{
				assert_eq!(scalars[reverse_bits(i, multilin.n_vars())], v);
			}
			offset += 1 << multilin.n_vars();
		}
	}
}
