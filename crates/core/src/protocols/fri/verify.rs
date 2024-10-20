// Copyright 2024 Ulvetanna Inc.

use super::{common::TerminateCodeword, error::Error, QueryProof, VerificationError};
use crate::{
	merkle_tree::VectorCommitScheme,
	polynomial::MultilinearQuery,
	protocols::fri::common::{
		calculate_fold_arities, fold_chunk, fold_interleaved_chunk, validate_common_fri_arguments,
		QueryRoundProof,
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField};
use binius_hal::make_portable_backend;
use binius_utils::bail;
use itertools::izip;
use p3_util::log2_strict_usize;
use std::iter;
use tracing::instrument;

/// A verifier for the FRI query phase.
///
/// The verifier is instantiated after the folding rounds and is used to test consistency of the
/// round messages and the original purported codeword.
#[derive(Debug)]
pub struct FRIVerifier<'a, F, FA, VCS>
where
	F: BinaryField,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	/// The Reed-Solomon code the verifier is testing proximity to.
	committed_rs_code: &'a ReedSolomonCode<FA>,
	/// Vector commitment scheme for the codeword oracle.
	committed_codeword_vcs: &'a VCS,
	log_batch_size: usize,
	/// Vector commitment scheme for the round oracles.
	round_vcss: &'a [VCS],
	/// Received commitment to the codeword.
	codeword_commitment: &'a VCS::Commitment,
	/// Received commitments to the round messages.
	round_commitments: &'a [VCS::Commitment],
	/// The challenges for each round.
	interleave_tensor: Vec<F>,
	/// The challenges for each round.
	fold_challenges: &'a [F],
	/// The termination codeword, up to which the prover has performed the folding before sending it to the verifier,
	/// must be folded to a dimension-1 codeword to check the protocol's correctness.
	terminate_codeword: TerminateCodeword<F>,
	/// The reduction arities between each oracle sent to the verifier.
	fold_arities: Vec<usize>,
}

impl<'a, F, FA, VCS> FRIVerifier<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		committed_rs_code: &'a ReedSolomonCode<FA>,
		log_batch_size: usize,
		committed_codeword_vcs: &'a VCS,
		round_vcss: &'a [VCS],
		codeword_commitment: &'a VCS::Commitment,
		round_commitments: &'a [VCS::Commitment],
		challenges: &'a [F],
		terminate_codeword: TerminateCodeword<F>,
	) -> Result<Self, Error> {
		if round_commitments.len() != round_vcss.len() {
			bail!(Error::InvalidArgs(format!(
				"got {} round commitments, expected {}",
				round_commitments.len(),
				round_vcss.len(),
			)));
		}

		let n_fold_rounds = log_batch_size + committed_rs_code.log_dim();
		if challenges.len() != n_fold_rounds {
			bail!(Error::InvalidArgs(format!(
				"got {} folding challenges, expected {}",
				challenges.len(),
				n_fold_rounds,
			)));
		}

		validate_common_fri_arguments(committed_rs_code, committed_codeword_vcs, round_vcss)?;

		let fold_arities = calculate_fold_arities(
			committed_rs_code.log_len(),
			log_batch_size,
			round_vcss
				.iter()
				.map(|vcs| log2_strict_usize(vcs.vector_len())),
		)?;

		let (interleave_challenges, fold_challenges) = challenges.split_at(log_batch_size);

		let backend = make_portable_backend();
		let interleave_tensor =
			MultilinearQuery::<F, _>::with_full_query(interleave_challenges, &backend)
				.expect("number of challenges is less than 32")
				.into_expansion();

		Ok(Self {
			committed_rs_code,
			committed_codeword_vcs,
			round_vcss,
			codeword_commitment,
			round_commitments,
			interleave_tensor,
			fold_challenges,
			terminate_codeword,
			fold_arities,
			log_batch_size,
		})
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.round_vcss.len()
	}

	/// Verifies that the last oracle sent is a codeword.
	///
	/// Returns the fully-folded message value.
	pub fn verify_last_oracle(&self) -> Result<F, Error> {
		let repetition_codeword = if let Some(last_vcs) = self.round_vcss.last() {
			// TODO: verify terminate_codeword against last_vcs

			let n_final_challenges =
				log2_strict_usize(last_vcs.vector_len()) - self.committed_rs_code.log_inv_rate();
			let n_prior_challenges = self.fold_challenges.len() - n_final_challenges;
			let final_challenges = &self.fold_challenges[n_prior_challenges..];
			let mut scratch_buffer = vec![F::default(); 1 << n_final_challenges];

			self.terminate_codeword
				.chunks(1 << n_final_challenges)
				.enumerate()
				.map(|(i, coset_values)| {
					fold_chunk(
						self.committed_rs_code,
						n_prior_challenges,
						i,
						coset_values,
						final_challenges,
						&mut scratch_buffer,
					)
				})
				.collect::<Vec<_>>()
		} else {
			// When the prover did not send any round oracles, fold the original interleaved
			// codeword.

			// TODO: verify terminate_codeword against last_vcs
			let _last_vcs = self.committed_codeword_vcs;

			let fold_arity = self.committed_rs_code.log_dim() + self.log_batch_size;
			let mut scratch_buffer = vec![F::default(); 2 * (1 << fold_arity)];
			self.terminate_codeword
				.chunks(1 << fold_arity)
				.enumerate()
				.map(|(i, chunk)| {
					fold_interleaved_chunk(
						self.committed_rs_code,
						self.log_batch_size,
						i,
						chunk,
						&self.interleave_tensor,
						self.fold_challenges,
						&mut scratch_buffer,
					)
				})
				.collect::<Vec<_>>()
		};

		let final_value = repetition_codeword[0];

		// Check that the fully-folded purported codeword is a repetition codeword.
		if repetition_codeword[1..]
			.iter()
			.any(|&entry| entry != final_value)
		{
			return Err(VerificationError::IncorrectDegree.into());
		}

		Ok(final_value)
	}

	/// Verifies a FRI challenge query.
	///
	/// A FRI challenge query tests for consistency between all consecutive oracles sent by the
	/// prover. The verifier has full access to the last oracle sent, and this is probabilistically
	/// verified to be a codeword by `Self::verify_last_oracle`.
	///
	/// ## Arguments
	///
	/// * `index` - an index into the original codeword domain
	/// * `proof` - a query proof
	#[instrument(skip_all, name = "fri::FRIVerifier::verify_query")]
	pub fn verify_query(
		&self,
		mut index: usize,
		proof: QueryProof<F, VCS::Proof>,
	) -> Result<(), Error> {
		if proof.len() != self.n_oracles() {
			return Err(VerificationError::IncorrectQueryProofLength {
				expected: self.n_oracles(),
			}
			.into());
		}

		let arities = self.fold_arities.iter().copied();
		let mut proof_and_arities_iter = iter::zip(proof, arities.clone());

		let Some((first_query_proof, first_fold_arity)) = proof_and_arities_iter.next() else {
			// If there are no query proofs, that means that no oracles were sent during the FRI
			// fold rounds. In that case, the original interleaved codeword is decommitted and
			// the only checks that need to be performed are in `verify_last_oracle`.
			return Ok(());
		};

		// Create scratch buffer used in `fold_chunk`.
		let max_arity = arities.clone().max().unwrap_or_default();
		let max_buffer_size = 2 * (1 << max_arity);
		let mut scratch_buffer = vec![F::default(); max_buffer_size];

		// This is the round of the folding phase that the codeword to be folded is committed to.
		let mut fold_round = 0;

		// Check the first fold round before the main loop. It is special because in the first
		// round we need to fold as an interleaved chunk instead of a regular coset.
		let log_coset_size = first_fold_arity - self.log_batch_size;
		let coset_index = index >> log_coset_size;
		let values = verify_interleaved_opening(
			self.committed_codeword_vcs,
			self.codeword_commitment,
			coset_index,
			log_coset_size,
			self.log_batch_size,
			first_query_proof,
		)?;

		let mut next_value = fold_interleaved_chunk(
			self.committed_rs_code,
			self.log_batch_size,
			coset_index,
			&values,
			&self.interleave_tensor,
			&self.fold_challenges[fold_round..fold_round + log_coset_size],
			&mut scratch_buffer,
		);
		index = coset_index;
		fold_round += log_coset_size;

		for (i, (vcs, commitment, (round_proof, arity))) in
			izip!(self.round_vcss.iter(), self.round_commitments.iter(), proof_and_arities_iter)
				.enumerate()
		{
			let coset_index = index >> arity;
			let values =
				verify_coset_opening(vcs, commitment, i + 1, coset_index, arity, round_proof)?;

			if next_value != values[index % (1 << arity)] {
				return Err(VerificationError::IncorrectFold {
					query_round: i,
					index,
				}
				.into());
			}

			next_value = fold_chunk(
				self.committed_rs_code,
				fold_round,
				coset_index,
				&values,
				&self.fold_challenges[fold_round..fold_round + arity],
				&mut scratch_buffer,
			);
			index = coset_index;
			fold_round += arity;
		}

		if next_value != self.terminate_codeword[index] {
			return Err(VerificationError::IncorrectFold {
				query_round: self.n_oracles() - 1,
				index,
			}
			.into());
		}

		Ok(())
	}
}

/// Verifies that the coset opening provided in the proof is consistent with the VCS commitment.
fn verify_coset_opening<F: BinaryField, VCS: VectorCommitScheme<F>>(
	vcs: &VCS,
	commitment: &VCS::Commitment,
	round: usize,
	coset_index: usize,
	log_coset_size: usize,
	proof: QueryRoundProof<F, VCS::Proof>,
) -> Result<Vec<F>, Error> {
	let QueryRoundProof { values, vcs_proof } = proof;

	if values.len() != 1 << log_coset_size {
		return Err(VerificationError::IncorrectQueryProofValuesLength {
			round,
			coset_size: 1 << log_coset_size,
		}
		.into());
	}

	let range = (coset_index << log_coset_size)..((coset_index + 1) << log_coset_size);
	vcs.verify_range_batch_opening(commitment, range, vcs_proof, iter::once(values.as_slice()))
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	// This condition should be guaranteed by the VCS verification.
	debug_assert_eq!(values.len(), 1 << log_coset_size);

	Ok(values)
}

/// Verifies that the coset opening provided in the proof is consistent with the VCS commitment.
fn verify_interleaved_opening<F: BinaryField, VCS: VectorCommitScheme<F>>(
	vcs: &VCS,
	commitment: &VCS::Commitment,
	coset_index: usize,
	log_coset_size: usize,
	log_batch_size: usize,
	proof: QueryRoundProof<F, VCS::Proof>,
) -> Result<Vec<F>, Error> {
	let QueryRoundProof { values, vcs_proof } = proof;

	if values.len() != 1 << (log_coset_size + log_batch_size) {
		return Err(VerificationError::IncorrectQueryProofValuesLength {
			round: 0,
			coset_size: 1 << (log_coset_size + log_batch_size),
		}
		.into());
	}

	let range = (coset_index << log_coset_size)..((coset_index + 1) << log_coset_size);
	vcs.verify_range_batch_opening(
		commitment,
		range,
		vcs_proof,
		(0..1 << log_batch_size).map(|i| {
			(0..1 << log_coset_size)
				.map(|j| values[i + (j << log_batch_size)])
				.collect::<Vec<_>>()
		}),
	)
	.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(values)
}
