// Copyright 2024 Ulvetanna Inc.

use super::{
	common::{
		calculate_fold_chunk_start_rounds, calculate_fold_commit_rounds, calculate_folding_arities,
		FinalCodeword, FinalMessage,
	},
	error::Error,
	QueryProof, VerificationError,
};
use crate::{
	linear_code::LinearCode,
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_chunk, QueryRoundProof},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField};
use itertools::{izip, Itertools};
use std::{iter, ops::Range};
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
	/// Vector commitment scheme for the round oracles.
	round_vcss: &'a [VCS],
	/// Received commitment to the codeword.
	codeword_commitment: &'a VCS::Commitment,
	/// Received commitments to the round messages.
	round_commitments: &'a [VCS::Commitment],
	/// The challenges for each round.
	challenges: &'a [F],
	/// The final message, which must be re-encoded to start fold consistency checks.
	final_codeword: FinalCodeword<F>,
	/// The start round of each fold chunk call made by the FRIFolder.
	fold_chunk_start_rounds: Vec<usize>,
	/// The arity of each fold chunk call made by the FRIFolder.
	folding_arities: Vec<usize>,
	/// The range of challenges used in each fold chunk call made by the FRIFolder.
	challenge_ranges: Vec<Range<usize>>,
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
		final_rs_code: &'a ReedSolomonCode<F>,
		committed_codeword_vcs: &'a VCS,
		round_vcss: &'a [VCS],
		codeword_commitment: &'a VCS::Commitment,
		round_commitments: &'a [VCS::Commitment],
		challenges: &'a [F],
		final_message: FinalMessage<F>,
	) -> Result<Self, Error> {
		let final_codeword = final_rs_code.encode(final_message)?;

		if round_commitments.len() != round_vcss.len() {
			return Err(Error::InvalidArgs(format!(
				"got {} round commitments, expected {}",
				round_commitments.len(),
				round_vcss.len(),
			)));
		}

		if challenges.len() != committed_rs_code.log_dim() {
			return Err(Error::InvalidArgs(format!(
				"got {} folding challenges, expected {}",
				challenges.len(),
				committed_rs_code.log_dim()
			)));
		}

		// TODO: With future STIR-like optimizations, this check should be removed.
		if final_rs_code.log_inv_rate() != committed_rs_code.log_inv_rate() {
			return Err(Error::InvalidArgs(
				"final RS code must have the same rate as the committed RS code".to_string(),
			));
		}

		let commitment_fold_rounds = calculate_fold_commit_rounds(
			committed_rs_code,
			final_rs_code,
			committed_codeword_vcs,
			round_vcss,
		)?;
		let fold_chunk_start_rounds = calculate_fold_chunk_start_rounds(&commitment_fold_rounds);
		let folding_arities =
			calculate_folding_arities(committed_rs_code.log_dim(), &fold_chunk_start_rounds);
		let challenge_ranges =
			calculate_challenge_ranges(committed_rs_code.log_dim(), &fold_chunk_start_rounds);

		Ok(Self {
			committed_rs_code,
			committed_codeword_vcs,
			round_vcss,
			codeword_commitment,
			round_commitments,
			challenges,
			final_codeword,
			fold_chunk_start_rounds,
			folding_arities,
			challenge_ranges,
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.round_vcss.len() + 1
	}

	/// Verifies a FRI challenge query.
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
		if proof.len() != self.n_rounds() {
			return Err(VerificationError::IncorrectQueryProofLength {
				expected: self.n_rounds(),
			}
			.into());
		}

		let max_arity = self.folding_arities.iter().max().copied().unwrap();
		let max_buffer_size = 1 << max_arity;
		let mut scratch_buffer = vec![F::default(); max_buffer_size];

		let mut proof_iter = proof.into_iter();
		let round_proof = proof_iter
			.next()
			.expect("verified that proof is non-empty above");

		let coset_index = index >> self.folding_arities[0];
		let values = verify_coset_opening(
			self.committed_codeword_vcs,
			self.codeword_commitment,
			0,
			coset_index,
			self.folding_arities[0],
			round_proof,
		)?;

		let mut next_value = fold_chunk(
			self.committed_rs_code,
			0,
			coset_index,
			&values,
			&self.challenges[self.challenge_ranges[0].start..self.challenge_ranges[0].end],
			&mut scratch_buffer[..values.len()],
		);
		index = coset_index;

		for (query_round, vcs, commitment, round_proof) in izip!(
			1..=self.round_vcss.len(),
			self.round_vcss.iter(),
			self.round_commitments.iter(),
			proof_iter,
		) {
			let folding_arity = self.folding_arities[query_round];
			let challenge_range = &self.challenge_ranges[query_round];
			let folding_challenges = &self.challenges[challenge_range.start..challenge_range.end];
			let fold_start_round = self.fold_chunk_start_rounds[query_round];
			let coset_index = index >> folding_arity;

			let values = verify_coset_opening(
				vcs,
				commitment,
				fold_start_round,
				coset_index,
				folding_arity,
				round_proof,
			)?;

			if next_value != values[index % (1 << folding_arity)] {
				return Err(VerificationError::IncorrectFold { query_round, index }.into());
			}

			next_value = fold_chunk(
				self.committed_rs_code,
				fold_start_round,
				coset_index,
				&values,
				folding_challenges,
				&mut scratch_buffer[..values.len()],
			);
			index = coset_index;
		}

		// Since this implementation currently runs FRI until the final message is one element, we
		// know that the Reed-Solomon encoding of this message is simply that final message element
		// repeated up to the codeword length.
		if next_value != self.final_codeword[index] {
			return Err(VerificationError::IncorrectFold {
				query_round: self.n_rounds() - 1,
				index,
			}
			.into());
		}

		Ok(())
	}
}

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

	let start_index = coset_index << log_coset_size;

	let range = start_index..start_index + (1 << log_coset_size);

	vcs.verify_range_batch_opening(commitment, range, vcs_proof, iter::once(values.as_slice()))
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	debug_assert_eq!(values.len(), 1 << log_coset_size);
	Ok(values)
}

fn calculate_challenge_ranges(
	total_fold_rounds: usize,
	fold_chunk_start_rounds: &[usize],
) -> Vec<Range<usize>> {
	fold_chunk_start_rounds
		.iter()
		.chain(std::iter::once(&total_fold_rounds))
		.tuple_windows()
		.map(|(prev_start_round, next_start_round)| Range {
			start: *prev_start_round,
			end: *next_start_round,
		})
		.collect()
}
