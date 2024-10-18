// Copyright 2024 Ulvetanna Inc.

use super::{common::TerminateCodeword, error::Error, QueryProof, VerificationError};
use crate::{
	merkle_tree::VectorCommitScheme,
	polynomial::MultilinearQuery,
	protocols::fri::common::{
		calculate_fold_arities, fold_chunk, fold_codeword, fold_interleaved_chunk,
		validate_common_fri_arguments, QueryRoundProof,
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
	/// The reduction arities between each query round.
	fold_arities: Vec<usize>,
	/// The constant value of the dimension-1 codeword
	final_message: F,
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

		let verify_fold_rounds = log_batch_size + committed_rs_code.log_dim();
		if challenges.len() != verify_fold_rounds {
			bail!(Error::InvalidArgs(format!(
				"got {} folding challenges, expected {}",
				challenges.len(),
				verify_fold_rounds,
			)));
		}

		validate_common_fri_arguments(committed_rs_code, committed_codeword_vcs, round_vcss)?;

		let (interleave_challenges, fold_challenges) = challenges.split_at(log_batch_size);

		let n_dim_one_fold_challenges = log2_strict_usize(
			round_vcss
				.last()
				.unwrap_or(committed_codeword_vcs)
				.vector_len(),
		) - committed_rs_code.log_inv_rate();

		let (fold_challenges, dim_one_fold_challenges) =
			fold_challenges.split_at(fold_challenges.len() - n_dim_one_fold_challenges);

		let dim_one_codeword = fold_codeword(
			committed_rs_code,
			&terminate_codeword,
			committed_rs_code.log_dim(),
			dim_one_fold_challenges,
		);

		if dim_one_codeword.iter().any(|e| *e != dim_one_codeword[0]) {
			bail!(VerificationError::IncorrectDegree);
		}

		let backend = make_portable_backend();
		let interleave_tensor =
			MultilinearQuery::<F, _>::with_full_query(interleave_challenges, &backend)
				.expect("number of challenges is less than 32")
				.into_expansion();

		let fold_arities = calculate_fold_arities(
			committed_rs_code.log_len(),
			committed_rs_code.log_inv_rate(),
			round_vcss
				.iter()
				.map(|vcs| log2_strict_usize(vcs.vector_len())),
			log_batch_size,
		)?;

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
			final_message: dim_one_codeword[0],
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.round_vcss.len()
	}

	pub fn final_message(&self) -> F {
		self.final_message
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

		let mut proof_iter = proof.into_iter();
		let mut arities = self.fold_arities.iter().copied();

		// Create scratch buffer used in `fold_chunk`.
		let max_arity = arities
			.clone()
			.max()
			.expect("iter_fold_arities returns non-empty iterator");
		let max_buffer_size = 2 * (1 << max_arity);
		let mut scratch_buffer = vec![F::default(); max_buffer_size];

		// The number of fold rounds until the first folded codeword is committed or sent.
		let arity = arities
			.next()
			.expect("iter_fold_arities returns non-empty iterator");

		let round_proof = proof_iter
			.next()
			.expect("verified that proof is non-empty above");

		// This is the round of the folding phase that the codeword to be folded is committed to.
		let mut fold_round = 0;

		let coset_index = index >> arity;
		let values = verify_interleaved_opening(
			self.committed_codeword_vcs,
			self.codeword_commitment,
			coset_index,
			arity - self.log_batch_size,
			self.log_batch_size,
			round_proof,
		)?;

		let mut next_value = fold_interleaved_chunk(
			self.committed_rs_code,
			self.log_batch_size,
			coset_index,
			&values,
			&self.interleave_tensor,
			&self.fold_challenges[fold_round..fold_round + arity - self.log_batch_size],
			&mut scratch_buffer,
		);
		index = coset_index;
		fold_round += arity - self.log_batch_size;

		for (i, (vcs, commitment, round_proof, arity)) in
			izip!(self.round_vcss.iter(), self.round_commitments.iter(), proof_iter, arities)
				.enumerate()
		{
			let query_round = i + 1;
			let coset_index = index >> arity;

			let values = verify_coset_opening(
				vcs,
				commitment,
				query_round,
				coset_index,
				arity,
				round_proof,
			)?;

			if next_value != values[index % (1 << arity)] {
				return Err(VerificationError::IncorrectFold { query_round, index }.into());
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

		// Since this implementation currently runs FRI until the final message is one element, we
		// know that the Reed-Solomon encoding of this message is simply that final message element
		// repeated up to the codeword length.
		if next_value != self.terminate_codeword[index] {
			return Err(VerificationError::IncorrectFold {
				query_round: self.n_rounds() - 1,
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
