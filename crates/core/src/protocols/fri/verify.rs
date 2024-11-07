// Copyright 2024 Irreducible Inc.

use super::{common::FRIProof, error::Error, QueryProof, VerificationError};
use crate::{
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_chunk, fold_interleaved_chunk, FRIParams, QueryRoundProof},
};
use binius_field::{BinaryField, ExtensionField};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_utils::bail;
use itertools::izip;
use p3_challenger::CanSampleBits;
use std::iter;
use tracing::instrument;

/// A verifier for the FRI query phase.
///
/// The verifier is instantiated after the folding rounds and is used to test consistency of the
/// round messages and the original purported codeword.
#[derive(Debug)]
pub struct FRIVerifier<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	params: &'a FRIParams<F, FA, VCS>,
	/// Received commitment to the codeword.
	codeword_commitment: &'a VCS::Commitment,
	/// Received commitments to the round messages.
	round_commitments: &'a [VCS::Commitment],
	/// The challenges for each round.
	interleave_tensor: Vec<F>,
	/// The challenges for each round.
	fold_challenges: &'a [F],
}

impl<'a, F, FA, VCS> FRIVerifier<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		params: &'a FRIParams<F, FA, VCS>,
		codeword_commitment: &'a VCS::Commitment,
		round_commitments: &'a [VCS::Commitment],
		challenges: &'a [F],
	) -> Result<Self, Error> {
		if round_commitments.len() != params.n_oracles() {
			bail!(Error::InvalidArgs(format!(
				"got {} round commitments, expected {}",
				round_commitments.len(),
				params.n_oracles(),
			)));
		}

		if challenges.len() != params.n_fold_rounds() {
			bail!(Error::InvalidArgs(format!(
				"got {} folding challenges, expected {}",
				challenges.len(),
				params.n_fold_rounds(),
			)));
		}

		let (interleave_challenges, fold_challenges) = challenges.split_at(params.log_batch_size());

		let backend = make_portable_backend();
		let interleave_tensor = backend
			.tensor_product_full_query(interleave_challenges)
			.expect("number of challenges is less than 32");

		Ok(Self {
			params,
			codeword_commitment,
			round_commitments,
			interleave_tensor,
			fold_challenges,
		})
	}

	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.params.n_oracles()
	}

	pub fn verify<Challenger>(
		&self,
		fri_proof: FRIProof<F, VCS::Proof>,
		mut challenger: Challenger,
	) -> Result<F, Error>
	where
		Challenger: CanSampleBits<usize>,
	{
		let FRIProof {
			terminate_codeword,
			proofs,
		} = fri_proof;

		let final_value = self.verify_last_oracle(&terminate_codeword)?;

		let indexes_iter =
			std::iter::repeat_with(|| challenger.sample_bits(self.params.index_bits()))
				.take(self.params.n_test_queries());

		for (index, proof) in indexes_iter.zip(proofs) {
			self.verify_query(index, proof, &terminate_codeword)?
		}
		Ok(final_value)
	}

	/// Verifies that the last oracle sent is a codeword.
	///
	/// Returns the fully-folded message value.
	pub fn verify_last_oracle(&self, terminate_codeword: &[F]) -> Result<F, Error> {
		let repetition_codeword = if let Some(last_vcs) = self.params.round_vcss().last() {
			let commitment = self.round_commitments.last().expect(
				"round_commitments and round_vcss are checked to have the same length in the constructor;
				when round_vcss is non-empty, round_commitments must be as well",
			);

			last_vcs
				.verify_interleaved(commitment, terminate_codeword)
				.map_err(|err| Error::VectorCommit(Box::new(err)))?;

			let n_final_challenges = self.params.n_final_challenges();
			let n_prior_challenges = self.fold_challenges.len() - n_final_challenges;
			let final_challenges = &self.fold_challenges[n_prior_challenges..];
			let mut scratch_buffer = vec![F::default(); 1 << n_final_challenges];

			terminate_codeword
				.chunks(1 << n_final_challenges)
				.enumerate()
				.map(|(i, coset_values)| {
					fold_chunk(
						self.params.rs_code(),
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

			self.params
				.codeword_vcs()
				.verify_interleaved(self.codeword_commitment, terminate_codeword)
				.map_err(|err| Error::VectorCommit(Box::new(err)))?;

			let fold_arity = self.params.rs_code().log_dim() + self.params.log_batch_size();
			let mut scratch_buffer = vec![F::default(); 2 * (1 << fold_arity)];
			terminate_codeword
				.chunks(1 << fold_arity)
				.enumerate()
				.map(|(i, chunk)| {
					fold_interleaved_chunk(
						self.params.rs_code(),
						self.params.log_batch_size(),
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
		terminate_codeword: &[F],
	) -> Result<(), Error> {
		if proof.len() != self.n_oracles() {
			return Err(VerificationError::IncorrectQueryProofLength {
				expected: self.n_oracles(),
			}
			.into());
		}

		let arities = self.params.fold_arities().iter().copied();
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
		let log_coset_size = first_fold_arity - self.params.log_batch_size();
		let values = verify_coset_opening(
			self.params.codeword_vcs(),
			self.codeword_commitment,
			0,
			index,
			first_fold_arity,
			first_query_proof,
		)?;

		let mut next_value = fold_interleaved_chunk(
			self.params.rs_code(),
			self.params.log_batch_size(),
			index,
			&values,
			&self.interleave_tensor,
			&self.fold_challenges[fold_round..fold_round + log_coset_size],
			&mut scratch_buffer,
		);
		fold_round += log_coset_size;

		for (i, (vcs, commitment, (round_proof, arity))) in izip!(
			self.params.round_vcss().iter(),
			self.round_commitments.iter(),
			proof_and_arities_iter
		)
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
				self.params.rs_code(),
				fold_round,
				coset_index,
				&values,
				&self.fold_challenges[fold_round..fold_round + arity],
				&mut scratch_buffer,
			);
			index = coset_index;
			fold_round += arity;
		}

		if next_value != terminate_codeword[index] {
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

	vcs.verify_batch_opening(commitment, coset_index, vcs_proof, values.iter().copied())
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(values)
}
