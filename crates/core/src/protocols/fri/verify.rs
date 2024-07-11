// Copyright 2024 Ulvetanna Inc.

use super::{error::Error, QueryProof, VerificationError};
use crate::{
	linear_code::LinearCode,
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_pair, QueryRoundProof},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField};
use itertools::izip;
use std::iter;

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
	rs_code: &'a ReedSolomonCode<FA>,
	/// Vector commitment scheme for the codeword oracle.
	codeword_vcs: &'a VCS,
	/// Vector commitment scheme for the round oracles.
	round_vcss: &'a [VCS],
	/// Received commitment to the codeword.
	codeword_commitment: &'a VCS::Commitment,
	/// Received commitments to the round messages.
	round_commitments: &'a [VCS::Commitment],
	/// The challenges for each round.
	challenges: &'a [F],
	/// The final message, which is a 1-element vector. Its encoding must be a vector of the final
	/// value repeated up to the codeword length.
	final_value: F,
}

impl<'a, F, FA, VCS> FRIVerifier<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	pub fn new(
		rs_code: &'a ReedSolomonCode<FA>,
		codeword_vcs: &'a VCS,
		round_vcss: &'a [VCS],
		codeword_commitment: &'a VCS::Commitment,
		round_commitments: &'a [VCS::Commitment],
		challenges: &'a [F],
		final_value: F,
	) -> Result<Self, Error> {
		if rs_code.len() != codeword_vcs.vector_len() {
			return Err(Error::InvalidArgs(
				"Reed–Solomon code length must match codeword vector commitment length".to_string(),
			));
		}

		// This is a tricky case to handle well.
		// TODO: Change interface to support dimension-1 messages.
		if rs_code.log_dim() == 0 {
			return Err(Error::MessageDimensionIsOne);
		}
		if round_vcss.len() != rs_code.log_dim() - 1 {
			return Err(Error::InvalidArgs(format!(
				"got {} round vector commitment schemes, expected {}",
				round_vcss.len(),
				rs_code.log_dim() - 1
			)));
		}
		if round_commitments.len() != rs_code.log_dim() - 1 {
			return Err(Error::InvalidArgs(format!(
				"got {} round commitments, expected {}",
				round_vcss.len(),
				rs_code.log_dim() - 1
			)));
		}
		if challenges.len() != rs_code.log_dim() {
			return Err(Error::InvalidArgs(format!(
				"got {} folding challenges, expected {}",
				challenges.len(),
				rs_code.log_dim()
			)));
		}

		for (round, round_vcs) in round_vcss.iter().enumerate() {
			if round_vcs.vector_len() != 1 << (rs_code.log_len() - round - 1) {
				return Err(Error::InvalidArgs(format!(
					"round {round} vector commitment length is incorrect, expected {}",
					1 << (rs_code.log_len() - round - 1)
				)));
			}
		}

		Ok(Self {
			rs_code,
			codeword_vcs,
			round_vcss,
			codeword_commitment,
			round_commitments,
			challenges,
			final_value,
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
	pub fn verify_query(
		&self,
		mut index: usize,
		proof: QueryProof<F, VCS::Proof>,
	) -> Result<(), Error> {
		const LOG_COSET_SIZE: usize = 1;

		if proof.len() != self.n_rounds() {
			return Err(VerificationError::IncorrectQueryProofLength {
				expected: self.n_rounds(),
			}
			.into());
		}

		let mut proof_iter = proof.into_iter();

		let round_proof = proof_iter
			.next()
			.expect("verified that proof is non-empty above");

		let coset_index = index >> LOG_COSET_SIZE;
		let values = verify_coset_opening(
			self.codeword_vcs,
			self.codeword_commitment,
			0,
			coset_index,
			LOG_COSET_SIZE,
			round_proof,
		)?;

		let mut next_value =
			fold_pair(self.rs_code, 0, coset_index, (values[0], values[1]), self.challenges[0]);
		index = coset_index;

		for (round, (vcs, commitment, r_i, round_proof)) in izip!(
			self.round_vcss.iter(),
			self.round_commitments.iter(),
			self.challenges[1..].iter().copied(),
			proof_iter
		)
		.enumerate()
		{
			let coset_index = index >> LOG_COSET_SIZE;
			let values = verify_coset_opening(
				vcs,
				commitment,
				round + 1,
				coset_index,
				LOG_COSET_SIZE,
				round_proof,
			)?;

			if next_value != values[index % (1 << LOG_COSET_SIZE)] {
				return Err(VerificationError::IncorrectFold { round, index }.into());
			}

			next_value =
				fold_pair(self.rs_code, round + 1, coset_index, (values[0], values[1]), r_i);
			index = coset_index;
		}

		// Since this implementation currently runs FRI until the final message is one element, we
		// know that the Reed-Solomon encoding of this message is simply that final message element
		// repeated up to the codeword length.
		if next_value != self.final_value {
			return Err(VerificationError::IncorrectFold {
				round: self.n_rounds() - 1,
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

	Ok(values)
}
