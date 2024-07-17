// Copyright 2024 Ulvetanna Inc.

use super::error::Error;
use crate::{
	linear_code::{LinearCode, LinearCodeWithExtensionEncoding},
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_pair, QueryProof, QueryRoundProof},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedFieldIndexable};
use rayon::prelude::*;
use std::iter;

#[derive(Debug)]
pub struct CommitOutput<P, VCSCommitment, VCSCommitted> {
	pub commitment: VCSCommitment,
	pub committed: VCSCommitted,
	pub codeword: Vec<P>,
}

/// Encodes and commits the input message.
pub fn commit_message<F, FA, P, PA, VCS>(
	rs_code: &ReedSolomonCode<PA>,
	vcs: &VCS,
	message: &[P],
) -> Result<CommitOutput<P, VCS::Commitment, VCS::Committed>, Error>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	P: PackedFieldIndexable<Scalar = F> + PackedExtension<FA, PackedSubfield = PA>,
	PA: PackedFieldIndexable<Scalar = FA>,
	VCS: VectorCommitScheme<F>,
{
	if message.len() * P::WIDTH != rs_code.dim() {
		return Err(Error::InvalidArgs("message length does not match code dimension".to_string()));
	}
	if vcs.vector_len() != rs_code.len() {
		return Err(Error::InvalidArgs(
			"code length does not vector commitment length".to_string(),
		));
	}

	let mut encoded = vec![P::zero(); message.len() << rs_code.log_inv_rate()];
	encoded[..message.len()].copy_from_slice(message);
	rs_code.encode_extension_inplace(&mut encoded)?;

	let (commitment, vcs_committed) = vcs
		.commit_batch(&[P::unpack_scalars(&encoded)])
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(CommitOutput {
		commitment,
		committed: vcs_committed,
		codeword: encoded,
	})
}

/// A stateful prover for the FRI fold phase.
pub struct FRIFolder<'a, F, FA, VCS>
where
	FA: BinaryField,
	F: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	rs_code: &'a ReedSolomonCode<FA>,
	codeword: &'a [F],
	codeword_vcs: &'a VCS,
	round_vcss: &'a [VCS],
	codeword_committed: &'a VCS::Committed,
	round_committed: Vec<(Vec<F>, VCS::Committed)>,
}

impl<'a, F, FA, VCS> FRIFolder<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F> + Sync,
	VCS::Committed: Send + Sync,
{
	/// Constructs a new folder.
	pub fn new(
		rs_code: &'a ReedSolomonCode<FA>,
		codeword: &'a [F],
		codeword_vcs: &'a VCS,
		round_vcss: &'a [VCS],
		committed: &'a VCS::Committed,
	) -> Result<Self, Error> {
		if rs_code.len() != codeword_vcs.vector_len() {
			return Err(Error::InvalidArgs(
				"Reed–Solomon code length must match codeword vector commitment length".to_string(),
			));
		}
		if rs_code.len() != codeword.len() {
			return Err(Error::InvalidArgs(
				"Reed–Solomon code length must match codeword length".to_string(),
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
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed: committed,
			round_committed: Vec::with_capacity(round_vcss.len()),
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.rs_code.log_dim()
	}

	/// Number of times `execute_fold_round` has been called.
	pub fn round(&self) -> usize {
		self.round_committed.len()
	}

	/// Executes the next fold round and returns the folded codeword commitment.
	pub fn execute_fold_round(&mut self, challenge: F) -> Result<VCS::Commitment, Error> {
		let round_vcs =
			self.round_vcss
				.get(self.round())
				.ok_or_else(|| Error::TooManyRoundExecutions {
					max_rounds: self.n_rounds() - 1,
				})?;

		let last_codeword = self
			.round_committed
			.last()
			.map(|(codeword, _)| codeword.as_slice())
			.unwrap_or(self.codeword);

		let folded = last_codeword
			.par_chunks(2)
			.enumerate()
			.map(|(coset_index, coset_values)| {
				fold_pair(
					self.rs_code,
					self.round(),
					coset_index,
					(coset_values[0], coset_values[1]),
					challenge,
				)
			})
			.collect::<Vec<_>>();

		let (commitment, committed) = round_vcs
			.commit_batch(&[&folded])
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;

		self.round_committed.push((folded, committed));

		Ok(commitment)
	}

	/// Finishes the FRI folding process, folding with the final challenge.
	///
	/// This returns the final message and a query prover instance.
	pub fn finish(self, challenge: F) -> Result<(F, FRIQueryProver<'a, F, VCS>), Error> {
		if self.round() != self.n_rounds() - 1 {
			return Err(Error::EarlyProverFinish);
		}

		let last_codeword = self
			.round_committed
			.last()
			.map(|(codeword, _)| codeword.as_slice())
			.unwrap_or(self.codeword);

		let final_value = fold_pair(
			self.rs_code,
			self.round(),
			0,
			(last_codeword[0], last_codeword[1]),
			challenge,
		);

		let Self {
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed,
			round_committed,
			..
		} = self;

		let query_prover = FRIQueryProver {
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed,
			round_committed,
		};
		Ok((final_value, query_prover))
	}
}

/// A prover for the FRI query phase.
pub struct FRIQueryProver<'a, F: BinaryField, VCS: VectorCommitScheme<F>> {
	codeword: &'a [F],
	codeword_vcs: &'a VCS,
	round_vcss: &'a [VCS],
	codeword_committed: &'a VCS::Committed,
	round_committed: Vec<(Vec<F>, VCS::Committed)>,
}

impl<'a, F: BinaryField, VCS: VectorCommitScheme<F>> FRIQueryProver<'a, F, VCS> {
	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.round_vcss.len() + 1
	}

	/// Proves a FRI challenge query.
	///
	/// ## Arguments
	///
	/// * `index` - an index into the original codeword domain
	pub fn prove_query(&self, index: usize) -> Result<QueryProof<F, VCS::Proof>, Error> {
		const LOG_COSET_SIZE: usize = 1;

		let mut round_proofs = Vec::with_capacity(self.n_rounds());

		let mut coset_index = index >> LOG_COSET_SIZE;
		round_proofs.push(prove_coset_opening(
			self.codeword_vcs,
			self.codeword,
			self.codeword_committed,
			coset_index,
			LOG_COSET_SIZE,
		)?);

		for (vcs, (codeword, committed)) in
			iter::zip(self.round_vcss.iter(), self.round_committed.iter())
		{
			coset_index >>= LOG_COSET_SIZE;
			round_proofs.push(prove_coset_opening(
				vcs,
				codeword,
				committed,
				coset_index,
				LOG_COSET_SIZE,
			)?);
		}

		Ok(round_proofs)
	}
}

fn prove_coset_opening<F: BinaryField, VCS: VectorCommitScheme<F>>(
	vcs: &VCS,
	codeword: &[F],
	committed: &VCS::Committed,
	coset_index: usize,
	log_coset_size: usize,
) -> Result<QueryRoundProof<F, VCS::Proof>, Error> {
	let start_index = coset_index << log_coset_size;

	let range = start_index..start_index + (1 << log_coset_size);

	let vcs_proof = vcs
		.prove_range_batch_opening(committed, range.clone())
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(QueryRoundProof {
		values: codeword[range].to_vec(),
		vcs_proof,
	})
}
