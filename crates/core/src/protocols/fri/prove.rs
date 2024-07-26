// Copyright 2024 Ulvetanna Inc.

use super::error::Error;
use crate::{
	linear_code::{LinearCode, LinearCodeWithExtensionEncoding},
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_chunk, QueryProof, QueryRoundProof},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedFieldIndexable};
use rayon::prelude::*;
use std::iter;

fn fold_codeword<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	codeword: &[F],
	round: usize,
	folding_challenges: &[F],
) -> Vec<F>
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	// Preconditions
	assert!(codeword.len() % (1 << folding_challenges.len()) == 0);
	assert!(round + 1 >= folding_challenges.len());
	assert!(round < rs_code.log_dim());
	assert!(!folding_challenges.is_empty());

	let start_round = round + 1 - folding_challenges.len();
	let chunk_size = 1 << folding_challenges.len();

	// For each chunk of size 2^folding_challenges.len() in the codeword, fold it with the folding challenges
	codeword
		.par_chunks(chunk_size)
		.enumerate()
		.map_init(
			|| vec![F::default(); chunk_size],
			|scratch_buffer, (chunk_index, chunk)| {
				fold_chunk(
					rs_code,
					start_round,
					chunk_index,
					chunk,
					folding_challenges,
					scratch_buffer,
				)
			},
		)
		.collect()
}

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

pub enum FoldRoundOutput<VCSCommitment> {
	NoCommitment,
	Commitment(VCSCommitment),
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
	folding_arity: usize,
	curr_round: usize,
	unprocessed_challenges: Vec<F>,
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
		folding_arity: usize,
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
		let n_rounds = rs_code.log_dim();
		// TODO: Relax this condition when Early FRI Termination
		if n_rounds % folding_arity != 0 {
			return Err(Error::InvalidArgs(format!(
				"Reed–Solomon code dimension {} must be a multiple of the folding arity {}",
				n_rounds, folding_arity
			)));
		}
		let n_round_commitments = (n_rounds / folding_arity) - 1;
		if round_vcss.len() != n_round_commitments {
			return Err(Error::InvalidArgs(format!(
				"got {} round vector commitment schemes, expected {}",
				round_vcss.len(),
				n_round_commitments,
			)));
		}

		for (round, round_vcs) in round_vcss.iter().enumerate() {
			let expected_folded_dimension = rs_code.log_len() - (round + 1) * folding_arity;
			if round_vcs.vector_len() != 1 << expected_folded_dimension {
				return Err(Error::InvalidArgs(format!(
					"round {round} vector commitment length is incorrect, expected {}",
					1 << expected_folded_dimension
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
			folding_arity,
			curr_round: 0,
			unprocessed_challenges: Vec::with_capacity(folding_arity),
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.rs_code.log_dim()
	}

	/// Number of times `execute_fold_round` has been called
	pub fn curr_round(&self) -> usize {
		self.curr_round
	}

	fn prev_codeword(&self) -> &[F] {
		self.round_committed
			.last()
			.map(|(codeword, _)| codeword.as_slice())
			.unwrap_or(self.codeword)
	}

	fn is_commitment_round(&self) -> bool {
		(self.curr_round + 1) % self.folding_arity == 0
	}

	/// Executes the next fold round and returns the folded codeword commitment.
	pub fn execute_fold_round(
		&mut self,
		challenge: F,
	) -> Result<FoldRoundOutput<VCS::Commitment>, Error> {
		self.unprocessed_challenges.push(challenge);
		if !self.is_commitment_round() {
			self.curr_round += 1;
			return Ok(FoldRoundOutput::NoCommitment);
		}

		// Fold the last codeword with the accumulated folding challenges.
		let folded_codeword = fold_codeword(
			self.rs_code,
			self.prev_codeword(),
			self.curr_round,
			&self.unprocessed_challenges,
		);
		self.unprocessed_challenges.clear();

		let round_vcs = self
			.round_vcss
			.get(self.round_committed.len())
			.ok_or_else(|| Error::TooManyFoldExecutions {
				max_folds: self.round_vcss.len() - 1,
			})?;

		let (commitment, committed) = round_vcs
			.commit_batch(&[&folded_codeword])
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;
		self.round_committed.push((folded_codeword, committed));

		self.curr_round += 1;
		Ok(FoldRoundOutput::Commitment(commitment))
	}

	/// Finishes the FRI folding process, folding with the final challenge.
	///
	/// This returns the final message and a query prover instance.
	pub fn finish(mut self, challenge: F) -> Result<(F, FRIQueryProver<'a, F, VCS>), Error> {
		if self.curr_round != self.n_rounds() - 1 {
			return Err(Error::EarlyProverFinish);
		}
		self.unprocessed_challenges.push(challenge);

		// Technically, each of these folded codewords are codewords for RS code with small dimension but
		// the same rate as the original RS code. The final folded codeword would be a codeword for an RS
		// code of dimension 1, meaning this RS code is simply a repetition code.
		// At the end of FRI Folding, we only care about the final folded codeword's underlying
		// message (also the codeword's first element).
		//
		// Here we do a trick where we treat the first $2^{\theta}$ elements of the previous codeword
		// as a codeword for an RS code with dimension $2^{\theta}$ and rate $1$. We then fold this
		// codeword with the final $\theta$ challenges to get the final message.
		let prev_codeword = &self.prev_codeword()[..1 << self.folding_arity];
		let final_codeword = fold_codeword(
			self.rs_code,
			prev_codeword,
			self.curr_round,
			&self.unprocessed_challenges,
		);
		let final_value = final_codeword[0];
		self.unprocessed_challenges.clear();

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
			log_coset_size: self.folding_arity,
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
	log_coset_size: usize,
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
		let mut round_proofs = Vec::with_capacity(self.n_rounds());

		let mut coset_index = index >> self.log_coset_size;
		round_proofs.push(prove_coset_opening(
			self.codeword_vcs,
			self.codeword,
			self.codeword_committed,
			coset_index,
			self.log_coset_size,
		)?);

		for (vcs, (codeword, committed)) in
			iter::zip(self.round_vcss.iter(), self.round_committed.iter())
		{
			coset_index >>= self.log_coset_size;
			round_proofs.push(prove_coset_opening(
				vcs,
				codeword,
				committed,
				coset_index,
				self.log_coset_size,
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
