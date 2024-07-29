// Copyright 2024 Ulvetanna Inc.

use super::{
	common::{
		calculate_fold_chunk_start_rounds, calculate_fold_commit_rounds, calculate_folding_arities,
		FinalMessage,
	},
	error::Error,
};
use crate::{
	linear_code::{LinearCode, LinearCodeWithExtensionEncoding},
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_chunk, QueryProof, QueryRoundProof},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedFieldIndexable};
use itertools::izip;
use rayon::prelude::*;
use tracing::instrument;

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
	curr_round: usize,
	unprocessed_challenges: Vec<F>,
	commitment_fold_rounds: Vec<usize>,
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

		let commitment_fold_rounds = calculate_fold_commit_rounds(rs_code, round_vcss)?;

		Ok(Self {
			rs_code,
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed: committed,
			round_committed: Vec::with_capacity(round_vcss.len()),
			curr_round: 0,
			unprocessed_challenges: Vec::with_capacity(rs_code.log_dim()),
			commitment_fold_rounds,
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
		let n_commitments = self.round_committed.len();
		n_commitments < self.round_vcss.len()
			&& self.commitment_fold_rounds[n_commitments] == self.curr_round
	}

	/// Executes the next fold round and returns the folded codeword commitment.
	///
	/// As a memory efficient optimization, this method may not actually do the folding, but instead accumulate the
	/// folding challenge for processing at a later time. This saves us from storing intermediate folded codewords.
	#[instrument(skip_all, name = "fri::FRIFolder::execute_fold_round")]
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

	/// Finalizes the FRI folding process.
	///
	/// This step will process any unprocessed folding challenges to produce the
	/// final folded codeword. Then it will decode this final folded codeword
	/// to get the final message. The result is the final message and a query prover instance.
	///
	/// This returns the final message and a query prover instance.
	#[instrument(skip_all, name = "fri::FRIFolder::finalize")]
	pub fn finalize(mut self) -> Result<(FinalMessage<F>, FRIQueryProver<'a, F, VCS>), Error> {
		if self.curr_round != self.n_rounds() {
			return Err(Error::EarlyProverFinish);
		}

		// TODO: This code will need to be generalized in early FRI termination and handle appropriate decoding.
		const FINAL_MESSAGE_DIM: usize = 0;

		// NB: The idea behind the following is that we should do the minimal amount of work necessary to
		// get the final message. Specifically, we do not need a final codeword with rate lower than 1.
		let final_message = if self.unprocessed_challenges.is_empty() {
			// In this case, final_codeword is interpreted as taking a prefix of the previous codeword and claiming
			// this is a codeword for an RS code with rate 1 and dimension FINAL_MESSAGE_DIM.
			let final_codeword = &self.prev_codeword()[..1 << FINAL_MESSAGE_DIM];
			// We decode the final codeword to get the final message.
			final_codeword[0]
		} else {
			// In this case, unfolded_codeword is interpreted as taking a prefix of the previous codeword and claiming
			// this is a codeword for an RS code with dimension FINAL_MESSAGE_DIM + unprocessed_challenges.len()
			// and rate 1.
			let unfolded_codeword_len =
				1 << (self.unprocessed_challenges.len() + FINAL_MESSAGE_DIM);
			let unfolded_codeword = &self.prev_codeword()[..unfolded_codeword_len];
			// We then fold this codeword with the unprocessed challenges to get a final codeword
			// for an RS code with rate 1 and dimension FINAL_MESSAGE_DIM.
			let final_codeword = fold_codeword(
				self.rs_code,
				unfolded_codeword,
				self.curr_round - 1,
				&self.unprocessed_challenges,
			);
			// We decode this final codeword to get the final message.
			final_codeword[0]
		};
		self.unprocessed_challenges.clear();

		let Self {
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed,
			round_committed,
			commitment_fold_rounds,
			rs_code,
			..
		} = self;

		let query_prover = FRIQueryProver {
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed,
			round_committed,
			commitment_fold_rounds,
			n_fold_rounds: rs_code.log_dim(),
		};
		Ok((final_message, query_prover))
	}
}

/// A prover for the FRI query phase.
pub struct FRIQueryProver<'a, F: BinaryField, VCS: VectorCommitScheme<F>> {
	codeword: &'a [F],
	codeword_vcs: &'a VCS,
	round_vcss: &'a [VCS],
	codeword_committed: &'a VCS::Committed,
	round_committed: Vec<(Vec<F>, VCS::Committed)>,
	commitment_fold_rounds: Vec<usize>,
	n_fold_rounds: usize,
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
	#[instrument(skip_all, name = "fri::FRIQueryProver::prove_query")]
	pub fn prove_query(&self, index: usize) -> Result<QueryProof<F, VCS::Proof>, Error> {
		let mut round_proofs = Vec::with_capacity(self.n_rounds());
		let fold_chunk_start_rounds =
			calculate_fold_chunk_start_rounds(&self.commitment_fold_rounds);
		let folding_arities =
			calculate_folding_arities(self.n_fold_rounds, &fold_chunk_start_rounds);

		let mut coset_index = index >> folding_arities[0];
		round_proofs.push(prove_coset_opening(
			self.codeword_vcs,
			self.codeword,
			self.codeword_committed,
			coset_index,
			folding_arities[0],
		)?);

		for (query_rd, vcs, (codeword, committed)) in
			izip!((1..=self.round_vcss.len()), self.round_vcss.iter(), self.round_committed.iter())
		{
			coset_index >>= folding_arities[query_rd];
			round_proofs.push(prove_coset_opening(
				vcs,
				codeword,
				committed,
				coset_index,
				folding_arities[query_rd],
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
