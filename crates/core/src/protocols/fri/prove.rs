// Copyright 2024 Ulvetanna Inc.

use super::{common::FinalMessage, error::Error};
use crate::{
	linear_code::LinearCode,
	merkle_tree::VectorCommitScheme,
	polynomial::MultilinearQuery,
	protocols::fri::common::{
		calculate_fold_arities, fold_chunk, fold_interleaved_chunk, validate_common_fri_arguments,
		QueryProof, QueryRoundProof,
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedFieldIndexable};
use binius_hal::make_portable_backend;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use itertools::izip;
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use tracing::instrument;

fn fold_codeword<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	codeword: &[F],
	// Round is the number of total folding challenges received so far.
	round: usize,
	folding_challenges: &[F],
) -> Vec<F>
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	// Preconditions
	assert_eq!(codeword.len() % (1 << folding_challenges.len()), 0);
	assert!(round >= folding_challenges.len());
	assert!(round <= rs_code.log_dim());

	if folding_challenges.is_empty() {
		return codeword.to_vec();
	}

	let start_round = round - folding_challenges.len();
	let chunk_size = 1 << folding_challenges.len();

	// For each chunk of size `2^chunk_size` in the codeword, fold it with the folding challenges
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

/// Fold the interleaved codeword into a single codeword with the same block length.
///
/// ## Arguments
///
/// * `rs_code` - the Reed–Solomon code the protocol tests proximity to.
/// * `codeword` - an interleaved codeword.
/// * `challenges` - the folding challenges. The length must be at least `log_batch_size`.
/// * `log_batch_size` - the base-2 logarithm of the batch size of the interleaved code.
fn fold_interleaved<F, FS>(
	rs_code: &ReedSolomonCode<FS>,
	codeword: &[F],
	challenges: &[F],
	log_batch_size: usize,
) -> Vec<F>
where
	F: BinaryField + ExtensionField<FS>,
	FS: BinaryField,
{
	assert_eq!(codeword.len(), 1 << (rs_code.log_len() + log_batch_size));
	assert!(challenges.len() >= log_batch_size);

	let backend = make_portable_backend();

	let (interleave_challenges, fold_challenges) = challenges.split_at(log_batch_size);
	let tensor = MultilinearQuery::<F, _>::with_full_query(interleave_challenges, &backend)
		.expect("number of challenges is less than 32")
		.into_expansion();

	// For each chunk of size `2^chunk_size` in the codeword, fold it with the folding challenges
	let fold_chunk_size = 1 << fold_challenges.len();
	let interleave_chunk_size = 1 << log_batch_size;
	let chunk_size = fold_chunk_size * interleave_chunk_size;
	codeword
		.par_chunks(chunk_size)
		.enumerate()
		.map_init(
			|| vec![F::default(); 2 * fold_chunk_size],
			|scratch_buffer, (i, chunk)| {
				fold_interleaved_chunk(
					rs_code,
					log_batch_size,
					i,
					chunk,
					&tensor,
					fold_challenges,
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
///
/// ## Arguments
///
/// * `rs_code` - the Reed-Solomon code to use for encoding
/// * `log_batch_size` - the base-2 logarithm of the batch size of the interleaved code.
/// * `vcs` - the vector commitment scheme to use for committing
/// * `message` - the interleaved message to encode and commit
pub fn commit_interleaved<F, FA, P, PA, VCS>(
	rs_code: &ReedSolomonCode<PA>,
	log_batch_size: usize,
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
	let n_elems = message.len() * P::WIDTH;
	if n_elems != rs_code.dim() << log_batch_size {
		bail!(Error::InvalidArgs(
			"interleaved message length does not match code parameters".to_string()
		));
	}
	if vcs.vector_len() != rs_code.len() {
		bail!(Error::InvalidArgs("code length does not vector commitment length".to_string(),));
	}

	let mut encoded = zeroed_vec(message.len() << rs_code.log_inv_rate());
	encoded[..message.len()].copy_from_slice(message);
	rs_code.encode_ext_batch_inplace(&mut encoded, log_batch_size)?;

	let (commitment, vcs_committed) = vcs
		.commit_interleaved(P::unpack_scalars(&encoded))
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
	committed_rs_code: &'a ReedSolomonCode<FA>,
	log_batch_size: usize,
	final_rs_code: &'a ReedSolomonCode<F>,
	codeword: &'a [F],
	codeword_vcs: &'a VCS,
	round_vcss: &'a [VCS],
	codeword_committed: &'a VCS::Committed,
	round_committed: Vec<(Vec<F>, VCS::Committed)>,
	fold_arities: Vec<usize>,
	curr_round: usize,
	next_commit_round: Option<usize>,
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
		committed_rs_code: &'a ReedSolomonCode<FA>,
		final_rs_code: &'a ReedSolomonCode<F>,
		log_batch_size: usize,
		committed_codeword: &'a [F],
		committed_codeword_vcs: &'a VCS,
		round_vcss: &'a [VCS],
		committed: &'a VCS::Committed,
	) -> Result<Self, Error> {
		if committed_codeword.len() != committed_rs_code.len() << log_batch_size {
			bail!(Error::InvalidArgs(
				"Reed–Solomon code length must match interleaved codeword length".to_string(),
			));
		}

		validate_common_fri_arguments(
			committed_rs_code,
			final_rs_code,
			committed_codeword_vcs,
			round_vcss,
		)?;

		let fold_arities = calculate_fold_arities(
			committed_rs_code.log_len(),
			final_rs_code.log_len(),
			round_vcss
				.iter()
				.map(|vcs| log2_strict_usize(vcs.vector_len())),
			log_batch_size,
		)?;

		let next_commit_round = fold_arities.first().copied();
		Ok(Self {
			committed_rs_code,
			log_batch_size,
			final_rs_code,
			codeword: committed_codeword,
			codeword_vcs: committed_codeword_vcs,
			round_vcss,
			codeword_committed: committed,
			round_committed: Vec::with_capacity(round_vcss.len()),
			fold_arities,
			curr_round: 0,
			next_commit_round,
			unprocessed_challenges: Vec::with_capacity(committed_rs_code.log_dim()),
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.committed_rs_code.log_dim() + self.log_batch_size - self.final_rs_code.log_dim()
	}

	/// Number of times `execute_fold_round` has been called.
	pub fn curr_round(&self) -> usize {
		self.curr_round
	}

	fn is_commitment_round(&self) -> bool {
		self.next_commit_round
			.is_some_and(|round| round == self.curr_round)
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
		self.curr_round += 1;

		if !self.is_commitment_round() {
			return Ok(FoldRoundOutput::NoCommitment);
		}

		// Fold the last codeword with the accumulated folding challenges.
		let folded_codeword = match self.round_committed.last() {
			Some((prev_codeword, _)) => {
				// Fold a full codeword committed in the previous FRI round into a codeword with
				// reduced dimension and rate.
				fold_codeword(
					self.committed_rs_code,
					prev_codeword,
					self.curr_round - self.log_batch_size,
					&self.unprocessed_challenges,
				)
			}
			None => {
				// Fold the interleaved codeword that was originally committed into a single
				// codeword with the same or reduced block length, depending on the sequence of
				// fold rounds.
				fold_interleaved(
					self.committed_rs_code,
					self.codeword,
					&self.unprocessed_challenges,
					self.log_batch_size,
				)
			}
		};
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

		self.next_commit_round = self.next_commit_round.take().and_then(|next_commit_round| {
			let n_commitments = self.round_committed.len();
			if n_commitments < self.fold_arities.len() - 1 {
				Some(next_commit_round + self.fold_arities[n_commitments])
			} else {
				None
			}
		});
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
			bail!(Error::EarlyProverFinish);
		}

		if self.final_rs_code.log_dim() != 0 {
			todo!("handle the case when the FRI protocol terminates before folding to a length-1 message");
		}

		// NB: The idea behind the following is that we should do the minimal amount of work
		// necessary to get the final message. We fold the final codeword, but first truncate it
		// so that the rate is 1. Since the prover is honest, we can decode to the final message
		// with an iNTT.
		let log_inv_rate = self.committed_rs_code.log_inv_rate();
		let final_codeword = match self.round_committed.last() {
			Some((prev_codeword, _)) => {
				// Fold a full codeword committed in the previous FRI round into a codeword with
				// reduced dimension and rate.
				let truncated_len = prev_codeword.len() >> log_inv_rate;
				fold_codeword(
					self.committed_rs_code,
					&prev_codeword[..truncated_len],
					self.curr_round - self.log_batch_size,
					&self.unprocessed_challenges,
				)
			}
			None => {
				// Fold the interleaved codeword that was originally committed into a single
				// codeword with the same or reduced block length, depending on the sequence of
				// fold rounds.
				let truncated_len = self.codeword.len() >> log_inv_rate;
				fold_interleaved(
					self.committed_rs_code,
					&self.codeword[..truncated_len],
					&self.unprocessed_challenges,
					self.log_batch_size,
				)
			}
		};

		// Because the final codeword has dimension 1 and rate 1, the decoding procedure is the
		// trival identity map.
		let final_message = final_codeword;

		self.unprocessed_challenges.clear();

		let Self {
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed,
			round_committed,
			log_batch_size,
			fold_arities,
			..
		} = self;

		let query_prover = FRIQueryProver {
			codeword,
			codeword_vcs,
			round_vcss,
			codeword_committed,
			round_committed,
			log_batch_size,
			fold_arities,
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
	log_batch_size: usize,
	fold_arities: Vec<usize>,
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
		let mut arities = self.fold_arities.iter().copied();

		let arity = arities
			.next()
			.expect("iter_fold_arities returns non-empty iterator");

		let mut coset_index = index >> arity;
		round_proofs.push(prove_interleaved_opening(
			self.codeword_vcs,
			self.codeword,
			self.codeword_committed,
			coset_index,
			arity - self.log_batch_size,
			self.log_batch_size,
		)?);

		for (vcs, (codeword, committed), arity) in
			izip!(self.round_vcss.iter(), self.round_committed.iter(), arities)
		{
			coset_index >>= arity;
			round_proofs.push(prove_coset_opening(vcs, codeword, committed, coset_index, arity)?);
		}

		Ok(round_proofs)
	}
}

fn prove_interleaved_opening<F: BinaryField, VCS: VectorCommitScheme<F>>(
	vcs: &VCS,
	codeword: &[F],
	committed: &VCS::Committed,
	coset_index: usize,
	log_coset_size: usize,
	log_batch_size: usize,
) -> Result<QueryRoundProof<F, VCS::Proof>, Error> {
	let vcs_range = (coset_index << log_coset_size)..((coset_index + 1) << log_coset_size);
	let vcs_proof = vcs
		.prove_range_batch_opening(committed, vcs_range.clone())
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	let code_range = (coset_index << (log_coset_size + log_batch_size))
		..((coset_index + 1) << (log_coset_size + log_batch_size));
	Ok(QueryRoundProof {
		values: codeword[code_range].to_vec(),
		vcs_proof,
	})
}

fn prove_coset_opening<F: BinaryField, VCS: VectorCommitScheme<F>>(
	vcs: &VCS,
	codeword: &[F],
	committed: &VCS::Committed,
	coset_index: usize,
	log_coset_size: usize,
) -> Result<QueryRoundProof<F, VCS::Proof>, Error> {
	let range = (coset_index << log_coset_size)..((coset_index + 1) << log_coset_size);

	let vcs_proof = vcs
		.prove_range_batch_opening(committed, range.clone())
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(QueryRoundProof {
		values: codeword[range].to_vec(),
		vcs_proof,
	})
}
