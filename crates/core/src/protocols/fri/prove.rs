// Copyright 2024 Irreducible Inc.

use super::{
	common::{FRIParams, FRIProof},
	error::Error,
	TerminateCodeword,
};
use crate::{
	linear_code::LinearCode,
	merkle_tree::VectorCommitScheme,
	protocols::fri::common::{fold_chunk, fold_interleaved_chunk, QueryProof, QueryRoundProof},
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{
	packed::iter_packed_slice, BinaryField, ExtensionField, PackedExtension, PackedField,
};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_utils::bail;
use bytemuck::zeroed_vec;
use itertools::izip;
use p3_challenger::CanSampleBits;
use rayon::prelude::*;
use tracing::instrument;

#[instrument(skip_all, level = "debug")]
pub fn fold_codeword<F, FS>(
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
#[instrument(skip_all, level = "debug")]
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
	let tensor = backend
		.tensor_product_full_query(interleave_challenges)
		.expect("number of challenges is less than 32");

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

/// Creates a parallel iterator over scalars of subfield elementsAssumes chunk_size to be a power of two
pub fn to_par_scalar_big_chunks<P>(
	packed_slice: &[P],
	chunk_size: usize,
) -> impl IndexedParallelIterator<Item = impl Iterator<Item = P::Scalar> + Send + '_>
where
	P: PackedField,
{
	packed_slice
		.par_chunks(chunk_size / P::WIDTH)
		.map(|chunk| iter_packed_slice(chunk))
}

pub fn to_par_scalar_small_chunks<P>(
	packed_slice: &[P],
	chunk_size: usize,
) -> impl IndexedParallelIterator<Item = impl Iterator<Item = P::Scalar> + Send + '_>
where
	P: PackedField,
{
	(0..packed_slice.len() * P::WIDTH)
		.into_par_iter()
		.step_by(chunk_size)
		.map(move |start_index| {
			let packed_item = &packed_slice[start_index / P::WIDTH];
			packed_item
				.iter()
				.skip(start_index % P::WIDTH)
				.take(chunk_size)
		})
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
	P: PackedField<Scalar = F> + PackedExtension<FA, PackedSubfield = PA>,
	PA: PackedField<Scalar = FA>,
	VCS: VectorCommitScheme<F>,
{
	let n_elems = message.len() * P::WIDTH;
	if n_elems != rs_code.dim() << log_batch_size {
		bail!(Error::InvalidArgs(
			"interleaved message length does not match code parameters".to_string()
		));
	}

	if !vcs.vector_len().is_power_of_two() {
		bail!(Error::InvalidArgs("vector commitment length is not a power of two".to_string()));
	}
	if rs_code.len() < vcs.vector_len() {
		bail!(Error::InvalidArgs(
			"Reed–Solomon code length must be at least the vector commitment length".to_string(),
		));
	}

	let mut encoded = tracing::debug_span!("allocate codeword")
		.in_scope(|| zeroed_vec(message.len() << rs_code.log_inv_rate()));
	encoded[..message.len()].copy_from_slice(message);
	rs_code.encode_ext_batch_inplace(&mut encoded, log_batch_size)?;

	let batch_size = encoded.len() * P::WIDTH / vcs.vector_len();

	let (commitment, vcs_committed) = if batch_size > P::WIDTH {
		let iterated_big_chunks = to_par_scalar_big_chunks(&encoded, batch_size);

		vcs.commit_iterated(iterated_big_chunks, batch_size)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?
	} else {
		let iterated_small_chunks = to_par_scalar_small_chunks(&encoded, batch_size);

		vcs.commit_iterated(iterated_small_chunks, batch_size)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?
	};

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
	params: &'a FRIParams<F, FA, VCS>,
	codeword: &'a [F],
	codeword_committed: &'a VCS::Committed,
	round_committed: Vec<(Vec<F>, VCS::Committed)>,
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
		params: &'a FRIParams<F, FA, VCS>,
		committed_codeword: &'a [F],
		committed: &'a VCS::Committed,
	) -> Result<Self, Error> {
		if committed_codeword.len() != 1 << (params.rs_code().log_len() + params.log_batch_size()) {
			bail!(Error::InvalidArgs(
				"Reed–Solomon code length must match interleaved codeword length".to_string(),
			));
		}

		let next_commit_round = params.fold_arities().first().copied();
		Ok(Self {
			params,
			codeword: committed_codeword,
			codeword_committed: committed,
			round_committed: Vec::with_capacity(params.n_oracles()),
			curr_round: 0,
			next_commit_round,
			unprocessed_challenges: Vec::with_capacity(params.rs_code().log_dim()),
		})
	}

	/// Number of fold rounds, including the final fold.
	pub fn n_rounds(&self) -> usize {
		self.params.n_fold_rounds()
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
					self.params.rs_code(),
					prev_codeword,
					self.curr_round - self.params.log_batch_size(),
					&self.unprocessed_challenges,
				)
			}
			None => {
				// Fold the interleaved codeword that was originally committed into a single
				// codeword with the same or reduced block length, depending on the sequence of
				// fold rounds.
				fold_interleaved(
					self.params.rs_code(),
					self.codeword,
					&self.unprocessed_challenges,
					self.params.log_batch_size(),
				)
			}
		};
		self.unprocessed_challenges.clear();

		let round_vcs = &self.params.round_vcss()[self.round_committed.len()];

		let (commitment, committed) = round_vcs
			.commit_interleaved(&folded_codeword)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;
		self.round_committed.push((folded_codeword, committed));

		self.next_commit_round = self.next_commit_round.take().and_then(|next_commit_round| {
			let arity = self.params.fold_arities().get(self.round_committed.len())?;
			Some(next_commit_round + arity)
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
	#[allow(clippy::type_complexity)]
	pub fn finalize(
		mut self,
	) -> Result<(TerminateCodeword<F>, FRIQueryProver<'a, F, FA, VCS>), Error> {
		if self.curr_round != self.n_rounds() {
			bail!(Error::EarlyProverFinish);
		}

		let terminate_codeword = self
			.round_committed
			.last()
			.map(|(codeword, _)| codeword.to_vec())
			.unwrap_or(self.codeword.to_vec());

		self.unprocessed_challenges.clear();

		let Self {
			params,
			codeword,
			codeword_committed,
			round_committed,
			..
		} = self;

		let query_prover = FRIQueryProver {
			params,
			codeword,
			codeword_committed,
			round_committed,
		};
		Ok((terminate_codeword, query_prover))
	}

	pub fn finish_proof<Challenger>(
		self,
		mut challenger: Challenger,
	) -> Result<FRIProof<F, VCS::Proof>, Error>
	where
		Challenger: CanSampleBits<usize>,
	{
		let (terminate_codeword, query_prover) = self.finalize()?;

		let params = query_prover.params;

		let indexes_iter = std::iter::repeat_with(|| challenger.sample_bits(params.index_bits()))
			.take(params.n_test_queries());

		let proofs = indexes_iter
			.map(|index| query_prover.prove_query(index))
			.collect::<Result<Vec<_>, _>>()?;

		Ok(FRIProof {
			terminate_codeword,
			proofs,
		})
	}
}

/// A prover for the FRI query phase.
pub struct FRIQueryProver<'a, F, FA, VCS>
where
	F: BinaryField,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	params: &'a FRIParams<F, FA, VCS>,
	codeword: &'a [F],
	codeword_committed: &'a VCS::Committed,
	round_committed: Vec<(Vec<F>, VCS::Committed)>,
}

impl<'a, F, FA, VCS> FRIQueryProver<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: VectorCommitScheme<F>,
{
	/// Number of oracles sent during the fold rounds.
	pub fn n_oracles(&self) -> usize {
		self.params.n_oracles()
	}

	/// Proves a FRI challenge query.
	///
	/// ## Arguments
	///
	/// * `index` - an index into the original codeword domain
	#[instrument(skip_all, name = "fri::FRIQueryProver::prove_query")]
	pub fn prove_query(&self, mut index: usize) -> Result<QueryProof<F, VCS::Proof>, Error> {
		let mut round_proofs = Vec::with_capacity(self.n_oracles());
		let mut arities = self.params.fold_arities().iter().copied();

		let Some(first_fold_arity) = arities.next() else {
			// If there are no query proofs, that means that no oracles were sent during the FRI
			// fold rounds. In that case, the original interleaved codeword is decommitted and
			// the only checks that need to be performed are in `verify_last_oracle`.
			return Ok(round_proofs);
		};

		round_proofs.push(prove_coset_opening(
			self.params.codeword_vcs(),
			self.codeword,
			self.codeword_committed,
			index,
			first_fold_arity,
		)?);

		for (vcs, (codeword, committed), arity) in
			izip!(self.params.round_vcss().iter(), self.round_committed.iter(), arities)
		{
			index >>= arity;
			round_proofs.push(prove_coset_opening(vcs, codeword, committed, index, arity)?);
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
	let vcs_proof = vcs
		.prove_batch_opening(committed, coset_index)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	let range = (coset_index << log_coset_size)..((coset_index + 1) << log_coset_size);
	Ok(QueryRoundProof {
		values: codeword[range].to_vec(),
		vcs_proof,
	})
}
