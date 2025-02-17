// Copyright 2024-2025 Irreducible Inc.

use binius_field::{
	BinaryField, ExtensionField, PackedExtension, PackedField, SerializeCanonical, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_maybe_rayon::prelude::*;
use binius_utils::bail;
use bytemuck::zeroed_vec;
use bytes::BufMut;
use itertools::izip;
use tracing::instrument;

use super::{
	common::{vcs_optimal_layers_depths_iter, FRIParams},
	error::Error,
	TerminateCodeword,
};
use crate::{
	fiat_shamir::{CanSampleBits, Challenger},
	merkle_tree::{MerkleTreeProver, MerkleTreeScheme},
	protocols::fri::common::{fold_chunk, fold_interleaved_chunk},
	reed_solomon::reed_solomon::ReedSolomonCode,
	transcript::{ProverTranscript, TranscriptWriter},
};

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
) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + '_>
where
	P: PackedField,
{
	packed_slice
		.par_chunks(chunk_size / P::WIDTH)
		.map(|chunk| PackedField::iter_slice(chunk))
}

pub fn to_par_scalar_small_chunks<P>(
	packed_slice: &[P],
	chunk_size: usize,
) -> impl IndexedParallelIterator<Item: Iterator<Item = P::Scalar> + Send + '_>
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
/// * `params` - common FRI protocol parameters.
/// * `merkle_prover` - the merke tree prover to use for committing
/// * `message` - the interleaved message to encode and commit
#[instrument(skip_all, level = "debug")]
pub fn commit_interleaved<F, FA, P, PA, MerkleProver, VCS>(
	rs_code: &ReedSolomonCode<PA>,
	params: &FRIParams<F, FA>,
	merkle_prover: &MerkleProver,
	message: &[P],
) -> Result<CommitOutput<P, VCS::Digest, MerkleProver::Committed>, Error>
where
	F: BinaryField,
	FA: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<FA, PackedSubfield = PA>,
	PA: PackedField<Scalar = FA>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	let n_elems = rs_code.dim() << params.log_batch_size();
	if message.len() * P::WIDTH != n_elems {
		bail!(Error::InvalidArgs(
			"interleaved message length does not match code parameters".to_string()
		));
	}

	commit_interleaved_with(rs_code, params, merkle_prover, move |buffer| {
		buffer.copy_from_slice(message)
	})
}

/// Encodes and commits the input message with a closure for writing the message.
///
/// ## Arguments
///
/// * `rs_code` - the Reed-Solomon code to use for encoding
/// * `params` - common FRI protocol parameters.
/// * `merkle_prover` - the Merkle tree prover to use for committing
/// * `message_writer` - a closure that writes the interleaved message to encode and commit
#[instrument(skip_all, level = "debug")]
pub fn commit_interleaved_with<F, FA, P, PA, MerkleProver, VCS>(
	rs_code: &ReedSolomonCode<PA>,
	params: &FRIParams<F, FA>,
	merkle_prover: &MerkleProver,
	message_writer: impl FnOnce(&mut [P]),
) -> Result<CommitOutput<P, VCS::Digest, MerkleProver::Committed>, Error>
where
	F: BinaryField,
	FA: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<FA, PackedSubfield = PA>,
	PA: PackedField<Scalar = FA>,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	let log_batch_size = params.log_batch_size();
	let log_elems = rs_code.log_dim() + log_batch_size;
	if log_elems < P::LOG_WIDTH {
		todo!("can't handle this case well");
	}

	let mut encoded = tracing::debug_span!("allocate codeword")
		.in_scope(|| zeroed_vec(1 << (log_elems - P::LOG_WIDTH + rs_code.log_inv_rate())));
	message_writer(&mut encoded[..1 << (log_elems - P::LOG_WIDTH)]);
	rs_code.encode_ext_batch_inplace(&mut encoded, log_batch_size)?;

	// take the first arity as coset_log_len, or use log_inv_rate if arities are empty
	let coset_log_len = params
		.fold_arities()
		.first()
		.copied()
		.unwrap_or_else(|| rs_code.log_inv_rate());

	let log_len = params.log_len() - coset_log_len;

	let (commitment, vcs_committed) = if coset_log_len > P::LOG_WIDTH {
		let iterated_big_chunks = to_par_scalar_big_chunks(&encoded, 1 << coset_log_len);

		merkle_prover
			.commit_iterated(iterated_big_chunks, log_len)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?
	} else {
		let iterated_small_chunks = to_par_scalar_small_chunks(&encoded, 1 << coset_log_len);

		merkle_prover
			.commit_iterated(iterated_small_chunks, log_len)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?
	};

	Ok(CommitOutput {
		commitment: commitment.root,
		committed: vcs_committed,
		codeword: encoded,
	})
}

pub enum FoldRoundOutput<VCSCommitment> {
	NoCommitment,
	Commitment(VCSCommitment),
}

/// A stateful prover for the FRI fold phase.
pub struct FRIFolder<'a, F, FA, MerkleProver, VCS>
where
	FA: BinaryField,
	F: BinaryField,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	params: &'a FRIParams<F, FA>,
	merkle_prover: &'a MerkleProver,
	codeword: &'a [F],
	codeword_committed: &'a MerkleProver::Committed,
	round_committed: Vec<(Vec<F>, MerkleProver::Committed)>,
	curr_round: usize,
	next_commit_round: Option<usize>,
	unprocessed_challenges: Vec<F>,
}

impl<'a, F, FA, MerkleProver, VCS> FRIFolder<'a, F, FA, MerkleProver, VCS>
where
	F: TowerField + ExtensionField<FA>,
	FA: BinaryField,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F, Digest: SerializeCanonical>,
{
	/// Constructs a new folder.
	pub fn new(
		params: &'a FRIParams<F, FA>,
		merkle_prover: &'a MerkleProver,
		committed_codeword: &'a [F],
		committed: &'a MerkleProver::Committed,
	) -> Result<Self, Error> {
		if committed_codeword.len() != 1 << params.log_len() {
			bail!(Error::InvalidArgs(
				"Reed–Solomon code length must match interleaved codeword length".to_string(),
			));
		}

		let next_commit_round = params.fold_arities().first().copied();
		Ok(Self {
			params,
			merkle_prover,
			codeword: committed_codeword,
			codeword_committed: committed,
			round_committed: Vec::with_capacity(params.n_oracles()),
			curr_round: 0,
			next_commit_round,
			unprocessed_challenges: Vec::with_capacity(params.rs_code().log_dim()),
		})
	}

	/// Number of fold rounds, including the final fold.
	pub const fn n_rounds(&self) -> usize {
		self.params.n_fold_rounds()
	}

	/// Number of times `execute_fold_round` has been called.
	pub const fn curr_round(&self) -> usize {
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
	#[instrument(skip_all, name = "fri::FRIFolder::execute_fold_round", level = "debug")]
	pub fn execute_fold_round(
		&mut self,
		challenge: F,
	) -> Result<FoldRoundOutput<VCS::Digest>, Error> {
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

		// take the first arity as coset_log_len, or use inv_rate if arities are empty
		let coset_size = self
			.params
			.fold_arities()
			.get(self.round_committed.len() + 1)
			.map(|log| 1 << log)
			.unwrap_or_else(|| self.params.rs_code().inv_rate());

		let (commitment, committed) = self
			.merkle_prover
			.commit(&folded_codeword, coset_size)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;

		self.round_committed.push((folded_codeword, committed));

		self.next_commit_round = self.next_commit_round.take().and_then(|next_commit_round| {
			let arity = self.params.fold_arities().get(self.round_committed.len())?;
			Some(next_commit_round + arity)
		});
		Ok(FoldRoundOutput::Commitment(commitment.root))
	}

	/// Finalizes the FRI folding process.
	///
	/// This step will process any unprocessed folding challenges to produce the
	/// final folded codeword. Then it will decode this final folded codeword
	/// to get the final message. The result is the final message and a query prover instance.
	///
	/// This returns the final message and a query prover instance.
	#[instrument(skip_all, name = "fri::FRIFolder::finalize", level = "debug")]
	#[allow(clippy::type_complexity)]
	pub fn finalize(
		mut self,
	) -> Result<(TerminateCodeword<F>, FRIQueryProver<'a, F, FA, MerkleProver, VCS>), Error> {
		if self.curr_round != self.n_rounds() {
			bail!(Error::EarlyProverFinish);
		}

		let terminate_codeword = self
			.round_committed
			.last()
			.map(|(codeword, _)| codeword.clone())
			.unwrap_or_else(|| self.codeword.to_vec());

		self.unprocessed_challenges.clear();

		let Self {
			params,
			codeword,
			codeword_committed,
			round_committed,
			merkle_prover,
			..
		} = self;

		let query_prover = FRIQueryProver {
			params,
			codeword,
			codeword_committed,
			round_committed,
			merkle_prover,
		};
		Ok((terminate_codeword, query_prover))
	}

	pub fn finish_proof<Challenger_>(
		self,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> Result<(), Error>
	where
		Challenger_: Challenger,
	{
		let (terminate_codeword, query_prover) = self.finalize()?;
		let mut advice = transcript.decommitment();
		advice.write_scalar_slice(&terminate_codeword);

		let layers = query_prover.vcs_optimal_layers()?;
		for layer in layers {
			advice.write_slice(&layer);
		}

		let params = query_prover.params;

		for _ in 0..params.n_test_queries() {
			let index = transcript.sample_bits(params.index_bits());
			query_prover.prove_query(index, transcript.decommitment())?;
		}

		Ok(())
	}
}

/// A prover for the FRI query phase.
pub struct FRIQueryProver<'a, F, FA, MerkleProver, VCS>
where
	F: BinaryField,
	FA: BinaryField,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	params: &'a FRIParams<F, FA>,
	codeword: &'a [F],
	codeword_committed: &'a MerkleProver::Committed,
	round_committed: Vec<(Vec<F>, MerkleProver::Committed)>,
	merkle_prover: &'a MerkleProver,
}

impl<F, FA, MerkleProver, VCS> FRIQueryProver<'_, F, FA, MerkleProver, VCS>
where
	F: TowerField + ExtensionField<FA>,
	FA: BinaryField,
	MerkleProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
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
	#[instrument(skip_all, name = "fri::FRIQueryProver::prove_query", level = "debug")]
	pub fn prove_query<B>(
		&self,
		mut index: usize,
		mut advice: TranscriptWriter<B>,
	) -> Result<(), Error>
	where
		B: BufMut,
	{
		let mut arities_and_optimal_layers_depths = self
			.params
			.fold_arities()
			.iter()
			.copied()
			.zip(vcs_optimal_layers_depths_iter(self.params, self.merkle_prover.scheme()));

		let Some((first_fold_arity, first_optimal_layer_depth)) =
			arities_and_optimal_layers_depths.next()
		else {
			// If there are no query proofs, that means that no oracles were sent during the FRI
			// fold rounds. In that case, the original interleaved codeword is decommitted and
			// the only checks that need to be performed are in `verify_last_oracle`.
			return Ok(());
		};

		prove_coset_opening(
			self.merkle_prover,
			self.codeword,
			self.codeword_committed,
			index,
			first_fold_arity,
			first_optimal_layer_depth,
			&mut advice,
		)?;

		for ((codeword, committed), (arity, optimal_layer_depth)) in
			izip!(self.round_committed.iter(), arities_and_optimal_layers_depths)
		{
			index >>= arity;
			prove_coset_opening(
				self.merkle_prover,
				codeword,
				committed,
				index,
				arity,
				optimal_layer_depth,
				&mut advice,
			)?;
		}

		Ok(())
	}

	pub fn vcs_optimal_layers(&self) -> Result<Vec<Vec<VCS::Digest>>, Error> {
		let committed_iter = std::iter::once(self.codeword_committed)
			.chain(self.round_committed.iter().map(|(_, committed)| committed));

		committed_iter
			.zip(vcs_optimal_layers_depths_iter(self.params, self.merkle_prover.scheme()))
			.map(|(committed, optimal_layer_depth)| {
				self.merkle_prover
					.layer(committed, optimal_layer_depth)
					.map(|layer| layer.to_vec())
					.map_err(|err| Error::VectorCommit(Box::new(err)))
			})
			.collect::<Result<Vec<_>, _>>()
	}
}

fn prove_coset_opening<F, MTProver, B>(
	merkle_prover: &MTProver,
	codeword: &[F],
	committed: &MTProver::Committed,
	coset_index: usize,
	log_coset_size: usize,
	optimal_layer_depth: usize,
	advice: &mut TranscriptWriter<B>,
) -> Result<(), Error>
where
	F: TowerField,
	MTProver: MerkleTreeProver<F>,
	B: BufMut,
{
	let values = &codeword[(coset_index << log_coset_size)..((coset_index + 1) << log_coset_size)];
	advice.write_scalar_slice(values);

	merkle_prover
		.prove_opening(committed, optimal_layer_depth, coset_index, advice)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(())
}
