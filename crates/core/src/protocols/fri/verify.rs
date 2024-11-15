// Copyright 2024 Irreducible Inc.

use super::{
	common::{vcs_optimal_layers_depths_iter, FRIProof},
	error::Error,
	QueryProof, VerificationError,
};
use crate::{
	merkle_tree_vcs::MerkleTreeScheme,
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
	VCS: MerkleTreeScheme<F>,
{
	vcs: &'a VCS,
	params: &'a FRIParams<F, FA>,
	/// Received commitment to the codeword.
	codeword_commitment: &'a VCS::Digest,
	/// Received commitments to the round messages.
	round_commitments: &'a [VCS::Digest],
	/// The challenges for each round.
	interleave_tensor: Vec<F>,
	/// The challenges for each round.
	fold_challenges: &'a [F],
}

impl<'a, F, FA, VCS> FRIVerifier<'a, F, FA, VCS>
where
	F: BinaryField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: MerkleTreeScheme<F>,
{
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		params: &'a FRIParams<F, FA>,
		vcs: &'a VCS,
		codeword_commitment: &'a VCS::Digest,
		round_commitments: &'a [VCS::Digest],
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
			vcs,
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
		fri_proof: FRIProof<F, VCS>,
		mut challenger: Challenger,
	) -> Result<F, Error>
	where
		Challenger: CanSampleBits<usize>,
	{
		let FRIProof {
			terminate_codeword,
			proofs,
			layers,
		} = fri_proof;

		let final_value = self.verify_last_oracle(&terminate_codeword)?;

		let indexes_iter =
			std::iter::repeat_with(|| challenger.sample_bits(self.params.index_bits()))
				.take(self.params.n_test_queries());

		// Verify that the provided layers match the commitments.
		for (commitment, layer_depth, layer) in izip!(
			iter::once(self.codeword_commitment).chain(self.round_commitments),
			vcs_optimal_layers_depths_iter(self.params, self.vcs),
			&layers
		) {
			self.vcs
				.verify_layer(commitment, layer_depth, layer)
				.map_err(|err| Error::VectorCommit(Box::new(err)))?;
		}

		let mut scratch_buffer = self.create_scratch_buffer();

		for (index, proof) in indexes_iter.zip(proofs) {
			self.verify_query_internal(
				index,
				proof,
				&terminate_codeword,
				&layers,
				&mut scratch_buffer,
			)?
		}
		Ok(final_value)
	}

	/// Verifies that the last oracle sent is a codeword.
	///
	/// Returns the fully-folded message value.
	pub fn verify_last_oracle(&self, terminate_codeword: &[F]) -> Result<F, Error> {
		self.vcs
			.verify_vector(
				self.round_commitments
					.last()
					.unwrap_or(self.codeword_commitment),
				terminate_codeword,
				1 << self.params.rs_code().log_inv_rate(),
			)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;

		let repetition_codeword = if self.n_oracles() != 0 {
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
	pub fn verify_query(
		&self,
		index: usize,
		proof: QueryProof<F, VCS::Proof>,
		terminate_codeword: &[F],
		layers: &[Vec<VCS::Digest>],
	) -> Result<(), Error> {
		self.verify_query_internal(
			index,
			proof,
			terminate_codeword,
			layers,
			&mut self.create_scratch_buffer(),
		)
	}

	#[instrument(skip_all, name = "fri::FRIVerifier::verify_query", level = "debug")]
	fn verify_query_internal(
		&self,
		mut index: usize,
		proof: QueryProof<F, VCS::Proof>,
		terminate_codeword: &[F],
		layers: &[Vec<VCS::Digest>],
		scratch_buffer: &mut [F],
	) -> Result<(), Error> {
		if proof.len() != self.n_oracles() {
			return Err(VerificationError::IncorrectQueryProofLength {
				expected: self.n_oracles(),
			}
			.into());
		}

		let arities = self.params.fold_arities().iter().copied();
		let mut proof_and_arities_iter = iter::zip(proof, arities.clone());

		let mut layer_digest_and_optimal_layer_depth =
			iter::zip(layers, vcs_optimal_layers_depths_iter(self.params, self.vcs));

		let Some((first_query_proof, first_fold_arity)) = proof_and_arities_iter.next() else {
			// If there are no query proofs, that means that no oracles were sent during the FRI
			// fold rounds. In that case, the original interleaved codeword is decommitted and
			// the only checks that need to be performed are in `verify_last_oracle`.
			return Ok(());
		};

		let (first_layer, first_optimal_layer_depth) = layer_digest_and_optimal_layer_depth
			.next()
			.expect("The length should be the same as the amount of proofs.");

		// This is the round of the folding phase that the codeword to be folded is committed to.
		let mut fold_round = 0;

		let mut log_n_cosets = self.params.index_bits();

		// Check the first fold round before the main loop. It is special because in the first
		// round we need to fold as an interleaved chunk instead of a regular coset.
		let log_coset_size = first_fold_arity - self.params.log_batch_size();
		let values = verify_coset_opening(
			self.vcs,
			0,
			index,
			first_fold_arity,
			first_query_proof,
			first_optimal_layer_depth,
			log_n_cosets,
			first_layer,
		)?;
		let mut next_value = fold_interleaved_chunk(
			self.params.rs_code(),
			self.params.log_batch_size(),
			index,
			&values,
			&self.interleave_tensor,
			&self.fold_challenges[fold_round..fold_round + log_coset_size],
			scratch_buffer,
		);
		fold_round += log_coset_size;

		for (i, ((round_proof, arity), (layer, optimal_layer_depth))) in
			izip!(proof_and_arities_iter, layer_digest_and_optimal_layer_depth).enumerate()
		{
			let coset_index = index >> arity;

			log_n_cosets -= arity;

			let values = verify_coset_opening(
				self.vcs,
				i + 1,
				coset_index,
				arity,
				round_proof,
				optimal_layer_depth,
				log_n_cosets,
				layer,
			)?;

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
				scratch_buffer,
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

	// scratch buffer used in `fold_chunk`.
	fn create_scratch_buffer(&self) -> Vec<F> {
		let max_arity = self
			.params
			.fold_arities()
			.iter()
			.cloned()
			.max()
			.unwrap_or_default();
		let max_buffer_size = 2 * (1 << max_arity);
		vec![F::default(); max_buffer_size]
	}
}

/// Verifies that the coset opening provided in the proof is consistent with the VCS commitment.
#[allow(clippy::too_many_arguments)]
fn verify_coset_opening<F: BinaryField, VCS: MerkleTreeScheme<F>>(
	vcs: &VCS,
	round: usize,
	coset_index: usize,
	log_coset_size: usize,
	proof: QueryRoundProof<F, VCS::Proof>,
	optimal_layer_depth: usize,
	tree_depth: usize,
	layer_digests: &[VCS::Digest],
) -> Result<Vec<F>, Error> {
	let QueryRoundProof { values, vcs_proof } = proof;

	if values.len() != 1 << log_coset_size {
		return Err(VerificationError::IncorrectQueryProofValuesLength {
			round,
			coset_size: 1 << log_coset_size,
		}
		.into());
	}
	vcs.verify_opening(
		coset_index,
		&values,
		optimal_layer_depth,
		tree_depth,
		layer_digests,
		vcs_proof,
	)
	.map_err(|err| Error::VectorCommit(Box::new(err)))?;

	Ok(values)
}
