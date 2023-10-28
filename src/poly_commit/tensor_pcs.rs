// Copyright 2023 Ulvetanna Inc.

use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_matrix::{dense::RowMajorMatrix, MatrixRowSlices};
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	iter,
	iter::repeat_with,
	marker::PhantomData,
};

use super::error::{Error, VerificationError};
use crate::{
	field::{
		get_packed_slice, unpack_scalars, unpack_scalars_mut, BinaryField8b, ExtensionField, Field,
		PackedExtensionField, PackedField,
	},
	hash::{hash, GroestlDigest, GroestlDigestCompression, GroestlHasher, Hasher},
	linear_code::LinearCodeWithExtensionEncoding,
	merkle_tree::{MerkleTreeVCS, VectorCommitScheme},
	poly_commit::PolyCommitScheme,
	polynomial::{Error as PolynomialError, MultilinearPoly},
};

#[derive(Debug)]
pub struct Proof<'a, F, PE, VCSProof>
where
	PE: PackedField,
{
	pub t_prime: MultilinearPoly<'a, PE>,
	pub vcs_proofs: Vec<(Vec<F>, VCSProof)>,
}

/// The multilinear polynomial commitment scheme specified in [DP23].
///
/// [DP23]: https://eprint.iacr.org/2023/630
#[derive(Copy, Clone)]
pub struct TensorPCS<F, P, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F>,
	PE: PackedField,
	PE::Scalar: ExtensionField<F>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
	H: Hasher<F>,
	VCS: VectorCommitScheme<H::Digest>,
{
	log_rows: usize,
	code: LC,
	vcs: VCS,
	_h_marker: PhantomData<H>,
	_ext_marker: PhantomData<PE>,
}

impl<F, P, PE, LC, H, VCS> Debug for TensorPCS<F, P, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F>,
	PE: PackedField,
	PE::Scalar: ExtensionField<F>,
	LC: LinearCodeWithExtensionEncoding<P = P> + Debug,
	H: Hasher<F>,
	VCS: VectorCommitScheme<H::Digest>,
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		f.debug_struct("PolyCommitScheme")
			.field("log_rows", &self.log_rows)
			.field("code", &self.code)
			.finish()
	}
}

impl<F, P, PE, LC>
	TensorPCS<
		F,
		P,
		PE,
		LC,
		GroestlHasher<F>,
		MerkleTreeVCS<
			GroestlDigest,
			GroestlDigest,
			GroestlHasher<GroestlDigest>,
			GroestlDigestCompression,
		>,
	> where
	F: ExtensionField<BinaryField8b>
		+ PackedField<Scalar = F>
		+ PackedExtensionField<BinaryField8b>,
	P: PackedField<Scalar = F> + Sync,
	PE: PackedExtensionField<P>,
	PE::Scalar: ExtensionField<F>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
{
	pub fn new_using_groestl_merkle_tree(log_rows: usize, code: LC) -> Result<Self, Error> {
		if !code.len().is_power_of_two() {
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		let log_len = log2_strict_usize(code.len());
		Self::new(log_rows, code, MerkleTreeVCS::new(log_len, GroestlDigestCompression))
	}
}

impl<F, P, FE, PE, LC, H, VCS> PolyCommitScheme<P, FE> for TensorPCS<F, P, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F> + PackedExtensionField<F> + Send + Sync,
	FE: ExtensionField<F>,
	PE: PackedField<Scalar = FE> + PackedExtensionField<P> + PackedExtensionField<FE>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
	H: Hasher<F>,
	H::Digest: Copy + Default + Send,
	VCS: VectorCommitScheme<H::Digest>,
{
	type Commitment = VCS::Commitment;
	type Committed = (RowMajorMatrix<P>, VCS::Committed);
	type Proof = Proof<'static, F, FE, VCS::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.n_vars()
	}

	fn commit(
		&self,
		poly: &MultilinearPoly<P>,
	) -> Result<(Self::Commitment, Self::Committed), Error> {
		// This condition is checked by the constructor.
		debug_assert!(self.code.dim() >= P::WIDTH);

		if poly.n_vars() != self.n_vars() {
			return Err(Error::IncorrectPolynomialSize {
				expected: self.n_vars(),
			});
		}

		let n_rows = 1 << self.log_rows;
		let n_cols = self.code.len();

		let mut encoded = vec![P::default(); n_rows * n_cols / P::WIDTH];
		transpose::transpose(
			unpack_scalars(poly.evals()),
			unpack_scalars_mut(&mut encoded[..n_rows * self.code.dim() / P::WIDTH]),
			self.code.dim(),
			n_rows,
		);

		self.code
			.encode_batch_inplace(&mut encoded, self.log_rows)
			.map_err(|err| Error::EncodeError(Box::new(err)))?;

		let mut digests = vec![H::Digest::default(); n_cols];
		encoded
			.par_chunks_exact(n_rows / P::WIDTH)
			.map(|col| hash::<_, H>(unpack_scalars(col)))
			.collect_into_vec(&mut digests);

		let encoded_mat = RowMajorMatrix::new(encoded, n_rows / P::WIDTH);
		let (commitment, vcs_committed) = self
			.vcs
			.commit_batch(iter::once(&digests))
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;
		Ok((commitment, (encoded_mat, vcs_committed)))
	}

	/// Generate an evaluation proof at a *random* challenge point.
	///
	/// Follows the notation from Construction 4.6 in [DP23].
	///
	/// Precondition: The queried point must already be observed by the challenger.
	///
	/// [DP23]: https://eprint.iacr.org/2023/630
	fn prove_evaluation<CH>(
		&self,
		challenger: &mut CH,
		committed: &Self::Committed,
		poly: &MultilinearPoly<P>,
		query: &[FE],
	) -> Result<Self::Proof, Error>
	where
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>,
	{
		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}

		let code_len_bits = log2_strict_usize(self.code.len());

		let t = poly;
		let t_prime = t.evaluate_partial(&query[self.code.dim_bits()..])?;

		challenger.observe_slice(t_prime.evals());
		let merkle_proofs = repeat_with(|| challenger.sample_bits(code_len_bits))
			.take(self.code.n_test_queries())
			.map(|index| {
				let (col_major_mat, ref vcs_committed) = committed;
				let col = unpack_scalars(col_major_mat.row_slice(index)).to_vec();

				let vcs_proof = self
					.vcs
					.prove_batch_opening(vcs_committed, index)
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;
				Ok((col, vcs_proof))
			})
			.collect::<Result<_, Error>>()?;

		Ok(Proof {
			t_prime,
			vcs_proofs: merkle_proofs,
		})
	}

	/// Verify an evaluation proof at a *random* challenge point.
	///
	/// Follows the notation from Construction 4.6 in [DP23].
	///
	/// Precondition: The queried point must already be observed by the challenger.
	///
	/// [DP23]: https://eprint.iacr.org/2023/630
	fn verify_evaluation<CH>(
		&self,
		challenger: &mut CH,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		value: FE,
	) -> Result<(), Error>
	where
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>,
	{
		// TODO: Check this during construction. Should be an invariant of the struct.
		assert!(self.code.dim() >= P::WIDTH);
		assert!(self.code.dim() >= PE::WIDTH);

		assert_eq!(self.code.len() % P::WIDTH, 0);
		assert_eq!(self.code.len() % PE::WIDTH, 0);

		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}

		self.check_proof_shape(&proof)?;

		if !self.code.len().is_power_of_two() {
			// This requirement is just to make sampling indices easier. With a little work it
			// could be relaxed, but power-of-two code lengths are more convenient to work with.
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		let code_len_bits = log2_strict_usize(self.code.len());

		challenger.observe_slice(proof.t_prime.evals());

		// Check evaluation of t' matches the claimed value
		let computed_value = proof
			.t_prime
			.evaluate(&query[..self.code.dim_bits()])
			.expect("query is the correct size by check_proof_shape checks");
		if computed_value != value {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// Encode t' into u'
		let mut u_prime = vec![PE::default(); self.code.len() / PE::WIDTH];
		unpack_scalars_mut::<_, FE>(&mut u_prime[..self.code.dim() / PE::WIDTH])
			.copy_from_slice(proof.t_prime.evals());

		self.code
			.encode_extension_inplace(&mut u_prime)
			.map_err(|err| Error::EncodeError(Box::new(err)))?;

		// Check vector commitment openings and get a sequence of column tests.
		let column_tests = proof
			.vcs_proofs
			.into_iter()
			.map(|(col, vcs_proof)| {
				let index = challenger.sample_bits(code_len_bits);
				let leaf_digest = hash::<_, H>(&col);

				self.vcs
					.verify_batch_opening(commitment, index, vcs_proof, iter::once(leaf_digest))
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				let u_prime_i = get_packed_slice(&u_prime, index);
				Ok((u_prime_i, col))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		// Batch evaluate all opened columns
		let leaf_evaluations = MultilinearPoly::batch_evaluate(
			column_tests.iter().map(|(_, leaf)| {
				MultilinearPoly::from_values_slice(leaf)
					.expect("leaf is guaranteed power of two length due to check_proof_shape")
			}),
			&query[self.code.dim_bits()..],
		);

		// Check that opened column evaluations match u'
		for ((expected, _), leaf_eval_result) in column_tests.iter().zip(leaf_evaluations) {
			let leaf_eval = leaf_eval_result
				.expect("leaf polynomials are the correct length by check_proof_shape");
			if leaf_eval != *expected {
				return Err(VerificationError::IncorrectPartialEvaluation.into());
			}
		}

		Ok(())
	}
}

impl<F, P, PE, LC, H, VCS> TensorPCS<F, P, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F>,
	PE: PackedField,
	PE::Scalar: ExtensionField<F>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
	LC::EncodeError: 'static,
	H: Hasher<F>,
	VCS: VectorCommitScheme<H::Digest>,
{
	/// Construct a [`TensorPCS`].
	///
	/// Throws if the linear code block length is not a power of 2.
	/// Throws if the packing width does not divide the code dimension.
	pub fn new(log_rows: usize, code: LC, vcs: VCS) -> Result<Self, Error> {
		if !code.len().is_power_of_two() {
			// This requirement is just to make sampling indices easier. With a little work it
			// could be relaxed, but power-of-two code lengths are more convenient to work with.
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		if (1 << log_rows) % P::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideNumberOfRows);
		}
		if code.dim() % P::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideCodeDimension);
		}
		if code.dim() % PE::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideCodeDimension);
		}

		Ok(Self {
			log_rows,
			code,
			vcs,
			_h_marker: PhantomData,
			_ext_marker: PhantomData,
		})
	}

	fn n_vars(&self) -> usize {
		self.log_rows + self.code.dim_bits()
	}

	fn check_proof_shape(
		&self,
		proof: &Proof<F, PE::Scalar, VCS::Proof>,
	) -> Result<(), VerificationError> {
		let n_rows = 1 << self.log_rows;
		let n_queries = self.code.n_test_queries();

		if proof.vcs_proofs.len() != n_queries {
			return Err(VerificationError::NumberOfOpeningProofs {
				expected: n_queries,
			});
		}
		for (i, (col, _)) in proof.vcs_proofs.iter().enumerate() {
			if col.len() != n_rows {
				return Err(VerificationError::OpenedColumnSize {
					index: i,
					expected: n_rows,
				});
			}
		}
		if proof.t_prime.n_vars() != self.code.dim_bits() {
			return Err(VerificationError::PartialEvaluationSize);
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{PackedBinaryField16x8b, PackedBinaryField1x128b},
		reed_solomon::reed_solomon::ReedSolomonCode,
	};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	#[test]
	fn test_commit_prove_verify_without_error() {
		type Packed = PackedBinaryField16x8b;

		let rs_code = ReedSolomonCode::new(5, 2, 12).unwrap();
		let pcs =
			<TensorPCS<_, Packed, PackedBinaryField1x128b, _, _, _>>::new_using_groestl_merkle_tree(4, rs_code).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| Packed::random(&mut rng))
			.take((1 << pcs.n_vars()) / Packed::WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearPoly::from_values(evals).unwrap();

		let (commitment, committed) = pcs.commit(&poly).unwrap();

		let mut challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
		let query = repeat_with(|| challenger.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();

		let value = poly.evaluate(&query).unwrap();

		let mut prove_challenger = challenger.clone();
		let proof = pcs
			.prove_evaluation(&mut prove_challenger, &committed, &poly, &query)
			.unwrap();

		let mut verify_challenger = challenger.clone();
		pcs.verify_evaluation(&mut verify_challenger, &commitment, &query, proof, value)
			.unwrap();
	}
}
