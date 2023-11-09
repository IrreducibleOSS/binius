// Copyright 2023 Ulvetanna Inc.

use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_matrix::{dense::RowMajorMatrix, MatrixRowSlices};
use p3_util::log2_strict_usize;
use rayon::prelude::*;
use std::{iter, iter::repeat_with, marker::PhantomData};

use super::error::{Error, VerificationError};
use crate::{
	field::{
		get_packed_slice, square_transpose, transpose_scalars, unpack_scalars, unpack_scalars_mut,
		BinaryField8b, ExtensionField, Field, PackedExtensionField, PackedField,
	},
	hash::{hash, GroestlDigest, GroestlDigestCompression, GroestlHasher, Hasher},
	linear_code::LinearCode,
	merkle_tree::{MerkleTreeVCS, VectorCommitScheme},
	poly_commit::PolyCommitScheme,
	polynomial::{Error as PolynomialError, MultilinearPoly},
};

/// Evaluation proof data for the `TensorPCS` polynomial commitment scheme.
///
/// # Type Parameters
///
/// * `PI`: The packed intermediate field type.
/// * `PE`: The packed extension field type.
/// * `VCSProof`: The vector commitment scheme proof type.
#[derive(Debug)]
pub struct Proof<'a, PI, PE, VCSProof>
where
	PE: PackedField,
{
	pub t_prime: MultilinearPoly<'a, PE>,
	pub vcs_proofs: Vec<(Vec<PI>, VCSProof)>,
}

/// The multilinear polynomial commitment scheme specified in [DP23].
///
/// # Type Parameters
///
/// * `P`: The base field type of committed elements.
/// * `PI`: The intermediate field type used for encoding.
/// * `PE`: The extension field type used for cryptographic challenges.
///
/// [DP23]: https://eprint.iacr.org/2023/630
#[derive(Debug, Copy, Clone)]
pub struct TensorPCS<P, PI, PE, LC, H, VCS>
where
	P: PackedField,
	PI: PackedField,
	PE: PackedField,
	LC: LinearCode<P = PI>,
	H: Hasher<PI>,
	VCS: VectorCommitScheme<H::Digest>,
{
	log_rows: usize,
	code: LC,
	vcs: VCS,
	_p_marker: PhantomData<P>,
	_h_marker: PhantomData<H>,
	_ext_marker: PhantomData<PE>,
}

impl<P, PI, PE, LC>
	TensorPCS<
		P,
		PI,
		PE,
		LC,
		GroestlHasher<PI>,
		MerkleTreeVCS<
			GroestlDigest,
			GroestlDigest,
			GroestlHasher<GroestlDigest>,
			GroestlDigestCompression,
		>,
	> where
	P: PackedField,
	PI: PackedField + PackedExtensionField<BinaryField8b> + Sync,
	PI::Scalar: ExtensionField<P::Scalar> + ExtensionField<BinaryField8b>,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	LC: LinearCode<P = PI>,
{
	pub fn new_using_groestl_merkle_tree(log_rows: usize, code: LC) -> Result<Self, Error> {
		// Check power of two length because MerkleTreeVCS requires it
		if !code.len().is_power_of_two() {
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		let log_len = log2_strict_usize(code.len());
		Self::new(log_rows, code, MerkleTreeVCS::new(log_len, GroestlDigestCompression))
	}
}

impl<F, P, FI, PI, FE, PE, LC, H, VCS> PolyCommitScheme<P, FE> for TensorPCS<P, PI, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F> + Send,
	FI: ExtensionField<P::Scalar>,
	PI: PackedField<Scalar = FI> + PackedExtensionField<FI> + PackedExtensionField<P> + Sync,
	FE: ExtensionField<F> + ExtensionField<FI>,
	PE: PackedField<Scalar = FE> + PackedExtensionField<PI> + PackedExtensionField<FE>,
	LC: LinearCode<P = PI>,
	H: Hasher<PI>,
	H::Digest: Copy + Default + Send,
	VCS: VectorCommitScheme<H::Digest>,
{
	type Commitment = VCS::Commitment;
	type Committed = (RowMajorMatrix<PI>, VCS::Committed);
	type Proof = Proof<'static, PI, PE, VCS::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.log_rows() + self.log_cols()
	}

	fn commit(
		&self,
		poly: &MultilinearPoly<P>,
	) -> Result<(Self::Commitment, Self::Committed), Error> {
		if poly.n_vars() != self.n_vars() {
			return Err(Error::IncorrectPolynomialSize {
				expected: self.n_vars(),
			});
		}

		// This condition is checked by the constructor.
		debug_assert_eq!(self.code.dim() % PI::WIDTH, 0);

		// Dimensions as an intermediate field matrix.
		let n_rows = 1 << self.log_rows;
		let n_cols_enc = self.code.len();

		let mut encoded = vec![PI::default(); n_rows * n_cols_enc / PI::WIDTH];
		let poly_vals_packed =
			PI::try_cast_to_ext(poly.evals()).ok_or_else(|| Error::UnalignedMessage)?;

		transpose::transpose(
			unpack_scalars(poly_vals_packed),
			unpack_scalars_mut(&mut encoded[..n_rows * self.code.dim() / PI::WIDTH]),
			1 << self.code.dim_bits(),
			1 << self.log_rows,
		);

		// TODO: Parallelize
		self.code
			.encode_batch_inplace(&mut encoded, self.log_rows)
			.map_err(|err| Error::EncodeError(Box::new(err)))?;

		let mut digests = vec![H::Digest::default(); n_cols_enc];
		encoded
			.par_chunks_exact(n_rows / PI::WIDTH)
			.map(hash::<_, H>)
			.collect_into_vec(&mut digests);

		let encoded_mat = RowMajorMatrix::new(encoded, n_rows / PI::WIDTH);
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
		let log_degree = log2_strict_usize(FI::DEGREE);
		let log_n_cols = self.code.dim_bits() + log_degree;

		let t = poly;
		let t_prime = t.evaluate_partial_high(&query[log_n_cols..])?;

		challenger.observe_slice(unpack_scalars(t_prime.evals()));
		let merkle_proofs = repeat_with(|| challenger.sample_bits(code_len_bits))
			.take(self.code.n_test_queries())
			.map(|index| {
				let (col_major_mat, ref vcs_committed) = committed;

				let vcs_proof = self
					.vcs
					.prove_batch_opening(vcs_committed, index)
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				let col = col_major_mat.row_slice(index);
				Ok((col.to_vec(), vcs_proof))
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
		// These are all checked during construction, so it is safe to assert as a defensive
		// measure.
		debug_assert_eq!(self.code.dim() % PI::WIDTH, 0);
		debug_assert_eq!((1 << self.log_rows) % P::WIDTH, 0);
		debug_assert_eq!((1 << self.log_rows) % PI::WIDTH, 0);
		debug_assert_eq!(self.code.dim() % PI::WIDTH, 0);
		debug_assert_eq!(self.code.dim() % PE::WIDTH, 0);

		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}

		self.check_proof_shape(&proof)?;

		// Code length is checked to be a power of two in the constructor
		let code_len_bits = log2_strict_usize(self.code.len());
		let log_degree = log2_strict_usize(FI::DEGREE);
		let log_n_cols = self.code.dim_bits() + log_degree;

		let n_rows = 1 << self.log_rows;

		challenger.observe_slice(unpack_scalars(proof.t_prime.evals()));

		// Check evaluation of t' matches the claimed value
		let computed_value = proof
			.t_prime
			.evaluate(&query[..log_n_cols])
			.expect("query is the correct size by check_proof_shape checks");
		if computed_value != value {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// Encode t' into u'
		let mut u_prime = vec![PE::default(); (1 << (code_len_bits + log_degree)) / PE::WIDTH];
		self.encode_ext(proof.t_prime.evals(), &mut u_prime)?;

		// Check vector commitment openings.
		let columns = proof
			.vcs_proofs
			.into_iter()
			.map(|(col, vcs_proof)| {
				let index = challenger.sample_bits(code_len_bits);
				let leaf_digest = hash::<_, H>(&col);

				self.vcs
					.verify_batch_opening(commitment, index, vcs_proof, iter::once(leaf_digest))
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				Ok((index, col))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		// Get the sequence of column tests.
		let column_tests = columns
			.into_iter()
			.flat_map(|(index, col)| {
				// Checked by check_proof_shape
				debug_assert_eq!(col.len(), n_rows / PI::WIDTH);

				// The columns are committed to and provided by the prover as packed vectors of
				// intermediate field elements. We need to transpose them into packed base field
				// elements to perform the consistency checks. Allocate col_transposed as packed
				// intermediate field elements to guarantee alignment.
				let mut col_transposed = vec![PI::default(); n_rows / PI::WIDTH];
				let base_cols = PackedExtensionField::<P>::cast_to_bases_mut(&mut col_transposed);
				transpose_scalars(&col, base_cols).expect(
					"guaranteed safe because of parameter checks in constructor; \
						alignment is guaranteed the cast from a PI slice",
				);

				debug_assert_eq!(base_cols.len(), n_rows / P::WIDTH * FI::DEGREE);

				(0..FI::DEGREE)
					.zip(base_cols.chunks_exact(n_rows / P::WIDTH))
					.map(|(j, col)| {
						let u_prime_i = get_packed_slice(&u_prime, index << log_degree | j);
						(u_prime_i, col.to_vec())
					})
					.collect::<Vec<_>>()
					.into_iter()
			})
			.collect::<Vec<_>>();

		// Batch evaluate all opened columns
		let leaf_evaluations = MultilinearPoly::batch_evaluate(
			column_tests.iter().map(|(_, leaf)| {
				MultilinearPoly::from_values_slice(leaf)
					.expect("leaf is guaranteed power of two length due to check_proof_shape")
			}),
			&query[log_n_cols..],
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

impl<F, P, FI, PI, FE, PE, LC, H, VCS> TensorPCS<P, PI, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F>,
	FI: ExtensionField<F>,
	PI: PackedField<Scalar = FI>,
	FE: ExtensionField<F>,
	PE: PackedField<Scalar = FE>,
	LC: LinearCode<P = PI>,
	H: Hasher<PI>,
	VCS: VectorCommitScheme<H::Digest>,
{
	/// Construct a [`TensorPCS`].
	///
	/// The constructor checks the validity of the type arguments and constructor arguments.
	///
	/// Throws if the linear code block length is not a power of 2.
	/// Throws if the packing width does not divide the code dimension.
	pub fn new(log_rows: usize, code: LC, vcs: VCS) -> Result<Self, Error> {
		if !code.len().is_power_of_two() {
			// This requirement is just to make sampling indices easier. With a little work it
			// could be relaxed, but power-of-two code lengths are more convenient to work with.
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}

		if !FI::DEGREE.is_power_of_two() {
			return Err(Error::ExtensionDegreePowerOfTwoRequired);
		}
		if !FE::DEGREE.is_power_of_two() {
			return Err(Error::ExtensionDegreePowerOfTwoRequired);
		}

		if (1 << log_rows) % P::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideNumberOfRows);
		}
		if (1 << log_rows) % PI::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideNumberOfRows);
		}
		if code.dim() % PI::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideCodeDimension);
		}
		if code.dim() % PE::WIDTH != 0 {
			return Err(Error::PackingWidthMustDivideCodeDimension);
		}

		Ok(Self {
			log_rows,
			code,
			vcs,
			_p_marker: PhantomData,
			_h_marker: PhantomData,
			_ext_marker: PhantomData,
		})
	}

	/// The base-2 logarithm of the number of rows in the committed matrix.
	pub fn log_rows(&self) -> usize {
		self.log_rows
	}

	/// The base-2 logarithm of the number of columns in the pre-encoded matrix.
	pub fn log_cols(&self) -> usize {
		self.code.dim_bits() + log2_strict_usize(FI::DEGREE)
	}
}

// Helper functions for PolyCommitScheme implementation.
impl<F, P, FI, PI, FE, PE, LC, H, VCS> TensorPCS<P, PI, PE, LC, H, VCS>
where
	F: Field,
	P: PackedField<Scalar = F> + Send,
	FI: ExtensionField<P::Scalar>,
	PI: PackedField<Scalar = FI> + PackedExtensionField<FI> + PackedExtensionField<P> + Sync,
	FE: ExtensionField<F> + ExtensionField<FI>,
	PE: PackedField<Scalar = FE> + PackedExtensionField<PI> + PackedExtensionField<FE>,
	LC: LinearCode<P = PI>,
	H: Hasher<PI>,
	H::Digest: Copy + Default + Send,
	VCS: VectorCommitScheme<H::Digest>,
{
	fn check_proof_shape(
		&self,
		proof: &Proof<PI, PE, VCS::Proof>,
	) -> Result<(), VerificationError> {
		let n_rows = 1 << self.log_rows;
		let log_n_cols = self.code.dim_bits() + log2_strict_usize(FI::DEGREE);
		let n_queries = self.code.n_test_queries();

		if proof.vcs_proofs.len() != n_queries {
			return Err(VerificationError::NumberOfOpeningProofs {
				expected: n_queries,
			});
		}
		for (i, (col, _)) in proof.vcs_proofs.iter().enumerate() {
			if col.len() * PI::WIDTH != n_rows {
				return Err(VerificationError::OpenedColumnSize {
					index: i,
					expected: n_rows,
				});
			}
		}

		if proof.t_prime.n_vars() != log_n_cols {
			return Err(VerificationError::PartialEvaluationSize);
		}

		Ok(())
	}

	fn encode_ext(&self, t_prime: &[PE], u_prime: &mut [PE]) -> Result<(), Error> {
		let code_len_bits = log2_strict_usize(self.code.len());
		let log_degree = log2_strict_usize(FI::DEGREE);
		let log_n_cols = self.code.dim_bits() + log_degree;

		assert_eq!(t_prime.len(), (1 << log_n_cols) / PE::WIDTH);
		assert_eq!(u_prime.len(), (1 << (code_len_bits + log_degree)) / PE::WIDTH);

		u_prime[..(1 << log_n_cols) / PE::WIDTH].copy_from_slice(t_prime);

		// View u' as a vector of packed base field elements and transpose into packed intermediate
		// field elements in order to apply the extension encoding.
		{
			// TODO: This requirement is necessary for how we perform the following transpose.
			// It should be relaxed by providing yet another PackedField type as a generic
			// parameter for which this is true.
			assert!(P::WIDTH <= <FE as ExtensionField<F>>::DEGREE);

			let f_view = PackedExtensionField::<P>::cast_to_bases_mut(
				PackedExtensionField::<PI>::cast_to_bases_mut(
					&mut u_prime[..(1 << log_n_cols) / PE::WIDTH],
				),
			);
			f_view
				.par_chunks_exact_mut(FI::DEGREE)
				.try_for_each(|chunk| square_transpose(log_degree, chunk))?;
		}

		// View u' as a vector of packed intermediate field elements and batch encode.
		{
			let fi_view = PackedExtensionField::<PI>::cast_to_bases_mut(u_prime);
			let log_batch_size = log2_strict_usize(<FE as ExtensionField<F>>::DEGREE);
			self.code
				.encode_batch_inplace(fi_view, log_batch_size)
				.map_err(|err| Error::EncodeError(Box::new(err)))?;
		}

		{
			// TODO: This requirement is necessary for how we perform the following transpose.
			// It should be relaxed by providing yet another PackedField type as a generic
			// parameter for which this is true.
			assert!(P::WIDTH <= <FE as ExtensionField<F>>::DEGREE);

			let f_view = PackedExtensionField::<P>::cast_to_bases_mut(
				PackedExtensionField::<PI>::cast_to_bases_mut(u_prime),
			);
			f_view
				.par_chunks_exact_mut(FI::DEGREE)
				.try_for_each(|chunk| square_transpose(log_degree, chunk))?;
		}

		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		challenger::HashChallenger,
		field::{PackedBinaryField128x1b, PackedBinaryField16x8b, PackedBinaryField1x128b},
		reed_solomon::reed_solomon::ReedSolomonCode,
	};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	#[test]
	fn test_simple_commit_prove_verify_without_error() {
		type Packed = PackedBinaryField16x8b;

		let rs_code = ReedSolomonCode::new(5, 2, 12).unwrap();
		let pcs =
			<TensorPCS<Packed, Packed, PackedBinaryField1x128b, _, _, _>>::new_using_groestl_merkle_tree(4, rs_code).unwrap();

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

	#[test]
	fn test_packed_1b_commit_prove_verify_without_error() {
		let rs_code = ReedSolomonCode::new(5, 2, 12).unwrap();
		let pcs = <TensorPCS<
			PackedBinaryField128x1b,
			PackedBinaryField16x8b,
			PackedBinaryField1x128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(8, rs_code)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| PackedBinaryField128x1b::random(&mut rng))
			.take((1 << pcs.n_vars()) / PackedBinaryField128x1b::WIDTH)
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
