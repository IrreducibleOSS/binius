// Copyright 2023 Ulvetanna Inc.

use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_commit::DirectMmcs;
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use p3_merkle_tree::MerkleTreeMmcs;
use std::{
	fmt::{self, Debug, Formatter},
	iter::repeat_with,
	marker::PhantomData,
};

use super::error::{Error, VerificationError};
use crate::{
	field::{
		get_packed_slice, unpack_scalars_mut, BinaryField8b, ExtensionField, PackedExtensionField,
		PackedField,
	},
	hash::{GroestlDigest, GroestlDigestCompression, GroestlHash},
	linear_code::LinearCodeWithExtensionEncoding,
	poly_commit::PolyCommitScheme,
	polynomial::{Error as PolynomialError, MultilinearPoly},
	util::log2,
};

#[derive(Debug)]
pub struct Proof<'a, P, PE, MMCSProof>
where
	PE: PackedField,
{
	pub t_prime: MultilinearPoly<'a, PE>,
	pub mmcs_proofs: Vec<(Vec<Vec<P>>, MMCSProof)>,
}

#[derive(Copy, Clone)]
pub struct TensorPCS<P, PE, LC, MMCS>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
	MMCS: DirectMmcs<P>,
{
	log_rows: usize,
	code: LC,
	mmcs: MMCS,
	_ext_marker: PhantomData<PE>,
}

impl<P, PE, LC, MMCS> Debug for TensorPCS<P, PE, LC, MMCS>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	LC: LinearCodeWithExtensionEncoding<P = P> + Debug,
	MMCS: DirectMmcs<P>,
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		f.debug_struct("PolyCommitScheme")
			.field("log_rows", &self.log_rows)
			.field("code", &self.code)
			.finish()
	}
}

impl<P, PE, LC>
	TensorPCS<P, PE, LC, MerkleTreeMmcs<P, GroestlDigest, GroestlHash, GroestlDigestCompression>>
where
	P: PackedExtensionField<BinaryField8b>,
	P::Scalar: ExtensionField<BinaryField8b>,
	PE: PackedExtensionField<P>,
	PE::Scalar: ExtensionField<P::Scalar>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
{
	pub fn new_using_groestl(log_rows: usize, code: LC) -> Self {
		Self {
			log_rows,
			code,
			mmcs: MerkleTreeMmcs::new(GroestlHash, GroestlDigestCompression),
			_ext_marker: PhantomData,
		}
	}
}

impl<P, FE, PE, LC, MMCS> PolyCommitScheme<P, FE> for TensorPCS<P, PE, LC, MMCS>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
	PE: PackedField<Scalar = FE>
		+ PackedExtensionField<P>
		+ PackedExtensionField<FE>
		+ PackedExtensionField<P::Scalar>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
	LC::EncodeError: 'static,
	MMCS: DirectMmcs<P>,
{
	type Commitment = MMCS::Commitment;
	type Committed = MMCS::ProverData;
	type Proof = Proof<'static, P, FE, MMCS::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.n_vars()
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

		// TODO: Check this during construction. Should be an invariant of the struct.
		assert!(self.code.dim() >= P::WIDTH);

		let n_cols_packed = self.code.len() / P::WIDTH;
		let mut encoded = RowMajorMatrix::new(
			vec![P::default(); (1 << self.log_rows) * n_cols_packed],
			n_cols_packed,
		);

		// TODO: Create better interface/impl for batch encoding & batch NTTs
		poly.iter_subpolynomials(self.code.dim_bits())?
			.zip(encoded.rows_mut())
			.try_for_each(|(subpoly, codeword)| {
				let msg = subpoly.evals();
				codeword[..msg.len()].copy_from_slice(msg);
				self.code.encode_inplace(codeword)
			})
			.map_err(|err| Error::EncodeError(Box::new(err)))?;

		Ok(self.mmcs.commit_matrix(encoded))
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
		if !self.code.len().is_power_of_two() {
			// This requirement is just to make sampling indices easier. With a little work it
			// could be relaxed, but power-of-two code lengths are more convenient to work with.
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		let code_len_bits = log2(self.code.len());

		let t = poly;
		let t_prime = t.evaluate_partial(&query[self.code.dim()..])?;

		challenger.observe_slice(t_prime.evals());
		let merkle_proofs = repeat_with(|| challenger.sample_bits(code_len_bits))
			.take(self.code.n_test_queries())
			.map(|index| self.mmcs.open_batch(index, committed))
			.collect();

		Ok(Proof {
			t_prime,
			mmcs_proofs: merkle_proofs,
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
		proof: &Self::Proof,
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

		let dimensions = Dimensions {
			width: self.code.len() / P::WIDTH,
			height: 1 << self.log_rows,
		};

		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}

		self.check_proof_shape(proof)?;

		if !self.code.len().is_power_of_two() {
			// This requirement is just to make sampling indices easier. With a little work it
			// could be relaxed, but power-of-two code lengths are more convenient to work with.
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		let code_len_bits = log2(self.code.len());

		challenger.observe_slice(proof.t_prime.evals());

		let computed_value = proof
			.t_prime
			.evaluate(&query[..self.code.dim_bits()])
			.expect("query is the correct size by check_proof_shape checks");
		if computed_value != value {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		let mut u_prime = vec![PE::default(); self.code.len() / PE::WIDTH];
		unpack_scalars_mut::<_, FE>(&mut u_prime[..self.code.dim() / PE::WIDTH])
			.copy_from_slice(proof.t_prime.evals());

		self.code
			.encode_extension_inplace(&mut u_prime)
			.map_err(|err| Error::EncodeError(Box::new(err)))?;

		let column_tests = (0..self.code.n_test_queries())
			.map(|i| {
				let index = challenger.sample_bits(code_len_bits);

				// Indices guaranteed in range by check_proof_shape
				let (leaves, branch) = &proof.mmcs_proofs[i];

				self.mmcs
					.verify_batch(commitment, &[dimensions], index, leaves, branch)
					.map_err(|_| VerificationError::MerkleProof)?;

				Ok((get_packed_slice(&u_prime, index), &leaves[0]))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		// TODO: Check evaluations of leaves match t_prime
		let leaf_evaluations = MultilinearPoly::batch_evaluate(
			column_tests.iter().map(|(_, leaf)| {
				MultilinearPoly::from_values_slice(leaf)
					.expect("leaf is guaranteed power of two length due to check_proof_shape")
			}),
			&query[self.code.dim_bits()..],
		);

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

impl<P, PE, LC, MMCS> TensorPCS<P, PE, LC, MMCS>
where
	P: PackedField,
	PE: PackedField,
	PE::Scalar: ExtensionField<P::Scalar>,
	LC: LinearCodeWithExtensionEncoding<P = P>,
	LC::EncodeError: 'static,
	MMCS: DirectMmcs<P>,
{
	fn n_vars(&self) -> usize {
		self.log_rows + self.code.dim_bits()
	}

	fn check_proof_shape(
		&self,
		proof: &Proof<P, PE::Scalar, MMCS::Proof>,
	) -> Result<(), VerificationError> {
		assert!(self.code.dim() >= P::WIDTH);

		let dimensions = Dimensions {
			width: self.code.len() / P::WIDTH,
			height: 1 << self.log_rows,
		};

		if proof.mmcs_proofs.len() != self.code.n_test_queries() {
			return Err(VerificationError::NumberOfMerkleProofs);
		}
		for (leaves, _) in proof.mmcs_proofs.iter() {
			if leaves.len() != 1 {
				return Err(VerificationError::NumberOfMerkleProofs);
			}
			if leaves[0].len() != dimensions.height {
				return Err(VerificationError::MerkleLeafSize);
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
		field::{PackedBinaryField16x8b, PackedBinaryField1x128b},
		reed_solomon::reed_solomon::ReedSolomonCode,
	};
	use rand::{rngs::StdRng, SeedableRng};
	use std::iter::repeat_with;

	#[test]
	fn test_commit_without_error() {
		type Packed = PackedBinaryField16x8b;

		let rs_code = ReedSolomonCode::new(5, 2).unwrap();
		let pcs = <TensorPCS<Packed, PackedBinaryField1x128b, _, _>>::new_using_groestl(3, rs_code);

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| Packed::random(&mut rng))
			.take((1 << pcs.n_vars()) / Packed::WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearPoly::from_values(evals).unwrap();

		let _ = pcs.commit(&poly).unwrap();
	}
}
