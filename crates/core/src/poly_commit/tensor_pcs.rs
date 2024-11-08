// Copyright 2023-2024 Irreducible Inc.

use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	linear_code::LinearCode,
	merkle_tree::{MerkleTreeVCS, VectorCommitScheme},
	poly_commit::PolyCommitScheme,
	polynomial::Error as PolynomialError,
	reed_solomon::reed_solomon::ReedSolomonCode,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::{get_packed_slice, iter_packed_slice},
	square_transpose, transpose_scalars,
	underlier::Divisible,
	util::inner_product_unchecked,
	BinaryField, BinaryField8b, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable, TowerField,
};
use binius_hal::{ComputationBackend, ComputationBackendExt};
use binius_hash::{
	GroestlDigest, GroestlDigestCompression, GroestlHasher, HashDigest, HasherDigest,
};
use binius_math::MultilinearExtension;
use binius_ntt::{NTTOptions, ThreadingSettings};
use binius_utils::bail;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use rayon::prelude::*;
use std::{cmp::min, iter::repeat_with, marker::PhantomData, mem, ops::Deref};
use tracing::instrument;

/// Creates a new multilinear from a batch of multilinears and a mixing challenge
///
/// REQUIRES:
///     All inputted multilinear polynomials have $\mu := \text{n_vars}$ variables
///     t_primes.len() == mixing_coeffs.len()
/// ENSURES:
///     Given a batch of $m$ multilinear polynomials $t_i$'s, and $n$ mixing coeffs $c_i$,
///     this function computes the multilinear polynomial $t$ such that
///     $\forall v \in \{0, 1\}^{\mu}$, $t(v) = \sum_{i=0}^{n-1} c_i * t_i(v)$
fn mix_t_primes<F, P>(
	n_vars: usize,
	t_primes: &[MultilinearExtension<P>],
	mixing_coeffs: &[F],
) -> Result<MultilinearExtension<P>, Error>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	for t_prime_i in t_primes {
		if t_prime_i.n_vars() != n_vars {
			bail!(Error::IncorrectPolynomialSize { expected: n_vars });
		}
	}

	let mixed_evals = (0..(1 << n_vars) / P::WIDTH)
		.into_par_iter()
		.map(|i| {
			t_primes
				.iter()
				.map(|t_prime| t_prime.evals()[i])
				.zip(mixing_coeffs.iter().copied())
				.map(|(t_prime_i, coeff)| t_prime_i * coeff)
				.sum()
		})
		.collect::<Vec<_>>();

	let mixed_t_prime = MultilinearExtension::from_values(mixed_evals)?;
	Ok(mixed_t_prime)
}

/// Type alias for a collection of VCS proofs for a batch of polynomials.
pub type VCSProofs<P, VCSProof> = Vec<(Vec<Vec<P>>, VCSProof)>;

/// Evaluation proof data for the `TensorPCS` polynomial commitment scheme.
///
/// # Type Parameters
///
/// * `PI`: The packed intermediate field type.
/// * `PE`: The packed extension field type.
/// * `VCSProof`: The vector commitment scheme proof type.
#[derive(Debug)]
pub struct Proof<U, FI, FE, VCSProof>
where
	U: PackScalar<FI> + PackScalar<FE>,
	FI: Field,
	FE: Field,
{
	/// Number of distinct multilinear polynomials in the batch opening proof
	pub n_polys: usize,
	/// Represents a mixing of individual polynomial t_primes
	///
	/// Let $n$ denote n_polys. Define $l = \lceil\log_2(n)\rceil$.
	/// Let $\alpha_0, \ldots, \alpha_{l-1}$ be the sampled mixing challenges.
	/// Then $c := \otimes_{i=0}^{l-1} (1 - \alpha_i, \alpha_i)$ are the $2^l$ mixing coefficients,
	/// denoting the $i$-th coefficient by $c_i$.
	/// Let $t'_i$ denote the $t'$ for the $i$-th polynomial in the batch opening proof.
	/// This value represents the multilinear polynomial such that $\forall v \in \{0, 1\}^{\mu}$,
	/// $v \rightarrow \sum_{i=0}^{n-1} c_i * t'_i(v)$
	pub mixed_t_prime: MultilinearExtension<PackedType<U, FE>>,
	/// Opening proofs for chosen columns of the encoded matrices
	///
	/// Let $j_1, \ldots, j_k$ be the indices of the columns that are opened.
	/// The ith element is a tuple of:
	/// * A vector (size=n_polys) of the $j_i$th columns (one from each polynomial's encoded matrix)
	/// * A proof that these columns are consistent with the vector commitment
	pub vcs_proofs: VCSProofs<PackedType<U, FI>, VCSProof>,
}

/// The multilinear polynomial commitment scheme specified in [DP23].
///
/// # Type Parameters
///
/// * `P`: The base field type of committed elements.
/// * `PA`: The field type of the encoding alphabet.
/// * `PI`: The intermediate field type that base field elements are packed into.
/// * `PE`: The extension field type used for cryptographic challenges.
///
/// [DP23]: https://eprint.iacr.org/2023/630
#[derive(Debug, Copy, Clone)]
pub struct TensorPCS<U, F, FA, FI, FE, LC, H, VCS>
where
	U: PackScalar<F> + PackScalar<FA> + PackScalar<FI> + PackScalar<FE>,
	F: Field,
	FA: Field,
	FI: Field,
	FE: Field,
	LC: LinearCode<P = PackedType<U, FA>>,
	H: HashDigest<PackedType<U, FI>>,
	VCS: VectorCommitScheme<H::Digest>,
{
	log_rows: usize,
	n_test_queries: usize,
	code: LC,
	vcs: VCS,
	_u_marker: PhantomData<U>,
	_f_marker: PhantomData<F>,
	_fa_marker: PhantomData<FA>,
	_fi_marker: PhantomData<FI>,
	_h_marker: PhantomData<H>,
	_ext_marker: PhantomData<FE>,
}

type GroestlMerkleTreeVCS = MerkleTreeVCS<
	GroestlDigest<BinaryField8b>,
	GroestlDigest<BinaryField8b>,
	GroestlHasher<GroestlDigest<BinaryField8b>>,
	GroestlDigestCompression<BinaryField8b>,
>;

impl<U, F, FA, FI, FE, LC>
	TensorPCS<
		U,
		F,
		FA,
		FI,
		FE,
		LC,
		HasherDigest<PackedType<U, FI>, GroestlHasher<PackedType<U, FI>>>,
		GroestlMerkleTreeVCS,
	>
where
	U: PackScalar<F>
		+ PackScalar<FA>
		+ PackScalar<FI>
		+ PackScalar<FE>
		+ PackScalar<BinaryField8b>
		+ binius_field::underlier::Divisible<u8>,
	F: Field,
	FA: Field,
	FI: Field + ExtensionField<BinaryField8b> + ExtensionField<F> + Sync,
	FE: BinaryField + ExtensionField<F>,
	LC: LinearCode<P = PackedType<U, FA>>,
{
	pub fn new_using_groestl_merkle_tree(
		log_rows: usize,
		code: LC,
		n_test_queries: usize,
	) -> Result<Self, Error> {
		// Check power of two length because MerkleTreeVCS requires it
		if !code.len().is_power_of_two() {
			return Err(Error::CodeLengthPowerOfTwoRequired);
		}
		let log_len = log2_strict_usize(code.len());
		// Compute optimal cap height
		let cap_height = calculate_optimal_cap_height(log_len, n_test_queries);
		Self::new(
			log_rows,
			code,
			n_test_queries,
			MerkleTreeVCS::new(
				log_len,
				cap_height,
				GroestlDigestCompression::<BinaryField8b>::default(),
			),
		)
	}
}

impl<U, F, FA, FI, FE, LC, H, VCS> PolyCommitScheme<PackedType<U, F>, FE>
	for TensorPCS<U, F, FA, FI, FE, LC, H, VCS>
where
	U: PackScalar<F>
		+ PackScalar<FA>
		+ PackScalar<FI, Packed: PackedFieldIndexable>
		+ PackScalar<FE, Packed: PackedFieldIndexable>,
	F: Field,
	FA: Field,
	FI: ExtensionField<F> + ExtensionField<FA>,
	FE: ExtensionField<F> + ExtensionField<FI> + TowerField,
	LC: LinearCode<P = PackedType<U, FA>> + Sync,
	H: HashDigest<PackedType<U, FI>> + Sync,
	H::Digest: Copy + Default + Send,
	VCS: VectorCommitScheme<H::Digest> + Sync,
{
	type Commitment = VCS::Commitment;
	type Committed = (Vec<RowMajorMatrix<PackedType<U, FI>>>, VCS::Committed);
	type Proof = Proof<U, FI, FE, VCS::Proof>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.log_rows() + self.log_cols()
	}

	#[instrument(skip_all, name = "tensor_pcs::commit", level = "debug")]
	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<PackedType<U, F>, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Error>
	where
		Data: Deref<Target = [PackedType<U, F>]> + Send + Sync,
	{
		for poly in polys {
			if poly.n_vars() != self.n_vars() {
				bail!(Error::IncorrectPolynomialSize {
					expected: self.n_vars(),
				});
			}
		}

		// These conditions are checked by the constructor, so are safe to assert defensively
		let pi_width = PackedType::<U, FI>::WIDTH;
		debug_assert_eq!(self.code.dim() % pi_width, 0);

		// Dimensions as an intermediate field matrix.
		let n_rows = 1 << self.log_rows;
		let n_cols_enc = self.code.len();

		let results = polys
			.par_iter()
			.map(|poly| -> Result<_, Error> {
				let mut encoded =
					vec![PackedType::<U, FI>::default(); n_rows * n_cols_enc / pi_width];
				let poly_vals_packed =
					<PackedType<U, FI> as PackedExtension<F>>::cast_exts(poly.evals());

				transpose::transpose(
					PackedType::<U, FI>::unpack_scalars(poly_vals_packed),
					PackedType::<U, FI>::unpack_scalars_mut(
						&mut encoded[..n_rows * self.code.dim() / pi_width],
					),
					1 << self.code.dim_bits(),
					1 << self.log_rows,
				);

				self.code
					.encode_batch_inplace(
						<PackedType<U, FI> as PackedExtension<FA>>::cast_bases_mut(&mut encoded),
						self.log_rows + log2_strict_usize(<FI as ExtensionField<FA>>::DEGREE),
					)
					.map_err(|err| Error::EncodeError(Box::new(err)))?;

				let mut digests = vec![H::Digest::default(); n_cols_enc];
				encoded
					.par_chunks_exact(n_rows / pi_width)
					.map(H::hash)
					.collect_into_vec(&mut digests);

				let encoded_mat = RowMajorMatrix::new(encoded, n_rows / pi_width);

				Ok((digests, encoded_mat))
			})
			.collect::<Vec<_>>();

		let mut encoded_mats = Vec::with_capacity(polys.len());
		let mut all_digests = Vec::with_capacity(polys.len());
		for result in results {
			let (digests, encoded_mat) = result?;
			all_digests.push(digests);
			encoded_mats.push(encoded_mat);
		}

		let (commitment, vcs_committed) = self
			.vcs
			.commit_batch(&all_digests)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;
		Ok((commitment, (encoded_mats, vcs_committed)))
	}

	/// Generate an evaluation proof at a *random* challenge point.
	///
	/// Follows the notation from Construction 4.6 in [DP23].
	///
	/// Precondition: The queried point must already be observed by the challenger.
	///
	/// [DP23]: https://eprint.iacr.org/2023/630
	#[instrument(skip_all, name = "tensor_pcs::prove_evaluation", level = "debug")]
	fn prove_evaluation<Data, CH, Backend>(
		&self,
		challenger: &mut CH,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<PackedType<U, F>, Data>],
		query: &[FE],
		backend: &Backend,
	) -> Result<Self::Proof, Error>
	where
		Data: Deref<Target = [PackedType<U, F>]> + Send + Sync,
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		let n_polys = polys.len();
		let n_challenges = log2_ceil_usize(n_polys);
		let mixing_challenges = challenger.sample_vec(n_challenges);
		let mixing_coefficients =
			&backend.tensor_product_full_query(&mixing_challenges)?[..n_polys];

		let (col_major_mats, ref vcs_committed) = committed;
		if col_major_mats.len() != n_polys {
			bail!(Error::NumBatchedMismatchError {
				err_str: format!("In prove_evaluation: number of polynomials {} must match number of committed matrices {}", n_polys, col_major_mats.len()),
			});
		}

		if query.len() != self.n_vars() {
			bail!(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			});
		}

		let code_len_bits = log2_strict_usize(self.code.len());
		let log_block_size = log2_strict_usize(<FI as ExtensionField<F>>::DEGREE);
		let log_n_cols = self.code.dim_bits() + log_block_size;

		let partial_query = backend.multilinear_query(&query[log_n_cols..])?;
		let ts = polys;
		let t_primes = ts
			.iter()
			.map(|t| t.evaluate_partial_high(&partial_query))
			.collect::<Result<Vec<_>, _>>()?;
		let t_prime = mix_t_primes(log_n_cols, &t_primes, mixing_coefficients)?;

		challenger.observe_slice(PackedType::<U, FE>::unpack_scalars(t_prime.evals()));
		let merkle_proofs = repeat_with(|| challenger.sample_bits(code_len_bits))
			.take(self.n_test_queries)
			.map(|index| {
				let vcs_proof = self
					.vcs
					.prove_batch_opening(vcs_committed, index)
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				let cols: Vec<_> = col_major_mats
					.iter()
					.map(|col_major_mat| col_major_mat.row_slice(index).to_vec())
					.collect();

				Ok((cols, vcs_proof))
			})
			.collect::<Result<_, Error>>()?;

		Ok(Proof {
			n_polys,
			mixed_t_prime: t_prime,
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
	#[instrument(skip_all, name = "tensor_pcs::verify_evaluation", level = "debug")]
	fn verify_evaluation<CH, Backend>(
		&self,
		challenger: &mut CH,
		commitment: &Self::Commitment,
		query: &[FE],
		proof: Self::Proof,
		values: &[FE],
		backend: &Backend,
	) -> Result<(), Error>
	where
		CH: CanObserve<FE> + CanSample<FE> + CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		// These are all checked during construction, so it is safe to assert as a defensive
		// measure.
		let p_width = PackedType::<U, F>::WIDTH;
		let pi_width = PackedType::<U, FI>::WIDTH;
		let pe_width = PackedType::<U, FE>::WIDTH;
		debug_assert_eq!(self.code.dim() % pi_width, 0);
		debug_assert_eq!((1 << self.log_rows) % pi_width, 0);
		debug_assert_eq!(self.code.dim() % pi_width, 0);
		debug_assert_eq!(self.code.dim() % pe_width, 0);

		if values.len() != proof.n_polys {
			bail!(Error::NumBatchedMismatchError {
				err_str:
					format!("In verify_evaluation: proof number of polynomials {} must match number of opened values {}", proof.n_polys, values.len()),
			});
		}

		let n_challenges = log2_ceil_usize(proof.n_polys);
		let mixing_challenges = challenger.sample_vec(n_challenges);
		let mixing_coefficients = &backend
			.tensor_product_full_query::<PackedType<U, FE>>(&mixing_challenges)?[..proof.n_polys];
		let value =
			inner_product_unchecked(values.iter().copied(), iter_packed_slice(mixing_coefficients));

		if query.len() != self.n_vars() {
			bail!(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			});
		}

		self.check_proof_shape(&proof)?;

		// Code length is checked to be a power of two in the constructor
		let code_len_bits = log2_strict_usize(self.code.len());
		let block_size = <FI as ExtensionField<F>>::DEGREE;
		let log_block_size = log2_strict_usize(block_size);
		let log_n_cols = self.code.dim_bits() + log_block_size;

		let n_rows = 1 << self.log_rows;

		challenger.observe_slice(<PackedType<U, FE>>::unpack_scalars(proof.mixed_t_prime.evals()));

		// Check evaluation of t' matches the claimed value
		let multilin_query =
			backend.multilinear_query::<PackedType<U, FE>>(&query[..log_n_cols])?;
		let computed_value = proof
			.mixed_t_prime
			.evaluate(&multilin_query)
			.expect("query is the correct size by check_proof_shape checks");
		if computed_value != value {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// Encode t' into u'
		let mut u_prime = vec![
			PackedType::<U, FE>::default();
			(1 << (code_len_bits + log_block_size)) / pe_width
		];
		self.encode_ext(proof.mixed_t_prime.evals(), &mut u_prime)?;

		// Check vector commitment openings.
		let columns = proof
			.vcs_proofs
			.into_iter()
			.map(|(cols, vcs_proof)| {
				let index = challenger.sample_bits(code_len_bits);

				let leaf_digests = cols.iter().map(H::hash);

				self.vcs
					.verify_batch_opening(commitment, index, vcs_proof, leaf_digests)
					.map_err(|err| Error::VectorCommit(Box::new(err)))?;

				Ok((index, cols))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		// Get the sequence of column tests.
		let column_tests = columns
			.into_iter()
			.flat_map(|(index, cols)| {
				let mut batched_column_test = (0..block_size)
					.map(|j| {
						let u_prime_i = get_packed_slice(&u_prime, index << log_block_size | j);
						let base_cols = Vec::with_capacity(proof.n_polys);
						(u_prime_i, base_cols)
					})
					.collect::<Vec<_>>();

				for mut col in cols {
					// Checked by check_proof_shape
					debug_assert_eq!(col.len(), n_rows / pi_width);

					// Pad column with empty elements to accommodate the following scalar transpose.
					if n_rows < p_width {
						col.resize(p_width, Default::default());
					}

					// The columns are committed to and provided by the prover as packed vectors of
					// intermediate field elements. We need to transpose them into packed base field
					// elements to perform the consistency checks. Allocate col_transposed as packed
					// intermediate field elements to guarantee alignment.
					let mut col_transposed = col.clone();
					let base_cols = <PackedType<U, FI> as PackedExtension<F>>::cast_bases_mut(
						&mut col_transposed,
					);
					transpose_scalars(&col, base_cols).expect(
						"guaranteed safe because of parameter checks in constructor; \
							alignment is guaranteed the cast from a PI slice",
					);

					for (j, col) in base_cols
						.chunks_exact(base_cols.len() / block_size)
						.enumerate()
					{
						// Trim off padding rows by converting from packed vec to scalar vec.
						let scalars_col = iter_packed_slice(col).take(n_rows).collect::<Vec<_>>();
						batched_column_test[j].1.push(scalars_col);
					}
				}
				batched_column_test
			})
			.collect::<Vec<_>>();

		// Batch evaluate all opened columns
		let multilin_query =
			backend.multilinear_query::<PackedType<U, FE>>(&query[log_n_cols..])?;
		let incorrect_evaluation = column_tests
			.par_iter()
			.map(|(expected, leaves)| {
				let actual_evals =
					leaves
						.par_iter()
						.map(|leaf| {
							MultilinearExtension::from_values_slice(leaf)
						.expect("leaf is guaranteed power of two length due to check_proof_shape")
						.evaluate(&multilin_query)
						.expect("failed to evaluate")
						})
						.collect::<Vec<_>>();
				(expected, actual_evals)
			})
			.any(|(expected_result, unmixed_actual_results)| {
				// Check that opened column evaluations match u'
				let actual_result = inner_product_unchecked(
					unmixed_actual_results.into_iter(),
					iter_packed_slice(mixing_coefficients),
				);
				actual_result != *expected_result
			});

		if incorrect_evaluation {
			return Err(VerificationError::IncorrectPartialEvaluation.into());
		} else {
			Ok(())
		}
	}

	fn proof_size(&self, n_polys: usize) -> usize {
		let pe_width = PackedType::<U, FE>::WIDTH;
		let pi_width = PackedType::<U, FI>::WIDTH;
		let t_prime_size = (mem::size_of::<U>() << self.log_cols()) / pe_width;
		let column_size = (mem::size_of::<U>() << self.log_rows()) / pi_width;
		t_prime_size + (n_polys * column_size + self.vcs.proof_size(n_polys)) * self.n_test_queries
	}
}

impl<U, F, FA, FI, FE, LC, H, VCS> TensorPCS<U, F, FA, FI, FE, LC, H, VCS>
where
	U: PackScalar<F> + PackScalar<FA> + PackScalar<FI> + PackScalar<FE>,
	F: Field,
	FA: Field,
	FI: ExtensionField<F>,
	FE: ExtensionField<F>,
	LC: LinearCode<P = PackedType<U, FA>>,
	H: HashDigest<PackedType<U, FI>>,
	VCS: VectorCommitScheme<H::Digest>,
{
	/// The base-2 logarithm of the number of rows in the committed matrix.
	pub fn log_rows(&self) -> usize {
		self.log_rows
	}

	/// The base-2 logarithm of the number of columns in the pre-encoded matrix.
	pub fn log_cols(&self) -> usize {
		self.code.dim_bits() + log2_strict_usize(FI::DEGREE)
	}
}

impl<U, F, FA, FI, FE, LC, H, VCS> TensorPCS<U, F, FA, FI, FE, LC, H, VCS>
where
	U: PackScalar<F> + PackScalar<FA> + PackScalar<FI> + PackScalar<FE>,
	F: Field,
	FA: Field,
	FI: ExtensionField<F>,
	FE: ExtensionField<F> + BinaryField,
	LC: LinearCode<P = PackedType<U, FA>>,
	H: HashDigest<PackedType<U, FI>>,
	VCS: VectorCommitScheme<H::Digest>,
{
	/// Construct a [`TensorPCS`].
	///
	/// The constructor checks the validity of the type arguments and constructor arguments.
	///
	/// Throws if the linear code block length is not a power of 2.
	/// Throws if the packing width does not divide the code dimension.
	pub fn new(log_rows: usize, code: LC, n_test_queries: usize, vcs: VCS) -> Result<Self, Error> {
		if !code.len().is_power_of_two() {
			// This requirement is just to make sampling indices easier. With a little work it
			// could be relaxed, but power-of-two code lengths are more convenient to work with.
			bail!(Error::CodeLengthPowerOfTwoRequired);
		}

		let fi_degree = <FI as ExtensionField<F>>::DEGREE;
		let fe_degree = <FE as ExtensionField<F>>::DEGREE;
		if !fi_degree.is_power_of_two() {
			bail!(Error::ExtensionDegreePowerOfTwoRequired);
		}
		if !fe_degree.is_power_of_two() {
			bail!(Error::ExtensionDegreePowerOfTwoRequired);
		}

		let pi_width = PackedType::<U, FI>::WIDTH;
		let pe_width = PackedType::<U, FE>::WIDTH;
		if (1 << log_rows) % pi_width != 0 {
			bail!(Error::PackingWidthMustDivideNumberOfRows);
		}
		if code.dim() % pi_width != 0 {
			bail!(Error::PackingWidthMustDivideCodeDimension);
		}
		if code.dim() % pe_width != 0 {
			bail!(Error::PackingWidthMustDivideCodeDimension);
		}

		Ok(Self {
			log_rows,
			n_test_queries,
			code,
			vcs,
			_u_marker: PhantomData,
			_f_marker: PhantomData,
			_fa_marker: PhantomData,
			_fi_marker: PhantomData,
			_h_marker: PhantomData,
			_ext_marker: PhantomData,
		})
	}
}

// Helper functions for PolyCommitScheme implementation.
impl<U, F, FA, FI, FE, LC, H, VCS> TensorPCS<U, F, FA, FI, FE, LC, H, VCS>
where
	U: PackScalar<F, Packed: Send>
		+ PackScalar<FA>
		+ PackScalar<FI, Packed: PackedFieldIndexable>
		+ PackScalar<FE, Packed: PackedFieldIndexable + Sync>,
	F: Field,
	FA: Field,
	FI: ExtensionField<F> + ExtensionField<FA>,
	FE: ExtensionField<F> + ExtensionField<FI>,
	LC: LinearCode<P = PackedType<U, FA>>,
	H: HashDigest<PackedType<U, FI>>,
	H::Digest: Copy + Default + Send,
	VCS: VectorCommitScheme<H::Digest>,
{
	fn check_proof_shape(&self, proof: &Proof<U, FI, FE, VCS::Proof>) -> Result<(), Error> {
		let n_rows = 1 << self.log_rows;
		let log_block_size = log2_strict_usize(<FI as ExtensionField<F>>::DEGREE);
		let log_n_cols = self.code.dim_bits() + log_block_size;
		let n_queries = self.n_test_queries;

		if proof.vcs_proofs.len() != n_queries {
			return Err(VerificationError::NumberOfOpeningProofs {
				expected: n_queries,
			}
			.into());
		}
		for (col_idx, (polys_col, _)) in proof.vcs_proofs.iter().enumerate() {
			if polys_col.len() != proof.n_polys {
				bail!(Error::NumBatchedMismatchError {
					err_str: format!(
						"Expected {} polynomials, but VCS proof at col_idx {} found {} polynomials instead",
						proof.n_polys,
						col_idx,
						polys_col.len()
					),
				});
			}

			for (poly_idx, poly_col) in polys_col.iter().enumerate() {
				let pi_width = PackedType::<U, FI>::WIDTH;
				if poly_col.len() * pi_width != n_rows {
					return Err(VerificationError::OpenedColumnSize {
						col_index: col_idx,
						poly_index: poly_idx,
						expected: n_rows,
						actual: poly_col.len() * pi_width,
					}
					.into());
				}
			}
		}

		if proof.mixed_t_prime.n_vars() != log_n_cols {
			return Err(VerificationError::PartialEvaluationSize.into());
		}

		Ok(())
	}

	#[instrument(skip_all, level = "debug")]
	fn encode_ext(
		&self,
		t_prime: &[PackedType<U, FE>],
		u_prime: &mut [PackedType<U, FE>],
	) -> Result<(), Error> {
		let code_len_bits = log2_strict_usize(self.code.len());
		let block_size = <FI as ExtensionField<F>>::DEGREE;
		let log_block_size = log2_strict_usize(block_size);
		let log_n_cols = self.code.dim_bits() + log_block_size;

		let pe_width = PackedType::<U, FE>::WIDTH;
		let p_width = PackedType::<U, F>::WIDTH;
		assert_eq!(t_prime.len(), (1 << log_n_cols) / pe_width);
		assert_eq!(u_prime.len(), (1 << (code_len_bits + log_block_size)) / pe_width);

		u_prime[..(1 << log_n_cols) / pe_width].copy_from_slice(t_prime);

		// View u' as a vector of packed base field elements and transpose into packed intermediate
		// field elements in order to apply the extension encoding.
		if log_block_size > 0 {
			// TODO: This requirement is necessary for how we perform the following transpose.
			// It should be relaxed by providing yet another PackedField type as a generic
			// parameter for which this is true.
			assert!(p_width <= <FE as ExtensionField<F>>::DEGREE);

			let f_view = <PackedType<U, FI> as PackedExtension<F>>::cast_bases_mut(
				<PackedType<U, FE> as PackedExtension<FI>>::cast_bases_mut(
					&mut u_prime[..(1 << log_n_cols) / pe_width],
				),
			);
			f_view
				.par_chunks_exact_mut(block_size)
				.try_for_each(|chunk| square_transpose(log_block_size, chunk))?;
		}

		// View u' as a vector of packed intermediate field elements and batch encode.
		{
			let fi_view = <PackedType<U, FE> as PackedExtension<FI>>::cast_bases_mut(u_prime);
			let log_batch_size = log2_strict_usize(<FE as ExtensionField<F>>::DEGREE);
			self.code
				.encode_batch_inplace(
					<PackedType<U, FI> as PackedExtension<FA>>::cast_bases_mut(fi_view),
					log_batch_size + log2_strict_usize(<FI as ExtensionField<FA>>::DEGREE),
				)
				.map_err(|err| Error::EncodeError(Box::new(err)))?;
		}

		if log_block_size > 0 {
			// TODO: This requirement is necessary for how we perform the following transpose.
			// It should be relaxed by providing yet another PackedField type as a generic
			// parameter for which this is true.
			assert!(p_width <= <FE as ExtensionField<F>>::DEGREE);

			let f_view = <PackedType<U, FI> as PackedExtension<F>>::cast_bases_mut(
				<PackedType<U, FE> as PackedExtension<FI>>::cast_bases_mut(u_prime),
			);
			f_view
				.par_chunks_exact_mut(block_size)
				.try_for_each(|chunk| square_transpose(log_block_size, chunk))?;
		}

		Ok(())
	}
}

/// The basic multilinear polynomial commitment scheme from [DP23].
///
/// The basic scheme follows Construction 3.7. In this case, the encoding alphabet is a subfield of
/// the polynomial's coefficient field.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
pub type BasicTensorPCS<U, F, FA, FE, LC, H, VCS> = TensorPCS<U, F, FA, F, FE, LC, H, VCS>;

/// The multilinear polynomial commitment scheme from [DP23] with block-level encoding.
///
/// The basic scheme follows Construction 3.11. In this case, the encoding alphabet is an extension
/// field of the polynomial's coefficient field.
///
/// [DP23]: <https://eprint.iacr.org/2023/1784>
pub type BlockTensorPCS<U, F, FA, FE, LC, H, VCS> = TensorPCS<U, F, FA, FA, FE, LC, H, VCS>;

pub fn calculate_n_test_queries<F: BinaryField, LC: LinearCode>(
	security_bits: usize,
	log_rows: usize,
	code: &LC,
) -> Result<usize, Error> {
	// Assume we are limited by the non-proximal error term
	let relative_dist = code.min_dist() as f64 / code.len() as f64;
	let non_proximal_per_query_err = 1.0 - (relative_dist / 3.0);
	let mut n_queries =
		(-(security_bits as f64) / non_proximal_per_query_err.log2()).ceil() as usize;
	for _ in 0..10 {
		if calculate_error_bound::<F, _>(log_rows, code, n_queries) >= security_bits {
			return Ok(n_queries);
		}
		n_queries += 1;
	}
	Err(Error::ParameterError)
}

/// Calculates the base-2 log soundness error bound when using general linear codes.
///
/// Returns the number of bits of security achieved with the given parameters. This is computed
/// using the formulae in Section 3.5 of [DP23].
///
/// [DP23]: https://eprint.iacr.org/2023/1784
fn calculate_error_bound<F: BinaryField, LC: LinearCode>(
	log_rows: usize,
	code: &LC,
	n_queries: usize,
) -> usize {
	let e = (code.min_dist() - 1) / 3;
	let relative_dist = code.min_dist() as f64 / code.len() as f64;
	let tensor_batching_err = (2 * log_rows * (e + 1)) as f64 / 2.0_f64.powi(F::N_BITS as i32);
	let non_proximal_err = (1.0 - relative_dist / 3.0).powi(n_queries as i32);
	let proximal_err = (1.0 - 2.0 * relative_dist / 3.0).powi(n_queries as i32);
	let total_err = (tensor_batching_err + non_proximal_err).max(proximal_err);
	-total_err.log2() as usize
}

pub fn calculate_n_test_queries_reed_solomon<F, FE, P>(
	security_bits: usize,
	log_rows: usize,
	code: &ReedSolomonCode<P>,
) -> Result<usize, Error>
where
	F: BinaryField,
	FE: BinaryField + ExtensionField<F>,
	P: PackedField<Scalar = F> + PackedExtension<F> + PackedFieldIndexable,
	P::Scalar: BinaryField,
{
	// Assume we are limited by the non-proximal error term
	let relative_dist = code.min_dist() as f64 / code.len() as f64;
	let non_proximal_per_query_err = 1.0 - (relative_dist / 2.0);
	let mut n_queries =
		(-(security_bits as f64) / non_proximal_per_query_err.log2()).ceil() as usize;
	for _ in 0..10 {
		if calculate_error_bound_reed_solomon::<_, FE, _>(log_rows, code, n_queries)
			>= security_bits
		{
			return Ok(n_queries);
		}
		n_queries += 1;
	}
	Err(Error::ParameterError)
}

/// Calculates the base-2 log soundness error bound when using Reed–Solomon codes.
///
/// Returns the number of bits of security achieved with the given parameters. This is computed
/// using the formulae in Section 3.5 of [DP23]. We use the improved proximity gap result for
/// Reed–Solomon codes, following Remark 3.18 in [DP23].
///
/// [DP23]: https://eprint.iacr.org/2023/1784
fn calculate_error_bound_reed_solomon<F, FE, P>(
	log_rows: usize,
	code: &ReedSolomonCode<P>,
	n_queries: usize,
) -> usize
where
	F: BinaryField,
	FE: BinaryField + ExtensionField<F>,
	P: PackedField<Scalar = F> + PackedExtension<F> + PackedFieldIndexable,
	P::Scalar: BinaryField,
{
	let e = (code.min_dist() - 1) / 2;
	let relative_dist = code.min_dist() as f64 / code.len() as f64;
	let tensor_batching_err = (2 * log_rows * (e + 1)) as f64 / 2.0_f64.powi(FE::N_BITS as i32);
	let non_proximal_err = (1.0 - (relative_dist / 2.0)).powi(n_queries as i32);
	let proximal_err = (1.0 - relative_dist / 2.0).powi(n_queries as i32);
	let total_err = (tensor_batching_err + non_proximal_err).max(proximal_err);
	-total_err.log2() as usize
}

/// Calculate the optimal Merkle-cap height for a given log length and number of test queries.
///
/// It is explained in the concrete soundness subsection of section 5 of
/// FRI-Binius that the optimal cap height is $\lceil \log_2(\gamma)\rceil$, where $\gamma$
/// is the number of test queries. However, there is a maximal cap height of `log_len`, which
/// corresponds to sending all of the leaves of the Merkle tree, so the optimal cap height
/// is the minimum of the two.
fn calculate_optimal_cap_height(log_len: usize, n_test_queries: usize) -> usize {
	let cap_height = (n_test_queries as f64).log2().ceil() as usize;
	min(cap_height, log_len)
}

/// Find the TensorPCS parameterization that optimizes proof size.
///
/// This constructs a TensorPCS using a Reed-Solomon code and a Merkle tree using Groestl.
#[allow(clippy::type_complexity)]
pub fn find_proof_size_optimal_pcs<U, F, FA, FI, FE>(
	security_bits: usize,
	n_vars: usize,
	n_polys: usize,
	log_inv_rate: usize,
	conservative_testing: bool,
) -> Option<
	TensorPCS<
		U,
		F,
		FA,
		FI,
		FE,
		ReedSolomonCode<PackedType<U, FA>>,
		HasherDigest<PackedType<U, FI>, GroestlHasher<PackedType<U, FI>>>,
		GroestlMerkleTreeVCS,
	>,
>
where
	U: PackScalar<F>
		+ PackScalar<FA, Packed: PackedFieldIndexable>
		+ PackScalar<FI, Packed: PackedFieldIndexable>
		+ PackScalar<FE, Packed: PackedFieldIndexable>
		+ PackScalar<BinaryField8b>
		+ Divisible<u8>,
	F: Field,
	FA: BinaryField,
	FI: ExtensionField<F> + ExtensionField<FA> + ExtensionField<BinaryField8b>,
	FE: TowerField + ExtensionField<F> + ExtensionField<FA> + ExtensionField<FI>,
{
	#[derive(Clone, Copy)]
	struct Params {
		log_rows: usize,
		log_dim: usize,
		n_test_queries: usize,
		proof_size: usize,
	}

	let mut best = None;
	let log_degree = log2_strict_usize(<FI as ExtensionField<F>>::DEGREE);

	for log_rows in 0..=(n_vars - log_degree) {
		let log_dim = n_vars - log_rows - log_degree;
		// While we are brute-force checking various PCS instances for proof size, use the default
		// NTTOptions, which makes the RS code the fastest to construct. Later when we return the
		// best PCS, we will reconstruct with faster NTT options that use twiddle precomputation.
		let rs_code = match ReedSolomonCode::new(log_dim, log_inv_rate, NTTOptions::default()) {
			Ok(rs_code) => rs_code,
			Err(_) => continue,
		};

		let n_test_queries_result = if conservative_testing {
			calculate_n_test_queries::<FE, _>(security_bits, log_rows, &rs_code)
		} else {
			calculate_n_test_queries_reed_solomon::<_, FE, _>(security_bits, log_rows, &rs_code)
		};
		let n_test_queries = match n_test_queries_result {
			Ok(n_test_queries) => n_test_queries,
			Err(_) => continue,
		};

		let pcs = match TensorPCS::<U, F, FA, FI, FE, _, _, _>::new_using_groestl_merkle_tree(
			log_rows,
			rs_code,
			n_test_queries,
		) {
			Ok(pcs) => pcs,
			Err(_) => continue,
		};

		let new_params = Params {
			log_rows,
			log_dim,
			n_test_queries,
			proof_size: pcs.proof_size(n_polys),
		};
		match best {
			None => best = Some(new_params),
			Some(current_best) => {
				if new_params.proof_size < current_best.proof_size {
					best = Some(new_params);
				}
			}
		}
	}

	let Params {
		log_rows,
		log_dim,
		n_test_queries,
		proof_size: _,
	} = best?;

	// New create the final PCS result. Instead of saving the PCS that we constructed above,
	// create one with the fastest NTT parameters.
	let ntt_opts = NTTOptions {
		precompute_twiddles: true,
		thread_settings: ThreadingSettings::MultithreadedDefault,
	};
	let rs_code = ReedSolomonCode::new(log_dim, log_inv_rate, ntt_opts).ok()?;
	let pcs = TensorPCS::new_using_groestl_merkle_tree(log_rows, rs_code, n_test_queries).ok()?;

	Some(pcs)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the polynomial must have {expected} variables")]
	IncorrectPolynomialSize { expected: usize },
	#[error("linear encoding error: {0}")]
	EncodeError(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("the polynomial commitment scheme requires a power of two code block length")]
	CodeLengthPowerOfTwoRequired,
	#[error("the polynomial commitment scheme requires a power of two extension degree")]
	ExtensionDegreePowerOfTwoRequired,
	#[error("cannot commit unaligned message")]
	UnalignedMessage,
	#[error("packing width must divide code dimension")]
	PackingWidthMustDivideCodeDimension,
	#[error("packing width must divide the number of rows")]
	PackingWidthMustDivideNumberOfRows,
	#[error("error in batching: {err_str}")]
	NumBatchedMismatchError { err_str: String },
	#[error("cannot calculate parameters satisfying the security target")]
	ParameterError,
	#[error("field error: {0}")]
	Field(#[from] binius_field::Error),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("vector commit error: {0}")]
	VectorCommit(#[source] Box<dyn std::error::Error + Send + Sync>),
	#[error("transpose error: {0}")]
	Transpose(#[from] binius_field::transpose::Error),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
	#[error("HAL error: {0}")]
	HalError(#[from] binius_hal::Error),
	#[error("Math error: {0}")]
	MathError(#[from] binius_math::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("incorrect number of vector commitment opening proofs, expected {expected}")]
	NumberOfOpeningProofs { expected: usize },
	#[error("column opening at poly_index {poly_index}, col_index {col_index} has incorrect size, got {actual} expected {expected}")]
	OpenedColumnSize {
		poly_index: usize,
		col_index: usize,
		expected: usize,
		actual: usize,
	},
	#[error("evaluation is incorrect")]
	IncorrectEvaluation,
	#[error("partial evaluation is incorrect")]
	IncorrectPartialEvaluation,
	#[error("partial evaluation (t') is the wrong size")]
	PartialEvaluationSize,
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		fiat_shamir::HasherChallenger,
		transcript::{AdviceWriter, TranscriptWriter},
	};
	use binius_field::{
		arch::OptimalUnderlier128b, BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b,
		PackedBinaryField128x1b, PackedBinaryField16x8b, PackedBinaryField1x128b,
		PackedBinaryField4x32b,
	};
	use binius_hal::make_portable_backend;
	use groestl_crypto::Groestl256;
	use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

	#[test]
	fn test_simple_commit_prove_verify_without_error() {
		type Packed = PackedBinaryField16x8b;

		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 4, &rs_code)
				.unwrap();
		let pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			BinaryField8b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(4, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| Packed::random(&mut rng))
			.take((1 << pcs.n_vars()) / Packed::WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearExtension::from_values(evals).unwrap();
		let polys = [poly.to_ref()];

		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};

		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();

		let backend = make_portable_backend();
		let multilin_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();
		let value = poly.evaluate(&multilin_query).unwrap();
		let values = vec![value];

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}

	#[test]
	fn test_simple_commit_prove_verify_batch_without_error() {
		type Packed = PackedBinaryField16x8b;

		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 4, &rs_code)
				.unwrap();
		let pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			BinaryField8b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(4, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let batch_size = thread_rng().gen_range(1..=10);
		let polys = repeat_with(|| {
			let evals = repeat_with(|| Packed::random(&mut rng))
				.take((1 << pcs.n_vars()) / Packed::WIDTH)
				.collect::<Vec<_>>();
			MultilinearExtension::from_values(evals).unwrap()
		})
		.take(batch_size)
		.collect::<Vec<_>>();
		let backend = make_portable_backend();

		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();
		let multilin_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();

		let values = polys
			.iter()
			.map(|poly| poly.evaluate(multilin_query.to_ref()).unwrap())
			.collect::<Vec<_>>();

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}

	#[test]
	fn test_packed_1b_commit_prove_verify_without_error() {
		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 8, &rs_code)
				.unwrap();
		let pcs = <BlockTensorPCS<
			OptimalUnderlier128b,
			BinaryField1b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(8, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| PackedBinaryField128x1b::random(&mut rng))
			.take((1 << pcs.n_vars()) / PackedBinaryField128x1b::WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearExtension::from_values(evals).unwrap();
		let polys = [poly.to_ref()];

		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();

		let backend = make_portable_backend();
		let multilin_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();
		let value = poly.evaluate(&multilin_query).unwrap();
		let values = vec![value];

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}

	#[test]
	fn test_packed_1b_commit_prove_verify_batch_without_error() {
		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 8, &rs_code)
				.unwrap();
		let pcs = <BlockTensorPCS<
			OptimalUnderlier128b,
			BinaryField1b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(8, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let batch_size = thread_rng().gen_range(1..=10);
		let polys = repeat_with(|| {
			let evals = repeat_with(|| PackedBinaryField128x1b::random(&mut rng))
				.take((1 << pcs.n_vars()) / PackedBinaryField128x1b::WIDTH)
				.collect::<Vec<_>>();
			MultilinearExtension::from_values(evals).unwrap()
		})
		.take(batch_size)
		.collect::<Vec<_>>();
		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let backend = make_portable_backend();
		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();
		let multilinear_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();

		let values = polys
			.iter()
			.map(|poly| poly.evaluate(multilinear_query.to_ref()).unwrap())
			.collect::<Vec<_>>();

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}

	#[test]
	fn test_packed_32b_commit_prove_verify_without_error() {
		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 8, &rs_code)
				.unwrap();
		let pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			BinaryField32b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(8, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
			.take((1 << pcs.n_vars()) / PackedBinaryField4x32b::WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearExtension::from_values(evals).unwrap();
		let polys = [poly.to_ref()];

		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();
		let backend = make_portable_backend();
		let multilin_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();
		let value = poly.evaluate(&multilin_query).unwrap();
		let values = vec![value];

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}

	#[test]
	fn test_packed_32b_commit_prove_verify_batch_without_error() {
		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 8, &rs_code)
				.unwrap();
		let pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			BinaryField32b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(8, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let batch_size = thread_rng().gen_range(1..=10);
		let polys = repeat_with(|| {
			let evals = repeat_with(|| PackedBinaryField4x32b::random(&mut rng))
				.take((1 << pcs.n_vars()) / PackedBinaryField4x32b::WIDTH)
				.collect::<Vec<_>>();
			MultilinearExtension::from_values(evals).unwrap()
		})
		.take(batch_size)
		.collect::<Vec<_>>();
		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let backend = make_portable_backend();
		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();
		let multilin_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();

		let values = polys
			.iter()
			.map(|poly| poly.evaluate(multilin_query.to_ref()).unwrap())
			.collect::<Vec<_>>();

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}

	#[test]
	fn test_proof_size() {
		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 8, &rs_code)
				.unwrap();
		let pcs = <BasicTensorPCS<
			OptimalUnderlier128b,
			BinaryField32b,
			BinaryField8b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(8, rs_code, n_test_queries)
		.unwrap();

		assert_eq!(pcs.proof_size(1), 150016);
		assert_eq!(pcs.proof_size(2), 299520);
	}

	#[test]
	fn test_proof_size_optimal_block_pcs() {
		let pcs = find_proof_size_optimal_pcs::<
			OptimalUnderlier128b,
			BinaryField1b,
			BinaryField16b,
			BinaryField16b,
			BinaryField128b,
		>(100, 28, 1, 2, false)
		.unwrap();
		assert_eq!(pcs.n_vars(), 28);
		assert_eq!(pcs.log_rows(), 12);
		assert_eq!(pcs.log_cols(), 16);

		// Matrix should be wider with more polynomials per batch.
		let pcs = find_proof_size_optimal_pcs::<
			OptimalUnderlier128b,
			BinaryField1b,
			BinaryField16b,
			BinaryField16b,
			BinaryField128b,
		>(100, 28, 8, 2, false)
		.unwrap();
		assert_eq!(pcs.n_vars(), 28);
		assert_eq!(pcs.log_rows(), 10);
		assert_eq!(pcs.log_cols(), 18);
	}

	#[test]
	fn test_proof_size_optimal_basic_pcs() {
		let pcs = find_proof_size_optimal_pcs::<
			OptimalUnderlier128b,
			BinaryField32b,
			BinaryField32b,
			BinaryField32b,
			BinaryField128b,
		>(100, 28, 1, 2, false)
		.unwrap();
		assert_eq!(pcs.n_vars(), 28);
		assert_eq!(pcs.log_rows(), 11);
		assert_eq!(pcs.log_cols(), 17);

		// Matrix should be wider with more polynomials per batch.
		let pcs = find_proof_size_optimal_pcs::<
			OptimalUnderlier128b,
			BinaryField32b,
			BinaryField32b,
			BinaryField32b,
			BinaryField128b,
		>(100, 28, 8, 2, false)
		.unwrap();
		assert_eq!(pcs.n_vars(), 28);
		assert_eq!(pcs.log_rows(), 10);
		assert_eq!(pcs.log_cols(), 18);
	}

	#[test]
	fn test_commit_prove_verify_with_num_rows_below_packing_width() {
		type Packed = PackedBinaryField128x1b;

		let rs_code = ReedSolomonCode::new(5, 2, NTTOptions::default()).unwrap();
		let n_test_queries =
			calculate_n_test_queries_reed_solomon::<_, BinaryField128b, _>(100, 4, &rs_code)
				.unwrap();
		let pcs = <BlockTensorPCS<
			OptimalUnderlier128b,
			BinaryField1b,
			BinaryField16b,
			BinaryField128b,
			_,
			_,
			_,
		>>::new_using_groestl_merkle_tree(4, rs_code, n_test_queries)
		.unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let evals = repeat_with(|| Packed::random(&mut rng))
			.take((1 << pcs.n_vars()) / Packed::WIDTH)
			.collect::<Vec<_>>();
		let poly = MultilinearExtension::from_values(evals).unwrap();
		let polys = [poly.to_ref()];

		let (commitment, committed) = pcs.commit(&polys).unwrap();

		let mut prove_challenger = crate::transcript::Proof {
			transcript: TranscriptWriter::<HasherChallenger<Groestl256>>::default(),
			advice: AdviceWriter::default(),
		};
		let query = repeat_with(|| prove_challenger.transcript.sample())
			.take(pcs.n_vars())
			.collect::<Vec<_>>();
		let backend = make_portable_backend();
		let multilin_query = backend
			.multilinear_query::<PackedBinaryField1x128b>(&query)
			.unwrap();
		let value = poly.evaluate(&multilin_query).unwrap();
		let values = vec![value];

		let proof = pcs
			.prove_evaluation(
				&mut prove_challenger.transcript,
				&committed,
				&polys,
				&query,
				&backend,
			)
			.unwrap();

		let mut verify_challenger = prove_challenger.into_verifier();
		let _: Vec<BinaryField128b> = verify_challenger.transcript.sample_vec(pcs.n_vars());
		pcs.verify_evaluation(
			&mut verify_challenger.transcript,
			&commitment,
			&query,
			proof,
			&values,
			&backend,
		)
		.unwrap();
	}
}
