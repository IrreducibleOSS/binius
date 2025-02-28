// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField};
use binius_math::BinarySubspace;

use super::error::Error;

/// The binary field additive NTT.
///
/// A number-theoretic transform (NTT) is a linear transformation on a finite field analogous to
/// the discrete fourier transform. The version of the additive NTT we use is originally described
/// in [LCH14]. In [DP24] Section 3.1, the authors present the LCH additive NTT algorithm in a way
/// that makes apparent its compatibility with the FRI proximity test. Throughout the
/// documentation, we will refer to the notation used in [DP24].
///
/// The additive NTT is parameterized by a binary field $K$ and $\mathbb{F}_2$-linear subspace. We
/// write $\beta_0, \ldots, \beta_{\ell-1}$ for the ordered basis elements of the subspace and
/// require $\beta_0 = 1$. The basis determines a novel polynomial basis and an evaluation domain.
/// In the forward direction, the additive NTT transforms a vector of polynomial coefficients, with
/// respect to the novel polynomial basis, into a vector of their evaluations over the evaluation
/// domain. The inverse transformation interpolates polynomial values over the domain into novel
/// polynomial basis coefficients.
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub trait AdditiveNTT<F: BinaryField> {
	/// Base-2 logarithm of the maximum size of the NTT domain, $\ell$.
	fn log_domain_size(&self) -> usize;

	/// Returns the binary subspace $S^(i)$.
	///
	/// The domain will have dimension $\ell - i$.
	///
	/// ## Preconditions
	///
	/// * `i` must be less than `self.log_domain_size()`
	fn subspace(&self, i: usize) -> BinarySubspace<F>;

	/// Get the normalized subspace polynomial evaluation $\hat{W}_i(\beta_j)$.
	///
	/// ## Preconditions
	///
	/// * `i` must be less than `self.log_domain_size()`
	/// * `j` must be less than `self.log_domain_size() - i`
	fn get_subspace_eval(&self, i: usize, j: usize) -> F;

	/// Forward transformation defined in [LCH14] on a batch of inputs.
	///
	/// Input is the vector of polynomial coefficients in novel basis, output is in Lagrange basis.
	/// The batched inputs are interleaved, which improves the cache-efficiency of the computation.
	///
	/// [LCH14]: <https://arxiv.org/abs/1404.3458>
	fn forward_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>;

	/// Inverse transformation defined in [LCH14] on a batch of inputs.
	///
	/// Input is the vector of polynomial coefficients in Lagrange basis, output is in novel basis.
	/// The batched inputs are interleaved, which improves the cache-efficiency of the computation.
	///
	/// [LCH14]: https://arxiv.org/abs/1404.3458
	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>;

	fn forward_transform_ext<PE: PackedExtension<F>>(
		&self,
		data: &mut [PE],
		coset: u32,
	) -> Result<(), Error> {
		self.forward_transform(PE::cast_bases_mut(data), coset, PE::Scalar::LOG_DEGREE)
	}

	fn inverse_transform_ext<PE: PackedExtension<F>>(
		&self,
		data: &mut [PE],
		coset: u32,
	) -> Result<(), Error> {
		self.inverse_transform(PE::cast_bases_mut(data), coset, PE::Scalar::LOG_DEGREE)
	}
}
