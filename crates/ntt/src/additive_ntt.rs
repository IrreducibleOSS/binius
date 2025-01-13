// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField};
use binius_utils::checked_arithmetics::log2_strict_usize;

use super::error::Error;

/// The additive NTT defined in [LCH14].
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
pub trait AdditiveNTT<F: BinaryField> {
	/// Base-2 logarithm of the size of the NTT domain.
	fn log_domain_size(&self) -> usize;

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
	fn forward_transform<P>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F>;

	/// Inverse transformation defined in [LCH14] on a batch of inputs.
	///
	/// Input is the vector of polynomial coefficients in Lagrange basis, output is in novel basis.
	/// The batched inputs are interleaved, which improves the cache-efficiency of the computation.
	///
	/// [LCH14]: https://arxiv.org/abs/1404.3458
	fn inverse_transform<P>(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>
	where
		P: PackedField<Scalar = F>;

	fn forward_transform_ext<PE: PackedExtension<F>>(&self, data: &mut [PE], coset: u32) -> Result<(), Error> {
		self.forward_transform(PE::cast_bases_mut(data), coset, log_batch_size)
	}

	fn inverse_transform_ext<PE: PackedExtension<F>>(&self, data: &mut [PE], coset: u32) -> Result<(), Error> {
		self.inverse_transform(PE::cast_bases_mut(data), coset, PE::Scalar::LOG_DEGREE)
	}
}
