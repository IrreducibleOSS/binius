// Copyright 2024 Irreducible Inc.

use super::error::Error;
use binius_field::{ExtensionField, PackedField, RepackedExtension};
use p3_util::log2_strict_usize;

/// The additive NTT defined defined in [LCH14].
///
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
pub trait AdditiveNTT<P: PackedField> {
	/// Base-2 logarithm of the size of the NTT domain.
	fn log_domain_size(&self) -> usize;

	/// Get the normalized subspace polynomial evaluation $\hat{W}_i(\beta_j)$.
	///
	/// ## Preconditions
	///
	/// * `i` must be less than `self.log_domain_size()`
	/// * `j` must be less than `self.log_domain_size() - i`
	fn get_subspace_eval(&self, i: usize, j: usize) -> P::Scalar;

	/// Forward transformation defined in [LCH14] on a batch of inputs.
	///
	/// Input is the vector of polynomial coefficients in novel basis, output is in Lagrange basis.
	/// The batched inputs are interleaved, which improves the cache-efficiency of the computation.
	///
	/// [LCH14]: <https://arxiv.org/abs/1404.3458>
	fn forward_transform(
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
	fn inverse_transform(
		&self,
		data: &mut [P],
		coset: u32,
		log_batch_size: usize,
	) -> Result<(), Error>;

	fn forward_transform_ext<PE>(&self, data: &mut [PE], coset: u32) -> Result<(), Error>
	where
		PE: RepackedExtension<P>,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if !PE::Scalar::DEGREE.is_power_of_two() {
			return Err(Error::PowerOfTwoExtensionDegreeRequired);
		}

		let log_batch_size = log2_strict_usize(PE::Scalar::DEGREE);
		self.forward_transform(PE::cast_bases_mut(data), coset, log_batch_size)
	}

	fn inverse_transform_ext<PE>(&self, data: &mut [PE], coset: u32) -> Result<(), Error>
	where
		PE: RepackedExtension<P>,
		PE::Scalar: ExtensionField<P::Scalar>,
	{
		if !PE::Scalar::DEGREE.is_power_of_two() {
			return Err(Error::PowerOfTwoExtensionDegreeRequired);
		}

		let log_batch_size = log2_strict_usize(PE::Scalar::DEGREE);
		self.inverse_transform(PE::cast_bases_mut(data), coset, log_batch_size)
	}
}
