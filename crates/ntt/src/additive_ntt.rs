// Copyright 2024-2025 Irreducible Inc.

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField};
use binius_math::BinarySubspace;

use super::error::Error;

/// Shape of a batched NTT operation on a 3-dimensional tensor.
///
/// The tensor has three dimensions. Elements are indexed first along axis X (width),
/// then axis Y (height), then axis Z (depth). In other words, the range of elements sharing Y and
/// Z coordinates are contiguous in memory, elements sharing X and Z coordinates are strided by the
/// tensor width, and elements sharing X and Y coordinates are strided by the width times height.
///
/// The tensor has power-of-two length in each dimension.
///
/// The NTT operation performs a parallel NTT along the _Y axis_, meaning each the operation
/// transforms each column independently.
#[derive(Debug, Default, Clone, Copy)]
pub struct NTTShape {
	/// Log length along the X axis (width). This is the interleaved batch size.
	pub log_x: usize,
	/// Log length along the Y axis (height). This is the size of the NTT transform.
	pub log_y: usize,
	/// Log length along the Z axis (depth). This is the consecutive batch size.
	pub log_z: usize,
}

/// The binary field additive NTT.
///
/// A number-theoretic transform (NTT) is a linear transformation on a finite field analogous to
/// the discrete fourier transform. The version of the additive NTT we use is originally described
/// in [LCH14]. In [DP24] Section 3.1, the authors present the LCH additive NTT algorithm in a way
/// that makes apparent its compatibility with the FRI proximity test. Throughout the
/// documentation, we will refer to the notation used in [DP24].
///
/// The additive NTT is parameterized by a binary field $K$ and $\mathbb{F}\_2$-linear subspace. We
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

	/// Returns the binary subspace $S^{(i)}$.
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

	/// Batched forward transformation defined in [LCH14].
	///
	/// The scalars of `data`, viewed in natural order, represent a tensor of `shape` dimensions.
	/// See [`NTTShape`] for layout details. The transform is inplace, output adheres to `shape`.
	///
	/// [LCH14]: <https://arxiv.org/abs/1404.3458>
	fn forward_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		shape: NTTShape,
		coset: u32,
	) -> Result<(), Error>;

	/// Batched inverse transformation defined in [LCH14].
	///
	/// The scalars of `data`, viewed in natural order, represent a tensor of `shape` dimensions.
	/// See [`NTTShape`] for layout details. The transform is inplace, output adheres to `shape`.
	///
	/// [LCH14]: https://arxiv.org/abs/1404.3458
	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		shape: NTTShape,
		coset: u32,
	) -> Result<(), Error>;

	fn forward_transform_ext<PE: PackedExtension<F>>(
		&self,
		data: &mut [PE],
		shape: NTTShape,
		coset: u32,
	) -> Result<(), Error> {
		let shape_ext = NTTShape {
			log_x: shape.log_x + PE::Scalar::LOG_DEGREE,
			..shape
		};
		self.forward_transform(PE::cast_bases_mut(data), shape_ext, coset)
	}

	fn inverse_transform_ext<PE: PackedExtension<F>>(
		&self,
		data: &mut [PE],
		shape: NTTShape,
		coset: u32,
	) -> Result<(), Error> {
		let shape_ext = NTTShape {
			log_x: shape.log_x + PE::Scalar::LOG_DEGREE,
			..shape
		};
		self.inverse_transform(PE::cast_bases_mut(data), shape_ext, coset)
	}
}
