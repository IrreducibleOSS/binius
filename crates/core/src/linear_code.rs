// Copyright 2023-2024 Irreducible Inc.

//! Linear error-correcting code traits.

use binius_field::{ExtensionField, PackedField, RepackedExtension};
use binius_utils::checked_arithmetics::checked_log_2;

/// An encodable [linear error-correcting code](https://en.wikipedia.org/wiki/Linear_code) intended
/// for use in a Brakedown-style polynomial commitment scheme.
///
/// This trait represents linear codes with a dimension that is a power of 2, as that property is
/// required for the Brakedown polynomial commitment scheme.
///
/// Requirements:
/// - `len()` is a multiple of `dim()`
/// - `dim()` is a power of 2
/// - `dim()` is a multiple of `P::WIDTH`
#[allow(clippy::len_without_is_empty)]
pub trait LinearCode {
	type P: PackedField;
	type EncodeError: std::error::Error + Send + Sync + 'static;

	/// The block length.
	fn len(&self) -> usize {
		self.dim() * self.inv_rate()
	}

	/// The base-2 log of the dimension.
	fn dim_bits(&self) -> usize;

	/// The dimension.
	fn dim(&self) -> usize {
		1 << self.dim_bits()
	}

	/// The minimum distance between codewords.
	fn min_dist(&self) -> usize;

	/// The reciprocal of the rate, ie. `self.len() / self.dim()`.
	fn inv_rate(&self) -> usize;

	/// Encode a batch of interleaved messages in-place in a provided buffer.
	///
	/// The message symbols are interleaved in the buffer, which improves the cache-efficiency of
	/// the encoding procedure. The interleaved codeword is stored in the buffer when the method
	/// completes.
	///
	/// ## Throws
	///
	/// * If the `code` buffer does not have capacity for `len() << log_batch_size` field
	///   elements.
	fn encode_batch_inplace(
		&self,
		code: &mut [Self::P],
		log_batch_size: usize,
	) -> Result<(), Self::EncodeError>;

	/// Encode a message in-place in a provided buffer.
	///
	/// ## Throws
	///
	/// * If the `code` buffer does not have capacity for `len()` field elements.
	fn encode_inplace(&self, code: &mut [Self::P]) -> Result<(), Self::EncodeError> {
		self.encode_batch_inplace(code, 0)
	}

	/// Encode a message provided as a vector of packed field elements.
	fn encode(&self, mut msg: Vec<Self::P>) -> Result<Vec<Self::P>, Self::EncodeError> {
		msg.resize(msg.len() * self.inv_rate(), Self::P::default());
		self.encode_inplace(&mut msg)?;
		Ok(msg)
	}

	/// Encode a batch of interleaved messages of extension field elements in-place in a provided
	/// buffer.
	///
	/// A linear code can be naturally extended to a code over extension fields by encoding each
	/// dimension of the extension as a vector-space separately.
	///
	/// ## Preconditions
	///
	/// * `PE::Scalar::DEGREE` must be a power of two.
	///
	/// ## Throws
	///
	/// * If the `code` buffer does not have capacity for `len() << log_batch_size` field elements.
	fn encode_ext_batch_inplace<PE>(
		&self,
		code: &mut [PE],
		log_batch_size: usize,
	) -> Result<(), Self::EncodeError>
	where
		PE: RepackedExtension<Self::P>,
		PE::Scalar: ExtensionField<<Self::P as PackedField>::Scalar>,
	{
		let log_degree = checked_log_2(PE::Scalar::DEGREE);
		self.encode_batch_inplace(PE::cast_bases_mut(code), log_batch_size + log_degree)
	}

	/// Encode a message of extension field elements in-place in a provided buffer.
	///
	/// See [`Self::encode_ext_batch_inplace`] for more details.
	///
	/// ## Throws
	///
	/// * If the `code` buffer does not have capacity for `len()` field elements.
	fn encode_ext_inplace<PE>(&self, code: &mut [PE]) -> Result<(), Self::EncodeError>
	where
		PE: RepackedExtension<Self::P>,
		PE::Scalar: ExtensionField<<Self::P as PackedField>::Scalar>,
	{
		self.encode_ext_batch_inplace(code, 0)
	}

	/// Encode a message of extension field elements provided as a vector of packed field elements.
	///
	/// See [`Self::encode_ext_inplace`] for more details.
	fn encode_extension<PE>(&self, mut msg: Vec<PE>) -> Result<Vec<PE>, Self::EncodeError>
	where
		PE: RepackedExtension<Self::P>,
		PE::Scalar: ExtensionField<<Self::P as PackedField>::Scalar>,
	{
		msg.resize(msg.len() * self.inv_rate(), PE::default());
		self.encode_ext_inplace(&mut msg)?;
		Ok(msg)
	}
}
