// Copyright 2023 Ulvetanna Inc.

use binius_field::{ExtensionField, PackedExtensionField, PackedField};

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

	/// Encode a message in-place in a provided buffer.
	///
	/// Returns an error if the `code` buffer does not have capacity for `len()` field elements.
	fn encode_inplace(&self, code: &mut [Self::P]) -> Result<(), Self::EncodeError> {
		self.encode_batch_inplace(code, 0)
	}

	/// Encode a message in-place in a provided buffer.
	///
	/// Returns an error if the `code` buffer does not have capacity for `len()` field elements.
	fn encode_batch_inplace(
		&self,
		code: &mut [Self::P],
		log_batch_size: usize,
	) -> Result<(), Self::EncodeError>;

	/// Encode a message provided as a vector of packed field elements.
	fn encode(&self, mut msg: Vec<Self::P>) -> Result<Vec<Self::P>, Self::EncodeError> {
		msg.resize(msg.len() * self.inv_rate(), Self::P::default());
		self.encode_inplace(&mut msg)?;
		Ok(msg)
	}
}

/// A linear code the with additional ability to encode packed extension field elements.
///
/// A linear code can be naturally extended to a code over extension fields by encoding each
/// dimension of the extension as a vector-space separately. However, a naive encoding procedure
/// would not be able to access elements in the most memory-efficient manner, hence the separate
/// trait.
pub trait LinearCodeWithExtensionEncoding: LinearCode {
	/// Encode a message of extension field elements in-place in a provided buffer.
	///
	/// Returns an error if the `code` buffer does not have capacity for `len()` field elements.
	fn encode_extension_inplace<PE>(&self, code: &mut [PE]) -> Result<(), Self::EncodeError>
	where
		PE: PackedExtensionField<Self::P>,
		PE::Scalar: ExtensionField<<Self::P as PackedField>::Scalar>;

	/// Encode a message of extension field elements provided as a vector of packed field elements.
	fn encode_extension<PE>(&self, mut msg: Vec<PE>) -> Result<Vec<PE>, Self::EncodeError>
	where
		PE: PackedExtensionField<Self::P>,
		PE::Scalar: ExtensionField<<Self::P as PackedField>::Scalar>,
	{
		msg.resize(msg.len() * self.inv_rate(), PE::default());
		self.encode_extension_inplace(&mut msg)?;
		Ok(msg)
	}
}
