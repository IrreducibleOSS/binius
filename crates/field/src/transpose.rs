// Copyright 2023-2024 Irreducible Inc.

use super::{packed::PackedField, ExtensionField, PackedFieldIndexable, RepackedExtension};
use p3_util::log2_strict_usize;

/// Error thrown when a transpose operation fails.
#[derive(Clone, thiserror::Error, Debug)]
pub enum Error {
	#[error("the \"{param}\" argument's size is invalid: {msg}")]
	InvalidBufferSize { param: &'static str, msg: String },
	#[error("dimension n of square blocks must divide packing width")]
	SquareBlockDimensionMustDivideWidth,
	#[error("destination buffer must be castable to a packed extension field buffer")]
	UnalignedDestination,
}

/// Transpose square blocks of elements within packed field elements in place.
///
/// The input elements are interpreted as a rectangular matrix with height `n = 2^n` in row-major
/// order. This matrix is interpreted as a vector of square matrices of field elements, and each
/// square matrix is transposed in-place.
///
/// # Arguments
///
/// * `log_n`: The base-2 logarithm of the dimension of the n x n square matrix. Must be less than
///   or equal to the base-2 logarithm of the packing width.
/// * `elems`: The packed field elements, length is a power-of-two multiple of `1 << log_n`.
pub fn square_transpose<P: PackedField>(log_n: usize, elems: &mut [P]) -> Result<(), Error> {
	if P::LOG_WIDTH < log_n {
		return Err(Error::SquareBlockDimensionMustDivideWidth);
	}

	let size = elems.len();
	if !size.is_power_of_two() {
		return Err(Error::InvalidBufferSize {
			param: "elems",
			msg: "power of two size required".to_string(),
		});
	}
	let log_size = log2_strict_usize(size);
	if log_size < log_n {
		return Err(Error::InvalidBufferSize {
			param: "elems",
			msg: "must have length at least 2^log_n".to_string(),
		});
	}

	let log_w = log_size - log_n;

	// See Hacker's Delight, Section 7-3.
	// https://dl.acm.org/doi/10.5555/2462741
	for i in 0..log_n {
		for j in 0..1 << (log_n - i - 1) {
			for k in 0..1 << (log_w + i) {
				let idx0 = (j << (log_w + i + 1)) | k;
				let idx1 = idx0 | (1 << (log_w + i));

				let v0 = elems[idx0];
				let v1 = elems[idx1];
				let (v0, v1) = v0.interleave(v1, i);
				elems[idx0] = v0;
				elems[idx1] = v1;
			}
		}
	}

	Ok(())
}

/// Transpose the scalars within a slice of packed extension field elements.
///
/// The `src` buffer is vector of `n` field extension field elements, or alternatively viewed as an
/// n x d matrix of base field elements, where `d` is the extension degree. This transposes the
/// base field elements into a d x n matrix in row-major order.
pub fn transpose_scalars<P, FE, PE>(src: &[PE], dst: &mut [P]) -> Result<(), Error>
where
	P: PackedField,
	FE: ExtensionField<P::Scalar>,
	PE: PackedFieldIndexable<Scalar = FE> + RepackedExtension<P>,
{
	let len = src.len();
	if !len.is_power_of_two() {
		return Err(Error::InvalidBufferSize {
			param: "elems",
			msg: "power of two size required".to_string(),
		});
	}
	if dst.len() != len {
		return Err(Error::InvalidBufferSize {
			param: "dst",
			msg: "must have equal length to src buffer".to_string(),
		});
	}

	let log_d = FE::LOG_DEGREE;
	let log_n = log2_strict_usize(src.len()) + PE::LOG_WIDTH;

	if log_n < log_d {
		return Err(Error::InvalidBufferSize {
			param: "src",
			msg: "must have length at least 2^{d - w} where d is the extension degree and w is \
			the extension packing width"
				.to_string(),
		});
	}

	{
		let dst_ext = PE::cast_exts_mut(dst);
		transpose::transpose(
			PE::unpack_scalars(src),
			PE::unpack_scalars_mut(dst_ext),
			1 << log_d,
			1 << (log_n - log_d),
		);
	}
	square_transpose(log_d, dst)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		BinaryField32b, PackedBinaryField128x1b, PackedBinaryField16x8b, PackedBinaryField4x32b,
		PackedBinaryField64x2b, PackedExtension,
	};

	#[test]
	fn test_square_transpose_128x1b() {
		let mut elems = [
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		square_transpose(3, &mut elems).unwrap();

		let expected = [
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
			PackedBinaryField128x1b::from(0xf0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0u128),
		];
		assert_eq!(elems, expected);
	}

	#[test]
	fn test_square_transpose_128x1b_multi_row() {
		let mut elems = [
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField128x1b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		square_transpose(1, &mut elems).unwrap();

		let expected = [
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
			PackedBinaryField128x1b::from(0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaau128),
		];
		assert_eq!(elems, expected);
	}

	#[test]
	fn test_square_transpose_64x2b() {
		let mut elems = [
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0x00000000000000000000000000000000u128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
			PackedBinaryField64x2b::from(0xffffffffffffffffffffffffffffffffu128),
		];
		square_transpose(3, &mut elems).unwrap();

		let expected = [
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
			0xff00ff00ff00ff00ff00ff00ff00ff00u128,
		]
		.map(PackedBinaryField64x2b::from);
		assert_eq!(elems, expected);
	}

	#[test]
	#[rustfmt::skip]
	fn test_transpose_scalars() {
		let elems = [
			[
				0x03020100,
				0x07060504,
				0x0b0a0908,
				0x0f0e0d0c,
			],
			[
				0x13121110,
				0x17161514,
				0x1b1a1918,
				0x1f1e1d1c,
			],
			[
				0x23222120,
				0x27262524,
				0x2b2a2928,
				0x2f2e2d2c,
			],
			[
				0x33323130,
				0x37363534,
				0x3b3a3938,
				0x3f3e3d3c,
			],
			[
				0x43424140,
				0x47464544,
				0x4b4a4948,
				0x4f4e4d4c,
			],
			[
				0x53525150,
				0x57565554,
				0x5b5a5958,
				0x5f5e5d5c,
			],
			[
				0x63626160,
				0x67666564,
				0x6b6a6968,
				0x6f6e6d6c,
			],
			[
				0x73727170,
				0x77767574,
				0x7b7a7978,
				0x7f7e7d7c,
			],
		].map(|vals| PackedBinaryField4x32b::from_scalars(vals.map(BinaryField32b::new)));

		let expected = [
			[0x0c080400, 0x1c181410, 0x2c282420, 0x3c383430],
			[0x4c484440, 0x5c585450, 0x6c686460, 0x7c787470],

			[0x0d090501, 0x1d191511, 0x2d292521, 0x3d393531],
			[0x4d494541, 0x5d595551, 0x6d696561, 0x7d797571],

			[0x0e0a0602, 0x1e1a1612, 0x2e2a2622, 0x3e3a3632],
			[0x4e4a4642, 0x5e5a5652, 0x6e6a6662, 0x7e7a7672],

			[0x0f0b0703, 0x1f1b1713, 0x2f2b2723, 0x3f3b3733],
			[0x4f4b4743, 0x5f5b5753, 0x6f6b6763, 0x7f7b7773],
		].map(|vals| PackedBinaryField4x32b::from_scalars(vals.map(BinaryField32b::new)));

		let mut dst = [PackedBinaryField4x32b::default(); 8];
		transpose_scalars::<PackedBinaryField16x8b,_,_>(&elems, PackedBinaryField4x32b::cast_bases_mut(&mut dst)).unwrap();
		assert_eq!(dst, expected);
	}
}
