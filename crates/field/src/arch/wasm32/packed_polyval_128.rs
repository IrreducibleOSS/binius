// Copyright 2023-2025 Irreducible Inc.
// Copyright (c) 2019-2023 RustCrypto Developers

//! ARMv8 `PMULL`-accelerated implementation of POLYVAL.
//!
//! Based on this C intrinsics implementation:
//! <https://github.com/noloader/AES-Intrinsics/blob/master/clmul-arm.c>
//!
//! Original C written and placed in public domain by Jeffrey Walton.
//! Based on code from ARM, and by Johannes Schneiders, Skip Hovsmith and
//! Barry O'Rourke for the mbedTLS project.
//!
//! For more information about PMULL, see:
//! - <https://developer.arm.com/documentation/100069/0608/A64-SIMD-Vector-Instructions/PMULL--PMULL2--vector->
//! - <https://eprint.iacr.org/2015/688.pdf>

use std::ops::Mul;

use super::{super::portable::packed::PackedPrimitiveType, m128::M128};
use crate::{
	BinaryField128bPolyval,
	arch::{PackedStrategy, PairwiseStrategy, ReuseMultiplyStrategy},
	arithmetic_traits::{InvertOrZero, impl_square_with, impl_transformation_with_strategy},
};

pub type PackedBinaryPolyval1x128b = PackedPrimitiveType<M128, BinaryField128bPolyval>;

type PortablePolyval = super::super::portable::packed_polyval_128::PackedBinaryPolyval1x128b;

// Define multiply
impl Mul for PackedBinaryPolyval1x128b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		Self::from_underlier(
			(PortablePolyval::from_underlier(self.0.into())
				* PortablePolyval::from_underlier(rhs.0.into()))
			.to_underlier()
			.into(),
		)
	}
}

// Define square
// TODO: implement a more optimal version for square case
impl_square_with!(PackedBinaryPolyval1x128b @ ReuseMultiplyStrategy);

// Define invert
impl InvertOrZero for PackedBinaryPolyval1x128b {
	fn invert_or_zero(self) -> Self {
		let portable = PortablePolyval::from_underlier(u128::from(self.0));

		Self::from_underlier(portable.invert_or_zero().0.into())
	}
}

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryPolyval1x128b, PackedStrategy);
