// Copyright 2024 Ulvetanna Inc.

use super::{
	m512::M512,
	packed_polyval_128::{simd_montgomery_multiply, PolyvalSimdType},
};
use crate::{
	arch::{
		portable::packed::{impl_conversion, impl_packed_extension_field, PackedPrimitiveType},
		PairwiseStrategy, ReuseMultiplyStrategy, SimdStrategy,
	},
	arithmetic_traits::{impl_invert_with, impl_square_with, impl_transformation_with_strategy},
	BinaryField128bPolyval,
};
use core::arch::x86_64::*;
use seq_macro::seq;
use std::ops::Mul;

/// Define packed type
pub type PackedBinaryPolyval4x128b = PackedPrimitiveType<M512, BinaryField128bPolyval>;

// Define conversion from type to underlier
impl_conversion!(M512, PackedBinaryPolyval4x128b);

impl From<PackedBinaryPolyval4x128b> for [u128; 4] {
	fn from(value: PackedBinaryPolyval4x128b) -> Self {
		value.0.into()
	}
}

// Define extension fields
impl_packed_extension_field!(PackedBinaryPolyval4x128b);

// Define multiplication
impl Mul for PackedBinaryPolyval4x128b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		let result = unsafe {
			let h: __m512i = self.into();
			let y: __m512i = rhs.into();

			simd_montgomery_multiply(h, y)
		};

		result.into()
	}
}

// Define square
impl_square_with!(PackedBinaryPolyval4x128b @ ReuseMultiplyStrategy);

// Define invert
impl_invert_with!(PackedBinaryPolyval4x128b @ PairwiseStrategy);

// Define affine transformations
impl_transformation_with_strategy!(PackedBinaryPolyval4x128b, SimdStrategy);

impl PolyvalSimdType for __m512i {
	#[inline(always)]
	unsafe fn shuffle_epi32<const IMM8: i32>(a: Self) -> Self {
		_mm512_shuffle_epi32::<IMM8>(a)
	}

	#[inline(always)]
	unsafe fn xor(a: Self, b: Self) -> Self {
		_mm512_xor_si512(a, b)
	}

	#[inline(always)]
	unsafe fn clmul_epi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		_mm512_clmulepi64_epi128::<IMM8>(a, b)
	}

	#[inline(always)]
	unsafe fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
		// This is a workaround for the problem that `_mm512_slli_epi64` and
		// `_mm256_slli_epi64` have different generic constant type and stable Rust
		// doesn't allow cast for const parameter.
		// All these `if`s will be eliminated by compiler because `IMM8` is known at compile time.
		seq!(N in 0..64 {
			if IMM8 == N {
				return _mm512_slli_epi64::<N>(a)
			}
		});

		unreachable!("bit shift count shouldn't exceed 63")
	}

	#[inline(always)]
	unsafe fn srli_epi64<const IMM8: i32>(a: Self) -> Self {
		seq!(N in 0..64 {
			if IMM8 == N {
				return _mm512_srli_epi64::<N>(a)
			}
		});

		unreachable!("bit shift count shouldn't exceed 63")
	}

	#[inline(always)]
	unsafe fn unpacklo_epi64(a: Self, b: Self) -> Self {
		_mm512_unpacklo_epi64(a, b)
	}
}
