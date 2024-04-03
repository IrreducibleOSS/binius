// Copyright 2024 Ulvetanna Inc.

use super::{
	m256::M256,
	polyval::{simd_montgomery_multiply, PolyvalSimdType},
};
use crate::{
	arch::{
		portable::packed::{impl_conversion, impl_packed_extension_field, PackedPrimitiveType},
		PairwiseStrategy,
	},
	arithmetic_traits::{impl_invert_with_strategy, impl_square_with_strategy},
	BinaryField128bPolyval,
};
use core::arch::x86_64::*;
use std::ops::Mul;

/// Define packed type
pub type PackedBinaryPolyval2x128b = PackedPrimitiveType<M256, BinaryField128bPolyval>;

// Define conversion from type to underlier
impl_conversion!(M256, PackedBinaryPolyval2x128b);

// Define extension fields
impl_packed_extension_field!(PackedBinaryPolyval2x128b);

// Define multiplication
impl Mul for PackedBinaryPolyval2x128b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		let result = unsafe {
			let h: __m256i = self.into();
			let y: __m256i = rhs.into();

			simd_montgomery_multiply(h, y)
		};

		result.into()
	}
}

// Define square
impl_square_with_strategy!(PackedBinaryPolyval2x128b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryPolyval2x128b, PairwiseStrategy);

impl PolyvalSimdType for __m256i {
	#[inline(always)]
	unsafe fn shuffle_epi32<const IMM8: i32>(a: Self) -> Self {
		_mm256_shuffle_epi32::<IMM8>(a)
	}

	#[inline(always)]
	unsafe fn xor(a: Self, b: Self) -> Self {
		_mm256_xor_si256(a, b)
	}

	#[inline(always)]
	unsafe fn clmul_epi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		_mm256_clmulepi64_epi128::<IMM8>(a, b)
	}

	#[inline(always)]
	unsafe fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
		_mm256_slli_epi64::<IMM8>(a)
	}

	#[inline(always)]
	unsafe fn srli_epi64<const IMM8: i32>(a: Self) -> Self {
		_mm256_srli_epi64::<IMM8>(a)
	}

	#[inline(always)]
	unsafe fn unpacklo_epi64(a: Self, b: Self) -> Self {
		_mm256_unpacklo_epi64(a, b)
	}
}
