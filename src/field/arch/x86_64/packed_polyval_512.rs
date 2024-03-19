use core::arch::x86_64::*;
use std::ops::Mul;

use crate::field::{
	arch::{
		portable::packed::{impl_conversion, impl_packed_extension_field, PackedPrimitiveType},
		PairwiseStrategy,
	},
	arithmetic_traits::{impl_invert_with_strategy, impl_square_with_strategy},
	BinaryField128bPolyval,
};

use super::{
	m512::M512,
	polyval::{simd_montgomery_multiply, SimdType},
};

/// Define packed type
pub type PackedBinaryPolyval4x128b = PackedPrimitiveType<M512, BinaryField128bPolyval>;

// Define conversion from type to underlier
impl_conversion!(M512, PackedBinaryPolyval4x128b);

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
impl_square_with_strategy!(PackedBinaryPolyval2x128b, PairwiseStrategy);

// Define invert
impl_invert_with_strategy!(PackedBinaryPolyval2x128b, PairwiseStrategy);

impl SimdType for __m512i {
	unsafe fn shuffle_epi32<const IMM8: i32>(a: Self) -> Self {
		_mm512_shuffle_epi32::<IMM8>(a)
	}

	unsafe fn xor(a: Self, b: Self) -> Self {
		_mm512_xor_si256(a, b)
	}

	unsafe fn clmul_epi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		_mm512_clmulepi64_epi128::<IMM8>(a, b)
	}

	unsafe fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
		_mm512_slli_epi64::<IMM8>(a)
	}

	unsafe fn srli_epi64<const IMM8: i32>(a: Self) -> Self {
		_mm512_srli_epi64::<IMM8>(a)
	}

	unsafe fn unpacklo_epi64(a: Self, b: Self) -> Self {
		_mm512_unpacklo_epi64(a, b)
	}
}
