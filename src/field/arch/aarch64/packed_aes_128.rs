// Copyright 2024 Ulvetanna Inc.

use super::m128::M128;
use crate::field::{
	aes_field::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
	},
	arch::{
		portable::{
			packed::{
				impl_conversion, impl_packed_extension_field, packed_binary_field_tower,
				PackedPrimitiveType,
			},
			packed_arithmetic::{alphas, impl_tower_constants},
		},
		PackedStrategy, PairwiseStrategy,
	},
	arithmetic_traits::{
		impl_invert_with_strategy, impl_mul_alpha_with_strategy, impl_mul_with_strategy,
		impl_square_with_strategy, MulAlpha, Square,
	},
};
use std::{arch::aarch64::*, ops::Mul};

// Define 128 bit packed field types
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;
pub type PackedAESBinaryField8x16b = PackedPrimitiveType<M128, AESTowerField16b>;
pub type PackedAESBinaryField4x32b = PackedPrimitiveType<M128, AESTowerField32b>;
pub type PackedAESBinaryField2x64b = PackedPrimitiveType<M128, AESTowerField64b>;
pub type PackedAESBinaryField1x128b = PackedPrimitiveType<M128, AESTowerField128b>;

// Define conversion from type to underlier;
impl_conversion!(M128, PackedAESBinaryField16x8b);
impl_conversion!(M128, PackedAESBinaryField8x16b);
impl_conversion!(M128, PackedAESBinaryField4x32b);
impl_conversion!(M128, PackedAESBinaryField2x64b);
impl_conversion!(M128, PackedAESBinaryField1x128b);

// Define tower
packed_binary_field_tower!(
	PackedAESBinaryField16x8b
	< PackedAESBinaryField8x16b
	< PackedAESBinaryField4x32b
	< PackedAESBinaryField2x64b
	< PackedAESBinaryField1x128b
);

// Define extension fields
impl_packed_extension_field!(PackedAESBinaryField16x8b);
impl_packed_extension_field!(PackedAESBinaryField8x16b);
impl_packed_extension_field!(PackedAESBinaryField4x32b);
impl_packed_extension_field!(PackedAESBinaryField2x64b);
impl_packed_extension_field!(PackedAESBinaryField1x128b);

// Define contants
// 0xD3 corresponds to 0x10 after isomorphism from BinaryField8b to AESField
impl_tower_constants!(AESTowerField8b, M128, { M128(0x00d300d300d300d300d300d300d300d3u128) });
impl_tower_constants!(AESTowerField16b, M128, { M128(alphas!(u128, 4)) });
impl_tower_constants!(AESTowerField32b, M128, { M128(alphas!(u128, 5)) });
impl_tower_constants!(AESTowerField64b, M128, { M128(alphas!(u128, 6)) });

// Define multiplication
impl_mul_with_strategy!(PackedAESBinaryField8x16b, PackedStrategy);
impl_mul_with_strategy!(PackedAESBinaryField4x32b, PackedStrategy);
impl_mul_with_strategy!(PackedAESBinaryField2x64b, PairwiseStrategy);
impl_mul_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);

impl Mul for PackedAESBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		//! Performs a multiplication in GF(2^8) on the packed bytes.
		//! See https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm_gf2p8mul_epi8.html
		unsafe {
			let a = vreinterpretq_p8_p128(self.0.into());
			let b = vreinterpretq_p8_p128(rhs.0.into());
			let c0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(a), vget_low_p8(b)));
			let c1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(a), vget_high_p8(b)));

			// Reduces the 16-bit output of a carryless multiplication to 8 bits using equation 22 in
			// https://www.intel.com/content/dam/develop/external/us/en/documents/clmul-wp-rev-2-02-2014-04-20.pdf

			// Since q+(x) doesn't fit into 8 bits, we right shift the polynomial (divide by x) and correct for this later.
			// This works because q+(x) is divisible by x/the last polynomial bit is 0.
			// q+(x)/x = (x^8 + x^4 + x^3 + x)/x = 0b100011010 >> 1 = 0b10001101 = 0x8d
			const QPLUS_RSH1: poly8x8_t = unsafe { std::mem::transmute(0x8d8d8d8d8d8d8d8d_u64) };

			// q*(x) = x^4 + x^3 + x + 1 = 0b00011011 = 0x1b
			const QSTAR: poly8x8_t = unsafe { std::mem::transmute(0x1b1b1b1b1b1b1b1b_u64) };

			let cl = vuzp1q_p8(c0, c1);
			let ch = vuzp2q_p8(c0, c1);

			let tmp0 = vmull_p8(vget_low_p8(ch), QPLUS_RSH1);
			let tmp1 = vmull_p8(vget_high_p8(ch), QPLUS_RSH1);

			// Correct for q+(x) having beed divided by x
			let tmp0 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp0), 1));
			let tmp1 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp1), 1));

			let tmp_hi = vuzp2q_p8(tmp0, tmp1);
			let tmp0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(tmp_hi), QSTAR));
			let tmp1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(tmp_hi), QSTAR));
			let tmp_lo = vuzp1q_p8(tmp0, tmp1);

			vreinterpretq_p128_p8(vaddq_p8(cl, tmp_lo)).into()
		}
	}
}

// Define square
impl_square_with_strategy!(PackedAESBinaryField2x64b, PairwiseStrategy);
impl_square_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);

impl Square for PackedAESBinaryField16x8b {
	fn square(self) -> Self {
		self * self
	}
}

impl Square for PackedAESBinaryField8x16b {
	fn square(self) -> Self {
		self * self
	}
}

impl Square for PackedAESBinaryField4x32b {
	fn square(self) -> Self {
		self * self
	}
}

// TODO: use more optimal SIMD implementation
// Define invert
impl_invert_with_strategy!(PackedAESBinaryField16x8b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField8x16b, PackedStrategy);
impl_invert_with_strategy!(PackedAESBinaryField4x32b, PackedStrategy);
impl_invert_with_strategy!(PackedAESBinaryField2x64b, PairwiseStrategy);
impl_invert_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);

// Define multiply by alpha
impl_mul_alpha_with_strategy!(PackedAESBinaryField8x16b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField4x32b, PackedStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField2x64b, PairwiseStrategy);
impl_mul_alpha_with_strategy!(PackedAESBinaryField1x128b, PairwiseStrategy);

impl MulAlpha for PackedAESBinaryField16x8b {
	fn mul_alpha(self) -> Self {
		// 0xD3 corresponds to 0x10 after isomorphism from BinaryField8b to AESField
		self * Self::from(0xd3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3u128)
	}
}
