// Copyright 2024 Ulvetanna Inc.

use crate::{
	arch::{portable::packed::PackedPrimitiveType, GfniStrategy},
	arithmetic_traits::{TaggedInvertOrZero, TaggedMul},
	underlier::UnderlierType,
	BinaryField,
};

#[rustfmt::skip]
const TOWER_TO_AES_MAP: i64 = u64::from_le_bytes([
	0b00111110,
	0b10011000,
	0b01001110,
	0b10010110,
	0b11101010,
	0b01101010,
	0b01010000,
	0b00110001,
]) as i64;

#[rustfmt::skip]
const AES_TO_TOWER_MAP: i64 = u64::from_le_bytes([
	0b00001100,
	0b01110000,
	0b10100010,
	0b01110010,
	0b00111110,
	0b10000110,
	0b11101000,
	0b11010001,
]) as i64;

#[rustfmt::skip]
pub const IDENTITY_MAP: i64 = u64::from_le_bytes([
	0b10000000,
	0b01000000,
	0b00100000,
	0b00010000,
	0b00001000,
	0b00000100,
	0b00000010,
	0b00000001,
]) as i64;

pub type GfniBinaryTowerStrategy = GfniStrategy<TOWER_TO_AES_MAP, AES_TO_TOWER_MAP>;
pub type GfniAESTowerStrategy = GfniStrategy<IDENTITY_MAP, IDENTITY_MAP>;

pub trait GfniType: Copy {
	fn set_epi_64(val: i64) -> Self;
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self;
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self;
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self;
}

#[inline(always)]
fn affine_transform<T: GfniType>(x: T, map: i64) -> T {
	let map = T::set_epi_64(map);
	T::gf2p8affine_epi64_epi8(x, map)
}

impl<
		const TO_AES_MAP: i64,
		const FROM_AES_MAP: i64,
		U: GfniType + UnderlierType,
		Scalar: BinaryField,
	> TaggedMul<GfniStrategy<TO_AES_MAP, FROM_AES_MAP>> for PackedPrimitiveType<U, Scalar>
{
	fn mul(self, rhs: Self) -> Self {
		let (lhs_gfni, rhs_gfni) = if TO_AES_MAP != IDENTITY_MAP {
			(
				affine_transform(self.to_underlier(), TO_AES_MAP),
				affine_transform(rhs.to_underlier(), TO_AES_MAP),
			)
		} else {
			(self.to_underlier(), rhs.to_underlier())
		};

		let prod_gfni = U::gf2p8mul_epi8(lhs_gfni, rhs_gfni);

		let prod_gfni = if FROM_AES_MAP != IDENTITY_MAP {
			affine_transform(prod_gfni, FROM_AES_MAP)
		} else {
			prod_gfni
		};

		prod_gfni.into()
	}
}

impl<
		const TO_AES_MAP: i64,
		const FROM_AES_MAP: i64,
		U: GfniType + UnderlierType,
		Scalar: BinaryField,
	> TaggedInvertOrZero<GfniStrategy<TO_AES_MAP, FROM_AES_MAP>> for PackedPrimitiveType<U, Scalar>
{
	fn invert_or_zero(self) -> Self {
		let val_gfni = if TO_AES_MAP != IDENTITY_MAP {
			affine_transform(self.to_underlier(), TO_AES_MAP)
		} else {
			self.to_underlier()
		};

		// Calculate inversion and affine transformation to the original field with a single instruction
		let identity = U::set_epi_64(FROM_AES_MAP);
		let inv_gfni = U::gf2p8affineinv_epi64_epi8(val_gfni, identity);

		inv_gfni.into()
	}
}

#[cfg(target_feature = "sse2")]
mod impl_128 {
	use super::*;
	use crate::arch::x86_64::m128::M128;
	use core::arch::x86_64::*;

	impl GfniType for M128 {
		#[inline(always)]
		fn set_epi_64(val: i64) -> Self {
			unsafe { _mm_set1_epi64x(val) }.into()
		}

		#[inline(always)]
		fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
			unsafe { _mm_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
		}

		#[inline(always)]
		fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
			unsafe { _mm_gf2p8mul_epi8(a.0, b.0) }.into()
		}

		#[inline(always)]
		fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
			unsafe { _mm_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
		}
	}
}

#[cfg(target_feature = "avx2")]
mod impl_256 {
	use super::*;
	use crate::arch::x86_64::m256::M256;
	use core::arch::x86_64::*;

	impl GfniType for M256 {
		#[inline(always)]
		fn set_epi_64(val: i64) -> Self {
			unsafe { _mm256_set1_epi64x(val) }.into()
		}

		#[inline(always)]
		fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
			unsafe { _mm256_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
		}

		#[inline(always)]
		fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
			unsafe { _mm256_gf2p8mul_epi8(a.0, b.0) }.into()
		}

		#[inline(always)]
		fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
			unsafe { _mm256_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
		}
	}
}

#[cfg(target_feature = "avx512f")]
mod impl_512 {
	use super::*;
	use crate::arch::x86_64::m512::M512;
	use core::arch::x86_64::*;

	impl GfniType for M512 {
		#[inline(always)]
		fn set_epi_64(val: i64) -> Self {
			unsafe { _mm512_set1_epi64(val) }.into()
		}

		#[inline(always)]
		fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
			unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
		}

		#[inline(always)]
		fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
			unsafe { _mm512_gf2p8mul_epi8(a.0, b.0) }.into()
		}

		#[inline(always)]
		fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
			unsafe { _mm512_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
		}
	}
}
