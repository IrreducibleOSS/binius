// Copyright 2024 Ulvetanna Inc.

use crate::field::{
	aes_field::AESTowerField8b,
	arch::{
		portable::{packed::PackedPrimitiveType, packed_arithmetic::PackedTowerField},
		SimdStrategy,
	},
	arithmetic_traits::TaggedMul,
	underlier::UnderlierType,
	BinaryField8b, PackedField, TowerField,
};
use std::{any::TypeId, arch::x86_64::*};

pub(super) trait TowerSimdType: Sized {
	fn shuffle_epi8(a: Self, b: Self) -> Self;
	fn xor(a: Self, b: Self) -> Self;

	fn dup_shuffle<Scalar: TowerField>() -> Self {
		let shuffle_mask_128 = unsafe {
			match Scalar::N_BITS.ilog2() {
				3 => _mm_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0),
				4 => _mm_set_epi8(13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0),
				5 => _mm_set_epi8(11, 10, 9, 8, 11, 10, 9, 8, 3, 2, 1, 0, 3, 2, 1, 0),
				6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(shuffle_mask_128)
	}

	fn flip_shuffle<Scalar: TowerField>() -> Self {
		let flip_mask_128 = unsafe {
			match Scalar::N_BITS.ilog2() {
				3 => _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
				4 => _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2),
				5 => _mm_set_epi8(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4),
				6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(flip_mask_128)
	}

	fn alpha<Scalar: TowerField>() -> Self {
		let alpha_128 = unsafe {
			match Scalar::TOWER_LEVEL {
				3 => {
					// Compiler will optimize this if out for each instantiation
					let type_id = TypeId::of::<Scalar>();
					let value = if type_id == TypeId::of::<BinaryField8b>() {
						0x10
					} else if type_id == TypeId::of::<AESTowerField8b>() {
						0xd3u8 as i8
					} else {
						panic!("tower field not supported")
					};
					_mm_set1_epi8(value)
				}
				4 => _mm_set1_epi16(0x0100),
				5 => _mm_set1_epi32(0x00010000),
				6 => _mm_set1_epi64x(0x0000000100000000),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(alpha_128)
	}

	fn even_mask<Scalar: TowerField>() -> Self {
		let mask_128 = unsafe {
			match Scalar::N_BITS.ilog2() {
				3 => _mm_set1_epi16(0x00FF),
				4 => _mm_set1_epi32(0x0000FFFF),
				5 => _mm_set1_epi64x(0x00000000FFFFFFFF),
				6 => _mm_set_epi64x(0, -1),
				_ => panic!("unsupported bit count"),
			}
		};

		Self::set1_epi128(mask_128)
	}

	fn set_alpha_even<Scalar: TowerField>(self) -> Self;
	fn set1_epi128(val: __m128i) -> Self;
}

impl<U, Scalar: TowerField> TaggedMul<SimdStrategy> for PackedPrimitiveType<U, Scalar>
where
	Self: PackedTowerField<Underlier = U>,
	U: TowerSimdType + UnderlierType,
{
	fn mul(self, rhs: Self) -> Self {
		// This fallback is needed to generically use SimdStrategy in benchmarks.
		if Scalar::TOWER_LEVEL <= 3 {
			return self * rhs;
		}

		let a = self.as_packed_subfield();
		let b = rhs.as_packed_subfield();

		// [a0_lo * b0_lo, a0_hi * b0_hi, a1_lo * b1_lo, a1_h1 * b1_hi, ...]
		let z0_even_z2_odd = a * b;

		// [a0_lo, b0_lo, a1_lo, b1_lo, ...]
		// [a0_hi, b0_hi, a1_hi, b1_hi, ...]
		let (lo, hi) = a.interleave(b, 0);
		// [a0_lo + a0_hi, b0_lo + b0_hi, a1_lo + a1_hi, b1lo + b1_hi, ...]
		let lo_plus_hi_a_even_b_odd = lo + hi;

		let alpha_even_z2_odd: <Self as PackedTowerField>::PackedDirectSubfield = z0_even_z2_odd
			.into()
			.set_alpha_even::<<Self as PackedTowerField>::DirectSubfield>()
			.into();
		let (lhs, rhs) = lo_plus_hi_a_even_b_odd.interleave(alpha_even_z2_odd, 0);
		let z1_xor_z0z2_even_z2a_odd = lhs * rhs;

		let z1_xor_z0z2 = U::shuffle_epi8(
			z1_xor_z0z2_even_z2a_odd.into(),
			U::dup_shuffle::<<Self as PackedTowerField>::DirectSubfield>(),
		);
		let zero_even_z1_xor_z2a_xor_z0z2_odd =
			U::xor(z1_xor_z0z2_even_z2a_odd.into(), z1_xor_z0z2);

		let z2_even_z0_odd = U::shuffle_epi8(
			z0_even_z2_odd.into(),
			U::flip_shuffle::<<Self as PackedTowerField>::DirectSubfield>(),
		);
		let z0z2 = U::xor(z0_even_z2_odd.into(), z2_even_z0_odd);

		Self::from(U::xor(zero_even_z1_xor_z2a_xor_z0z2_odd, z0z2))
	}
}
