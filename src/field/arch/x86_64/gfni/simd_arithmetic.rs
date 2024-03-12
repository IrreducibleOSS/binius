// Copyright 2024 Ulvetanna Inc.

use super::m128::M128;
use crate::field::{
	aes_field::AESTowerField8b,
	arch::portable::{packed::PackedPrimitiveType, packed_arithmetic::PackedTowerField},
	arithmetic_traits::TaggedMul,
	BinaryField8b, PackedField, TowerField,
};
use std::{any::TypeId, arch::x86_64::*};

unsafe fn dup_shuffle<Scalar: TowerField>() -> __m128i {
	match Scalar::N_BITS.ilog2() {
		3 => _mm_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0),
		4 => _mm_set_epi8(13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0),
		5 => _mm_set_epi8(11, 10, 9, 8, 11, 10, 9, 8, 3, 2, 1, 0, 3, 2, 1, 0),
		6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0),
		_ => panic!("unsupported bit count"),
	}
}

unsafe fn flip_shuffle<Scalar: TowerField>() -> __m128i {
	match Scalar::N_BITS.ilog2() {
		3 => _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
		4 => _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2),
		5 => _mm_set_epi8(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4),
		6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8),
		_ => panic!("unsupported bit count"),
	}
}

unsafe fn even_mask<Scalar: TowerField>() -> __m128i {
	match Scalar::N_BITS.ilog2() {
		3 => _mm_set_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1),
		4 => _mm_set_epi8(0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1),
		5 => _mm_set_epi8(0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1),
		6 => _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1),
		_ => panic!("unsupported bit count"),
	}
}

unsafe fn alpha<Scalar: TowerField>() -> __m128i {
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
			_mm_set_epi8(
				value, value, value, value, value, value, value, value, value, value, value, value,
				value, value, value, value,
			)
		}
		4 => _mm_set_epi8(
			0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
			0x01, 0x00,
		),
		5 => _mm_set_epi8(
			0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
			0x00, 0x00,
		),
		6 => _mm_set_epi8(
			0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
			0x00, 0x00,
		),
		_ => panic!("unsupported bit count"),
	}
}

/// Implement operation using SIMD
pub struct SimdStrategy;

impl<Scalar: TowerField> TaggedMul<SimdStrategy> for PackedPrimitiveType<M128, Scalar>
where
	Self: PackedTowerField<Underlier = M128>,
{
	fn mul(self, rhs: Self) -> Self {
		let a = self.as_packed_subfield();
		let b = rhs.as_packed_subfield();

		// [a0_lo * b0_lo, a0_hi * b0_hi, a1_lo * b1_lo, a1_h1 * b1_hi, ...]
		let z0_even_z2_odd = a * b;

		// [a0_lo, b0_lo, a1_lo, b1_lo, ...]
		// [a0_hi, b0_hi, a1_hi, b1_hi, ...]
		let (lo, hi) = a.interleave(b, 0);
		// [a0_lo + a0_hi, b0_lo + b0_hi, a1_lo + a1_hi, b1lo + b1_hi, ...]
		let lo_plus_hi_a_even_b_odd = lo + hi;

		let alpha_even_z2_odd: <Self as PackedTowerField>::PackedDirectSubfield = unsafe {
			let alpha = alpha::<<Self as PackedTowerField>::DirectSubfield>();
			let mask = even_mask::<<Self as PackedTowerField>::DirectSubfield>();
			// NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not a
			M128::from(_mm_blendv_epi8(Into::<M128>::into(z0_even_z2_odd).into(), alpha, mask))
				.into()
		};
		let (lhs, rhs) = lo_plus_hi_a_even_b_odd.interleave(alpha_even_z2_odd, 0);
		let z1_xor_z0z2_even_z2a_odd = lhs * rhs;

		unsafe {
			let z1_xor_z0z2 = _mm_shuffle_epi8(
				z1_xor_z0z2_even_z2a_odd.into().into(),
				dup_shuffle::<<Self as PackedTowerField>::DirectSubfield>(),
			);
			let zero_even_z1_xor_z2a_xor_z0z2_odd =
				_mm_xor_si128(z1_xor_z0z2_even_z2a_odd.into().into(), z1_xor_z0z2);

			let z2_even_z0_odd = _mm_shuffle_epi8(
				z0_even_z2_odd.into().into(),
				flip_shuffle::<<Self as PackedTowerField>::DirectSubfield>(),
			);
			let z0z2 = _mm_xor_si128(z0_even_z2_odd.into().into(), z2_even_z0_odd);

			M128::from(_mm_xor_si128(zero_even_z1_xor_z2a_xor_z0z2_odd, z0z2)).into()
		}
	}
}
