// Copyright 2024-2025 Irreducible Inc.

use std::arch::wasm32::*;

use crate::{
	BinaryField, TowerField,
	arch::{
		SimdStrategy,
		m128::M128,
		portable::{
			packed::PackedPrimitiveType,
			packed_arithmetic::{PackedTowerField, TowerConstants, UnderlierWithBitConstants},
		},
	},
	arithmetic_traits::{
		MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
	},
	underlier::WithUnderlier,
};

impl<PT> TaggedMul<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::DirectSubfield: TowerConstants<M128> + BinaryField,
{
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		let alphas = PT::DirectSubfield::ALPHAS_ODD;
		let odd_mask = M128::INTERLEAVE_ODD_MASK[PT::DirectSubfield::TOWER_LEVEL];
		let a = self.as_packed_subfield();
		let b = rhs.as_packed_subfield();
		let p1 = (a * b).to_underlier();
		let (lo, hi) =
			M128::interleave(a.to_underlier(), b.to_underlier(), PT::DirectSubfield::TOWER_LEVEL);
		let (lhs, rhs) =
			M128::interleave(lo ^ hi, alphas ^ (p1 & odd_mask), PT::DirectSubfield::TOWER_LEVEL);
		let p2 = (PT::PackedDirectSubfield::from_underlier(lhs)
			* PT::PackedDirectSubfield::from_underlier(rhs))
		.to_underlier();
		let q1 = p1 ^ flip_even_odd::<PT::DirectSubfield>(p1);
		let q2 = p2 ^ shift_left::<PT::DirectSubfield>(p2);
		Self::from_underlier(q1 ^ (q2 & odd_mask))
	}
}

impl<PT> TaggedMulAlpha<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::PackedDirectSubfield: MulAlpha,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		let a0_a1 = self.as_packed_subfield();
		let a0alpha_a1alpha: M128 = a0_a1.mul_alpha().to_underlier();
		let a1_a0 = flip_even_odd::<PT::DirectSubfield>(a0_a1.to_underlier());
		Self::from_underlier(blend_odd_even::<PT::DirectSubfield>(a1_a0 ^ a0alpha_a1alpha, a1_a0))
	}
}

impl<PT> TaggedSquare<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::PackedDirectSubfield: MulAlpha + Square,
{
	#[inline]
	fn square(self) -> Self {
		let a0_a1 = self.as_packed_subfield();
		let a0sq_a1sq = Square::square(a0_a1);
		let a1sq_a0sq = flip_even_odd::<PT::DirectSubfield>(a0sq_a1sq.to_underlier());
		let a0sq_plus_a1sq = a0sq_a1sq.to_underlier() ^ a1sq_a0sq;
		let a1_mul_alpha = a0sq_a1sq.mul_alpha();
		Self::from_underlier(blend_odd_even::<PT::DirectSubfield>(
			a1_mul_alpha.to_underlier(),
			a0sq_plus_a1sq,
		))
	}
}

impl<PT> TaggedInvertOrZero<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::PackedDirectSubfield: MulAlpha + Square,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		let a0_a1 = self.as_packed_subfield();
		let a1_a0 = a0_a1.mutate_underlier(flip_even_odd::<PT::DirectSubfield>);
		let a1alpha = a1_a0.mul_alpha();
		let a0_plus_a1alpha = a0_a1 + a1alpha;
		let a1sq_a0sq = Square::square(a1_a0);
		let delta = a1sq_a0sq + (a0_plus_a1alpha * a0_a1);
		let deltainv = delta.invert_or_zero();
		let deltainv_deltainv = deltainv.mutate_underlier(duplicate_odd::<PT::DirectSubfield>);
		let delta_multiplier = a0_a1.mutate_underlier(|a0_a1| {
			blend_odd_even::<PT::DirectSubfield>(a0_a1, a0_plus_a1alpha.to_underlier())
		});
		PT::from_packed_subfield(deltainv_deltainv * delta_multiplier)
	}
}

#[inline]
fn duplicate_odd<F: TowerField>(x: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=2 => {
			let t = x & M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
			t | shift_right::<F>(t)
		}
		3 => x.shuffle_u8([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]),
		4 => x.shuffle_u8([2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15]),
		5 => x.shuffle_u8([4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15]),
		6 => x.shuffle_u8([8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15]),
		_ => panic!("Unsupported tower level"),
	}
}

#[inline]
fn flip_even_odd<F: TowerField>(x: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=2 => {
			let m1 = M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
			let m2 = M128::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];
			shift_right::<F>(x & m1) | shift_left::<F>(x & m2)
		}
		3 => x.shuffle_u8([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]),
		4 => x.shuffle_u8([2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]),
		5 => x.shuffle_u8([4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]),
		6 => x.shuffle_u8([8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]),
		_ => panic!("Unsupported tower level"),
	}
}

#[inline]
fn blend_odd_even<F: TowerField>(x: M128, y: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=2 => {
			let m1 = M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
			let m2 = M128::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];
			(x & m1) | (y & m2)
		}
		3 => u8x16_shuffle::<16, 1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15>(x.0, y.0)
			.into(),
		4 => u16x8_shuffle::<8, 1, 10, 3, 12, 5, 14, 7>(x.0, y.0).into(),
		5 => u32x4_shuffle::<4, 1, 6, 3>(x.0, y.0).into(),
		6 => u64x2_shuffle::<2, 1>(x.0, y.0).into(),
		_ => panic!("Unsupported tower level"),
	}
}

#[inline(always)]
fn shift_left<F: TowerField>(x: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=5 => u64x2_shl(x.0, 1 << F::TOWER_LEVEL).into(),
		6 => u64x2(0, u64x2_extract_lane::<0>(x.0)).into(),
		_ => panic!("Unsupported tower level"),
	}
}

#[inline(always)]
fn shift_right<F: TowerField>(x: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=5 => u64x2_shr(x.0, 1 << F::TOWER_LEVEL).into(),
		6 => u64x2(u64x2_extract_lane::<1>(x.0), 0).into(),
		_ => panic!("Unsupported tower level"),
	}
}

#[cfg(test)]
mod tests {
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::{
		BinaryField1b, BinaryField2b, BinaryField4b, BinaryField8b, BinaryField16b, BinaryField32b,
		BinaryField64b,
		arch::portable::packed_arithmetic::{interleave_mask_even, interleave_mask_odd},
		underlier::{Random, UnderlierType},
	};

	fn check_flip_even_odd<F: TowerField>(x: M128) {
		let flipped = flip_even_odd::<F>(x);

		let m1 = M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
		let m2 = M128::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];
		let expected = ((x & m1) >> (1 << F::TOWER_LEVEL)) | ((x & m2) << (1 << F::TOWER_LEVEL));

		assert_eq!(
			flipped,
			expected,
			"Flip even-odd mismatch for {x:?} with tower level {}",
			F::TOWER_LEVEL
		);
	}

	#[test]
	fn test_flip_even_odd() {
		let mut rng = StdRng::from_seed([0u8; 32]);
		let value = M128::random(&mut rng);
		check_flip_even_odd::<BinaryField1b>(value);
		check_flip_even_odd::<BinaryField2b>(value);
		check_flip_even_odd::<BinaryField4b>(value);
		check_flip_even_odd::<BinaryField8b>(value);
		check_flip_even_odd::<BinaryField16b>(value);
		check_flip_even_odd::<BinaryField32b>(value);
		check_flip_even_odd::<BinaryField64b>(value);
	}

	fn check_duplicate_odd<F: TowerField>(x: M128) {
		let duplicated = duplicate_odd::<F>(x);

		let m1 = x & M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
		let expected = m1 | (m1 >> (1 << F::TOWER_LEVEL));

		assert_eq!(
			duplicated,
			expected,
			"duplicate odd mismatch for {x:?} with tower level {}",
			F::TOWER_LEVEL
		);
	}

	#[test]
	fn test_duplicate_odd() {
		let mut rng = StdRng::from_seed([0u8; 32]);
		let value = M128::random(&mut rng);
		check_duplicate_odd::<BinaryField1b>(value);
		check_duplicate_odd::<BinaryField2b>(value);
		check_duplicate_odd::<BinaryField4b>(value);
		check_duplicate_odd::<BinaryField8b>(value);
		check_duplicate_odd::<BinaryField16b>(value);
		check_duplicate_odd::<BinaryField32b>(value);
		check_duplicate_odd::<BinaryField64b>(value);
	}

	fn check_blend_odd_even<F: TowerField>(x: M128, y: M128) {
		let blended = blend_odd_even::<F>(x, y);

		let m1 = M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
		let m2 = M128::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];

		let expected = (x & m1) | (y & m2);

		assert_eq!(
			blended,
			expected,
			"blend odd-even mismatch for {x:?} and {y:?} with tower level {} with m1 = {}, m2 = {}",
			F::TOWER_LEVEL,
			m1,
			m2
		);
	}

	#[test]
	fn test_blend_odd_even() {
		let mut rng = StdRng::from_seed([0u8; 32]);
		let x = M128::random(&mut rng);
		let y = M128::random(&mut rng);
		check_blend_odd_even::<BinaryField1b>(x, y);
		check_blend_odd_even::<BinaryField2b>(x, y);
		check_blend_odd_even::<BinaryField4b>(x, y);
		check_blend_odd_even::<BinaryField8b>(x, y);
		check_blend_odd_even::<BinaryField16b>(x, y);
		check_blend_odd_even::<BinaryField32b>(x, y);
		check_blend_odd_even::<BinaryField64b>(x, y);
	}
}
