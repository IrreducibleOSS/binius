// Copyright 2024 Ulvetanna Inc.

use crate::field::{
	underlier::UnderlierType, BinaryField, ExtensionField, PackedField, TowerField,
};

use crate::field::arithmetic_traits::{
	MulAlpha, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
};

pub trait UnderlierWithConstants: UnderlierType
where
	Self: 'static,
{
	const ALPHAS_ODD: &'static [Self];
	const INTERLEAVE_EVEN_MASK: &'static [Self];
	const INTERLEAVE_ODD_MASK: &'static [Self];
	const ZERO_ELEMENT_MASKS: &'static [Self];
}
/// Abstraction for a packed tower field of height greater than 0.
pub trait PackedTowerField: PackedField + From<Self::Underlier> + Into<Self::Underlier> {
	type Underlier: UnderlierWithConstants;
	/// A scalar of a lower height
	type DirectSubfield: TowerField;
	/// Packed type with the same underlier with a lower height
	type PackedDirectSubfield: PackedField<Scalar = Self::DirectSubfield>
		+ From<Self::Underlier>
		+ Into<Self::Underlier>;

	/// Reinterpret value as a packed field over a lower height
	fn as_packed_subfield(self) -> Self::PackedDirectSubfield {
		Self::PackedDirectSubfield::from(Into::<Self::Underlier>::into(self))
	}
}

/// Packed strategy for arithmetic operations
pub struct PackedStrategy;

macro_rules! define_packed_ops_for_zero_height {
	($name:ty) => {
		impl
			$crate::field::arithmetic_traits::TaggedMul<
				$crate::field::arch::portable::packed_arithmetic::PackedStrategy,
			> for $name
		{
			fn mul(self: Self, b: Self) -> Self {
				(self.to_underlier() & b.to_underlier()).into()
			}
		}

		impl
			$crate::field::arithmetic_traits::TaggedMulAlpha<
				$crate::field::arch::portable::packed_arithmetic::PackedStrategy,
			> for $name
		{
			fn mul_alpha(self) -> Self {
				self
			}
		}

		impl
			$crate::field::arithmetic_traits::TaggedSquare<
				$crate::field::arch::portable::packed_arithmetic::PackedStrategy,
			> for $name
		{
			fn square(self) -> Self {
				self
			}
		}

		impl
			$crate::field::arithmetic_traits::TaggedInvertOrZero<
				$crate::field::arch::portable::packed_arithmetic::PackedStrategy,
			> for $name
		{
			fn invert_or_zero(self) -> Self {
				self
			}
		}
	};
}

pub(crate) use define_packed_ops_for_zero_height;

/// Compile-time known constants needed for packed multiply implementation.
pub(crate) trait PackedOpsConstants<U, F: TowerField> {
	const ALPHAS: U;
	const INTERLEAVE_EVEN_MASK: U;
	const INTERLEAVE_ODD_MASK: U;
}

impl<PT> TaggedMul<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::Underlier: UnderlierWithConstants,
{
	/// Optimized packed field multiplication algorithm
	fn mul(self, b: Self) -> Self {
		assert_ne!(PT::DirectSubfield::DEGREE, 0);

		let a = self;

		// a and b can be interpreted as packed subfield elements:
		// a = <a_lo_0, a_hi_0, a_lo_1, a_hi_1, ...>
		// b = <b_lo_0, b_hi_0, b_lo_1, b_hi_1, ...>//
		// ab is the product of a * b as packed subfield elements
		// ab = <a_lo_0 * b_lo_0, a_hi_0 * b_hi_0, a_lo_1 * b_lo_1, a_hi_1 * b_hi_1, ...>
		let repacked_a = a.as_packed_subfield();
		let repacked_b = b.as_packed_subfield();
		let z0_even_z2_odd = repacked_a * repacked_b;

		// lo = <a_lo_0, b_lo_0, a_lo_1, b_lo_1, ...>
		// hi = <a_hi_0, b_hi_0, a_hi_1, b_hi_1, ...>
		let (lo, hi) = interleave::<PT::Underlier, PT::DirectSubfield>(a.into(), b.into());

		// <a_lo_0 + a_hi_0, b_lo_0 + b_hi_0, a_lo_1 + a_hi_1, b_lo_1 + b_hi_1, ...>
		let lo_plus_hi_a_even_b_odd = lo ^ hi;

		let even_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = even_mask << PT::DirectSubfield::N_BITS;

		let alphas =
			<PT::Underlier as UnderlierWithConstants>::ALPHAS_ODD[PT::DirectSubfield::TOWER_LEVEL];

		// <α, z2_0, α, z2_1, ...>
		let alpha_even_z2_odd = alphas ^ (z0_even_z2_odd.into() & odd_mask);

		// a_lo_plus_hi_even_z2_odd    = <a_lo_0 + a_hi_0, z2_0, a_lo_1 + a_hi_1, z2_1, ...>
		// b_lo_plus_hi_even_alpha_odd = <b_lo_0 + b_hi_0,    α, a_lo_1 + a_hi_1,   αz, ...>
		let (a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd) =
			interleave::<PT::Underlier, PT::DirectSubfield>(
				lo_plus_hi_a_even_b_odd,
				alpha_even_z2_odd,
			);

		// <z1_0 + z0_0 + z2_0, z2a_0, z1_1 + z0_1 + z2_1, z2a_1, ...>
		let z1_plus_z0_plus_z2_even_z2a_odd =
			PT::PackedDirectSubfield::from(a_lo_plus_hi_even_alpha_odd)
				* PT::PackedDirectSubfield::from(b_lo_plus_hi_even_z2_odd);

		// <0, z1_0 + z2a_0 + z0_0 + z2_0, 0, z1_1 + z2a_1 + z0_1 + z2_1, ...>
		let zero_even_z1_plus_z2a_plus_z0_plus_z2_odd = (z1_plus_z0_plus_z2_even_z2a_odd.into()
			^ (z1_plus_z0_plus_z2_even_z2a_odd.into() << PT::DirectSubfield::N_BITS))
			& odd_mask;

		// <z0_0 + z2_0, z0_0 + z2_0, z0_1 + z2_1, z0_1 + z2_1, ...>
		let z0_plus_z2_dup =
			xor_adjacent::<PT::Underlier, PT::DirectSubfield>(z0_even_z2_odd.into());

		// <z0_0 + z2_0, z1_0 + z2a_0, z0_1 + z2_1, z1_1 + z2a_1, ...>
		(z0_plus_z2_dup ^ zero_even_z1_plus_z2a_plus_z0_plus_z2_odd).into()
	}
}

/// Generate the mask with alphas in the odd packed element positions and zeros in even
macro_rules! alphas {
	($underlier:ty, $tower_level:literal) => {{
		let mut alphas: $underlier = if $tower_level == 0 {
			1
		} else {
			1 << (1 << ($tower_level - 1))
		};

		let log_width = <$underlier>::LOG_BITS - $tower_level;
		let mut i = 1;
		while i < log_width {
			alphas |= alphas << (1 << ($tower_level + i));
			i += 1;
		}

		alphas
	}};
}

pub(crate) use alphas;

/// Generate the mask with ones in the odd packed element positions and zeros in even
macro_rules! interleave_mask_even {
	($underlier:ty, $tower_level:literal) => {{
		let scalar_bits = 1 << $tower_level;

		let mut mask: $underlier = (1 << scalar_bits) - 1;
		let log_width = <$underlier>::LOG_BITS - $tower_level;
		let mut i = 1;
		while i < log_width {
			mask |= mask << (scalar_bits << i);
			i += 1;
		}

		mask
	}};
}

pub(crate) use interleave_mask_even;

/// Generate the mask with ones in the even packed element positions and zeros in odd
macro_rules! interleave_mask_odd {
	($underlier:ty, $tower_level:literal) => {
		interleave_mask_even!($underlier, $tower_level) << (1 << $tower_level)
	};
}

pub(crate) use interleave_mask_odd;

/// Generate the mask for a single element
macro_rules! single_element_mask {
	($underlier:ty, $tower_level:literal) => {
		if <$underlier>::LOG_BITS == $tower_level {
			<$underlier>::MAX
		} else {
			((1 as $underlier) << (1 << $tower_level)) - 1
		}
	};
}

pub(crate) use single_element_mask;

/// View the inputs as vectors of packed binary tower elements and transpose as 2x2 square matrices.
/// Given vectors <a_0, a_1, a_2, a_3, ...> and <b_0, b_1, b_2, b_3, ...>, returns a tuple with
/// <a0, b0, a2, b2, ...> and <a1, b1, a3, b3>.
fn interleave<U: UnderlierWithConstants, F: TowerField>(a: U, b: U) -> (U, U) {
	let mask = U::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];

	let block_len = 1 << F::TOWER_LEVEL;
	let t = ((a >> block_len) ^ b) & mask;
	let c = a ^ (t << block_len);
	let d = b ^ t;

	(c, d)
}

/// View the input as a vector of packed binary tower elements and add the adjacent ones.
/// Given a vector <a_0, a_1, a_2, a_3, ...>, returns <a0 + a1, a0 + a1, a2 + a3, a2 + a3, ...>.
fn xor_adjacent<U: UnderlierWithConstants, F: TowerField>(a: U) -> U {
	let mask = U::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];

	let block_len = F::N_BITS;
	let t = ((a >> block_len) ^ a) & mask;

	t ^ (t << block_len)
}

impl<PT> TaggedMulAlpha<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::PackedDirectSubfield: MulAlpha,
{
	fn mul_alpha(self) -> Self {
		let block_len = PT::DirectSubfield::N_BITS;
		let even_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		let a = self.into();
		let a0 = a & even_mask;
		let a1 = a & odd_mask;
		let z1 = PT::PackedDirectSubfield::from(a1).mul_alpha().into();

		((a1 >> block_len) | ((a0 << block_len) ^ z1)).into()
	}
}

impl<PT> TaggedSquare<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::PackedDirectSubfield: MulAlpha,
{
	fn square(self) -> Self {
		let block_len = PT::DirectSubfield::N_BITS;
		let even_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		let z_02 = PackedField::square(self.as_packed_subfield());
		let z_2a = z_02.mul_alpha().into() & odd_mask;

		let z_0_xor_z_2 = (z_02.into() ^ (z_02.into() >> block_len)) & even_mask;

		(z_0_xor_z_2 | z_2a).into()
	}
}

impl<PT> TaggedInvertOrZero<PackedStrategy> for PT
where
	PT: PackedTowerField,
	PT::PackedDirectSubfield: MulAlpha,
{
	fn invert_or_zero(self) -> Self {
		let block_len = PT::DirectSubfield::N_BITS;
		let even_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_EVEN_MASK
			[PT::DirectSubfield::TOWER_LEVEL];
		let odd_mask = <PT::Underlier as UnderlierWithConstants>::INTERLEAVE_ODD_MASK
			[PT::DirectSubfield::TOWER_LEVEL];

		// has meaningful values in even positions
		let a_1_even = self.into() >> block_len;
		let intermediate =
			self.as_packed_subfield() + PT::PackedDirectSubfield::from(a_1_even).mul_alpha();
		let delta = self.as_packed_subfield() * intermediate
			+ <PT::PackedDirectSubfield as PackedField>::square(a_1_even.into());
		let delta_inv = PackedField::invert_or_zero(delta);

		// set values from even positions to odd as well
		let mut delta_inv_delta_inv = delta_inv.into() & even_mask;
		delta_inv_delta_inv |= delta_inv_delta_inv << block_len;

		let intermediate_a1 = (self.into() & odd_mask) | (intermediate.into() & even_mask);
		let result = PT::PackedDirectSubfield::from(delta_inv_delta_inv)
			* PT::PackedDirectSubfield::from(intermediate_a1);
		Into::<PT::Underlier>::into(result).into()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	use rand::thread_rng;
	use std::fmt::Debug;

	use proptest::{arbitrary::any, proptest};

	use crate::field::{
		arch::portable::packed_128::{
			PackedBinaryField16x8b, PackedBinaryField1x128b, PackedBinaryField2x64b,
			PackedBinaryField32x4b, PackedBinaryField4x32b, PackedBinaryField64x2b,
			PackedBinaryField8x16b,
		},
		test_utils::{define_invert_tests, define_multiply_tests, define_square_tests},
		PackedBinaryField128x1b,
	};

	use crate::field::{
		BinaryField16b, BinaryField1b, BinaryField2b, BinaryField32b, BinaryField4b,
		BinaryField64b, BinaryField8b,
	};

	const NUM_TESTS: u64 = 100;

	fn check_interleave<F: TowerField>(a: u128, b: u128, c: u128, d: u128) {
		assert_eq!(interleave::<u128, F>(a, b), (c, d));
		assert_eq!(interleave::<u128, F>(c, d), (a, b));
	}

	#[test]
	fn test_interleave() {
		check_interleave::<BinaryField1b>(
			0x0000000000000000FFFFFFFFFFFFFFFF,
			0xFFFFFFFFFFFFFFFF0000000000000000,
			0xAAAAAAAAAAAAAAAA5555555555555555,
			0xAAAAAAAAAAAAAAAA5555555555555555,
		);

		check_interleave::<BinaryField2b>(
			0x0000000000000000FFFFFFFFFFFFFFFF,
			0xFFFFFFFFFFFFFFFF0000000000000000,
			0xCCCCCCCCCCCCCCCC3333333333333333,
			0xCCCCCCCCCCCCCCCC3333333333333333,
		);

		check_interleave::<BinaryField4b>(
			0x0000000000000000FFFFFFFFFFFFFFFF,
			0xFFFFFFFFFFFFFFFF0000000000000000,
			0xF0F0F0F0F0F0F0F00F0F0F0F0F0F0F0F,
			0xF0F0F0F0F0F0F0F00F0F0F0F0F0F0F0F,
		);

		check_interleave::<BinaryField8b>(
			0x0F0E0D0C0B0A09080706050403020100,
			0x1F1E1D1C1B1A19181716151413121110,
			0x1E0E1C0C1A0A18081606140412021000,
			0x1F0F1D0D1B0B19091707150513031101,
		);

		check_interleave::<BinaryField16b>(
			0x0F0E0D0C0B0A09080706050403020100,
			0x1F1E1D1C1B1A19181716151413121110,
			0x1D1C0D0C191809081514050411100100,
			0x1F1E0F0E1B1A0B0A1716070613120302,
		);

		check_interleave::<BinaryField32b>(
			0x0F0E0D0C0B0A09080706050403020100,
			0x1F1E1D1C1B1A19181716151413121110,
			0x1B1A19180B0A09081312111003020100,
			0x1F1E1D1C0F0E0D0C1716151407060504,
		);

		check_interleave::<BinaryField64b>(
			0x0F0E0D0C0B0A09080706050403020100,
			0x1F1E1D1C1B1A19181716151413121110,
			0x17161514131211100706050403020100,
			0x1F1E1D1C1B1A19180F0E0D0C0B0A0908,
		);
	}

	fn test_packed_multiply_alpha<P>()
	where
		P: PackedField + TaggedMulAlpha<PackedStrategy> + Debug,
		P::Scalar: MulAlpha,
	{
		let mut rng = thread_rng();

		for _ in 0..NUM_TESTS {
			let a = P::random(&mut rng);

			let result = a.mul_alpha();
			for i in 0..P::WIDTH {
				assert_eq!(result.get(i), MulAlpha::mul_alpha(a.get(i)));
			}
		}
	}

	// TODO: currently gfni implementation doesn't implement packed operations
	#[cfg(not(target_feature = "gfni"))]
	#[test]
	fn test_multiply_alpha() {
		test_packed_multiply_alpha::<PackedBinaryField128x1b>();
		test_packed_multiply_alpha::<PackedBinaryField64x2b>();
		test_packed_multiply_alpha::<PackedBinaryField32x4b>();
		test_packed_multiply_alpha::<PackedBinaryField16x8b>();
		test_packed_multiply_alpha::<PackedBinaryField8x16b>();
		test_packed_multiply_alpha::<PackedBinaryField4x32b>();
		test_packed_multiply_alpha::<PackedBinaryField2x64b>();
		test_packed_multiply_alpha::<PackedBinaryField1x128b>();
	}

	trait PackedFieldWithPackedOps:
		PackedField
		+ TaggedMul<PackedStrategy>
		+ TaggedSquare<PackedStrategy>
		+ TaggedInvertOrZero<PackedStrategy>
	{
	}

	impl<P> PackedFieldWithPackedOps for P where
		P: PackedField
			+ PackedField
			+ TaggedMul<PackedStrategy>
			+ TaggedSquare<PackedStrategy>
			+ TaggedInvertOrZero<PackedStrategy>
	{
	}

	fn packed_mult<P: PackedFieldWithPackedOps>(a: P, b: P) -> P {
		<P as TaggedMul<PackedStrategy>>::mul(a, b)
	}

	// TODO: currently gfni implementation doesn't implement packed operations
	#[cfg(not(target_feature = "gfni"))]
	define_multiply_tests!(packed_mult, PackedFieldWithPackedOps);

	fn packed_square<P: PackedFieldWithPackedOps>(a: P) -> P {
		<P as TaggedSquare<PackedStrategy>>::square(a)
	}

	// TODO: currently gfni implementation doesn't implement packed operations
	#[cfg(not(target_feature = "gfni"))]
	define_square_tests!(packed_square, PackedFieldWithPackedOps);

	fn packed_invert<P: PackedFieldWithPackedOps>(a: P) -> P {
		<P as TaggedInvertOrZero<PackedStrategy>>::invert_or_zero(a)
	}

	// TODO: currently gfni implementation doesn't implement packed operations
	#[cfg(not(target_feature = "gfni"))]
	define_invert_tests!(packed_invert, PackedFieldWithPackedOps);
}
