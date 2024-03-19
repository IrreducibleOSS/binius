use std::{
	arch::x86_64::*,
	mem::transmute_copy,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use bytemuck::{must_cast, Pod, Zeroable};
use rand::{Rng, RngCore};
use subtle::{Choice, ConstantTimeEq};

use crate::field::{
	arch::portable::{
		packed::PackedPrimitiveType,
		packed_arithmetic::{interleave_mask_even, interleave_mask_odd, UnderlierWithBitConstants},
	},
	arithmetic_traits::Broadcast,
	underlier::{NumCast, Random, UnderlierType, WithUnderlier},
	BinaryField,
};

/// 512-bit value that is used for 512-bit SIMD operations
#[derive(Copy, Clone, Debug)]
pub struct M512(__m512i);

impl From<__m512i> for M512 {
	fn from(value: __m512i) -> Self {
		Self(value)
	}
}

impl From<[u128; 4]> for M512 {
	fn from(value: [u128; 4]) -> Self {
		Self(unsafe {
			_mm512_set_epi64(
				(value[3] >> 64) as i64,
				value[3] as i64,
				(value[2] >> 64) as i64,
				value[2] as i64,
				(value[1] >> 64) as i64,
				value[1] as i64,
				(value[0] >> 64) as i64,
				value[0] as i64,
			)
		})
	}
}

impl From<u128> for M512 {
	fn from(value: u128) -> Self {
		Self::from([value, 0, 0, 0])
	}
}

impl From<u64> for M512 {
	fn from(value: u64) -> Self {
		Self::from(value as u128)
	}
}

impl From<u32> for M512 {
	fn from(value: u32) -> Self {
		Self::from(value as u128)
	}
}

impl From<u16> for M512 {
	fn from(value: u16) -> Self {
		Self::from(value as u128)
	}
}

impl From<u8> for M512 {
	fn from(value: u8) -> Self {
		Self::from(value as u128)
	}
}
impl From<M512> for [u128; 4] {
	fn from(value: M512) -> Self {
		let result: [u128; 4] = unsafe { transmute_copy(&value.0) };

		result
	}
}

impl From<M512> for __m512i {
	fn from(value: M512) -> Self {
		value.0
	}
}
impl<U: NumCast<u128>> NumCast<M512> for U {
	fn num_cast_from(val: M512) -> Self {
		let [low, _, _, _] = val.into();
		Self::num_cast_from(low)
	}
}

impl Default for M512 {
	fn default() -> Self {
		Self(unsafe { _mm512_setzero_si512() })
	}
}

impl BitAnd for M512 {
	type Output = Self;

	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm512_and_si512(self.0, rhs.0) })
	}
}

impl BitAndAssign for M512 {
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs
	}
}

impl BitOr for M512 {
	type Output = Self;

	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm512_or_si512(self.0, rhs.0) })
	}
}

impl BitOrAssign for M512 {
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs
	}
}

impl BitXor for M512 {
	type Output = Self;

	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm512_xor_si512(self.0, rhs.0) })
	}
}

impl BitXorAssign for M512 {
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M512 {
	type Output = Self;

	fn not(self) -> Self::Output {
		const ONES: __m512i = m512_from_u128s!(u128::MAX, u128::MAX, u128::MAX, u128::MAX,);

		self ^ Self(ONES)
	}
}

impl Shl<usize> for M512 {
	type Output = Self;

	/// TODO: this is not the most efficient implementation
	fn shl(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 512 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut val_0, mut val_1, mut val_2, mut val_3]: [u128; 4] = self.into();
				if rhs >= 384 {
					val_3 = val_0 << (rhs - 384);
					val_2 = 0;
					val_1 = 0;
					val_0 = 0;
				} else if rhs > 256 {
					val_3 = (val_1 << (rhs - 256)) + (val_0 >> (128usize - (rhs - 256)));
					val_2 = val_0 << (rhs - 256);
					val_1 = 0;
					val_0 = 0;
				} else if rhs == 256 {
					val_3 = val_1;
					val_2 = val_0;
					val_1 = 0;
					val_0 = 0;
				} else if rhs > 128 {
					val_3 = (val_2 << (rhs - 128)) + (val_1 >> (128usize - (rhs - 128)));
					val_2 = (val_1 << (rhs - 128)) + (val_0 >> (128usize - (rhs - 128)));
					val_1 = val_0 << (rhs - 128);
					val_0 = 0;
				} else if rhs == 128 {
					val_3 = val_2;
					val_2 = val_1;
					val_1 = val_0;
					val_0 = 0;
				} else {
					val_3 = (val_3 << rhs) + (val_2 >> (128usize - rhs));
					val_2 = (val_2 << rhs) + (val_1 >> (128usize - rhs));
					val_1 = (val_1 << rhs) + (val_0 >> (128usize - rhs));
					val_0 <<= rhs;
				}
				[val_0, val_1, val_2, val_3].into()
			}
		}
	}
}

impl Shr<usize> for M512 {
	type Output = Self;

	/// TODO: this is not the most efficient implementation
	fn shr(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 512 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut val_0, mut val_1, mut val_2, mut val_3]: [u128; 4] = self.into();
				if rhs >= 384 {
					val_0 = val_3 >> (rhs - 384);
					val_1 = 0;
					val_2 = 0;
					val_3 = 0;
				} else if rhs > 256 {
					val_0 = (val_2 >> (rhs - 256)) + (val_3 << (128usize - (rhs - 256)));
					val_1 = val_3 >> (rhs - 256);
					val_2 = 0;
					val_3 = 0;
				} else if rhs == 256 {
					val_0 = val_2;
					val_1 = val_3;
					val_2 = 0;
					val_3 = 0;
				} else if rhs > 128 {
					val_0 = (val_1 >> (rhs - 128)) + (val_2 << (128usize - (rhs - 128)));
					val_1 = (val_2 >> (rhs - 128)) + (val_3 << (128usize - (rhs - 128)));
					val_2 = val_3 >> (rhs - 128);
					val_3 = 0;
				} else if rhs == 128 {
					val_0 = val_1;
					val_1 = val_2;
					val_2 = val_3;
					val_3 = 0;
				} else {
					val_0 = (val_0 >> rhs) + (val_1 << (128usize - rhs));
					val_1 = (val_1 >> rhs) + (val_2 << (128usize - rhs));
					val_2 = (val_2 >> rhs) + (val_3 << (128usize - rhs));
					val_3 >>= rhs;
				}
				[val_0, val_1, val_2, val_3].into()
			}
		}
	}
}

impl PartialEq for M512 {
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let pcmp = _mm512_cmpeq_epi32_mask(self.0, other.0);
			pcmp == 0xFFFF
		}
	}
}

impl Eq for M512 {}

impl ConstantTimeEq for M512 {
	fn ct_eq(&self, other: &Self) -> Choice {
		unsafe {
			let pcmp = _mm512_cmpeq_epi32_mask(self.0, other.0);
			pcmp.ct_eq(&0xFFFF)
		}
	}
}

impl Random for M512 {
	fn random(mut rng: impl RngCore) -> Self {
		let val: [u128; 4] = rng.gen();
		val.into()
	}
}

impl std::fmt::Display for M512 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: [u128; 4] = (*self).into();
		write!(f, "{data:02X?}")
	}
}

#[repr(align(64))]
pub struct AlignedData(pub [u128; 4]);

macro_rules! m512_from_u128s {
    ($($values:expr,)+) => {{
        let aligned_data = $crate::field::arch::x86_64::m512::AlignedData([$($values,)*]);
        unsafe {* (aligned_data.0.as_ptr() as *const __m512i)}
    }};
}

pub(super) use m512_from_u128s;

impl UnderlierType for M512 {
	const LOG_BITS: usize = 9;

	const ONE: Self = { Self(m512_from_u128s!(0, 0, 0, 1,)) };

	const ZERO: Self = { Self(m512_from_u128s!(0, 0, 0, 0,)) };

	fn fill_with_bit(val: u8) -> Self {
		Self(unsafe { _mm512_set1_epi8(val.wrapping_neg() as i8) })
	}
}

unsafe impl Zeroable for M512 {}

unsafe impl Pod for M512 {}

unsafe impl Send for M512 {}

unsafe impl Sync for M512 {}

impl<Scalar: BinaryField> From<__m512i> for PackedPrimitiveType<M512, Scalar> {
	fn from(value: __m512i) -> Self {
		PackedPrimitiveType::from(M512::from(value))
	}
}

impl<Scalar: BinaryField> From<[u128; 4]> for PackedPrimitiveType<M512, Scalar> {
	fn from(value: [u128; 4]) -> Self {
		PackedPrimitiveType::from(M512::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M512, Scalar>> for __m512i {
	fn from(value: PackedPrimitiveType<M512, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

impl<Scalar: BinaryField + WithUnderlier> Broadcast<Scalar> for PackedPrimitiveType<M512, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::N_BITS.ilog2() as usize;
		let mut value = u128::from(scalar.to_underlier());
		for n in tower_level..3 {
			value |= value << (1 << n);
		}

		match tower_level {
			0..=3 => unsafe { _mm512_broadcastb_epi8(must_cast(value)).into() },
			4 => unsafe { _mm512_broadcastw_epi16(must_cast(value)).into() },
			5 => unsafe { _mm512_broadcastd_epi32(must_cast(value)).into() },
			6 => unsafe { _mm512_broadcastq_epi64(must_cast(value)).into() },
			7 => [value, value, value, value].into(),
			_ => unreachable!(),
		}
	}
}

const fn from_equal_u128s(val: u128) -> M512 {
	unsafe { transmute_copy(&AlignedData([val, val, val, val])) }
}

// TODO: Add efficient interleave specialization for 512-bit values
impl UnderlierWithBitConstants for M512 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		from_equal_u128s(interleave_mask_even!(u128, 0)),
		from_equal_u128s(interleave_mask_even!(u128, 1)),
		from_equal_u128s(interleave_mask_even!(u128, 2)),
		from_equal_u128s(interleave_mask_even!(u128, 3)),
		from_equal_u128s(interleave_mask_even!(u128, 4)),
		from_equal_u128s(interleave_mask_even!(u128, 5)),
		from_equal_u128s(interleave_mask_even!(u128, 6)),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		from_equal_u128s(interleave_mask_odd!(u128, 0)),
		from_equal_u128s(interleave_mask_odd!(u128, 1)),
		from_equal_u128s(interleave_mask_odd!(u128, 2)),
		from_equal_u128s(interleave_mask_odd!(u128, 3)),
		from_equal_u128s(interleave_mask_odd!(u128, 4)),
		from_equal_u128s(interleave_mask_odd!(u128, 5)),
		from_equal_u128s(interleave_mask_odd!(u128, 6)),
	];
}

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use super::*;

	fn check_roundtrip<T>(val: M512)
	where
		T: From<M512>,
		M512: From<T>,
	{
		assert_eq!(M512::from(T::from(val)), val);
	}

	#[test]
	fn test_constants() {
		assert_eq!(M512::default(), M512::ZERO);
		assert_eq!(M512::from(0u128), M512::ZERO);
		assert_eq!(M512::from([0u128, 0u128, 0u128, 1u128]), M512::ONE);
	}

	#[derive(Default)]
	struct ByteData([u128; 4]);

	impl ByteData {
		fn get_bit(&self, i: usize) -> u8 {
			if self.0[i / 128] & (1u128 << (i % 128)) == 0 {
				0
			} else {
				1
			}
		}

		fn set_bit(&mut self, i: usize, val: u8) {
			self.0[i / 128] &= !(1 << (i % 128));
			self.0[i / 128] |= (val as u128) << (i % 128);
		}
	}

	impl From<ByteData> for M512 {
		fn from(value: ByteData) -> Self {
			let vals: [u128; 4] = unsafe { std::mem::transmute(value) };
			vals.into()
		}
	}

	impl From<[u128; 4]> for ByteData {
		fn from(value: [u128; 4]) -> Self {
			unsafe { std::mem::transmute(value) }
		}
	}

	impl Shl<usize> for ByteData {
		type Output = Self;

		fn shl(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..512 {
				if i >= rhs {
					result.set_bit(i, self.get_bit(i - rhs));
				}
			}

			result
		}
	}

	impl Shr<usize> for ByteData {
		type Output = Self;

		fn shr(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..512 {
				if i + rhs < 512 {
					result.set_bit(i, self.get_bit(i + rhs));
				}
			}

			result
		}
	}

	proptest! {
		#[test]
		fn test_conversion(a in any::<[u128; 4]>()) {
			check_roundtrip::<[u128; 4]>(a.into());
			check_roundtrip::<__m512i>(a.into());
		}

		#[test]
		fn test_binary_bit_operations([a, b] in any::<[[u128;4];2]>()) {
			assert_eq!(M512::from([a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]), M512::from(a) & M512::from(b));
			assert_eq!(M512::from([a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]), M512::from(a) | M512::from(b));
			assert_eq!(M512::from([a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]), M512::from(a) ^ M512::from(b));
		}

		#[test]
		fn test_negate(a in any::<[u128; 4]>()) {
			assert_eq!(M512::from([!a[0], !a[1], !a[2], !a[3]]), !M512::from(a))
		}

		#[test]
		fn test_shifts(a in any::<[u128; 4]>(), rhs in 0..255usize) {
			assert_eq!(M512::from(a) << rhs, M512::from(ByteData::from(a) << rhs));
			assert_eq!(M512::from(a) >> rhs, M512::from(ByteData::from(a) >> rhs));
		}
	}

	#[test]
	fn test_fill_with_bit() {
		assert_eq!(
			M512::fill_with_bit(1),
			M512::from([u128::MAX, u128::MAX, u128::MAX, u128::MAX])
		);
		assert_eq!(M512::fill_with_bit(0), M512::from(0u128));
	}

	#[test]
	fn test_eq() {
		let a = M512::from(0u128);
		let b = M512::from(42u128);
		let c = M512::from(u128::MAX);
		let d = M512::from([u128::MAX, u128::MAX, u128::MAX, u128::MAX]);

		assert_eq!(a, a);
		assert_eq!(b, b);
		assert_eq!(c, c);
		assert_eq!(d, d);

		assert_ne!(a, b);
		assert_ne!(a, c);
		assert_ne!(a, d);
		assert_ne!(b, c);
		assert_ne!(b, d);
		assert_ne!(c, d);
	}
}
