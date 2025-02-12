// Copyright 2023-2025 Irreducible Inc.

//! Traits for packed field elements which support SIMD implementations.
//!
//! Interfaces are derived from [`plonky2`](https://github.com/mir-protocol/plonky2).

use std::{
	fmt::Debug,
	iter::{self, Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use binius_utils::iter::IterExtensions;
use bytemuck::Zeroable;
use rand::RngCore;

use super::{
	arithmetic_traits::{Broadcast, MulAlpha, Square},
	binary_field_arithmetic::TowerFieldArithmetic,
	Error,
};
use crate::{
	arithmetic_traits::InvertOrZero, underlier::WithUnderlier, BinaryField, ExtensionField, Field,
	PackedExtension,
};

/// A packed field represents a vector of underlying field elements.
///
/// Arithmetic operations on packed field elements can be accelerated with SIMD CPU instructions.
/// The vector width is a constant, `WIDTH`. This trait requires that the width must be a power of
/// two.
pub trait PackedField:
	Default
	+ Debug
	+ Clone
	+ Copy
	+ Eq
	+ Sized
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Mul<Output = Self>
	+ AddAssign
	+ SubAssign
	+ MulAssign
	+ Add<Self::Scalar, Output = Self>
	+ Sub<Self::Scalar, Output = Self>
	+ Mul<Self::Scalar, Output = Self>
	+ AddAssign<Self::Scalar>
	+ SubAssign<Self::Scalar>
	+ MulAssign<Self::Scalar>
	// TODO: Get rid of Sum and Product. It's confusing with nested impls of Packed.
	+ Sum
	+ Product
	+ Send
	+ Sync
	+ Zeroable
	+ 'static
{
	type Scalar: Field;

	/// Base-2 logarithm of the number of field elements packed into one packed element.
	const LOG_WIDTH: usize;

	/// The number of field elements packed into one packed element.
	///
	/// WIDTH is guaranteed to equal 2^LOG_WIDTH.
	const WIDTH: usize = 1 << Self::LOG_WIDTH;

	/// Get the scalar at a given index without bounds checking.
	/// # Safety
	/// The caller must ensure that `i` is less than `WIDTH`.
	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar;

	/// Set the scalar at a given index without bounds checking.
	/// # Safety
	/// The caller must ensure that `i` is less than `WIDTH`.
	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar);

	/// Get the scalar at a given index.
	#[inline]
	fn get_checked(&self, i: usize) -> Result<Self::Scalar, Error> {
		(i < Self::WIDTH)
			.then_some(unsafe { self.get_unchecked(i) })
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
	}

	/// Set the scalar at a given index.
	#[inline]
	fn set_checked(&mut self, i: usize, scalar: Self::Scalar) -> Result<(), Error> {
		(i < Self::WIDTH)
			.then(|| unsafe { self.set_unchecked(i, scalar) })
			.ok_or(Error::IndexOutOfRange {
				index: i,
				max: Self::WIDTH,
			})
	}

	/// Get the scalar at a given index.
	#[inline]
	fn get(&self, i: usize) -> Self::Scalar {
		self.get_checked(i).expect("index must be less than width")
	}

	/// Set the scalar at a given index.
	#[inline]
	fn set(&mut self, i: usize, scalar: Self::Scalar) {
		self.set_checked(i, scalar).expect("index must be less than width")
	}

	#[inline]
	fn into_iter(self) -> impl Iterator<Item=Self::Scalar> + Send {
		(0..Self::WIDTH).map_skippable(move |i|
			// Safety: `i` is always less than `WIDTH`
			unsafe { self.get_unchecked(i) })
	}

	#[inline]
	fn iter(&self) -> impl Iterator<Item=Self::Scalar> + Send + '_ {
		(0..Self::WIDTH).map_skippable(move |i|
			// Safety: `i` is always less than `WIDTH`
			unsafe { self.get_unchecked(i) })
	}

	#[inline]
	fn iter_slice(slice: &[Self]) -> impl Iterator<Item=Self::Scalar> + Send + '_ {
		slice.iter().flat_map(Self::iter)
	}

	#[inline]
	fn zero() -> Self {
		Self::broadcast(Self::Scalar::ZERO)
	}

	#[inline]
	fn one() -> Self {
		Self::broadcast(Self::Scalar::ONE)
	}

	/// Initialize zero position with `scalar`, set other elements to zero.
	#[inline(always)]
	fn set_single(scalar: Self::Scalar) -> Self {
		let mut result = Self::default();
		result.set(0, scalar);

		result
	}

	fn random(rng: impl RngCore) -> Self;
	fn broadcast(scalar: Self::Scalar) -> Self;

	/// Construct a packed field element from a function that returns scalar values by index.
	fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self;

	/// Creates a packed field from a fallible function applied to each index.
	fn try_from_fn<E>(
            mut f: impl FnMut(usize) -> Result<Self::Scalar, E>,
        ) -> Result<Self, E> {
            let mut result = Self::default();
            for i in 0..Self::WIDTH {
                let scalar = f(i)?;
                unsafe {
                    result.set_unchecked(i, scalar);
                };
            }
            Ok(result)
        }

	/// Construct a packed field element from a sequence of scalars.
	///
	/// If the number of values in the sequence is less than the packing width, the remaining
	/// elements are set to zero. If greater than the packing width, the excess elements are
	/// ignored.
	fn from_scalars(values: impl IntoIterator<Item=Self::Scalar>) -> Self {
		let mut result = Self::default();
		for (i, val) in values.into_iter().take(Self::WIDTH).enumerate() {
			result.set(i, val);
		}
		result
	}

	/// Returns the value multiplied by itself
	fn square(self) -> Self;

	/// Returns the value to the power `exp`.
	fn pow(self, exp: u64) -> Self {
		let mut res = Self::one();
		for i in (0..64).rev() {
			res = res.square();
			if ((exp >> i) & 1) == 1 {
				res.mul_assign(self)
			}
		}
		res
	}

	/// Returns the packed inverse values or zeroes at indices where `self` is zero.
	fn invert_or_zero(self) -> Self;

	/// Interleaves blocks of this packed vector with another packed vector.
	///
	/// The operation can be seen as stacking the two vectors, dividing them into 2x2 matrices of
	/// blocks, where each block is 2^`log_block_width` elements, and transposing the matrices.
	///
	/// Consider this example, where `LOG_WIDTH` is 3 and `log_block_len` is 1:
	///     A = [a0, a1, a2, a3, a4, a5, a6, a7]
	///     B = [b0, b1, b2, b3, b4, b5, b6, b7]
	///
	/// The interleaved result is
	///     A' = [a0, a1, b0, b1, a4, a5, b4, b5]
	///     B' = [a2, a3, b2, b3, a6, a7, b6, b7]
	///
	/// ## Preconditions
	/// * `log_block_len` must be strictly less than `LOG_WIDTH`.
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self);

	/// Unzips interleaved blocks of this packed vector with another packed vector.
	/// 
	/// Consider this example, where `LOG_WIDTH` is 3 and `log_block_len` is 1:
	///    A = [a0, a1, b0, b1, a2, a3, b2, b3]
	///    B = [a4, a5, b4, b5, a6, a7, b6, b7]
	/// 
	/// The transposed result is
	///    A' = [a0, a1, a2, a3, a4, a5, a6, a7]
	///    B' = [b0, b1, b2, b3, b4, b5, b6, b7]
	///
	/// ## Preconditions
	/// * `log_block_len` must be strictly less than `LOG_WIDTH`.
	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self);

	/// Spread takes a block of elements within a packed field and repeats them to the full packing
	/// width.
	///
	/// Spread can be seen as an extension of the functionality of [`Self::broadcast`].
	///
	/// ## Examples
	///
	/// ```
	/// use binius_field::{BinaryField16b, PackedField, PackedBinaryField8x16b};
	///
	/// let input =
	///     PackedBinaryField8x16b::from_scalars([0, 1, 2, 3, 4, 5, 6, 7].map(BinaryField16b::new));
	/// assert_eq!(
	///     input.spread(0, 5),
	///     PackedBinaryField8x16b::from_scalars([5, 5, 5, 5, 5, 5, 5, 5].map(BinaryField16b::new))
	/// );
	/// assert_eq!(
	///     input.spread(1, 2),
	///     PackedBinaryField8x16b::from_scalars([4, 4, 4, 4, 5, 5, 5, 5].map(BinaryField16b::new))
	/// );
	/// assert_eq!(
	///     input.spread(2, 1),
	///     PackedBinaryField8x16b::from_scalars([4, 4, 5, 5, 6, 6, 7, 7].map(BinaryField16b::new))
	/// );
	/// assert_eq!(input.spread(3, 0), input);
	/// ```
	///
	/// ## Preconditions
	///
	/// * `log_block_len` must be less than or equal to `LOG_WIDTH`.
	/// * `block_idx` must be less than `2^(Self::LOG_WIDTH - log_block_len)`.
	#[inline]
	fn spread(self, log_block_len: usize, block_idx: usize) -> Self {
		assert!(log_block_len <= Self::LOG_WIDTH);
		assert!(block_idx < 1 << (Self::LOG_WIDTH - log_block_len));

		// Safety: is guaranteed by the preconditions.
		unsafe {
			self.spread_unchecked(log_block_len, block_idx)
		}
	}

	/// Unsafe version of [`Self::spread`].
	///
	/// # Safety
	/// The caller must ensure that `log_block_len` is less than or equal to `LOG_WIDTH` and `block_idx` is less than `2^(Self::LOG_WIDTH - log_block_len)`.
	#[inline]
	unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
		let block_len = 1 << log_block_len;
		let repeat = 1 << (Self::LOG_WIDTH - log_block_len);

		Self::from_scalars(
			self.iter().skip(block_idx * block_len).take(block_len).flat_map(|elem| iter::repeat_n(elem, repeat))
		)
	}
}

/// Iterate over scalar values in a packed field slice.
///
/// The iterator skips the first `offset` elements. This is more efficient than skipping elements of the iterator returned.
pub fn iter_packed_slice_with_offset<P: PackedField>(
	packed: &[P],
	offset: usize,
) -> impl Iterator<Item = P::Scalar> + '_ + Send {
	let (packed, offset): (&[P], usize) = if offset < packed.len() * P::WIDTH {
		(&packed[(offset / P::WIDTH)..], offset % P::WIDTH)
	} else {
		(&[], 0)
	};

	P::iter_slice(packed).skip(offset)
}

#[inline]
pub fn get_packed_slice<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	// Safety: `i % P::WIDTH` is always less than `P::WIDTH
	unsafe { packed[i / P::WIDTH].get_unchecked(i % P::WIDTH) }
}

/// Returns the scalar at the given index without bounds checking.
/// # Safety
/// The caller must ensure that `i` is less than `P::WIDTH * packed.len()`.
#[inline]
pub unsafe fn get_packed_slice_unchecked<P: PackedField>(packed: &[P], i: usize) -> P::Scalar {
	packed
		.get_unchecked(i / P::WIDTH)
		.get_unchecked(i % P::WIDTH)
}

pub fn get_packed_slice_checked<P: PackedField>(
	packed: &[P],
	i: usize,
) -> Result<P::Scalar, Error> {
	packed
		.get(i / P::WIDTH)
		.map(|el| el.get(i % P::WIDTH))
		.ok_or(Error::IndexOutOfRange {
			index: i,
			max: packed.len() * P::WIDTH,
		})
}

/// Sets the scalar at the given index without bounds checking.
/// # Safety
/// The caller must ensure that `i` is less than `P::WIDTH * packed.len()`.
pub unsafe fn set_packed_slice_unchecked<P: PackedField>(
	packed: &mut [P],
	i: usize,
	scalar: P::Scalar,
) {
	unsafe {
		packed
			.get_unchecked_mut(i / P::WIDTH)
			.set_unchecked(i % P::WIDTH, scalar)
	}
}

pub fn set_packed_slice<P: PackedField>(packed: &mut [P], i: usize, scalar: P::Scalar) {
	// Safety: `i % P::WIDTH` is always less than `P::WIDTH
	unsafe { packed[i / P::WIDTH].set_unchecked(i % P::WIDTH, scalar) }
}

pub fn set_packed_slice_checked<P: PackedField>(
	packed: &mut [P],
	i: usize,
	scalar: P::Scalar,
) -> Result<(), Error> {
	packed
		.get_mut(i / P::WIDTH)
		.map(|el| el.set(i % P::WIDTH, scalar))
		.ok_or(Error::IndexOutOfRange {
			index: i,
			max: packed.len() * P::WIDTH,
		})
}

pub const fn len_packed_slice<P: PackedField>(packed: &[P]) -> usize {
	packed.len() * P::WIDTH
}

/// Multiply packed field element by a subfield scalar.
pub fn mul_by_subfield_scalar<P, FS>(val: P, multiplier: FS) -> P
where
	P: PackedExtension<FS, Scalar: ExtensionField<FS>>,
	FS: Field,
{
	use crate::underlier::UnderlierType;

	// This is a workaround not to make the multiplication slower in certain cases.
	// TODO: implement efficient strategy to multiply packed field by a subfield scalar.
	let subfield_bits = FS::Underlier::BITS;
	let extension_bits = <<P as PackedField>::Scalar as WithUnderlier>::Underlier::BITS;

	if (subfield_bits == 1 && extension_bits > 8) || extension_bits >= 32 {
		P::from_fn(|i| unsafe { val.get_unchecked(i) } * multiplier)
	} else {
		P::cast_ext(P::cast_base(val) * P::PackedSubfield::broadcast(multiplier))
	}
}

impl<F: Field> Broadcast<F> for F {
	fn broadcast(scalar: F) -> Self {
		scalar
	}
}

impl<T: TowerFieldArithmetic> MulAlpha for T {
	#[inline]
	fn mul_alpha(self) -> Self {
		<Self as TowerFieldArithmetic>::multiply_alpha(self)
	}
}

impl<F: Field> PackedField for F {
	type Scalar = F;

	const LOG_WIDTH: usize = 0;

	#[inline]
	unsafe fn get_unchecked(&self, _i: usize) -> Self::Scalar {
		*self
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, _i: usize, scalar: Self::Scalar) {
		*self = scalar;
	}

	#[inline]
	fn iter(&self) -> impl Iterator<Item = Self::Scalar> + Send + '_ {
		iter::once(*self)
	}

	#[inline]
	fn into_iter(self) -> impl Iterator<Item = Self::Scalar> + Send {
		iter::once(self)
	}

	#[inline]
	fn iter_slice(slice: &[Self]) -> impl Iterator<Item = Self::Scalar> + Send + '_ {
		slice.iter().copied()
	}

	fn random(rng: impl RngCore) -> Self {
		<Self as Field>::random(rng)
	}

	fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
		panic!("cannot interleave when WIDTH = 1");
	}

	fn unzip(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
		panic!("cannot transpose when WIDTH = 1");
	}

	fn broadcast(scalar: Self::Scalar) -> Self {
		scalar
	}

	fn square(self) -> Self {
		<Self as Square>::square(self)
	}

	fn invert_or_zero(self) -> Self {
		<Self as InvertOrZero>::invert_or_zero(self)
	}

	#[inline]
	fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
		f(0)
	}

	#[inline]
	unsafe fn spread_unchecked(self, _log_block_len: usize, _block_idx: usize) -> Self {
		self
	}
}

/// A helper trait to make the generic bunds shorter
pub trait PackedBinaryField: PackedField<Scalar: BinaryField> {}

impl<PT> PackedBinaryField for PT where PT: PackedField<Scalar: BinaryField> {}

#[cfg(test)]
mod tests {
	use rand::{
		distributions::{Distribution, Uniform},
		rngs::StdRng,
		SeedableRng,
	};

	use super::*;
	use crate::{
		AESTowerField128b, AESTowerField16b, AESTowerField32b, AESTowerField64b, AESTowerField8b,
		BinaryField128b, BinaryField128bPolyval, BinaryField16b, BinaryField1b, BinaryField2b,
		BinaryField32b, BinaryField4b, BinaryField64b, BinaryField8b, ByteSlicedAES32x128b,
		ByteSlicedAES32x16b, ByteSlicedAES32x32b, ByteSlicedAES32x64b, ByteSlicedAES32x8b,
		PackedBinaryField128x1b, PackedBinaryField128x2b, PackedBinaryField128x4b,
		PackedBinaryField16x16b, PackedBinaryField16x1b, PackedBinaryField16x2b,
		PackedBinaryField16x32b, PackedBinaryField16x4b, PackedBinaryField16x8b,
		PackedBinaryField1x128b, PackedBinaryField1x16b, PackedBinaryField1x1b,
		PackedBinaryField1x2b, PackedBinaryField1x32b, PackedBinaryField1x4b,
		PackedBinaryField1x64b, PackedBinaryField1x8b, PackedBinaryField256x1b,
		PackedBinaryField256x2b, PackedBinaryField2x128b, PackedBinaryField2x16b,
		PackedBinaryField2x1b, PackedBinaryField2x2b, PackedBinaryField2x32b,
		PackedBinaryField2x4b, PackedBinaryField2x64b, PackedBinaryField2x8b,
		PackedBinaryField32x16b, PackedBinaryField32x1b, PackedBinaryField32x2b,
		PackedBinaryField32x4b, PackedBinaryField32x8b, PackedBinaryField4x128b,
		PackedBinaryField4x16b, PackedBinaryField4x1b, PackedBinaryField4x2b,
		PackedBinaryField4x32b, PackedBinaryField4x4b, PackedBinaryField4x64b,
		PackedBinaryField4x8b, PackedBinaryField512x1b, PackedBinaryField64x1b,
		PackedBinaryField64x2b, PackedBinaryField64x4b, PackedBinaryField64x8b,
		PackedBinaryField8x16b, PackedBinaryField8x1b, PackedBinaryField8x2b,
		PackedBinaryField8x32b, PackedBinaryField8x4b, PackedBinaryField8x64b,
		PackedBinaryField8x8b, PackedBinaryPolyval1x128b, PackedBinaryPolyval2x128b,
		PackedBinaryPolyval4x128b, PackedField,
	};

	trait PackedFieldTest {
		fn run<P: PackedField>(&self);
	}

	/// Run the test for all the packed fields defined in this crate.
	fn run_for_all_packed_fields(test: &impl PackedFieldTest) {
		// canonical tower

		test.run::<BinaryField1b>();
		test.run::<BinaryField2b>();
		test.run::<BinaryField4b>();
		test.run::<BinaryField8b>();
		test.run::<BinaryField16b>();
		test.run::<BinaryField32b>();
		test.run::<BinaryField64b>();
		test.run::<BinaryField128b>();

		// packed canonical tower
		test.run::<PackedBinaryField1x1b>();
		test.run::<PackedBinaryField2x1b>();
		test.run::<PackedBinaryField1x2b>();
		test.run::<PackedBinaryField4x1b>();
		test.run::<PackedBinaryField2x2b>();
		test.run::<PackedBinaryField1x4b>();
		test.run::<PackedBinaryField8x1b>();
		test.run::<PackedBinaryField4x2b>();
		test.run::<PackedBinaryField2x4b>();
		test.run::<PackedBinaryField1x8b>();
		test.run::<PackedBinaryField16x1b>();
		test.run::<PackedBinaryField8x2b>();
		test.run::<PackedBinaryField4x4b>();
		test.run::<PackedBinaryField2x8b>();
		test.run::<PackedBinaryField1x16b>();
		test.run::<PackedBinaryField32x1b>();
		test.run::<PackedBinaryField16x2b>();
		test.run::<PackedBinaryField8x4b>();
		test.run::<PackedBinaryField4x8b>();
		test.run::<PackedBinaryField2x16b>();
		test.run::<PackedBinaryField1x32b>();
		test.run::<PackedBinaryField64x1b>();
		test.run::<PackedBinaryField32x2b>();
		test.run::<PackedBinaryField16x4b>();
		test.run::<PackedBinaryField8x8b>();
		test.run::<PackedBinaryField4x16b>();
		test.run::<PackedBinaryField2x32b>();
		test.run::<PackedBinaryField1x64b>();
		test.run::<PackedBinaryField128x1b>();
		test.run::<PackedBinaryField64x2b>();
		test.run::<PackedBinaryField32x4b>();
		test.run::<PackedBinaryField16x8b>();
		test.run::<PackedBinaryField8x16b>();
		test.run::<PackedBinaryField4x32b>();
		test.run::<PackedBinaryField2x64b>();
		test.run::<PackedBinaryField1x128b>();
		test.run::<PackedBinaryField256x1b>();
		test.run::<PackedBinaryField128x2b>();
		test.run::<PackedBinaryField64x4b>();
		test.run::<PackedBinaryField32x8b>();
		test.run::<PackedBinaryField16x16b>();
		test.run::<PackedBinaryField8x32b>();
		test.run::<PackedBinaryField4x64b>();
		test.run::<PackedBinaryField2x128b>();
		test.run::<PackedBinaryField512x1b>();
		test.run::<PackedBinaryField256x2b>();
		test.run::<PackedBinaryField128x4b>();
		test.run::<PackedBinaryField64x8b>();
		test.run::<PackedBinaryField32x16b>();
		test.run::<PackedBinaryField16x32b>();
		test.run::<PackedBinaryField8x64b>();
		test.run::<PackedBinaryField4x128b>();

		// AES tower
		test.run::<AESTowerField8b>();
		test.run::<AESTowerField16b>();
		test.run::<AESTowerField32b>();
		test.run::<AESTowerField64b>();
		test.run::<AESTowerField128b>();

		// packed AES tower
		test.run::<PackedBinaryField1x8b>();
		test.run::<PackedBinaryField2x8b>();
		test.run::<PackedBinaryField1x16b>();
		test.run::<PackedBinaryField4x8b>();
		test.run::<PackedBinaryField2x16b>();
		test.run::<PackedBinaryField1x32b>();
		test.run::<PackedBinaryField8x8b>();
		test.run::<PackedBinaryField4x16b>();
		test.run::<PackedBinaryField2x32b>();
		test.run::<PackedBinaryField1x64b>();
		test.run::<PackedBinaryField16x8b>();
		test.run::<PackedBinaryField8x16b>();
		test.run::<PackedBinaryField4x32b>();
		test.run::<PackedBinaryField2x64b>();
		test.run::<PackedBinaryField1x128b>();
		test.run::<PackedBinaryField32x8b>();
		test.run::<PackedBinaryField16x16b>();
		test.run::<PackedBinaryField8x32b>();
		test.run::<PackedBinaryField4x64b>();
		test.run::<PackedBinaryField2x128b>();
		test.run::<PackedBinaryField64x8b>();
		test.run::<PackedBinaryField32x16b>();
		test.run::<PackedBinaryField16x32b>();
		test.run::<PackedBinaryField8x64b>();
		test.run::<PackedBinaryField4x128b>();
		test.run::<ByteSlicedAES32x8b>();
		test.run::<ByteSlicedAES32x64b>();
		test.run::<ByteSlicedAES32x16b>();
		test.run::<ByteSlicedAES32x32b>();
		test.run::<ByteSlicedAES32x128b>();

		// polyval tower
		test.run::<BinaryField128bPolyval>();

		// packed polyval tower
		test.run::<PackedBinaryPolyval1x128b>();
		test.run::<PackedBinaryPolyval2x128b>();
		test.run::<PackedBinaryPolyval4x128b>();
	}

	fn check_value_iteration<P: PackedField>(mut rng: impl RngCore) {
		let packed = P::random(&mut rng);
		let mut iter = packed.iter();
		for i in 0..P::WIDTH {
			assert_eq!(packed.get(i), iter.next().unwrap());
		}
		assert!(iter.next().is_none());
	}

	fn check_ref_iteration<P: PackedField>(mut rng: impl RngCore) {
		let packed = P::random(&mut rng);
		let mut iter = packed.into_iter();
		for i in 0..P::WIDTH {
			assert_eq!(packed.get(i), iter.next().unwrap());
		}
		assert!(iter.next().is_none());
	}

	fn check_slice_iteration<P: PackedField>(mut rng: impl RngCore) {
		for len in [0, 1, 5] {
			let packed = std::iter::repeat_with(|| P::random(&mut rng))
				.take(len)
				.collect::<Vec<_>>();

			let elements_count = len * P::WIDTH;
			for offset in [
				0,
				1,
				Uniform::new(0, elements_count.max(1)).sample(&mut rng),
				elements_count.saturating_sub(1),
				elements_count,
			] {
				let actual = iter_packed_slice_with_offset(&packed, offset).collect::<Vec<_>>();
				let expected = (offset..elements_count)
					.map(|i| get_packed_slice(&packed, i))
					.collect::<Vec<_>>();

				assert_eq!(actual, expected);
			}
		}
	}

	struct PackedFieldIterationTest;

	impl PackedFieldTest for PackedFieldIterationTest {
		fn run<P: PackedField>(&self) {
			let mut rng = StdRng::seed_from_u64(0);

			check_value_iteration::<P>(&mut rng);
			check_ref_iteration::<P>(&mut rng);
			check_slice_iteration::<P>(&mut rng);
		}
	}

	#[test]
	fn test_iteration() {
		run_for_all_packed_fields(&PackedFieldIterationTest);
	}
}
