// Copyright 2025 Irreducible Inc.

use core::arch::aarch64::*;
use std::{
	fmt::{Display, Formatter, Result as FmtResult},
	hash::{Hash, Hasher},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
	DeserializeBytes, SerializationMode, SerializeBytes,
	bytes::{Buf, BufMut},
};
use bytemuck::{Pod, Zeroable};

use crate::{
	BinaryField, BinaryField1b, BinaryField64b, ExtensionField, Field, PackedField, TowerField,
	arch::{
		SimdStrategy,
		aarch64::simd_arithmetic::{flip_even_odd, shift_left},
		m128::M128,
		portable::{packed::PackedPrimitiveType, packed_arithmetic::UnderlierWithBitConstants},
	},
	arithmetic_traits::{InvertOrZero, MulAlpha, Square, TaggedMul},
	tower_levels::TowerLevel2,
	underlier::WithUnderlier,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Zeroable, Pod, Default, Hash)]
#[repr(transparent)]
pub struct NISTBinaryField64(u64);

unsafe impl WithUnderlier for NISTBinaryField64 {
	type Underlier = u64;

	fn to_underlier(self) -> Self::Underlier {
		self.0
	}

	fn to_underlier_ref(&self) -> &Self::Underlier {
		&self.0
	}

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		&mut self.0
	}

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		bytemuck::must_cast_slice(val)
	}

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		bytemuck::must_cast_slice_mut(val)
	}

	fn from_underlier(val: Self::Underlier) -> Self {
		Self(val)
	}

	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		bytemuck::must_cast_ref(val)
	}

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		bytemuck::must_cast_mut(val)
	}

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		bytemuck::must_cast_slice(val)
	}

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		bytemuck::must_cast_slice_mut(val)
	}
}

impl Field for NISTBinaryField64 {
	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const CHARACTERISTIC: usize = 2;

	fn random(mut rng: impl rand::RngCore) -> Self {
		Self(rng.next_u64())
	}

	fn double(&self) -> Self {
		Self(0)
	}
}

impl TowerField for NISTBinaryField64 {
	const TOWER_LEVEL: usize = 6;
	type Canonical = BinaryField64b;

	fn min_tower_level(self) -> usize {
		todo!()
	}
}

impl From<BinaryField64b> for NISTBinaryField64 {
	fn from(value: BinaryField64b) -> Self {
		todo!()
	}
}

impl From<NISTBinaryField64> for BinaryField64b {
	fn from(value: NISTBinaryField64) -> Self {
		todo!()
	}
}

impl Display for NISTBinaryField64 {
	fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
		todo!()
	}
}

impl Neg for NISTBinaryField64 {
	type Output = Self;
	fn neg(self) -> Self::Output {
		todo!()
	}
}

impl<'a> Add<&'a NISTBinaryField64> for NISTBinaryField64 {
	type Output = Self;
	fn add(self, _rhs: &'a NISTBinaryField64) -> Self::Output {
		todo!()
	}
}

impl<'a> Sub<&'a NISTBinaryField64> for NISTBinaryField64 {
	type Output = Self;
	fn sub(self, _rhs: &'a NISTBinaryField64) -> Self::Output {
		todo!()
	}
}

impl<'a> Mul<&'a NISTBinaryField64> for NISTBinaryField64 {
	type Output = Self;
	fn mul(self, _rhs: &'a NISTBinaryField64) -> Self::Output {
		todo!()
	}
}

impl<'a> AddAssign<&'a NISTBinaryField64> for NISTBinaryField64 {
	fn add_assign(&mut self, _rhs: &'a NISTBinaryField64) {
		todo!()
	}
}

impl<'a> SubAssign<&'a NISTBinaryField64> for NISTBinaryField64 {
	fn sub_assign(&mut self, _rhs: &'a NISTBinaryField64) {
		todo!()
	}
}

impl<'a> MulAssign<&'a NISTBinaryField64> for NISTBinaryField64 {
	fn mul_assign(&mut self, _rhs: &'a NISTBinaryField64) {
		todo!()
	}
}

impl MulAssign for NISTBinaryField64 {
	fn mul_assign(&mut self, _rhs: Self) {
		todo!()
	}
}

impl Sum for NISTBinaryField64 {
	fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
		todo!()
	}
}

impl<'a> Sum<&'a NISTBinaryField64> for NISTBinaryField64 {
	fn sum<I: Iterator<Item = &'a NISTBinaryField64>>(_iter: I) -> Self {
		todo!()
	}
}

impl Product for NISTBinaryField64 {
	fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
		todo!()
	}
}

impl<'a> Product<&'a NISTBinaryField64> for NISTBinaryField64 {
	fn product<I: Iterator<Item = &'a NISTBinaryField64>>(_iter: I) -> Self {
		todo!()
	}
}

// Empty trait impls for required field traits
impl Square for NISTBinaryField64 {
	fn square(self) -> Self {
		todo!()
	}
}

impl InvertOrZero for NISTBinaryField64 {
	fn invert_or_zero(self) -> Self {
		todo!()
	}
}

impl MulAlpha for NISTBinaryField64 {
	fn mul_alpha(self) -> Self {
		let msb = self.0 >> 63;
		const POLY: u64 = (1u64 << 4) | (1u64 << 3) | 1u64;

		Self((self.0 << 1) ^ (msb.wrapping_neg() & POLY))
	}
}

impl ExtensionField<BinaryField1b> for NISTBinaryField64 {
	const LOG_DEGREE: usize = 6;

	fn basis_checked(i: usize) -> Result<Self, crate::Error> {
		unimplemented!()
	}

	fn from_bases_sparse(
		_base_elems: impl IntoIterator<Item = BinaryField1b>,
		_log_stride: usize,
	) -> Result<Self, crate::Error> {
		unimplemented!()
	}

	fn iter_bases(&self) -> impl Iterator<Item = BinaryField1b> {
		unimplemented!();
		std::iter::empty()
	}

	fn into_iter_bases(self) -> impl Iterator<Item = BinaryField1b> {
		unimplemented!();
		std::iter::empty()
	}

	unsafe fn get_base_unchecked(&self, _i: usize) -> BinaryField1b {
		unimplemented!()
	}

	unsafe fn set_base_unchecked(&mut self, _i: usize, _val: BinaryField1b) {
		unimplemented!()
	}
}

impl BinaryField for NISTBinaryField64 {
	const MULTIPLICATIVE_GENERATOR: Self = Self(1);
}

impl Add for NISTBinaryField64 {
	type Output = Self;

	fn add(self, other: Self) -> Self::Output {
		Self(self.0 ^ other.0)
	}
}

impl AddAssign for NISTBinaryField64 {
	fn add_assign(&mut self, other: Self) {
		self.0 ^= other.0;
	}
}

impl Sub for NISTBinaryField64 {
	type Output = Self;

	fn sub(self, other: Self) -> Self::Output {
		Self(self.0 ^ other.0)
	}
}

impl SubAssign for NISTBinaryField64 {
	fn sub_assign(&mut self, other: Self) {
		self.0 ^= other.0;
	}
}

impl Mul for NISTBinaryField64 {
	type Output = Self;

	fn mul(self, other: Self) -> Self::Output {
		let prod = unsafe { vmull_p64(self.0, other.0) };
		let mut lo = prod as u64;
		let hi = (prod >> 64) as u64;

		lo ^= hi;
		lo ^= hi << 1;
		lo ^= hi << 3;
		lo ^= hi << 4;

		Self(lo)
	}
}

impl DeserializeBytes for NISTBinaryField64 {
	fn deserialize(
		read_buf: impl Buf,
		mode: SerializationMode,
	) -> Result<Self, binius_utils::SerializationError>
	where
		Self: Sized,
	{
		todo!()
	}
}

impl SerializeBytes for NISTBinaryField64 {
	fn serialize(
		&self,
		mut write_buf: impl BufMut,
		mode: SerializationMode,
	) -> Result<(), binius_utils::SerializationError> {
		todo!()
	}
}

// Operator impls for NISTBinaryField64 <op> BinaryField1b
impl Add<BinaryField1b> for NISTBinaryField64 {
	type Output = Self;
	fn add(self, _rhs: BinaryField1b) -> Self::Output {
		todo!()
	}
}
impl Sub<BinaryField1b> for NISTBinaryField64 {
	type Output = Self;
	fn sub(self, _rhs: BinaryField1b) -> Self::Output {
		todo!()
	}
}
impl Mul<BinaryField1b> for NISTBinaryField64 {
	type Output = Self;
	fn mul(self, _rhs: BinaryField1b) -> Self::Output {
		todo!()
	}
}
impl AddAssign<BinaryField1b> for NISTBinaryField64 {
	fn add_assign(&mut self, _rhs: BinaryField1b) {
		todo!()
	}
}
impl SubAssign<BinaryField1b> for NISTBinaryField64 {
	fn sub_assign(&mut self, _rhs: BinaryField1b) {
		todo!()
	}
}
impl MulAssign<BinaryField1b> for NISTBinaryField64 {
	fn mul_assign(&mut self, _rhs: BinaryField1b) {
		todo!()
	}
}
// Conversion impls
impl From<BinaryField1b> for NISTBinaryField64 {
	fn from(_val: BinaryField1b) -> Self {
		todo!()
	}
}
impl From<NISTBinaryField64> for BinaryField1b {
	fn from(_val: NISTBinaryField64) -> Self {
		todo!()
	}
}

pub type PackedNISTBinaryField1x64b = PackedPrimitiveType<u64, NISTBinaryField64>;
pub type PackedNISTBinaryField2x64b = PackedPrimitiveType<u128, NISTBinaryField64>;

// Implement Broadcast for packed NIST field types
use crate::arithmetic_traits::Broadcast;

impl Broadcast<NISTBinaryField64> for PackedPrimitiveType<u64, NISTBinaryField64> {
	fn broadcast(scalar: NISTBinaryField64) -> Self {
		Self::from_underlier(scalar.0)
	}
}

impl Broadcast<NISTBinaryField64> for PackedPrimitiveType<u128, NISTBinaryField64> {
	fn broadcast(scalar: NISTBinaryField64) -> Self {
		Self::from_underlier((scalar.0 as u128) << 64 | (scalar.0 as u128))
	}
}

impl crate::arithmetic_traits::Square for PackedPrimitiveType<u64, NISTBinaryField64> {
	fn square(self) -> Self {
		todo!()
	}
}

impl crate::arithmetic_traits::Square for PackedPrimitiveType<u128, NISTBinaryField64> {
	fn square(self) -> Self {
		todo!()
	}
}

impl MulAlpha for PackedPrimitiveType<u128, NISTBinaryField64> {
	fn mul_alpha(self) -> Self {
		let mut res: uint64x2_t = M128::from(self.0).into();
		let msbs = unsafe { vshrq_n_u64(res, 63) };
		let zero = unsafe { vdupq_n_u64(0) };
		let poly = unsafe { vdupq_n_u64((1u64 << 4) | (1u64 << 3) | 1u64) };
		let mask = unsafe { vsubq_u64(zero, msbs) };

		Self::from_underlier(M128::from(unsafe { veorq_u64(res, vandq_u64(mask, poly)) }).into())
	}
}

impl crate::arithmetic_traits::InvertOrZero for PackedPrimitiveType<u64, NISTBinaryField64> {
	fn invert_or_zero(self) -> Self {
		todo!()
	}
}

impl crate::arithmetic_traits::InvertOrZero for PackedPrimitiveType<u128, NISTBinaryField64> {
	fn invert_or_zero(self) -> Self {
		todo!()
	}
}

impl std::ops::Mul for PackedPrimitiveType<u64, NISTBinaryField64> {
	type Output = Self;
	fn mul(self, rhs: Self) -> Self::Output {
		Self::set_single(NISTBinaryField64(self.0) * NISTBinaryField64(rhs.0))
	}
}

impl std::ops::Mul for PackedPrimitiveType<u128, NISTBinaryField64> {
	type Output = Self;
	fn mul(self, rhs: Self) -> Self::Output {
		let lhs: uint64x2_t = M128::from(self.0).into();
		let rhs: uint64x2_t = M128::from(rhs.0).into();

		unsafe {
			let lo = NISTBinaryField64(vgetq_lane_u64(lhs, 0))
				* NISTBinaryField64(vgetq_lane_u64(rhs, 0));
			let hi = NISTBinaryField64(vgetq_lane_u64(lhs, 1))
				* NISTBinaryField64(vgetq_lane_u64(rhs, 1));

			Self::from_underlier(std::mem::transmute([lo, hi]))
		}

		// let prod_0 = unsafe { vmull_p64(vgetq_lane_u64(lhs, 0), vgetq_lane_u64(rhs, 0)) };
		// let prod_1 = unsafe { vmull_p64(vgetq_lane_u64(lhs, 1), vgetq_lane_u64(rhs, 1)) };
		// let prod_0 = M128::from(prod_0);
		// let prod_1 = M128::from(prod_1);

		// let (lo, hi) = prod_0.interleave(prod_1, 6);

		// let (r_lo, r_hi): (uint64x2_t, uint64x2_t) = (lo.into(), hi.into());

		// unsafe {
		// 	let mut res = veorq_u64(
		// 		veorq_u64(veorq_u64(r_lo, r_hi), vshlq_n_u64(r_hi, 1)),
		// 		veorq_u64(vshlq_n_u64(r_hi, 3), vshlq_n_u64(r_hi, 4)),
		// 	);

		// 	let carry = vshrq_n_u64(r_hi, 60); // top-4 bits of each lane
		// 	let fold = veorq_u64(
		// 		veorq_u64(carry, vshlq_n_u64(carry, 1)),
		// 		veorq_u64(vshlq_n_u64(carry, 3), vshlq_n_u64(carry, 4)),
		// 	);

		// 	res = veorq_u64(res, fold);

		// 	Self::from_underlier(M128::from(res).into())
		// }
	}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Zeroable, Pod, Default, Hash)]
#[repr(transparent)]
pub struct NISTBinaryField128(u128);

unsafe impl WithUnderlier for NISTBinaryField128 {
	type Underlier = u128;

	fn to_underlier(self) -> Self::Underlier {
		self.0
	}

	fn to_underlier_ref(&self) -> &Self::Underlier {
		&self.0
	}

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		&mut self.0
	}

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		bytemuck::must_cast_slice(val)
	}

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		bytemuck::must_cast_slice_mut(val)
	}

	fn from_underlier(val: Self::Underlier) -> Self {
		Self(val)
	}

	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		bytemuck::must_cast_ref(val)
	}

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		bytemuck::must_cast_mut(val)
	}

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		bytemuck::must_cast_slice(val)
	}

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		bytemuck::must_cast_slice_mut(val)
	}
}

impl Field for NISTBinaryField128 {
	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const CHARACTERISTIC: usize = 2;
	fn random(_rng: impl rand::RngCore) -> Self {
		unimplemented!()
	}
	fn double(&self) -> Self {
		unimplemented!()
	}
}

impl Display for NISTBinaryField128 {
	fn fmt(&self, _f: &mut Formatter<'_>) -> FmtResult {
		unimplemented!()
	}
}

impl Neg for NISTBinaryField128 {
	type Output = Self;
	fn neg(self) -> Self::Output {
		unimplemented!()
	}
}

impl<'a> Add<&'a NISTBinaryField128> for NISTBinaryField128 {
	type Output = Self;
	fn add(self, _rhs: &'a NISTBinaryField128) -> Self::Output {
		unimplemented!()
	}
}

impl<'a> Sub<&'a NISTBinaryField128> for NISTBinaryField128 {
	type Output = Self;
	fn sub(self, _rhs: &'a NISTBinaryField128) -> Self::Output {
		unimplemented!()
	}
}

impl<'a> Mul<&'a NISTBinaryField128> for NISTBinaryField128 {
	type Output = Self;
	fn mul(self, _rhs: &'a NISTBinaryField128) -> Self::Output {
		unimplemented!()
	}
}

impl<'a> AddAssign<&'a NISTBinaryField128> for NISTBinaryField128 {
	fn add_assign(&mut self, _rhs: &'a NISTBinaryField128) {
		unimplemented!()
	}
}

impl<'a> SubAssign<&'a NISTBinaryField128> for NISTBinaryField128 {
	fn sub_assign(&mut self, _rhs: &'a NISTBinaryField128) {
		unimplemented!()
	}
}

impl<'a> MulAssign<&'a NISTBinaryField128> for NISTBinaryField128 {
	fn mul_assign(&mut self, _rhs: &'a NISTBinaryField128) {
		unimplemented!()
	}
}

impl MulAssign for NISTBinaryField128 {
	fn mul_assign(&mut self, _rhs: Self) {
		unimplemented!()
	}
}

impl Sum for NISTBinaryField128 {
	fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
		unimplemented!()
	}
}

impl<'a> Sum<&'a NISTBinaryField128> for NISTBinaryField128 {
	fn sum<I: Iterator<Item = &'a NISTBinaryField128>>(_iter: I) -> Self {
		unimplemented!()
	}
}

impl Product for NISTBinaryField128 {
	fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
		unimplemented!()
	}
}

impl<'a> Product<&'a NISTBinaryField128> for NISTBinaryField128 {
	fn product<I: Iterator<Item = &'a NISTBinaryField128>>(_iter: I) -> Self {
		unimplemented!()
	}
}

impl Square for NISTBinaryField128 {
	fn square(self) -> Self {
		unimplemented!()
	}
}

impl InvertOrZero for NISTBinaryField128 {
	fn invert_or_zero(self) -> Self {
		unimplemented!()
	}
}

impl ExtensionField<BinaryField1b> for NISTBinaryField128 {
	const LOG_DEGREE: usize = 7;
	fn basis_checked(_i: usize) -> Result<Self, crate::Error> {
		unimplemented!()
	}
	fn from_bases_sparse(
		_base_elems: impl IntoIterator<Item = BinaryField1b>,
		_log_stride: usize,
	) -> Result<Self, crate::Error> {
		unimplemented!()
	}
	fn iter_bases(&self) -> impl Iterator<Item = BinaryField1b> {
		std::iter::empty()
	}
	fn into_iter_bases(self) -> impl Iterator<Item = BinaryField1b> {
		std::iter::empty()
	}
	unsafe fn get_base_unchecked(&self, _i: usize) -> BinaryField1b {
		unimplemented!()
	}
	unsafe fn set_base_unchecked(&mut self, _i: usize, _val: BinaryField1b) {
		unimplemented!()
	}
}

impl BinaryField for NISTBinaryField128 {
	const MULTIPLICATIVE_GENERATOR: Self = Self(1);
}

impl Add for NISTBinaryField128 {
	type Output = Self;
	fn add(self, _other: Self) -> Self::Output {
		unimplemented!()
	}
}

impl AddAssign for NISTBinaryField128 {
	fn add_assign(&mut self, _other: Self) {
		unimplemented!()
	}
}

impl Sub for NISTBinaryField128 {
	type Output = Self;
	fn sub(self, _other: Self) -> Self::Output {
		unimplemented!()
	}
}

impl SubAssign for NISTBinaryField128 {
	fn sub_assign(&mut self, _other: Self) {
		unimplemented!()
	}
}

impl Mul for NISTBinaryField128 {
	type Output = Self;
	fn mul(self, _other: Self) -> Self::Output {
		unimplemented!()
	}
}

impl DeserializeBytes for NISTBinaryField128 {
	fn deserialize(
		_read_buf: impl Buf,
		_mode: SerializationMode,
	) -> Result<Self, binius_utils::SerializationError>
	where
		Self: Sized,
	{
		unimplemented!()
	}
}

impl SerializeBytes for NISTBinaryField128 {
	fn serialize(
		&self,
		_write_buf: impl BufMut,
		_mode: SerializationMode,
	) -> Result<(), binius_utils::SerializationError> {
		unimplemented!()
	}
}

impl Add<BinaryField1b> for NISTBinaryField128 {
	type Output = Self;
	fn add(self, _rhs: BinaryField1b) -> Self::Output {
		unimplemented!()
	}
}
impl Sub<BinaryField1b> for NISTBinaryField128 {
	type Output = Self;
	fn sub(self, _rhs: BinaryField1b) -> Self::Output {
		unimplemented!()
	}
}
impl Mul<BinaryField1b> for NISTBinaryField128 {
	type Output = Self;
	fn mul(self, _rhs: BinaryField1b) -> Self::Output {
		unimplemented!()
	}
}
impl AddAssign<BinaryField1b> for NISTBinaryField128 {
	fn add_assign(&mut self, _rhs: BinaryField1b) {
		unimplemented!()
	}
}
impl SubAssign<BinaryField1b> for NISTBinaryField128 {
	fn sub_assign(&mut self, _rhs: BinaryField1b) {
		unimplemented!()
	}
}
impl MulAssign<BinaryField1b> for NISTBinaryField128 {
	fn mul_assign(&mut self, _rhs: BinaryField1b) {
		unimplemented!()
	}
}
impl From<BinaryField1b> for NISTBinaryField128 {
	fn from(_val: BinaryField1b) -> Self {
		unimplemented!()
	}
}
impl From<NISTBinaryField128> for BinaryField1b {
	fn from(_val: NISTBinaryField128) -> Self {
		unimplemented!()
	}
}

pub type PackedNISTBinaryField1x128b = PackedPrimitiveType<u128, NISTBinaryField128>;

impl Broadcast<NISTBinaryField128> for PackedPrimitiveType<u128, NISTBinaryField128> {
	fn broadcast(scalar: NISTBinaryField128) -> Self {
		Self::from_underlier((scalar.0 as u128) << 64 | (scalar.0 as u128))
	}
}

impl Square for PackedPrimitiveType<u128, NISTBinaryField128> {
	fn square(self) -> Self {
		todo!()
	}
}

impl InvertOrZero for PackedPrimitiveType<u128, NISTBinaryField128> {
	fn invert_or_zero(self) -> Self {
		todo!()
	}
}

impl Mul for PackedPrimitiveType<u128, NISTBinaryField128> {
	type Output = Self;
	fn mul(self, rhs: Self) -> Self::Output {
		let lhs: uint64x2_t = M128::from(self.0).into();
		let rhs: uint64x2_t = M128::from(rhs.0).into();
		let a = unsafe { vgetq_lane_u64(lhs, 0) };
		let b = unsafe { vgetq_lane_u64(lhs, 1) };
		let c = unsafe { vgetq_lane_u64(rhs, 0) };
		let d = unsafe { vgetq_lane_u64(rhs, 1) };

		let ac = NISTBinaryField64(a) * NISTBinaryField64(c);
		let bd = NISTBinaryField64(b) * NISTBinaryField64(d);
		let ab_cd = NISTBinaryField64(a ^ b) * NISTBinaryField64(c ^ d);
		let bd_alpha = bd.mul_alpha();
		let res_0 = ac.0 ^ bd.0;
		let arr = [res_0, ab_cd.0 ^ res_0 ^ bd_alpha.0];
		let res: uint64x2_t = unsafe { vld1q_u64(arr.as_ptr()) };

		Self::from_underlier(M128::from(res).into())
	}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Zeroable, Pod, Default)]
#[repr(transparent)]
pub struct PackedNISTBinaryField2x128b {
	data: [PackedNISTBinaryField2x64b; 2],
}

impl PackedField for PackedNISTBinaryField2x128b {
	type Scalar = NISTBinaryField128;

	const LOG_WIDTH: usize = 1;

	unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
		todo!()
	}

	unsafe fn set_unchecked(&mut self, i: usize, scalar: Self::Scalar) {
		todo!()
	}

	fn random(mut rng: impl rand::RngCore) -> Self {
		Self {
			data: [PackedField::random(&mut rng), PackedField::random(&mut rng)],
		}
	}

	fn broadcast(scalar: Self::Scalar) -> Self {
		todo!()
	}

	fn from_fn(f: impl FnMut(usize) -> Self::Scalar) -> Self {
		todo!()
	}

	fn square(self) -> Self {
		todo!()
	}

	fn invert_or_zero(self) -> Self {
		todo!()
	}

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		todo!()
	}

	fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
		todo!()
	}
}

// --- Begin trait impls for PackedNISTBinaryField2x128b ---
// These are required for PackedField trait bounds and byte-sliced field compatibility.
// All methods are stubs with unimplemented!(), as requested.

impl std::ops::Add for PackedNISTBinaryField2x128b {
	type Output = Self;
	fn add(self, _rhs: Self) -> Self {
		unimplemented!()
	}
}
impl std::ops::AddAssign for PackedNISTBinaryField2x128b {
	fn add_assign(&mut self, _rhs: Self) {
		unimplemented!()
	}
}
impl std::ops::Sub for PackedNISTBinaryField2x128b {
	type Output = Self;
	fn sub(self, _rhs: Self) -> Self {
		unimplemented!()
	}
}
impl std::ops::SubAssign for PackedNISTBinaryField2x128b {
	fn sub_assign(&mut self, _rhs: Self) {
		unimplemented!()
	}
}
impl std::ops::Mul for PackedNISTBinaryField2x128b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		let ac = self.data[0] * rhs.data[0];
		let bd = self.data[1] * rhs.data[1];
		let ab_cd = (self.data[0] + self.data[1]) * (rhs.data[0] + rhs.data[1]);
		let bd_alpha = bd.mul_alpha();
		let res_0 = ac + bd;
		let res_1 = ab_cd + res_0 + bd_alpha;

		Self {
			data: [res_0, res_1],
		}
	}
}
impl std::ops::MulAssign for PackedNISTBinaryField2x128b {
	fn mul_assign(&mut self, _rhs: Self) {
		unimplemented!()
	}
}
impl std::ops::Add<NISTBinaryField128> for PackedNISTBinaryField2x128b {
	type Output = Self;
	fn add(self, _rhs: NISTBinaryField128) -> Self {
		unimplemented!()
	}
}
impl std::ops::AddAssign<NISTBinaryField128> for PackedNISTBinaryField2x128b {
	fn add_assign(&mut self, _rhs: NISTBinaryField128) {
		unimplemented!()
	}
}
impl std::ops::Sub<NISTBinaryField128> for PackedNISTBinaryField2x128b {
	type Output = Self;
	fn sub(self, _rhs: NISTBinaryField128) -> Self {
		unimplemented!()
	}
}
impl std::ops::SubAssign<NISTBinaryField128> for PackedNISTBinaryField2x128b {
	fn sub_assign(&mut self, _rhs: NISTBinaryField128) {
		unimplemented!()
	}
}
impl std::ops::Mul<NISTBinaryField128> for PackedNISTBinaryField2x128b {
	type Output = Self;
	fn mul(self, rhs: NISTBinaryField128) -> Self {
		unimplemented!()
	}
}
impl std::ops::MulAssign<NISTBinaryField128> for PackedNISTBinaryField2x128b {
	fn mul_assign(&mut self, _rhs: NISTBinaryField128) {
		unimplemented!()
	}
}
impl std::iter::Sum for PackedNISTBinaryField2x128b {
	fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
		unimplemented!()
	}
}
impl std::iter::Product for PackedNISTBinaryField2x128b {
	fn product<I: Iterator<Item = Self>>(_iter: I) -> Self {
		unimplemented!()
	}
}
// --- End trait impls for PackedNISTBinaryField2x128b ---
