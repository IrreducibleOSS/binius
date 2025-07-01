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
	arithmetic_traits::{InvertOrZero, Square, TaggedMul},
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
		let prod_0 = unsafe { vmull_p64(self.0 as u64, rhs.0 as u64) };
		let prod_1 = unsafe { vmull_p64((self.0 >> 64) as u64, (rhs.0 >> 64) as u64) };
		let prod_0 = M128::from(prod_0);
		let prod_1 = M128::from(prod_1);

		let (mut lo, hi) = prod_0.interleave(prod_1, 6);

		lo ^= hi;
		lo ^= hi << 1;
		lo ^= hi << 3;
		lo ^= hi << 4;

		Self::from_underlier(lo.0)
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

impl crate::arithmetic_traits::Square for PackedPrimitiveType<u128, NISTBinaryField128> {
	fn square(self) -> Self {
		todo!()
	}
}

impl crate::arithmetic_traits::InvertOrZero for PackedPrimitiveType<u128, NISTBinaryField128> {
	fn invert_or_zero(self) -> Self {
		todo!()
	}
}

impl std::ops::Mul for PackedPrimitiveType<u128, NISTBinaryField128> {
	type Output = Self;
	fn mul(self, rhs: Self) -> Self::Output {
		let alphas = M128::from(128737474576474u128);
		let odd_mask = M128::INTERLEAVE_ODD_MASK[6];
		let a = PackedPrimitiveType::<u128, NISTBinaryField64>::from_underlier(self.0);
		let b = PackedPrimitiveType::<u128, NISTBinaryField64>::from_underlier(rhs.0);
		let p1 = M128::from((a * b).to_underlier());
		let (lo, hi) = M128::interleave(a.0.into(), b.0.into(), 6);
		let (lhs, rhs) = M128::interleave(lo ^ hi, alphas ^ (p1 & odd_mask), 6);
		let p2: M128 = (PackedPrimitiveType::<u128, NISTBinaryField64>::from_underlier(lhs.into())
			* PackedPrimitiveType::<u128, NISTBinaryField64>::from_underlier(rhs.into()))
		.to_underlier()
		.into();
		let q1 = p1 ^ flip_even_odd::<NISTBinaryField64>(p1);
		let q2 = p2 ^ shift_left::<NISTBinaryField64>(p2);
		Self::from_underlier((q1 ^ (q2 & odd_mask)).into())
	}
}
