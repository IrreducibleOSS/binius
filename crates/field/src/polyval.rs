// Copyright 2023-2024 Irreducible Inc.

//! Binary field implementation of GF(2^128) with a modulus of X^128 + X^127 + X^126 + 1.

use super::{
	arithmetic_traits::InvertOrZero,
	binary_field::{BinaryField, BinaryField128b, BinaryField1b, TowerField},
	error::Error,
	extension::ExtensionField,
	underlier::WithUnderlier,
};
use crate::{
	arch::packed_polyval_128::PackedBinaryPolyval1x128b,
	arithmetic_traits::Square,
	binary_field_arithmetic::{
		invert_or_zero_using_packed, multiple_using_packed, square_using_packed,
	},
	linear_transformation::{FieldLinearTransformation, Transformation},
	underlier::UnderlierWithBitOps,
	Field,
};
use bytemuck::{Pod, TransparentWrapper, Zeroable};
use rand::{Rng, RngCore};
use std::{
	array,
	fmt::{self, Debug, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[derive(
	Default,
	Clone,
	Copy,
	PartialEq,
	Eq,
	PartialOrd,
	Ord,
	Hash,
	Zeroable,
	bytemuck::TransparentWrapper,
)]
#[repr(transparent)]
pub struct BinaryField128bPolyval(pub(crate) u128);

impl BinaryField128bPolyval {
	#[inline]
	pub fn new(value: u128) -> Self {
		Self(value).to_montgomery()
	}
}

unsafe impl WithUnderlier for BinaryField128bPolyval {
	type Underlier = u128;

	fn to_underlier(self) -> Self::Underlier {
		TransparentWrapper::peel(self)
	}

	fn to_underlier_ref(&self) -> &Self::Underlier {
		TransparentWrapper::peel_ref(self)
	}

	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		TransparentWrapper::peel_mut(self)
	}

	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		TransparentWrapper::peel_slice(val)
	}

	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		TransparentWrapper::peel_slice_mut(val)
	}

	fn from_underlier(val: Self::Underlier) -> Self {
		TransparentWrapper::wrap(val)
	}

	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		TransparentWrapper::wrap_ref(val)
	}

	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		TransparentWrapper::wrap_mut(val)
	}

	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		TransparentWrapper::wrap_slice(val)
	}

	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		TransparentWrapper::wrap_slice_mut(val)
	}
}

impl Neg for BinaryField128bPolyval {
	type Output = Self;

	#[inline]
	fn neg(self) -> Self::Output {
		self
	}
}

impl Add<Self> for BinaryField128bPolyval {
	type Output = Self;

	#[allow(clippy::suspicious_arithmetic_impl)]
	fn add(self, rhs: Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Add<&Self> for BinaryField128bPolyval {
	type Output = Self;

	#[allow(clippy::suspicious_arithmetic_impl)]
	fn add(self, rhs: &Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Sub<Self> for BinaryField128bPolyval {
	type Output = Self;

	#[allow(clippy::suspicious_arithmetic_impl)]
	fn sub(self, rhs: Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Sub<&Self> for BinaryField128bPolyval {
	type Output = Self;

	#[allow(clippy::suspicious_arithmetic_impl)]
	fn sub(self, rhs: &Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Mul<Self> for BinaryField128bPolyval {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self::Output {
		multiple_using_packed::<PackedBinaryPolyval1x128b>(self, rhs)
	}
}

impl Mul<&Self> for BinaryField128bPolyval {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: &Self) -> Self::Output {
		self * *rhs
	}
}

impl AddAssign<Self> for BinaryField128bPolyval {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl AddAssign<&Self> for BinaryField128bPolyval {
	#[inline]
	fn add_assign(&mut self, rhs: &Self) {
		*self = *self + rhs;
	}
}

impl SubAssign<Self> for BinaryField128bPolyval {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl SubAssign<&Self> for BinaryField128bPolyval {
	#[inline]
	fn sub_assign(&mut self, rhs: &Self) {
		*self = *self - rhs;
	}
}

impl MulAssign<Self> for BinaryField128bPolyval {
	#[inline]
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl MulAssign<&Self> for BinaryField128bPolyval {
	#[inline]
	fn mul_assign(&mut self, rhs: &Self) {
		*self = *self * rhs;
	}
}

impl Sum<Self> for BinaryField128bPolyval {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::ZERO, |acc, x| acc + x)
	}
}

impl<'a> Sum<&'a Self> for BinaryField128bPolyval {
	#[inline]
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::ZERO, |acc, x| acc + x)
	}
}

impl Product<Self> for BinaryField128bPolyval {
	#[inline]
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::ONE, |acc, x| acc * x)
	}
}

impl<'a> Product<&'a Self> for BinaryField128bPolyval {
	#[inline]
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::ONE, |acc, x| acc * x)
	}
}

impl ConstantTimeEq for BinaryField128bPolyval {
	#[inline]
	fn ct_eq(&self, other: &Self) -> Choice {
		self.0.ct_eq(&other.0)
	}
}

impl ConditionallySelectable for BinaryField128bPolyval {
	#[inline]
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		Self(ConditionallySelectable::conditional_select(&a.0, &b.0, choice))
	}
}

impl Square for BinaryField128bPolyval {
	#[inline]
	fn square(self) -> Self {
		square_using_packed::<PackedBinaryPolyval1x128b>(self)
	}
}

impl Field for BinaryField128bPolyval {
	const ZERO: Self = BinaryField128bPolyval(0);
	const ONE: Self = BinaryField128bPolyval(0xc2000000000000000000000000000001);

	fn random(mut rng: impl RngCore) -> Self {
		Self(rng.gen())
	}

	fn double(&self) -> Self {
		Self(0)
	}
}

impl InvertOrZero for BinaryField128bPolyval {
	#[inline]
	fn invert_or_zero(self) -> Self {
		invert_or_zero_using_packed::<PackedBinaryPolyval1x128b>(self)
	}
}

impl From<u128> for BinaryField128bPolyval {
	#[inline]
	fn from(value: u128) -> Self {
		Self(value)
	}
}

impl From<BinaryField128bPolyval> for u128 {
	#[inline]
	fn from(value: BinaryField128bPolyval) -> Self {
		value.0
	}
}

impl Display for BinaryField128bPolyval {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		write!(f, "0x{repr:0>32x}", repr = self.from_montgomery().0)
	}
}

impl Debug for BinaryField128bPolyval {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		write!(f, "BinaryField128bPolyval({})", self)
	}
}

impl BinaryField128bPolyval {
	#[inline]
	pub(super) fn to_montgomery(self) -> Self {
		self * Self(0x1e563df92ea7081b4563df92ea7081b5)
	}

	// Clippy wants us to use a reference here as API design
	// but we can ignore it as we'd need to copy otherwise.
	#[allow(clippy::wrong_self_convention)]
	pub(super) fn from_montgomery(self) -> Self {
		self * Self(1)
	}
}

unsafe impl Pod for BinaryField128bPolyval {}

impl TryInto<BinaryField1b> for BinaryField128bPolyval {
	type Error = ();

	#[inline]
	fn try_into(self) -> Result<BinaryField1b, Self::Error> {
		let result = CtOption::new(BinaryField1b::ZERO, self.ct_eq(&Self::ZERO))
			.or_else(|| CtOption::new(BinaryField1b::ONE, self.ct_eq(&Self::ONE)));
		Option::from(result).ok_or(())
	}
}

impl From<BinaryField1b> for BinaryField128bPolyval {
	#[inline]
	fn from(value: BinaryField1b) -> Self {
		debug_assert_eq!(Self::ZERO, Self(0));

		Self(Self::ONE.0 & u128::fill_with_bit(value.val().val()))
	}
}

impl Add<BinaryField1b> for BinaryField128bPolyval {
	type Output = Self;

	#[inline]
	fn add(self, rhs: BinaryField1b) -> Self::Output {
		self + Self::from(rhs)
	}
}

impl Sub<BinaryField1b> for BinaryField128bPolyval {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: BinaryField1b) -> Self::Output {
		self - Self::from(rhs)
	}
}

impl Mul<BinaryField1b> for BinaryField128bPolyval {
	type Output = Self;

	#[inline]
	#[allow(clippy::suspicious_arithmetic_impl)]
	fn mul(self, rhs: BinaryField1b) -> Self::Output {
		crate::tracing::trace_multiplication!(BinaryField128bPolyval, BinaryField1b);

		Self(self.0 & u128::fill_with_bit(u8::from(rhs.0)))
	}
}

impl AddAssign<BinaryField1b> for BinaryField128bPolyval {
	#[inline]
	fn add_assign(&mut self, rhs: BinaryField1b) {
		*self = *self + rhs;
	}
}

impl SubAssign<BinaryField1b> for BinaryField128bPolyval {
	#[inline]
	fn sub_assign(&mut self, rhs: BinaryField1b) {
		*self = *self - rhs;
	}
}

impl MulAssign<BinaryField1b> for BinaryField128bPolyval {
	#[inline]
	fn mul_assign(&mut self, rhs: BinaryField1b) {
		*self = *self * rhs;
	}
}

impl Add<BinaryField128bPolyval> for BinaryField1b {
	type Output = BinaryField128bPolyval;

	#[inline]
	fn add(self, rhs: BinaryField128bPolyval) -> Self::Output {
		rhs + self
	}
}

impl Sub<BinaryField128bPolyval> for BinaryField1b {
	type Output = BinaryField128bPolyval;

	#[inline]
	fn sub(self, rhs: BinaryField128bPolyval) -> Self::Output {
		rhs - self
	}
}

impl Mul<BinaryField128bPolyval> for BinaryField1b {
	type Output = BinaryField128bPolyval;

	#[inline]
	fn mul(self, rhs: BinaryField128bPolyval) -> Self::Output {
		rhs * self
	}
}

impl ExtensionField<BinaryField1b> for BinaryField128bPolyval {
	type Iterator = <[BinaryField1b; 128] as IntoIterator>::IntoIter;
	const LOG_DEGREE: usize = 7;

	#[inline]
	fn basis(i: usize) -> Result<Self, Error> {
		if i >= 128 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		Ok(Self::new(1 << i))
	}

	#[inline]
	fn from_bases_sparse(base_elems: &[BinaryField1b], log_stride: usize) -> Result<Self, Error> {
		if base_elems.len() > 128 || log_stride != 7 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		// REVIEW: is this actually correct for a monomial field?
		let value = base_elems
			.iter()
			.rev()
			.fold(0, |value, elem| value << 1 | elem.val().val() as u128);
		Ok(Self::new(value))
	}

	#[inline]
	fn iter_bases(&self) -> Self::Iterator {
		let base_elems = array::from_fn(|i| BinaryField1b::from((self.0 >> i) as u8));
		base_elems.into_iter()
	}
}

impl BinaryField for BinaryField128bPolyval {
	const MULTIPLICATIVE_GENERATOR: BinaryField128bPolyval =
		BinaryField128bPolyval(0x72bdf2504ce49c03105433c1c25a4a7);
}

impl TowerField for BinaryField128bPolyval {
	type Canonical = BinaryField128b;

	fn mul_primitive(self, _iota: usize) -> Result<Self, Error> {
		// This method could be implemented by multiplying by isomorphic alpha value
		// But it's not being used as for now
		unimplemented!()
	}
}

pub const BINARY_TO_POLYVAL_TRANSFORMATION: FieldLinearTransformation<BinaryField128bPolyval> =
	FieldLinearTransformation::new_const(&[
		BinaryField128bPolyval(0xc2000000000000000000000000000001),
		BinaryField128bPolyval(0x21a09a4bf26aadcd3eb19c5f1a06b528),
		BinaryField128bPolyval(0xe62f1a804db43b94852cef0e61d7353d),
		BinaryField128bPolyval(0xadcde131ca862a6ba378ea68e992a5b6),
		BinaryField128bPolyval(0x5474611d07bdcd1f72e9bdc82ec4fe6c),
		BinaryField128bPolyval(0xf9a472d4a4965f4caa3532aa6258c986),
		BinaryField128bPolyval(0x10bd76c920260f81877681ed1a50b210),
		BinaryField128bPolyval(0xe7f3264523858ca36ef84934fdd225f2),
		BinaryField128bPolyval(0x586704bda927015fedb8ddceb7f825d6),
		BinaryField128bPolyval(0x552dab8acfd831aeb65f8aaec9cef096),
		BinaryField128bPolyval(0xeccdac666a363defde6792e475892fb3),
		BinaryField128bPolyval(0x4a621d01701247f6e4a8327e33d95aa2),
		BinaryField128bPolyval(0x8ed5002fed1f4b9a9a11840f87149e2d),
		BinaryField128bPolyval(0x3c65abbd41c759f0302467db5a791e09),
		BinaryField128bPolyval(0xc2df68a5949a96b3aa643692e93caaab),
		BinaryField128bPolyval(0x4455027df88c165117daf9822eb57383),
		BinaryField128bPolyval(0xc50e3a207f91d7cd6dd1e116d55455fb),
		BinaryField128bPolyval(0xc89c3920b9b24b755fd08543d8caf5a2),
		BinaryField128bPolyval(0xfa583eb935de76a2ec180360b6548830),
		BinaryField128bPolyval(0xc4d3d3b9938f3af77800a5cd03690171),
		BinaryField128bPolyval(0xe1faff3b895be1e2bec91c0836143b44),
		BinaryField128bPolyval(0x256bd50f868b82cf1c83552eeb1cd844),
		BinaryField128bPolyval(0x82fd35d590073ae9595cab38e9b59d79),
		BinaryField128bPolyval(0x08dadd230bc90e192304a2533cdce9e6),
		BinaryField128bPolyval(0xf4400f37acedc7d9502abeff6cead84c),
		BinaryField128bPolyval(0x5438d34e2b5b90328cc88b7384deedfb),
		BinaryField128bPolyval(0x7d798db71ef80a3e447cd7d1d4a0385d),
		BinaryField128bPolyval(0xa50d5ef4e33979db8012303dc09cbf35),
		BinaryField128bPolyval(0x91c4b5e29de5759e0bb337efbc5b8115),
		BinaryField128bPolyval(0xbbb0d4aaba0fab72848f461ed0a4b110),
		BinaryField128bPolyval(0x3c9de86b9a306d6d11cc078904076865),
		BinaryField128bPolyval(0xb5f43a166aa1f15f99db6d689ca1b370),
		BinaryField128bPolyval(0xa26153cb8c150af8243ecbd46378e59e),
		BinaryField128bPolyval(0xccaa154bab1dd7aca876f81fe0c950ab),
		BinaryField128bPolyval(0x4185b7e3ee1dddbc761a6139cdb07755),
		BinaryField128bPolyval(0x2c9f95285b7aa574653ed207337325f2),
		BinaryField128bPolyval(0xc8ba616ab131bfd242195c4c82d54dbb),
		BinaryField128bPolyval(0x2a9b07221a34865faa36a28da1ab1c24),
		BinaryField128bPolyval(0x7e6e572804b548a88b92900e0196dd39),
		BinaryField128bPolyval(0x4e9060deff44c9ef9882a0015debd575),
		BinaryField128bPolyval(0x00a3a4d8c163c95ac7ac9a5b424e1c65),
		BinaryField128bPolyval(0xf67c7eb5dde73d96f8f5eecba6033679),
		BinaryField128bPolyval(0x54d78d187bbb57d19b536094ba539fde),
		BinaryField128bPolyval(0x76c553699edc5d4a033139975ab7f264),
		BinaryField128bPolyval(0x74ae8da43b2f587df3e41bbf5c6be650),
		BinaryField128bPolyval(0x8a2941b59774c41acd850aa6098e5fd2),
		BinaryField128bPolyval(0x9ddf65660a6f8f3c0058165a063de84c),
		BinaryField128bPolyval(0xbb52da733635cc3d1ff02ef96ee64cf3),
		BinaryField128bPolyval(0x564032a0d5d3773b7b7ed18bebf1c668),
		BinaryField128bPolyval(0xef5c765e64b24b1b00222054ff0040ef),
		BinaryField128bPolyval(0xade661c18acba6233d484726e6249bee),
		BinaryField128bPolyval(0x9939ba35c969cdeea29f2ef849c2d170),
		BinaryField128bPolyval(0x2b100b39761d4f23eb42d05b80174ce2),
		BinaryField128bPolyval(0xfbc25b179830f9eec765bd6229125d6c),
		BinaryField128bPolyval(0xb58e089ebe7ad0b2698e30184ab93141),
		BinaryField128bPolyval(0x53874933a148be94d12025afa876234c),
		BinaryField128bPolyval(0x41bbc7902188f4e9880f1d81fa580ffb),
		BinaryField128bPolyval(0xea4199916a5d127d25da1fe777b2dcbb),
		BinaryField128bPolyval(0xe7bc816547efbe987d9359ee0de0c287),
		BinaryField128bPolyval(0x02e0f1f67e7139835892155a7addd9da),
		BinaryField128bPolyval(0xdc6beb6eade9f875e74955ca950af235),
		BinaryField128bPolyval(0x786d616edeadfa356453a78d8f103230),
		BinaryField128bPolyval(0xe84e70191accaddac8034da936737487),
		BinaryField128bPolyval(0x012b8669ff3f451e5363edfddd37fb3c),
		BinaryField128bPolyval(0x756209f0893e96877833c194b9c943a0),
		BinaryField128bPolyval(0xb2ac9efc9a1891369f63bd1e0d1439ac),
		BinaryField128bPolyval(0x4de88e9a5bbb4c3df650cc3994c3d2d8),
		BinaryField128bPolyval(0x8de7b5c85c07f3359849e7c85e426b54),
		BinaryField128bPolyval(0xcadd54ae6a7e72a4f184e6761cf226d4),
		BinaryField128bPolyval(0xcdb182fb8d95496f55b5f3952f81bc30),
		BinaryField128bPolyval(0x40013bc3c81722753a05bb2aca01a02e),
		BinaryField128bPolyval(0x704e7ce55e9033883e97351591adf18a),
		BinaryField128bPolyval(0xf330cd9a74a5e884988c3f36567d26f4),
		BinaryField128bPolyval(0x18f4535304c0d74ac3bdf09d78cbde50),
		BinaryField128bPolyval(0xfe739c97fc26bed28885b838405c7e7e),
		BinaryField128bPolyval(0x492479260f2dcd8af980c3d74b3ec345),
		BinaryField128bPolyval(0x96b6440a34de0aad4ea2f744396691af),
		BinaryField128bPolyval(0x98355d1b4f7cfb03960a59aa564a7a26),
		BinaryField128bPolyval(0x2703fda0532095ca8b1886b12ca37d64),
		BinaryField128bPolyval(0x59c9dabe49bebf6b468c3c120f142822),
		BinaryField128bPolyval(0xf8f3c35c671bac841b14381a592e6cdd),
		BinaryField128bPolyval(0xd7b888791bd83b13d80d2e9324894861),
		BinaryField128bPolyval(0x113ab0405354dd1c5aab9658137fa73f),
		BinaryField128bPolyval(0xae56192d5e9c309e461f797121b28ce6),
		BinaryField128bPolyval(0xb7927ec7a84c2e04811a6dac6b997783),
		BinaryField128bPolyval(0x9e2f8d67fc600703ba9b4189ce751cb4),
		BinaryField128bPolyval(0x574e95df2d8bb9e2c8fc29729eb723ca),
		BinaryField128bPolyval(0x38bc6fc47739c06cd9fa20f9a5088f26),
		BinaryField128bPolyval(0x69d3b9b1d9483174b3c38d8f95ce7a5f),
		BinaryField128bPolyval(0xd6e4bb147cc82b6e90e27e882f18640d),
		BinaryField128bPolyval(0x027338db641804d985cd9fece12f7adc),
		BinaryField128bPolyval(0x523cb73968169ccce76f523928c4364e),
		BinaryField128bPolyval(0xcdcf898117f927208a11b0dcc941f2f6),
		BinaryField128bPolyval(0xc908287814c8cba67f7892fec7a5b217),
		BinaryField128bPolyval(0x92b99988bb26215d104968d4cbbb285a),
		BinaryField128bPolyval(0x4dbca8fd835d00ea4b95692534ef5068),
		BinaryField128bPolyval(0xcd8b92c8a6e0e65e167a2b851f32fd9c),
		BinaryField128bPolyval(0xc3473dfda9f97d6ac1e2d544628e7845),
		BinaryField128bPolyval(0x0260e7badc64dbfde0dc39a240365722),
		BinaryField128bPolyval(0x3966125b40fe2bca9719c80e41953868),
		BinaryField128bPolyval(0xac0211506eda3cba57b709a360d4a2c7),
		BinaryField128bPolyval(0x0e4f0e47d02fedd15b337fefa219c52b),
		BinaryField128bPolyval(0x1d5907ccdc659f7aace675511f754ee3),
		BinaryField128bPolyval(0x4ad5b368eaddc4bb097284863b2a5b6e),
		BinaryField128bPolyval(0x2eae07273b8c4fc5cef553a4a46cde5b),
		BinaryField128bPolyval(0x096a310e7b1e3a3179d4a3b5d8dd9396),
		BinaryField128bPolyval(0x8c81362eeb1656a91dde08d05018a353),
		BinaryField128bPolyval(0x387e59e44cc0d53fecf7f057b6fdba0b),
		BinaryField128bPolyval(0x9d29670bbd0e8051ac82d91ca97561d6),
		BinaryField128bPolyval(0xaf1310d0f5cac4e89714e48065be74a4),
		BinaryField128bPolyval(0x9b684a3865c2b59c411d14182a36fb6b),
		BinaryField128bPolyval(0x3e7de163516ffdcaca22b4e848340fbe),
		BinaryField128bPolyval(0x3c37dbe331de4b0dc2f5db315d5e7fda),
		BinaryField128bPolyval(0x19e7f4b53ff86990e3d5a1c40c3769a0),
		BinaryField128bPolyval(0x56469ab32b2b82e8cc93fdb1b14a4775),
		BinaryField128bPolyval(0x9c01cefde47816300d8ad49d260bb71b),
		BinaryField128bPolyval(0x6100101b8cebde7381366fec1e4e52c0),
		BinaryField128bPolyval(0xa28d30c3cbd8b69632143fa65158ee4f),
		BinaryField128bPolyval(0x3db7a902ec509e58151c45f71eee6368),
		BinaryField128bPolyval(0x42d5a505e8ab70097107d37d79ebbaba),
		BinaryField128bPolyval(0xe47b83247cb2b162c7d6d15c84cca8ce),
		BinaryField128bPolyval(0x076caf0e23541c753e4c87ff505737a5),
		BinaryField128bPolyval(0x590a8d1cdbd17ae83980f5d1d3b84a89),
		BinaryField128bPolyval(0x77d649ff61a7cd0da53497edd34c4204),
		BinaryField128bPolyval(0xefbe0c34eeab379ea4a8feed84fd3993),
		BinaryField128bPolyval(0x90540cf7957a8a3051629cdde777f968),
		BinaryField128bPolyval(0x8749050496dd288244c49c70aa92831f),
		BinaryField128bPolyval(0x0fc80b1d600406b2370368d94947961a),
	]);

impl From<BinaryField128b> for BinaryField128bPolyval {
	fn from(value: BinaryField128b) -> Self {
		BINARY_TO_POLYVAL_TRANSFORMATION.transform(&value)
	}
}

pub const POLYVAL_TO_BINARY_TRANSFORMATION: FieldLinearTransformation<BinaryField128b> =
	FieldLinearTransformation::new_const(&[
		BinaryField128b(0x66e1d645d7eb87dca8fc4d30a32dadcc),
		BinaryField128b(0x53ca87ba77172fd8c5675d78c59c1901),
		BinaryField128b(0x1a9cf63d31827dcda15acb755a948567),
		BinaryField128b(0xa8f28bdf6d29cee2474b0401a99f6c0a),
		BinaryField128b(0x4eefa9efe87ed19c06b39ca9799c8d73),
		BinaryField128b(0x06ec578f505abf1e9885a6b2bc494f3e),
		BinaryField128b(0x70ecdfe1f601f8509a96d3fb9cd3348a),
		BinaryField128b(0xcb0d16fc7f13733deb25f618fc3faf28),
		BinaryField128b(0x4e9a97aa2c84139ffcb578115fcbef3c),
		BinaryField128b(0xc6de6210afe8c6bd9a441bffe19219ad),
		BinaryField128b(0x73e3e8a7c59748601be5bf1e30c488d3),
		BinaryField128b(0x1f6d67e2e64bd6c4b39e7f4bb37dce9c),
		BinaryField128b(0xc34135d567eada885f5095b4c155f3b5),
		BinaryField128b(0x23f165958d59a55e4790b8e2e37330e4),
		BinaryField128b(0x4f2be978f16908e405b88802add08d17),
		BinaryField128b(0x6442b00f5bbf4009907936513c3a7d45),
		BinaryField128b(0xac63f0397d911a7a5d61b9f18137026f),
		BinaryField128b(0x8e70543ae0e43313edf07cbc6698e144),
		BinaryField128b(0xcb417a646d59f652aa5a07984066d026),
		BinaryField128b(0xf028de8dd616318735bd8f76de7bb84e),
		BinaryField128b(0x2e03a12472d21599f15b4bcaa9bf186c),
		BinaryField128b(0x54a376cc03e5b2cfa27d8e48d1b9ca76),
		BinaryField128b(0xd22894c253031b1b201b87da07cb58ae),
		BinaryField128b(0x6bc1416afea6308ff77d902dd5d2a563),
		BinaryField128b(0x9958ecd28adbebf850055f8ac3095121),
		BinaryField128b(0x595a1b37062233d7e6bb6f54c227fb91),
		BinaryField128b(0x41ffcfcdda4583c4f671558ee315d809),
		BinaryField128b(0x780c2490f3e5cb4763e982ec4b3e6ea2),
		BinaryField128b(0xf7a450b35931fa76722a6b9037b6db34),
		BinaryField128b(0xe21991100e84821328592772430ad07e),
		BinaryField128b(0x360d4079f62863cc60c65ec87d6f9277),
		BinaryField128b(0xd898bfa0b076cc4eaca590e7a60dbe92),
		BinaryField128b(0xcaacddd5e114fe5c2e1647fc34b549bf),
		BinaryField128b(0x3042e34911c28e90617776ddb2d3f888),
		BinaryField128b(0x3728a3b0da53cdfecfd8455b13cb9b14),
		BinaryField128b(0x2f2eb3d5bc7b2c48a7c643bffbddc6b2),
		BinaryField128b(0x3b71a5c04010c0aa501b04302706b908),
		BinaryField128b(0x0701845b090e79bb9be54df766e48c51),
		BinaryField128b(0x1e9eac7bf45b14c8db06fcfff7408f78),
		BinaryField128b(0x6b1b8e39a339423d0eb3bef69eee8b0b),
		BinaryField128b(0x8b06616385967df95d3a99cff1edcf0a),
		BinaryField128b(0x5d921137890a3ded58e1dd1a51fe6a30),
		BinaryField128b(0x828ed6fba42805b2628b705d38121acc),
		BinaryField128b(0x9b7a95220e9d5b0ff70ecb6116cabd81),
		BinaryField128b(0x0eb9055cb11711ed047f136cab751c88),
		BinaryField128b(0xd6f590777c17a6d0ca451290f7d5c78a),
		BinaryField128b(0x401a922a6461fbe691f910cb0893e71f),
		BinaryField128b(0x15a549308bc53902c927ebad9ed253f7),
		BinaryField128b(0x45dccafc72a584480f340a43f11a1b84),
		BinaryField128b(0x19d2a2c057d60656e6d3e20451335d5b),
		BinaryField128b(0x035af143a5827a0f99197c8b9a811454),
		BinaryField128b(0x7ee35d174ad7cc692191fd0e013f163a),
		BinaryField128b(0xc4c0401d841f965c9599fac8831effa9),
		BinaryField128b(0x63e809a843fc04f84acfca3fc5630691),
		BinaryField128b(0xdb2f3301594e3de49fb7d78e2d6643c4),
		BinaryField128b(0x1b31772535984ef93d709319cc130a7c),
		BinaryField128b(0x036dc9c884cd6d6c918071b62a0593f3),
		BinaryField128b(0x4700cd0e81c88045132360b078027103),
		BinaryField128b(0xdfa3f35eb236ea63b0350e17ed2d625d),
		BinaryField128b(0xf0fd7c7760099f1ac28be91822978e15),
		BinaryField128b(0x852a1eba3ad160e95034e9eed1f21205),
		BinaryField128b(0x4a07dd461892df45ca9efee1701763c3),
		BinaryField128b(0xadbbaa0add4c82fe85fd61b42f707384),
		BinaryField128b(0x5c63d0673f33c0f2c231db13f0e15600),
		BinaryField128b(0x24ddc1516501135626e0e794dd4b3076),
		BinaryField128b(0xb60c601bbf72924e38afd02d201fb05b),
		BinaryField128b(0x2ef68918f416caca84334bcf70649aeb),
		BinaryField128b(0x0b72a3124c504bcad815534c707343f2),
		BinaryField128b(0xcfd8b2076040c43d5d396f8523d80fe0),
		BinaryField128b(0x098d9daf64154a63504192bb27cc65e1),
		BinaryField128b(0x3ae44070642e6720283621f8fb6a6704),
		BinaryField128b(0x19cd9b2843d0ff936bfe2b373f47fd05),
		BinaryField128b(0x451e2e4159c78e65db10450431d26122),
		BinaryField128b(0x797b753e29b9d0e9423b36807c70f3ae),
		BinaryField128b(0xa8d0e8ba9bb634f6ea30600915664e22),
		BinaryField128b(0xdf8c74bbd66f86809c504cb944475b0a),
		BinaryField128b(0x32831a457ced3a417a5a94d498128018),
		BinaryField128b(0x1aca728985936a6147119b9b5f00350e),
		BinaryField128b(0x6f436d64b4ee1a556b66764ed05bb1db),
		BinaryField128b(0x25930eaed3fd982915e483cb21e5a1a2),
		BinaryField128b(0x21735f5eb346e56006bf1d7e151780ab),
		BinaryField128b(0x55fc6f607f10e17f805eb16d7bd5345c),
		BinaryField128b(0x4b4d289591f878114965292af4aeb57e),
		BinaryField128b(0x30608bc7444bcbaff67998c1883c1cf3),
		BinaryField128b(0xa12a72abe4152e4a657c6e6395404343),
		BinaryField128b(0x7579186d4e0959dec73f9cd68fb0e2fb),
		BinaryField128b(0xb5560ce63f7894cc965c822892b7bfda),
		BinaryField128b(0x6b06d7165072861eba63d9fd645995d7),
		BinaryField128b(0x359f439f5ec9107dde3c8ef8f9bf4e29),
		BinaryField128b(0xcbfe7985c6006a46105821cd8b55b06b),
		BinaryField128b(0x2110b3b51f5397ef1129fb9076474061),
		BinaryField128b(0x1928478b6f3275c944c33b275c388c47),
		BinaryField128b(0x23f978e6a0a54802437111aa4652421a),
		BinaryField128b(0xe8c526bf924dc5cd1dd32dbedd310f5b),
		BinaryField128b(0xa0ac29f901f79ed5f43c73d22a05c8e4),
		BinaryField128b(0x55e0871c6e97408f47f4635b747145ea),
		BinaryField128b(0x6c2114c3381f53667d3c2dfefd1ebcb3),
		BinaryField128b(0x42d23c18722fbd58863c3aceaaa3eef7),
		BinaryField128b(0xbb0821ab38d5de133838f8408a72fdf1),
		BinaryField128b(0x035d7239054762b131fa387773bb9153),
		BinaryField128b(0x8fa898aafe8b154f9ab652e8979139e7),
		BinaryField128b(0x6a383e5cd4a16923c658193f16cb726c),
		BinaryField128b(0x9948caa8c6cefb0182022f32ae3f68b9),
		BinaryField128b(0x8d2a8decf9855bd4df7bac577ed73b44),
		BinaryField128b(0x09c7b8300f0f984259d548c5aa959879),
		BinaryField128b(0x92e16d2d24e070efdca8b8e134047afc),
		BinaryField128b(0x47d8621457f4118aaf24877fb5031512),
		BinaryField128b(0x25576941a55f0a0c19583a966a85667f),
		BinaryField128b(0xb113cad79cd35f2e83fda3bc6285a8dc),
		BinaryField128b(0xc76968eecb2748d0c3e6318431ffe580),
		BinaryField128b(0x7211122aa7e7f6fe39e6618395b68416),
		BinaryField128b(0x88463599bf7d3e92f450d00a45146d11),
		BinaryField128b(0x6e12b7d5adf95da33bbb7f79a18ee123),
		BinaryField128b(0xe0a98ac4025bc568eaca7e7b7280ff16),
		BinaryField128b(0xc13fc79f6c35048df274057ac892ff77),
		BinaryField128b(0x93c1a3145d4e47dee39cae4de47eb505),
		BinaryField128b(0x780064be3036df98f1e5d7c53bdbd52b),
		BinaryField128b(0x48c467b5cec265628b709172ecaff561),
		BinaryField128b(0x5bbbab77ce5552ff7682094560524a7e),
		BinaryField128b(0x551537ef6048831fb128fec4e4a23a63),
		BinaryField128b(0xe7ef397fcc095ead439317a13568b284),
		BinaryField128b(0xbc5d2927eac0a720f9d75d62d92c6332),
		BinaryField128b(0x3bfeb420021f93e9b2bc992b5b59e61e),
		BinaryField128b(0xc651dc438e2f1bc64af1b7307b574ed9),
		BinaryField128b(0xbfe0a17ee2b777542a1ddb55413a4e43),
		BinaryField128b(0xa062da2427df3d1a7dfc01c05d732a32),
		BinaryField128b(0x1e4889fd72b70ecf93417ba0b085e1e8),
		BinaryField128b(0xc4f4769f4f9c2e33c26a6bf2ca842f17),
	]);

impl From<BinaryField128bPolyval> for BinaryField128b {
	fn from(value: BinaryField128bPolyval) -> BinaryField128b {
		POLYVAL_TO_BINARY_TRANSFORMATION.transform(&value)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		arch::{
			packed_polyval_256::PackedBinaryPolyval2x128b,
			packed_polyval_512::PackedBinaryPolyval4x128b,
		},
		binary_field::tests::is_binary_field_valid_generator,
		deserialize_canonical,
		linear_transformation::PackedTransformationFactory,
		serialize_canonical, PackedBinaryField1x128b, PackedBinaryField2x128b,
		PackedBinaryField4x128b, PackedField,
	};
	use bytes::BytesMut;
	use proptest::prelude::*;
	use rand::thread_rng;

	#[test]
	fn test_display() {
		assert_eq!(
			"0x00000000000000000000000000000001",
			format!("{}", BinaryField128bPolyval::ONE)
		);
		assert_eq!(
			"0x2a9055e4e69a61f0b5cfd6f4161087ba",
			format!("{}", BinaryField128bPolyval::new(0x2a9055e4e69a61f0b5cfd6f4161087ba))
		);
	}

	proptest! {
		#[test]
		fn test_multiplicative_identity(v in any::<u128>()) {
			let v = BinaryField128bPolyval::new(v);
			assert_eq!(v, v * BinaryField128bPolyval::ONE);
		}
	}

	#[test]
	fn test_mul() {
		assert_eq!(
			BinaryField128bPolyval::new(0x2a9055e4e69a61f0b5cfd6f4161087ba)
				* BinaryField128bPolyval::new(0x3843cf87fb7c84e18276983bed670337),
			BinaryField128bPolyval::new(0x5b2619c8a035206a12100d7a171aa988)
		);
	}

	#[test]
	fn test_sqr() {
		assert_eq!(
			Square::square(BinaryField128bPolyval::new(0x2a9055e4e69a61f0b5cfd6f4161087ba)),
			BinaryField128bPolyval::new(0x59aba0d4ffa9dca427b5b489f293e529)
		);
	}

	#[test]
	fn test_multiplicative_generator() {
		assert!(is_binary_field_valid_generator::<BinaryField128bPolyval>());
	}

	fn test_packed_conversion<
		PT1: PackedField<Scalar: BinaryField> + PackedTransformationFactory<PT2>,
		PT2: PackedField<Scalar: BinaryField>,
	>(
		val: PT1,
		transformation: FieldLinearTransformation<PT2::Scalar>,
	) {
		let expected = PT2::from_fn(|i| transformation.transform(&val.get(i)));
		let transformed =
			<PT1 as PackedTransformationFactory<PT2>>::make_packed_transformation(transformation)
				.transform(&val);
		assert_eq!(transformed, expected);
	}

	proptest! {
		#[test]
		fn test_to_from_tower_basis(a_val in any::<u128>(), b_val in any::<u128>()) {
			let a_tower = BinaryField128b::new(a_val);
			let b_tower = BinaryField128b::new(b_val);
			let a_polyval = BinaryField128bPolyval::from(a_tower);
			let b_polyval = BinaryField128bPolyval::from(b_tower);
			assert_eq!(BinaryField128b::from(a_polyval * b_polyval), a_tower * b_tower);
		}

		#[test]
		fn test_conversion_roundtrip(a in any::<u128>()) {
			let a_val = BinaryField128bPolyval(a);
			let converted = BinaryField128b::from(a_val);
			assert_eq!(a_val, BinaryField128bPolyval::from(converted));
		}

		#[test]
		fn test_conversion_128b(a in any::<u128>()) {
			test_packed_conversion::<PackedBinaryPolyval1x128b, PackedBinaryField1x128b>(a.into(), POLYVAL_TO_BINARY_TRANSFORMATION);
			test_packed_conversion::<PackedBinaryField1x128b, PackedBinaryPolyval1x128b>(a.into(), BINARY_TO_POLYVAL_TRANSFORMATION);
		}

		#[test]
		fn test_conversion_256b(a in any::<[u128; 2]>()) {
			test_packed_conversion::<PackedBinaryPolyval2x128b, PackedBinaryField2x128b>(a.into(), POLYVAL_TO_BINARY_TRANSFORMATION);
			test_packed_conversion::<PackedBinaryField2x128b, PackedBinaryPolyval2x128b>(a.into(), BINARY_TO_POLYVAL_TRANSFORMATION);
		}

		#[test]
		fn test_conversion_512b(a in any::<[u128; 4]>()) {
			test_packed_conversion::<PackedBinaryPolyval4x128b, PackedBinaryField4x128b>(PackedBinaryPolyval4x128b::from_underlier(a.into()), POLYVAL_TO_BINARY_TRANSFORMATION);
			test_packed_conversion::<PackedBinaryField4x128b, PackedBinaryPolyval4x128b>(PackedBinaryField4x128b::from_underlier(a.into()), BINARY_TO_POLYVAL_TRANSFORMATION);
		}


		#[test]
		fn test_invert_or_zero(a_val in any::<u128>()) {
			let a = BinaryField128bPolyval::new(a_val);
			let a_invert = InvertOrZero::invert_or_zero(a);
			if a != BinaryField128bPolyval::ZERO {
				assert_eq!(a * a_invert, BinaryField128bPolyval::ONE);
			} else {
				assert_eq!(a_invert, BinaryField128bPolyval::ZERO);
			}
		}
	}

	/// Test that `invert` method properly wraps `invert_or_zero`
	#[test]
	fn test_invert() {
		let x = BinaryField128bPolyval::new(2);
		let y = x.invert().unwrap();

		assert_eq!(x * y, BinaryField128bPolyval::ONE);
	}

	#[test]
	fn test_conversion_from_1b() {
		assert_eq!(
			BinaryField128bPolyval::from(BinaryField1b::from(0)),
			BinaryField128bPolyval::ZERO
		);
		assert_eq!(
			BinaryField128bPolyval::from(BinaryField1b::from(1)),
			BinaryField128bPolyval::ONE
		);
	}

	#[test]
	fn test_canonical_serialization() {
		let mut buffer = BytesMut::new();
		let mut rng = thread_rng();

		let b128_poly1 = <BinaryField128bPolyval as Field>::random(&mut rng);
		let b128_poly2 = <BinaryField128bPolyval as Field>::random(&mut rng);

		serialize_canonical(b128_poly1, &mut buffer).unwrap();
		serialize_canonical(b128_poly2, &mut buffer).unwrap();

		let mut read_buffer = buffer.freeze();

		assert_eq!(
			deserialize_canonical::<BinaryField128bPolyval, _>(&mut read_buffer).unwrap(),
			b128_poly1
		);
		assert_eq!(
			BinaryField128bPolyval::from(
				deserialize_canonical::<BinaryField128b, _>(&mut read_buffer).unwrap()
			),
			b128_poly2
		);
	}
}
