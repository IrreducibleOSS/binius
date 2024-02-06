// Copyright 2023 Ulvetanna Inc.

//! Binary field implementation of GF(2^128) with a modulus of X^128 + X^127 + X^126 + 1.

use super::binary_field::BinaryField128b;
use bytemuck::{Pod, Zeroable};
use ff::Field;
use rand::{Rng, RngCore};
use std::{
	array,
	fmt::{self, Display, Formatter},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use super::{
	arch::polyval,
	binary_field::{BinaryField, BinaryField1b},
	error::Error,
	extension::ExtensionField,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
#[repr(transparent)]
pub struct BinaryField128bPolyval(pub(crate) u128);

impl BinaryField128bPolyval {
	pub fn new(value: u128) -> Self {
		Self(value).to_montgomery()
	}
}

impl Neg for BinaryField128bPolyval {
	type Output = Self;

	fn neg(self) -> Self::Output {
		self
	}
}

impl Add<Self> for BinaryField128bPolyval {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Add<&Self> for BinaryField128bPolyval {
	type Output = Self;

	fn add(self, rhs: &Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Sub<Self> for BinaryField128bPolyval {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Sub<&Self> for BinaryField128bPolyval {
	type Output = Self;

	fn sub(self, rhs: &Self) -> Self::Output {
		Self(self.0 ^ rhs.0)
	}
}

impl Mul<Self> for BinaryField128bPolyval {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		Self(polyval::montgomery_multiply(self.0, rhs.0))
	}
}

impl Mul<&Self> for BinaryField128bPolyval {
	type Output = Self;

	fn mul(self, rhs: &Self) -> Self::Output {
		self * *rhs
	}
}

impl AddAssign<Self> for BinaryField128bPolyval {
	fn add_assign(&mut self, rhs: Self) {
		self.0 ^= rhs.0;
	}
}

impl AddAssign<&Self> for BinaryField128bPolyval {
	fn add_assign(&mut self, rhs: &Self) {
		self.0 ^= rhs.0;
	}
}

impl SubAssign<Self> for BinaryField128bPolyval {
	fn sub_assign(&mut self, rhs: Self) {
		self.0 ^= rhs.0;
	}
}

impl SubAssign<&Self> for BinaryField128bPolyval {
	fn sub_assign(&mut self, rhs: &Self) {
		self.0 ^= rhs.0;
	}
}

impl MulAssign<Self> for BinaryField128bPolyval {
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl MulAssign<&Self> for BinaryField128bPolyval {
	fn mul_assign(&mut self, rhs: &Self) {
		*self = *self * rhs;
	}
}

impl Sum<Self> for BinaryField128bPolyval {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::ZERO, |acc, x| acc + x)
	}
}

impl<'a> Sum<&'a Self> for BinaryField128bPolyval {
	fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::ZERO, |acc, x| acc + x)
	}
}

impl Product<Self> for BinaryField128bPolyval {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::ONE, |acc, x| acc * x)
	}
}

impl<'a> Product<&'a Self> for BinaryField128bPolyval {
	fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
		iter.fold(Self::ONE, |acc, x| acc * x)
	}
}

impl ConstantTimeEq for BinaryField128bPolyval {
	fn ct_eq(&self, other: &Self) -> Choice {
		self.0.ct_eq(&other.0)
	}
}

impl ConditionallySelectable for BinaryField128bPolyval {
	fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
		Self(ConditionallySelectable::conditional_select(&a.0, &b.0, choice))
	}
}

impl Field for BinaryField128bPolyval {
	const ZERO: Self = BinaryField128bPolyval(0);
	const ONE: Self = BinaryField128bPolyval(0xc2000000000000000000000000000001);

	fn random(mut rng: impl RngCore) -> Self {
		Self(rng.gen())
	}

	fn square(&self) -> Self {
		Self(polyval::montgomery_square(self.0))
	}

	fn double(&self) -> Self {
		Self(0)
	}

	fn invert(&self) -> CtOption<Self> {
		polyval::invert(self.0).map(Self)
	}

	fn sqrt_ratio(_num: &Self, _div: &Self) -> (Choice, Self) {
		todo!()
	}
}

impl From<u128> for BinaryField128bPolyval {
	fn from(value: u128) -> Self {
		Self(value)
	}
}

impl From<BinaryField128bPolyval> for u128 {
	fn from(value: BinaryField128bPolyval) -> Self {
		value.0
	}
}

impl Display for BinaryField128bPolyval {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		write!(f, "0x{repr:0>32x}", repr = self.from_montgomery().0)
	}
}

impl BinaryField128bPolyval {
	fn to_montgomery(self) -> Self {
		self * Self(0x1e563df92ea7081b4563df92ea7081b5)
	}

	// Clippy wants us to use a reference here as API design
	// but we can ignore it as we'd need to copy otherwise.
	#[allow(clippy::wrong_self_convention)]
	fn from_montgomery(self) -> Self {
		self * Self(1)
	}
}

unsafe impl Pod for BinaryField128bPolyval {}

impl TryInto<BinaryField1b> for BinaryField128bPolyval {
	type Error = ();

	fn try_into(self) -> Result<BinaryField1b, Self::Error> {
		let result = CtOption::new(BinaryField1b::ZERO, self.ct_eq(&Self::ZERO))
			.or_else(|| CtOption::new(BinaryField1b::ONE, self.ct_eq(&Self::ONE)));
		Option::from(result).ok_or(())
	}
}

impl From<BinaryField1b> for BinaryField128bPolyval {
	fn from(value: BinaryField1b) -> Self {
		Self::conditional_select(&Self::ZERO, &Self::ONE, value.into())
	}
}

impl Add<BinaryField1b> for BinaryField128bPolyval {
	type Output = Self;

	fn add(self, rhs: BinaryField1b) -> Self::Output {
		self + Self::from(rhs)
	}
}

impl Sub<BinaryField1b> for BinaryField128bPolyval {
	type Output = Self;

	fn sub(self, rhs: BinaryField1b) -> Self::Output {
		self - Self::from(rhs)
	}
}

impl Mul<BinaryField1b> for BinaryField128bPolyval {
	type Output = Self;

	fn mul(self, rhs: BinaryField1b) -> Self::Output {
		Self::conditional_select(&Self::ZERO, &self, rhs.0.into())
	}
}

impl AddAssign<BinaryField1b> for BinaryField128bPolyval {
	fn add_assign(&mut self, rhs: BinaryField1b) {
		*self = *self + rhs;
	}
}

impl SubAssign<BinaryField1b> for BinaryField128bPolyval {
	fn sub_assign(&mut self, rhs: BinaryField1b) {
		*self = *self - rhs;
	}
}

impl MulAssign<BinaryField1b> for BinaryField128bPolyval {
	fn mul_assign(&mut self, rhs: BinaryField1b) {
		*self = *self * rhs;
	}
}

impl Add<BinaryField128bPolyval> for BinaryField1b {
	type Output = BinaryField128bPolyval;

	fn add(self, rhs: BinaryField128bPolyval) -> Self::Output {
		rhs + self
	}
}

impl Sub<BinaryField128bPolyval> for BinaryField1b {
	type Output = BinaryField128bPolyval;

	fn sub(self, rhs: BinaryField128bPolyval) -> Self::Output {
		rhs - self
	}
}

impl Mul<BinaryField128bPolyval> for BinaryField1b {
	type Output = BinaryField128bPolyval;

	fn mul(self, rhs: BinaryField128bPolyval) -> Self::Output {
		rhs * self
	}
}

impl ExtensionField<BinaryField1b> for BinaryField128bPolyval {
	type Iterator = <[BinaryField1b; 128] as IntoIterator>::IntoIter;
	const DEGREE: usize = 128;

	fn basis(i: usize) -> Result<Self, Error> {
		if i >= 128 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		Ok(Self::new(1 << i))
	}

	fn from_bases(base_elems: &[BinaryField1b]) -> Result<Self, Error> {
		if base_elems.len() > 128 {
			return Err(Error::ExtensionDegreeMismatch);
		}
		let value = base_elems
			.iter()
			.rev()
			.fold(0, |value, elem| value << 1 | elem.val() as u128);
		Ok(Self::new(value))
	}

	fn iter_bases(&self) -> Self::Iterator {
		let base_elems = array::from_fn(|i| BinaryField1b(((self.0 >> i) as u8) & 1));
		base_elems.into_iter()
	}
}

impl BinaryField for BinaryField128bPolyval {}

impl From<BinaryField128b> for BinaryField128bPolyval {
	fn from(value: BinaryField128b) -> Self {
		const POLYVAL_BASES: [u128; 128] = [
			0xc2000000000000000000000000000001,
			0x21a09a4bf26aadcd3eb19c5f1a06b528,
			0xe62f1a804db43b94852cef0e61d7353d,
			0xadcde131ca862a6ba378ea68e992a5b6,
			0x5474611d07bdcd1f72e9bdc82ec4fe6c,
			0xf9a472d4a4965f4caa3532aa6258c986,
			0x10bd76c920260f81877681ed1a50b210,
			0xe7f3264523858ca36ef84934fdd225f2,
			0x586704bda927015fedb8ddceb7f825d6,
			0x552dab8acfd831aeb65f8aaec9cef096,
			0xeccdac666a363defde6792e475892fb3,
			0x4a621d01701247f6e4a8327e33d95aa2,
			0x8ed5002fed1f4b9a9a11840f87149e2d,
			0x3c65abbd41c759f0302467db5a791e09,
			0xc2df68a5949a96b3aa643692e93caaab,
			0x4455027df88c165117daf9822eb57383,
			0xc50e3a207f91d7cd6dd1e116d55455fb,
			0xc89c3920b9b24b755fd08543d8caf5a2,
			0xfa583eb935de76a2ec180360b6548830,
			0xc4d3d3b9938f3af77800a5cd03690171,
			0xe1faff3b895be1e2bec91c0836143b44,
			0x256bd50f868b82cf1c83552eeb1cd844,
			0x82fd35d590073ae9595cab38e9b59d79,
			0x08dadd230bc90e192304a2533cdce9e6,
			0xf4400f37acedc7d9502abeff6cead84c,
			0x5438d34e2b5b90328cc88b7384deedfb,
			0x7d798db71ef80a3e447cd7d1d4a0385d,
			0xa50d5ef4e33979db8012303dc09cbf35,
			0x91c4b5e29de5759e0bb337efbc5b8115,
			0xbbb0d4aaba0fab72848f461ed0a4b110,
			0x3c9de86b9a306d6d11cc078904076865,
			0xb5f43a166aa1f15f99db6d689ca1b370,
			0xa26153cb8c150af8243ecbd46378e59e,
			0xccaa154bab1dd7aca876f81fe0c950ab,
			0x4185b7e3ee1dddbc761a6139cdb07755,
			0x2c9f95285b7aa574653ed207337325f2,
			0xc8ba616ab131bfd242195c4c82d54dbb,
			0x2a9b07221a34865faa36a28da1ab1c24,
			0x7e6e572804b548a88b92900e0196dd39,
			0x4e9060deff44c9ef9882a0015debd575,
			0x00a3a4d8c163c95ac7ac9a5b424e1c65,
			0xf67c7eb5dde73d96f8f5eecba6033679,
			0x54d78d187bbb57d19b536094ba539fde,
			0x76c553699edc5d4a033139975ab7f264,
			0x74ae8da43b2f587df3e41bbf5c6be650,
			0x8a2941b59774c41acd850aa6098e5fd2,
			0x9ddf65660a6f8f3c0058165a063de84c,
			0xbb52da733635cc3d1ff02ef96ee64cf3,
			0x564032a0d5d3773b7b7ed18bebf1c668,
			0xef5c765e64b24b1b00222054ff0040ef,
			0xade661c18acba6233d484726e6249bee,
			0x9939ba35c969cdeea29f2ef849c2d170,
			0x2b100b39761d4f23eb42d05b80174ce2,
			0xfbc25b179830f9eec765bd6229125d6c,
			0xb58e089ebe7ad0b2698e30184ab93141,
			0x53874933a148be94d12025afa876234c,
			0x41bbc7902188f4e9880f1d81fa580ffb,
			0xea4199916a5d127d25da1fe777b2dcbb,
			0xe7bc816547efbe987d9359ee0de0c287,
			0x02e0f1f67e7139835892155a7addd9da,
			0xdc6beb6eade9f875e74955ca950af235,
			0x786d616edeadfa356453a78d8f103230,
			0xe84e70191accaddac8034da936737487,
			0x012b8669ff3f451e5363edfddd37fb3c,
			0x756209f0893e96877833c194b9c943a0,
			0xb2ac9efc9a1891369f63bd1e0d1439ac,
			0x4de88e9a5bbb4c3df650cc3994c3d2d8,
			0x8de7b5c85c07f3359849e7c85e426b54,
			0xcadd54ae6a7e72a4f184e6761cf226d4,
			0xcdb182fb8d95496f55b5f3952f81bc30,
			0x40013bc3c81722753a05bb2aca01a02e,
			0x704e7ce55e9033883e97351591adf18a,
			0xf330cd9a74a5e884988c3f36567d26f4,
			0x18f4535304c0d74ac3bdf09d78cbde50,
			0xfe739c97fc26bed28885b838405c7e7e,
			0x492479260f2dcd8af980c3d74b3ec345,
			0x96b6440a34de0aad4ea2f744396691af,
			0x98355d1b4f7cfb03960a59aa564a7a26,
			0x2703fda0532095ca8b1886b12ca37d64,
			0x59c9dabe49bebf6b468c3c120f142822,
			0xf8f3c35c671bac841b14381a592e6cdd,
			0xd7b888791bd83b13d80d2e9324894861,
			0x113ab0405354dd1c5aab9658137fa73f,
			0xae56192d5e9c309e461f797121b28ce6,
			0xb7927ec7a84c2e04811a6dac6b997783,
			0x9e2f8d67fc600703ba9b4189ce751cb4,
			0x574e95df2d8bb9e2c8fc29729eb723ca,
			0x38bc6fc47739c06cd9fa20f9a5088f26,
			0x69d3b9b1d9483174b3c38d8f95ce7a5f,
			0xd6e4bb147cc82b6e90e27e882f18640d,
			0x027338db641804d985cd9fece12f7adc,
			0x523cb73968169ccce76f523928c4364e,
			0xcdcf898117f927208a11b0dcc941f2f6,
			0xc908287814c8cba67f7892fec7a5b217,
			0x92b99988bb26215d104968d4cbbb285a,
			0x4dbca8fd835d00ea4b95692534ef5068,
			0xcd8b92c8a6e0e65e167a2b851f32fd9c,
			0xc3473dfda9f97d6ac1e2d544628e7845,
			0x0260e7badc64dbfde0dc39a240365722,
			0x3966125b40fe2bca9719c80e41953868,
			0xac0211506eda3cba57b709a360d4a2c7,
			0x0e4f0e47d02fedd15b337fefa219c52b,
			0x1d5907ccdc659f7aace675511f754ee3,
			0x4ad5b368eaddc4bb097284863b2a5b6e,
			0x2eae07273b8c4fc5cef553a4a46cde5b,
			0x096a310e7b1e3a3179d4a3b5d8dd9396,
			0x8c81362eeb1656a91dde08d05018a353,
			0x387e59e44cc0d53fecf7f057b6fdba0b,
			0x9d29670bbd0e8051ac82d91ca97561d6,
			0xaf1310d0f5cac4e89714e48065be74a4,
			0x9b684a3865c2b59c411d14182a36fb6b,
			0x3e7de163516ffdcaca22b4e848340fbe,
			0x3c37dbe331de4b0dc2f5db315d5e7fda,
			0x19e7f4b53ff86990e3d5a1c40c3769a0,
			0x56469ab32b2b82e8cc93fdb1b14a4775,
			0x9c01cefde47816300d8ad49d260bb71b,
			0x6100101b8cebde7381366fec1e4e52c0,
			0xa28d30c3cbd8b69632143fa65158ee4f,
			0x3db7a902ec509e58151c45f71eee6368,
			0x42d5a505e8ab70097107d37d79ebbaba,
			0xe47b83247cb2b162c7d6d15c84cca8ce,
			0x076caf0e23541c753e4c87ff505737a5,
			0x590a8d1cdbd17ae83980f5d1d3b84a89,
			0x77d649ff61a7cd0da53497edd34c4204,
			0xefbe0c34eeab379ea4a8feed84fd3993,
			0x90540cf7957a8a3051629cdde777f968,
			0x8749050496dd288244c49c70aa92831f,
			0x0fc80b1d600406b2370368d94947961a,
		];
		ExtensionField::<BinaryField1b>::iter_bases(&value)
			.zip(POLYVAL_BASES.iter())
			.fold(BinaryField128bPolyval::ZERO, |acc, (scalar, &basis_elem)| {
				acc + BinaryField128bPolyval(basis_elem) * scalar
			})
	}
}

impl From<BinaryField128bPolyval> for BinaryField128b {
	fn from(value: BinaryField128bPolyval) -> BinaryField128b {
		const TOWER_BASES: [u128; 128] = [
			0x66e1d645d7eb87dca8fc4d30a32dadcc,
			0x53ca87ba77172fd8c5675d78c59c1901,
			0x1a9cf63d31827dcda15acb755a948567,
			0xa8f28bdf6d29cee2474b0401a99f6c0a,
			0x4eefa9efe87ed19c06b39ca9799c8d73,
			0x06ec578f505abf1e9885a6b2bc494f3e,
			0x70ecdfe1f601f8509a96d3fb9cd3348a,
			0xcb0d16fc7f13733deb25f618fc3faf28,
			0x4e9a97aa2c84139ffcb578115fcbef3c,
			0xc6de6210afe8c6bd9a441bffe19219ad,
			0x73e3e8a7c59748601be5bf1e30c488d3,
			0x1f6d67e2e64bd6c4b39e7f4bb37dce9c,
			0xc34135d567eada885f5095b4c155f3b5,
			0x23f165958d59a55e4790b8e2e37330e4,
			0x4f2be978f16908e405b88802add08d17,
			0x6442b00f5bbf4009907936513c3a7d45,
			0xac63f0397d911a7a5d61b9f18137026f,
			0x8e70543ae0e43313edf07cbc6698e144,
			0xcb417a646d59f652aa5a07984066d026,
			0xf028de8dd616318735bd8f76de7bb84e,
			0x2e03a12472d21599f15b4bcaa9bf186c,
			0x54a376cc03e5b2cfa27d8e48d1b9ca76,
			0xd22894c253031b1b201b87da07cb58ae,
			0x6bc1416afea6308ff77d902dd5d2a563,
			0x9958ecd28adbebf850055f8ac3095121,
			0x595a1b37062233d7e6bb6f54c227fb91,
			0x41ffcfcdda4583c4f671558ee315d809,
			0x780c2490f3e5cb4763e982ec4b3e6ea2,
			0xf7a450b35931fa76722a6b9037b6db34,
			0xe21991100e84821328592772430ad07e,
			0x360d4079f62863cc60c65ec87d6f9277,
			0xd898bfa0b076cc4eaca590e7a60dbe92,
			0xcaacddd5e114fe5c2e1647fc34b549bf,
			0x3042e34911c28e90617776ddb2d3f888,
			0x3728a3b0da53cdfecfd8455b13cb9b14,
			0x2f2eb3d5bc7b2c48a7c643bffbddc6b2,
			0x3b71a5c04010c0aa501b04302706b908,
			0x0701845b090e79bb9be54df766e48c51,
			0x1e9eac7bf45b14c8db06fcfff7408f78,
			0x6b1b8e39a339423d0eb3bef69eee8b0b,
			0x8b06616385967df95d3a99cff1edcf0a,
			0x5d921137890a3ded58e1dd1a51fe6a30,
			0x828ed6fba42805b2628b705d38121acc,
			0x9b7a95220e9d5b0ff70ecb6116cabd81,
			0x0eb9055cb11711ed047f136cab751c88,
			0xd6f590777c17a6d0ca451290f7d5c78a,
			0x401a922a6461fbe691f910cb0893e71f,
			0x15a549308bc53902c927ebad9ed253f7,
			0x45dccafc72a584480f340a43f11a1b84,
			0x19d2a2c057d60656e6d3e20451335d5b,
			0x035af143a5827a0f99197c8b9a811454,
			0x7ee35d174ad7cc692191fd0e013f163a,
			0xc4c0401d841f965c9599fac8831effa9,
			0x63e809a843fc04f84acfca3fc5630691,
			0xdb2f3301594e3de49fb7d78e2d6643c4,
			0x1b31772535984ef93d709319cc130a7c,
			0x036dc9c884cd6d6c918071b62a0593f3,
			0x4700cd0e81c88045132360b078027103,
			0xdfa3f35eb236ea63b0350e17ed2d625d,
			0xf0fd7c7760099f1ac28be91822978e15,
			0x852a1eba3ad160e95034e9eed1f21205,
			0x4a07dd461892df45ca9efee1701763c3,
			0xadbbaa0add4c82fe85fd61b42f707384,
			0x5c63d0673f33c0f2c231db13f0e15600,
			0x24ddc1516501135626e0e794dd4b3076,
			0xb60c601bbf72924e38afd02d201fb05b,
			0x2ef68918f416caca84334bcf70649aeb,
			0x0b72a3124c504bcad815534c707343f2,
			0xcfd8b2076040c43d5d396f8523d80fe0,
			0x098d9daf64154a63504192bb27cc65e1,
			0x3ae44070642e6720283621f8fb6a6704,
			0x19cd9b2843d0ff936bfe2b373f47fd05,
			0x451e2e4159c78e65db10450431d26122,
			0x797b753e29b9d0e9423b36807c70f3ae,
			0xa8d0e8ba9bb634f6ea30600915664e22,
			0xdf8c74bbd66f86809c504cb944475b0a,
			0x32831a457ced3a417a5a94d498128018,
			0x1aca728985936a6147119b9b5f00350e,
			0x6f436d64b4ee1a556b66764ed05bb1db,
			0x25930eaed3fd982915e483cb21e5a1a2,
			0x21735f5eb346e56006bf1d7e151780ab,
			0x55fc6f607f10e17f805eb16d7bd5345c,
			0x4b4d289591f878114965292af4aeb57e,
			0x30608bc7444bcbaff67998c1883c1cf3,
			0xa12a72abe4152e4a657c6e6395404343,
			0x7579186d4e0959dec73f9cd68fb0e2fb,
			0xb5560ce63f7894cc965c822892b7bfda,
			0x6b06d7165072861eba63d9fd645995d7,
			0x359f439f5ec9107dde3c8ef8f9bf4e29,
			0xcbfe7985c6006a46105821cd8b55b06b,
			0x2110b3b51f5397ef1129fb9076474061,
			0x1928478b6f3275c944c33b275c388c47,
			0x23f978e6a0a54802437111aa4652421a,
			0xe8c526bf924dc5cd1dd32dbedd310f5b,
			0xa0ac29f901f79ed5f43c73d22a05c8e4,
			0x55e0871c6e97408f47f4635b747145ea,
			0x6c2114c3381f53667d3c2dfefd1ebcb3,
			0x42d23c18722fbd58863c3aceaaa3eef7,
			0xbb0821ab38d5de133838f8408a72fdf1,
			0x035d7239054762b131fa387773bb9153,
			0x8fa898aafe8b154f9ab652e8979139e7,
			0x6a383e5cd4a16923c658193f16cb726c,
			0x9948caa8c6cefb0182022f32ae3f68b9,
			0x8d2a8decf9855bd4df7bac577ed73b44,
			0x09c7b8300f0f984259d548c5aa959879,
			0x92e16d2d24e070efdca8b8e134047afc,
			0x47d8621457f4118aaf24877fb5031512,
			0x25576941a55f0a0c19583a966a85667f,
			0xb113cad79cd35f2e83fda3bc6285a8dc,
			0xc76968eecb2748d0c3e6318431ffe580,
			0x7211122aa7e7f6fe39e6618395b68416,
			0x88463599bf7d3e92f450d00a45146d11,
			0x6e12b7d5adf95da33bbb7f79a18ee123,
			0xe0a98ac4025bc568eaca7e7b7280ff16,
			0xc13fc79f6c35048df274057ac892ff77,
			0x93c1a3145d4e47dee39cae4de47eb505,
			0x780064be3036df98f1e5d7c53bdbd52b,
			0x48c467b5cec265628b709172ecaff561,
			0x5bbbab77ce5552ff7682094560524a7e,
			0x551537ef6048831fb128fec4e4a23a63,
			0xe7ef397fcc095ead439317a13568b284,
			0xbc5d2927eac0a720f9d75d62d92c6332,
			0x3bfeb420021f93e9b2bc992b5b59e61e,
			0xc651dc438e2f1bc64af1b7307b574ed9,
			0xbfe0a17ee2b777542a1ddb55413a4e43,
			0xa062da2427df3d1a7dfc01c05d732a32,
			0x1e4889fd72b70ecf93417ba0b085e1e8,
			0xc4f4769f4f9c2e33c26a6bf2ca842f17,
		];
		ExtensionField::<BinaryField1b>::iter_bases(&value)
			.zip(TOWER_BASES.iter())
			.fold(BinaryField128b::ZERO, |acc, (scalar, &basis_elem)| {
				acc + BinaryField128b::new(basis_elem) * scalar
			})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use proptest::prelude::*;

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
			BinaryField128bPolyval::new(0x2a9055e4e69a61f0b5cfd6f4161087ba).square(),
			BinaryField128bPolyval::new(0x59aba0d4ffa9dca427b5b489f293e529)
		);
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
	}
}
