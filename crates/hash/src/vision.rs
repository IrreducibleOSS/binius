// Copyright 2024-2025 Irreducible Inc.

use std::{array, iter::repeat, mem::MaybeUninit};

use binius_field::{
	linear_transformation::{
		FieldLinearTransformation, PackedTransformationFactory, Transformation,
	},
	make_aes_to_binary_packed_transformer, make_binary_to_aes_packed_transformer,
	underlier::WithUnderlier,
	AESTowerField32b, AESTowerField8b, AesToBinaryTransformation, BinaryField8b,
	BinaryToAesTransformation, ByteSlicedAES32x32b, ByteSlicedAES4x32x8b,
	PackedAESBinaryField32x8b, PackedAESBinaryField8x32b, PackedBinaryField32x8b,
	PackedBinaryField8x32b, PackedExtension, PackedExtensionIndexable, PackedField,
	PackedFieldIndexable, RepackedExtension,
};
use binius_ntt::{
	twiddle::{OnTheFlyTwiddleAccess, TwiddleAccess},
	SingleThreadedNTT,
};
use digest::consts::U32;
use lazy_static::lazy_static;
use stackalloc::helpers::slice_assume_init_mut;

use crate::{
	permutation::Permutation,
	vision_constants::{
		AFFINE_FWD_AES, AFFINE_FWD_CONST_AES, AFFINE_INV_AES, AFFINE_INV_CONST_AES, NUM_ROUNDS,
		ROUND_KEYS,
	},
	MultiDigest,
};

const RATE_AS_U32: usize = 16;
const RATE_AS_U8: usize = RATE_AS_U32 * std::mem::size_of::<u32>();

const SCALAR_FWD_TRANS_AES: FieldLinearTransformation<AESTowerField32b> =
	FieldLinearTransformation::new_const(&AFFINE_FWD_AES);
const SCALAR_INV_TRANS_AES: FieldLinearTransformation<AESTowerField32b> =
	FieldLinearTransformation::new_const(&AFFINE_INV_AES);

type PackedTransformationType8x32bAES = <PackedAESBinaryField8x32b as PackedTransformationFactory<
	PackedAESBinaryField8x32b,
>>::PackedTransformation<&'static [AESTowerField32b]>;
type PackedTransformationType32x32bAES = <ByteSlicedAES32x32b as PackedTransformationFactory<
	ByteSlicedAES32x32b,
>>::PackedTransformation<&'static [AESTowerField32b]>;
type AdditiveNTT8b = SingleThreadedNTT<AESTowerField8b, OnTheFlyTwiddleAccess<AESTowerField8b>>;

lazy_static! {
	/// We use this object only to calculate twiddles for the fast NTT.
	static ref ADDITIVE_NTT_AES: AdditiveNTT8b = {
		let log_h = 3;
		let log_rate = 1;
		SingleThreadedNTT::<AESTowerField8b>::with_domain_field::<BinaryField8b>(log_h + 2 + log_rate)
			.expect("log_domain_size is less than 32")
	};

	/// Specialized fast additive NTT to transform 3 x PackAESBinaryField8x32b with cosets [0, 1, 2]
	static ref INVERSE_FAST_TRANSFORM: FastNTT = FastNTT::new(&ADDITIVE_NTT_AES, [0, 1, 2]);
	/// Specialized fast additive NTT to transform 3 x PackAESBinaryField8x32b with cosets [3, 4, 5]
	static ref FORWARD_FAST_TRANSFORM: FastNTT = FastNTT::new(&ADDITIVE_NTT_AES, [3, 4, 5]);

	static ref INVERSE_FAST_TRANSFORM_BYTE_SLICED: FastNttByteSliced = FastNttByteSliced::new(&ADDITIVE_NTT_AES, [0, 1, 2]);
	static ref FORWARD_FAST_TRANSFORM_BYTE_SLICED: FastNttByteSliced = FastNttByteSliced::new(&ADDITIVE_NTT_AES, [3, 4, 5]);

	pub static ref FWD_PACKED_TRANS_AES: PackedTransformationType8x32bAES = <PackedAESBinaryField8x32b as PackedTransformationFactory<
		PackedAESBinaryField8x32b,
	>>::make_packed_transformation(SCALAR_FWD_TRANS_AES);
	pub static ref INV_PACKED_TRANS_AES: PackedTransformationType8x32bAES = <PackedAESBinaryField8x32b as PackedTransformationFactory<
		PackedAESBinaryField8x32b,
	>>::make_packed_transformation(SCALAR_INV_TRANS_AES);

	pub static ref FWD_PACKED_TRANS_AES_BYTE_SLICED: PackedTransformationType32x32bAES = <ByteSlicedAES32x32b as PackedTransformationFactory<
		ByteSlicedAES32x32b,
	>>::make_packed_transformation(SCALAR_FWD_TRANS_AES);
	pub static ref INV_PACKED_TRANS_AES_BYTE_SLICED: PackedTransformationType32x32bAES = <ByteSlicedAES32x32b as PackedTransformationFactory<
		ByteSlicedAES32x32b,
	>>::make_packed_transformation(SCALAR_INV_TRANS_AES);

	static ref FWD_CONST_AES: PackedAESBinaryField8x32b = PackedField::broadcast(AFFINE_FWD_CONST_AES);
	static ref INV_CONST_AES: PackedAESBinaryField8x32b = PackedField::broadcast(AFFINE_INV_CONST_AES);

	static ref ROUND_KEYS_PACKED_AES: [[PackedAESBinaryField8x32b; 3]; 2 * NUM_ROUNDS + 1] = ROUND_KEYS.map(|key| {
			let arr: [PackedAESBinaryField8x32b; 3] = key
				.chunks_exact(8)
				.map(|x| PackedAESBinaryField8x32b::from_fn(|i| AESTowerField32b::from(x[i])))
				.collect::<Vec<_>>()
				.try_into()
				.unwrap();
			arr
		});
	static ref ROUND_KEYS_PACKED_AES_BYTE_SLICED: [[ByteSlicedAES32x32b; 3]; 2 * NUM_ROUNDS + 1] = ROUND_KEYS_PACKED_AES.map(|round_consts| {
		round_consts.map(|x|
			ByteSlicedAES32x32b::from_scalars(x.clone().into_iter().cycle())
		)
	});

	static ref PERMUTATION: VisionPermutation = VisionPermutation::default();
	static ref PERMUTATION_BYTE_SLICED: VisionPermutationByteSliced = VisionPermutationByteSliced::default();

	static ref TRANS_AES_TO_CANONICAL: AesToBinaryTransformation<PackedAESBinaryField8x32b, PackedBinaryField8x32b> =
		make_aes_to_binary_packed_transformer::<PackedAESBinaryField8x32b, PackedBinaryField8x32b>();
	static ref TRANS_CANONICAL_TO_AES: BinaryToAesTransformation<PackedBinaryField8x32b, PackedAESBinaryField8x32b> =
		make_binary_to_aes_packed_transformer::<PackedBinaryField8x32b, PackedAESBinaryField8x32b>();
}

#[inline]
fn add_packed_768<P: PackedField>(a: &mut [P; 3], b: &[P; 3]) {
	for i in 0..3 {
		a[i] += b[i];
	}
}

/// The MDS step in the Vision Permutation which uses AdditiveNTT to compute matrix multiplication
/// of the state vector 24x32b
#[derive(Debug, Clone)]
pub struct Vision32MDSTransform<
	P: PackedField<Scalar = AESTowerField8b>,
	Ntt: NttExecutor<P> + 'static,
> {
	x: P,
	y: P,
	z: P,
	forward_ntt: &'static Ntt,
	inverse_ntt: &'static Ntt,
}

impl Default for Vision32MDSTransform<PackedAESBinaryField32x8b, FastNTT> {
	fn default() -> Self {
		Self::new(&*FORWARD_FAST_TRANSFORM, &*INVERSE_FAST_TRANSFORM)
	}
}

impl<P: PackedField<Scalar = AESTowerField8b>, Ntt: NttExecutor<P> + 'static>
	Vision32MDSTransform<P, Ntt>
{
	fn new(forward_ntt: &'static Ntt, inverse_ntt: &'static Ntt) -> Self {
		let x = P::broadcast(ADDITIVE_NTT_AES.get_subspace_eval(3, 1));
		let y = P::broadcast(ADDITIVE_NTT_AES.get_subspace_eval(3, 2));
		let z = P::broadcast(ADDITIVE_NTT_AES.get_subspace_eval(4, 1));

		Self {
			x,
			y,
			z,
			forward_ntt,
			inverse_ntt,
		}
	}

	pub fn transform(&self, data: &mut [P; 3]) {
		self.inverse_ntt.inverse(data);

		data[1] += data[0];
		let x = self.x * data[1];
		data[2] += x + data[0];

		let y = self.y * data[1];
		let z = self.z * data[2];

		let stash_0 = data[0];
		let stash_1 = data[1];
		data[0] += x + data[1] + data[2];
		data[1] = stash_0 + y + z;
		data[2] = data[1] + stash_1;

		self.forward_ntt.forward(data);
	}
}

/// This is the complete permutation function for the Vision hash which implements `Permutation`
/// and `CryptographicPermutation` traits over `PackedAESBinary8x32b` as well as `BinaryField32b`
pub struct Vision32bPermutation<P, PE, Ntt, LinearTransformation>
where
	P: PackedField<Scalar = AESTowerField8b>,
	PE: RepackedExtension<P, Scalar = AESTowerField32b>,
	Ntt: NttExecutor<P> + 'static,
	LinearTransformation: Transformation<PE, PE> + 'static,
{
	mds: Vision32MDSTransform<P, Ntt>,
	round_keys_packed: &'static [[PE; 3]; 2 * NUM_ROUNDS + 1],
	forward_transformation: &'static LinearTransformation,
	inverse_transformation: &'static LinearTransformation,
	forward_ntt: &'static Ntt,
	inverse_ntt: &'static Ntt,
	forward_constants: PE,
	inverse_constants: PE,
}

impl Default
	for Vision32bPermutation<
		PackedAESBinaryField32x8b,
		PackedAESBinaryField8x32b,
		FastNTT,
		PackedTransformationType8x32bAES,
	>
{
	fn default() -> Self {
		Self::new(
			&FORWARD_FAST_TRANSFORM,
			&INVERSE_FAST_TRANSFORM,
			&FWD_PACKED_TRANS_AES,
			&INV_PACKED_TRANS_AES,
			&ROUND_KEYS_PACKED_AES,
		)
	}
}

impl Default
	for Vision32bPermutation<
		ByteSlicedAES4x32x8b,
		ByteSlicedAES32x32b,
		FastNttByteSliced,
		PackedTransformationType32x32bAES,
	>
{
	fn default() -> Self {
		Self::new(
			&FORWARD_FAST_TRANSFORM_BYTE_SLICED,
			&INVERSE_FAST_TRANSFORM_BYTE_SLICED,
			&FWD_PACKED_TRANS_AES_BYTE_SLICED,
			&INV_PACKED_TRANS_AES_BYTE_SLICED,
			&ROUND_KEYS_PACKED_AES_BYTE_SLICED,
		)
	}
}

impl<P, PE, Ntt, LinearTransformation> Clone
	for Vision32bPermutation<P, PE, Ntt, LinearTransformation>
where
	P: PackedField<Scalar = AESTowerField8b>,
	PE: RepackedExtension<P, Scalar = AESTowerField32b>,
	Ntt: NttExecutor<P> + 'static,
	LinearTransformation: Transformation<PE, PE> + 'static,
{
	fn clone(&self) -> Self {
		Self {
			mds: self.mds.clone(),
			round_keys_packed: self.round_keys_packed,
			forward_transformation: self.forward_transformation,
			inverse_transformation: self.inverse_transformation,
			forward_ntt: self.forward_ntt,
			inverse_ntt: self.inverse_ntt,
			forward_constants: self.forward_constants,
			inverse_constants: self.inverse_constants,
		}
	}
}

impl<P, PE, Ntt, LinearTransformation> Permutation<[PE; 3]>
	for Vision32bPermutation<P, PE, Ntt, LinearTransformation>
where
	P: PackedField<Scalar = AESTowerField8b>,
	PE: RepackedExtension<P, Scalar = AESTowerField32b>,
	Ntt: NttExecutor<P> + 'static,
	LinearTransformation: Transformation<PE, PE> + 'static,
{
	fn permute_mut(&self, input: &mut [PE; 3]) {
		add_packed_768(input, &self.round_keys_packed[0]);
		for r in 0..NUM_ROUNDS {
			self.sbox_step(input, self.inverse_transformation, self.inverse_constants);

			let input_bases = PE::cast_bases_mut(input);
			self.mds
				.transform(input_bases.try_into().expect("input is 3 elements"));

			add_packed_768(input, &self.round_keys_packed[1 + 2 * r]);
			self.sbox_step(input, self.forward_transformation, self.forward_constants);
			let input_bases = PE::cast_bases_mut(input);
			self.mds
				.transform(input_bases.try_into().expect("input is 3 elements"));
			add_packed_768(input, &self.round_keys_packed[2 + 2 * r]);
		}
	}
}

impl<P, PE, Ntt, LinearTransformation> Vision32bPermutation<P, PE, Ntt, LinearTransformation>
where
	P: PackedField<Scalar = AESTowerField8b>,
	PE: RepackedExtension<P, Scalar = AESTowerField32b>,
	Ntt: NttExecutor<P> + 'static,
	LinearTransformation: Transformation<PE, PE> + 'static,
{
	pub fn new(
		forward_ntt: &'static Ntt,
		inverse_ntt: &'static Ntt,
		forward_transformation: &'static LinearTransformation,
		inverse_transformation: &'static LinearTransformation,
		round_keys_packed: &'static [[PE; 3]; 2 * NUM_ROUNDS + 1],
	) -> Self {
		Self {
			mds: Vision32MDSTransform::new(forward_ntt, inverse_ntt),
			round_keys_packed,
			forward_ntt,
			inverse_ntt,
			forward_transformation,
			inverse_transformation,
			forward_constants: PE::broadcast(AFFINE_FWD_CONST_AES),
			inverse_constants: PE::broadcast(AFFINE_INV_CONST_AES),
		}
	}

	#[inline]
	fn sbox_packed_affine(
		&self,
		chunk: &mut PE,
		packed_linear_trans: &impl Transformation<PE, PE>,
		constant: PE,
	) {
		let x_inv_eval = chunk.invert_or_zero();
		let result = packed_linear_trans.transform(&x_inv_eval);
		*chunk = result + constant
	}

	fn sbox_step(
		&self,
		d: &mut [PE; 3],
		packed_linear_trans: &impl Transformation<PE, PE>,
		constant: PE,
	) {
		for chunk in d.iter_mut() {
			self.sbox_packed_affine(chunk, packed_linear_trans, constant);
		}
	}
}

type VisionPermutation = Vision32bPermutation<
	PackedAESBinaryField32x8b,
	PackedAESBinaryField8x32b,
	FastNTT,
	PackedTransformationType8x32bAES,
>;
type VisionPermutationByteSliced = Vision32bPermutation<
	ByteSlicedAES4x32x8b,
	ByteSlicedAES32x32b,
	FastNttByteSliced,
	PackedTransformationType32x32bAES,
>;

#[derive(Clone)]
pub struct VisionHasherDigest {
	// The hashed state
	state: [PackedAESBinaryField8x32b; 3],
	buffer: [u8; RATE_AS_U8],
	filled_bytes: usize,
}

impl Default for VisionHasherDigest {
	fn default() -> Self {
		Self {
			state: [PackedAESBinaryField8x32b::zero(); 3],
			buffer: [0; RATE_AS_U8],
			filled_bytes: 0,
		}
	}
}

impl VisionHasherDigest {
	fn permute(state: &mut [PackedAESBinaryField8x32b; 3], data: &[u8]) {
		debug_assert_eq!(data.len(), RATE_AS_U8);

		let mut data_packed = [PackedBinaryField8x32b::zero(); 2];
		for (i, value_32) in WithUnderlier::to_underliers_ref_mut(
			PackedBinaryField8x32b::unpack_scalars_mut(&mut data_packed),
		)
		.iter_mut()
		.enumerate()
		{
			*value_32 =
				u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().expect("chunk is 4 bytes"));
		}

		for i in 0..2 {
			state[i] = TRANS_CANONICAL_TO_AES.transform(&data_packed[i]);
		}

		PERMUTATION.permute_mut(state);
	}
}

impl digest::HashMarker for VisionHasherDigest {}

impl digest::Update for VisionHasherDigest {
	fn update(&mut self, mut data: &[u8]) {
		if self.filled_bytes != 0 {
			let to_copy = std::cmp::min(data.len(), RATE_AS_U8 - self.filled_bytes);
			self.buffer[self.filled_bytes..self.filled_bytes + to_copy]
				.copy_from_slice(&data[..to_copy]);
			data = &data[to_copy..];
			self.filled_bytes += to_copy;

			if self.filled_bytes == RATE_AS_U8 {
				Self::permute(&mut self.state, &self.buffer);
				self.filled_bytes = 0;
			}
		}

		let mut chunks = data.chunks_exact(RATE_AS_U8);
		for chunk in &mut chunks {
			Self::permute(&mut self.state, chunk);
		}

		let remaining = chunks.remainder();
		if !remaining.is_empty() {
			self.buffer[..remaining.len()].copy_from_slice(remaining);
			self.filled_bytes = remaining.len();
		}
	}
}

impl digest::OutputSizeUser for VisionHasherDigest {
	type OutputSize = U32;
}

impl digest::FixedOutput for VisionHasherDigest {
	fn finalize_into(mut self, out: &mut digest::Output<Self>) {
		if self.filled_bytes != 0 {
			self.buffer[self.filled_bytes..].fill(0);
			Self::permute(&mut self.state, &self.buffer);
		}

		let canonical_tower: PackedBinaryField8x32b =
			TRANS_AES_TO_CANONICAL.transform(&self.state[0]);
		out.copy_from_slice(BinaryField8b::to_underliers_ref(
			PackedBinaryField8x32b::unpack_base_scalars(std::slice::from_ref(&canonical_tower)),
		));
	}
}

pub trait NttExecutor<P>: Clone + Sync {
	/// This method executes the transformation equivalent to `AdditiveNTT::forward_transform`.
	/// Each `data` element is treated as 32 8-bit AES field elements.
	fn forward(&self, data: &mut [P; 3]);

	/// This method executes the transformation equivalent to `AdditiveNTT::inverse_transform`.
	/// Each `data` element is treated as 32 8-bit AES field elements.
	fn inverse(&self, data: &mut [P; 3]);
}

/// This structure represents fast additive NTT transformation that transforms
/// 3 x `PackedAESBinaryField8x32b` with a different coset for each item in a single go.
#[derive(Clone)]
pub struct FastNTT {
	// Each of the arrays below contains [interleaved twiddles of cosets 0 and 1, broadcast twiddles for coset 2]
	round_0_twiddles: [PackedAESBinaryField32x8b; 2],
	round_1_twiddles: [PackedAESBinaryField32x8b; 2],
	round_2_twiddles: [PackedAESBinaryField32x8b; 2],
}

impl FastNTT {
	fn new(ntt: &AdditiveNTT8b, cosets: [u32; 3]) -> Self {
		let get_coset_twiddles_for_round =
			|round: usize| get_coset_twiddles_for_round(round, cosets, ntt);

		let cosets_twiddles_0 = get_coset_twiddles_for_round(0);
		let cosets_twiddles_1 = get_coset_twiddles_for_round(1);
		let cosets_twiddles_2 = get_coset_twiddles_for_round(2);

		let round_0_twiddles = [
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_0[0].get(0))
					.take(4)
					.chain(repeat(cosets_twiddles_0[1].get(0)).take(4))
					.chain(repeat(cosets_twiddles_0[0].get(1)).take(4))
					.chain(repeat(cosets_twiddles_0[1].get(1)).take(4))
					.chain(repeat(cosets_twiddles_0[0].get(2)).take(4))
					.chain(repeat(cosets_twiddles_0[1].get(2)).take(4))
					.chain(repeat(cosets_twiddles_0[0].get(3)).take(4))
					.chain(repeat(cosets_twiddles_0[1].get(3)).take(4)),
			),
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_0[2].get(0))
					.take(8)
					.chain(repeat(cosets_twiddles_0[2].get(1)).take(8))
					.chain(repeat(cosets_twiddles_0[2].get(2)).take(8))
					.chain(repeat(cosets_twiddles_0[2].get(3)).take(8))
					.cycle(),
			),
		];
		let round_1_twiddles = [
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_1[0].get(0))
					.take(8)
					.chain(repeat(cosets_twiddles_1[1].get(0)).take(8))
					.chain(repeat(cosets_twiddles_1[0].get(1)).take(8))
					.chain(repeat(cosets_twiddles_1[1].get(1)).take(8)),
			),
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_1[2].get(0))
					.take(16)
					.chain(repeat(cosets_twiddles_1[2].get(1)).take(16))
					.cycle(),
			),
		];
		let round_2_twiddles = [
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_2[0].get(0))
					.take(16)
					.chain(repeat(cosets_twiddles_2[1].get(0)).take(16))
					.cycle(),
			),
			PackedField::broadcast(cosets_twiddles_2[2].get(0)),
		];

		Self {
			round_0_twiddles,
			round_1_twiddles,
			round_2_twiddles,
		}
	}
}

impl NttExecutor<PackedAESBinaryField32x8b> for FastNTT {
	#[inline]
	fn forward(&self, data: &mut [PackedAESBinaryField32x8b; 3]) {
		let mut forward_round_simd = |twiddles: &[PackedAESBinaryField32x8b; 2], log_block_size| {
			(data[0], data[1]) =
				transform_forward_round_pair(twiddles[0], data[0], data[1], log_block_size);
			(data[2], _) = transform_forward_round_pair(
				twiddles[1],
				data[2],
				PackedAESBinaryField32x8b::zero(),
				log_block_size,
			);
		};

		// round 2
		forward_round_simd(&self.round_2_twiddles, 4);

		// round 1
		forward_round_simd(&self.round_1_twiddles, 3);

		// round 0
		forward_round_simd(&self.round_0_twiddles, 2);
	}

	#[inline]
	fn inverse(&self, data: &mut [PackedAESBinaryField32x8b; 3]) {
		let mut inverse_round = |twiddles: &[PackedAESBinaryField32x8b; 2], block_size| {
			(data[0], data[1]) =
				transform_inverse_round_pair(twiddles[0], data[0], data[1], block_size);
			(data[2], _) = transform_inverse_round_pair(
				twiddles[1],
				data[2],
				PackedAESBinaryField32x8b::zero(),
				block_size,
			);
		};

		// round 0
		inverse_round(&self.round_0_twiddles, 2);

		// round 1
		inverse_round(&self.round_1_twiddles, 3);

		// round 2
		inverse_round(&self.round_2_twiddles, 4);
	}
}

fn get_coset_twiddles_for_round(
	round: usize,
	cosets: [u32; 3],
	ntt: &AdditiveNTT8b,
) -> [impl TwiddleAccess<AESTowerField8b> + '_; 3] {
	cosets.map(|coset| ntt.twiddles()[round].coset(3, coset as _))
}

/// This method does a single forward NTT transformation round for two pairs (data, coset).
/// `twiddles` must contain interleaved twiddles for the given round.
/// Given an input:
/// twiddles = [coset_0_twiddles_0, coset_1_twiddles_0, coset_0_twiddles_1, coset_0_twiddles_1, ...].
/// data_0 = [data_0_0, data_0_1, data_0_2, data_0_3, ...]
/// data_1 = [data_1_0, data_1_1, data_1_2, data_1_3, ...]
/// returns
/// (
///   [data_0_0 + data_0_1 * coset_0_twiddles_0, data_0_1 + data_0_0 + data_0_1 * coset_0_twiddles_0, ...],
///   [data_1_0 + data_1_1 * coset_1_twiddles_0, data_1_1 + data_1_0 + data_1_1 * coset_1_twiddles_0, ...],
/// )
///
/// This method allows us to perform forward NTT transformation for two pieces of data using the same number of vector operations as for one piece.
fn transform_forward_round_pair(
	twiddles: PackedAESBinaryField32x8b,
	data_0: PackedAESBinaryField32x8b,
	data_1: PackedAESBinaryField32x8b,
	log_block_size: usize,
) -> (PackedAESBinaryField32x8b, PackedAESBinaryField32x8b) {
	let (odds, evens) = PackedField::interleave(data_0, data_1, log_block_size);
	let result_odds = odds + evens * twiddles;
	let result_evens = evens + result_odds;
	PackedField::interleave(result_odds, result_evens, log_block_size)
}

/// This method does a single inverse NTT transformation round for two pairs '(data, coset)'.
/// `twiddles` must contain interleaved twiddles for the given round.
/// Given an input:
/// twiddles = [coset_0_twiddles_0, coset_1_twiddles_0, coset_0_twiddles_1, coset_1_twiddles_1, ...].
/// data_0 = [data_0_0, data_0_1, data_0_2, data_0_3, ...]
/// data_1 = [data_1_0, data_1_1, data_1_2, data_1_3, ...]
/// returns
/// (
///   [data_0_0 + (data_0_0 + data_0_1) * coset_0_twiddles_0, data_0_0 + data_0_1, ...],
///   [data_1_0 + (data_1_0 + data_1_1) * coset_1_twiddles_0, data_1_0 + data_1_1, ...],
/// )
///
/// This method allows us to perform inverse NTT transformation for two pieces of data using the same number of vector operations as for one piece.
fn transform_inverse_round_pair(
	twiddles: PackedAESBinaryField32x8b,
	data_0: PackedAESBinaryField32x8b,
	data_1: PackedAESBinaryField32x8b,
	log_block_size: usize,
) -> (PackedAESBinaryField32x8b, PackedAESBinaryField32x8b) {
	let (odds, evens) = PackedField::interleave(data_0, data_1, log_block_size);
	let result_evens = odds + evens;
	let result_odds = odds + twiddles * result_evens;
	PackedField::interleave(result_odds, result_evens, log_block_size)
}

#[derive(Clone)]
struct FastNttByteSliced {
	// Each of the arrays below contains [interleaved twiddles of cosets 0 and 1, broadcast twiddles for coset 2]
	round_0_twiddles: [PackedAESBinaryField32x8b; 3],
	round_1_twiddles: [PackedAESBinaryField32x8b; 3],
	round_2_twiddles: [PackedAESBinaryField32x8b; 3],
}

impl FastNttByteSliced {
	fn new(ntt: &AdditiveNTT8b, cosets: [u32; 3]) -> Self {
		let get_coset_twiddles_for_round =
			|round: usize| get_coset_twiddles_for_round(round, cosets, ntt);

		let cosets_twiddles_0 = get_coset_twiddles_for_round(0);
		let cosets_twiddles_1 = get_coset_twiddles_for_round(1);
		let cosets_twiddles_2 = get_coset_twiddles_for_round(2);

		let round_0_twiddles = array::from_fn(|coset| {
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_0[coset].get(0))
					.take(2)
					.chain(repeat(cosets_twiddles_0[coset].get(1)).take(2))
					.chain(repeat(cosets_twiddles_0[coset].get(2)).take(2))
					.chain(repeat(cosets_twiddles_0[coset].get(3)).take(2))
					.cycle(),
			)
		});
		let round_1_twiddles = array::from_fn(|coset| {
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_1[coset].get(0))
					.take(4)
					.chain(repeat(cosets_twiddles_1[coset].get(1)).take(4))
					.cycle(),
			)
		});
		let round_2_twiddles = array::from_fn(|coset| {
			PackedAESBinaryField32x8b::from_scalars(
				repeat(cosets_twiddles_2[coset].get(0)).take(16).cycle(),
			)
		});

		Self {
			round_0_twiddles,
			round_1_twiddles,
			round_2_twiddles,
		}
	}
}

impl NttExecutor<ByteSlicedAES4x32x8b> for FastNttByteSliced {
	#[inline]
	fn forward(&self, data: &mut [ByteSlicedAES4x32x8b; 3]) {
		let mut forward_round_simd = |twiddles: &[PackedAESBinaryField32x8b; 3], log_block_size| {
			for (data, twiddles) in data.iter_mut().zip(twiddles.iter()) {
				for i in (0..4).step_by(2) {
					(data.data_mut()[i], data.data_mut()[i + 1]) = transform_forward_round_pair(
						*twiddles,
						data.data()[i],
						data.data()[i + 1],
						log_block_size,
					);
				}
			}
		};

		// round 2
		forward_round_simd(&self.round_2_twiddles, 2);

		// round 1
		forward_round_simd(&self.round_1_twiddles, 1);

		// round 0
		forward_round_simd(&self.round_0_twiddles, 0);
	}

	#[inline]
	fn inverse(&self, data: &mut [ByteSlicedAES4x32x8b; 3]) {
		let mut inverse_round = |twiddles: &[PackedAESBinaryField32x8b; 3], log_block_size| {
			for (data, twiddles) in data.iter_mut().zip(twiddles.iter()) {
				for i in (0..4).step_by(2) {
					(data.data_mut()[i], data.data_mut()[i + 1]) = transform_inverse_round_pair(
						*twiddles,
						data.data()[i],
						data.data()[i + 1],
						log_block_size,
					);
				}
			}
		};

		// round 0
		inverse_round(&self.round_0_twiddles, 0);

		// round 1
		inverse_round(&self.round_1_twiddles, 1);

		// round 2
		inverse_round(&self.round_2_twiddles, 2);
	}
}

#[derive(Clone)]
pub struct VisionHasherDigestByteSliced {
	// The hashed state
	state: [ByteSlicedAES32x32b; 3],
	buffer: [[u8; RATE_AS_U8]; 4],
	filled_bytes: usize,
}

impl Default for VisionHasherDigestByteSliced {
	fn default() -> Self {
		Self {
			state: [ByteSlicedAES32x32b::zero(); 3],
			buffer: [[0; RATE_AS_U8]; 4],
			filled_bytes: 0,
		}
	}
}

impl VisionHasherDigestByteSliced {
	fn permute(state: &mut [ByteSlicedAES32x32b; 3], data: [&[u8; RATE_AS_U8]; 4]) {
		for row in data.iter() {
			debug_assert_eq!(row.len(), RATE_AS_U8);
		}

		for state_element_index in 0..2 {
			let data_offset = state_element_index * 32;
			let byte_sliced_8b_data = array::from_fn(|row| {
				let canonical_value_8b = PackedBinaryField32x8b::from_fn(|col| {
					let index = row + col * 4;
					BinaryField8b::from(data[index / 32][data_offset + index % 32])
				});
				let canonical_value_32b = PackedBinaryField8x32b::cast_ext(canonical_value_8b);

				TRANS_CANONICAL_TO_AES.transform(&canonical_value_32b)
			});

			state[state_element_index] = ByteSlicedAES32x32b::from_raw_data(byte_sliced_8b_data);
		}

		PERMUTATION_BYTE_SLICED.permute_mut(state);
	}

	fn finalize(&mut self, out: &mut [digest::Output<VisionHasherDigest>; 4]) {
		if self.filled_bytes > 0 {
			for row in 0..4 {
				self.buffer[row][self.filled_bytes..].fill(0);
			}

			Self::permute(&mut self.state, array::from_fn(|i| &self.buffer[i]));
			self.filled_bytes = 0;
		}

		let byte_sliced_8b_canonical: [PackedBinaryField32x8b; 4] = self.state[0]
			.data()
			.map(|x| TRANS_AES_TO_CANONICAL.transform(&x));
		for (row_i, row) in byte_sliced_8b_canonical.into_iter().enumerate() {
			for (col, value) in row.into_iter().enumerate() {
				let index = row_i + col * 4;
				out[index / 32][index % 32] = value.to_underlier();
			}
		}
	}
}

impl MultiDigest<4> for VisionHasherDigestByteSliced {
	type Digest = VisionHasherDigest;

	fn update(&mut self, data: [&[u8]; 4]) {
		for row in 1..4 {
			debug_assert_eq!(data[row].len(), data[0].len());
		}

		let mut offset = if self.filled_bytes > 0 {
			let to_copy = std::cmp::min(data[0].len(), RATE_AS_U8 - self.filled_bytes);
			for row in 0..4 {
				self.buffer[row][self.filled_bytes..self.filled_bytes + to_copy]
					.copy_from_slice(&data[row][..to_copy]);
			}

			self.filled_bytes += to_copy;

			if self.filled_bytes == RATE_AS_U8 {
				Self::permute(&mut self.state, array::from_fn(|i| &self.buffer[i]));
				self.filled_bytes = 0;
			}

			to_copy
		} else {
			0
		};

		while offset + RATE_AS_U8 <= data[0].len() {
			let chunk = array::from_fn(|i| {
				(&data[i][offset..offset + RATE_AS_U8])
					.try_into()
					.expect("array is 32 bytes")
			});
			Self::permute(&mut self.state, chunk);
			offset += RATE_AS_U8;
		}

		if offset < data[0].len() {
			for row in 0..4 {
				self.buffer[row][..data[row].len() - offset].copy_from_slice(&data[row][offset..]);
			}

			self.filled_bytes = data[0].len() - offset;
		}
	}

	fn finalize_into(mut self, out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 4]) {
		let out = unsafe { slice_assume_init_mut(out) }
			.try_into()
			.expect("array is 4 elements");
		self.finalize(out);
	}

	fn finalize_into_reset(&mut self, out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 4]) {
		let out = unsafe { slice_assume_init_mut(out) }
			.try_into()
			.expect("array is 4 elements");
		self.finalize(out);
		self.reset();
	}

	fn reset(&mut self) {
		for v in self.state.iter_mut() {
			*v = ByteSlicedAES32x32b::zero();
		}
		self.filled_bytes = 0;
	}

	fn digest(data: [&[u8]; 4], out: &mut [MaybeUninit<digest::Output<Self::Digest>>; 4]) {
		let mut digest = Self::default();
		digest.update(data);
		digest.finalize_into(out);
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{packed::set_packed_slice, BinaryField32b, PackedExtension};
	use digest::Digest;
	use hex_literal::hex;

	use super::*;

	fn mds_transform(data: &mut [PackedAESBinaryField32x8b; 3]) {
		let vision = Vision32MDSTransform::new(&*FORWARD_FAST_TRANSFORM, &*INVERSE_FAST_TRANSFORM);
		vision.transform(data);
	}

	#[inline]
	fn from_u32_to_packed_256(elements: &[AESTowerField32b; 8]) -> PackedAESBinaryField8x32b {
		PackedAESBinaryField8x32b::from_fn(|i| elements[i])
	}

	#[inline]
	fn from_u32_to_packed_768(elements: &[AESTowerField32b; 24]) -> [PackedAESBinaryField8x32b; 3] {
		[
			from_u32_to_packed_256(elements[0..8].try_into().unwrap()),
			from_u32_to_packed_256(elements[8..16].try_into().unwrap()),
			from_u32_to_packed_256(elements[16..24].try_into().unwrap()),
		]
	}

	#[test]
	fn test_mds_matrix() {
		let mds: [[u32; 24]; 24] = [
			[
				0xad, 0x3b, 0xd4, 0x25, 0xab, 0x37, 0xd7, 0x2d, 0x9a, 0x4d, 0x6a, 0xd8, 0x90, 0x44,
				0x6b, 0xdb, 0x06, 0x0f, 0x0e, 0x04, 0x0d, 0x0c, 0x0a, 0x09,
			],
			[
				0x3b, 0xad, 0x25, 0xd4, 0x37, 0xab, 0x2d, 0xd7, 0x4d, 0x9a, 0xd8, 0x6a, 0x44, 0x90,
				0xdb, 0x6b, 0x0f, 0x06, 0x04, 0x0e, 0x0c, 0x0d, 0x09, 0x0a,
			],
			[
				0xd4, 0x25, 0xad, 0x3b, 0xd7, 0x2d, 0xab, 0x37, 0x6a, 0xd8, 0x9a, 0x4d, 0x6b, 0xdb,
				0x90, 0x44, 0x0e, 0x04, 0x06, 0x0f, 0x0a, 0x09, 0x0d, 0x0c,
			],
			[
				0x25, 0xd4, 0x3b, 0xad, 0x2d, 0xd7, 0x37, 0xab, 0xd8, 0x6a, 0x4d, 0x9a, 0xdb, 0x6b,
				0x44, 0x90, 0x04, 0x0e, 0x0f, 0x06, 0x09, 0x0a, 0x0c, 0x0d,
			],
			[
				0xab, 0x37, 0xd7, 0x2d, 0xad, 0x3b, 0xd4, 0x25, 0x90, 0x44, 0x6b, 0xdb, 0x9a, 0x4d,
				0x6a, 0xd8, 0x0d, 0x0c, 0x0a, 0x09, 0x06, 0x0f, 0x0e, 0x04,
			],
			[
				0x37, 0xab, 0x2d, 0xd7, 0x3b, 0xad, 0x25, 0xd4, 0x44, 0x90, 0xdb, 0x6b, 0x4d, 0x9a,
				0xd8, 0x6a, 0x0c, 0x0d, 0x09, 0x0a, 0x0f, 0x06, 0x04, 0x0e,
			],
			[
				0xd7, 0x2d, 0xab, 0x37, 0xd4, 0x25, 0xad, 0x3b, 0x6b, 0xdb, 0x90, 0x44, 0x6a, 0xd8,
				0x9a, 0x4d, 0x0a, 0x09, 0x0d, 0x0c, 0x0e, 0x04, 0x06, 0x0f,
			],
			[
				0x2d, 0xd7, 0x37, 0xab, 0x25, 0xd4, 0x3b, 0xad, 0xdb, 0x6b, 0x44, 0x90, 0xd8, 0x6a,
				0x4d, 0x9a, 0x09, 0x0a, 0x0c, 0x0d, 0x04, 0x0e, 0x0f, 0x06,
			],
			[
				0xa9, 0x0f, 0x7d, 0x24, 0x23, 0x14, 0x45, 0xed, 0x54, 0xdf, 0x62, 0xc0, 0x67, 0xf8,
				0x22, 0xf7, 0xd5, 0x47, 0x06, 0xf2, 0x93, 0x83, 0x8b, 0xff,
			],
			[
				0x0f, 0xa9, 0x24, 0x7d, 0x14, 0x23, 0xed, 0x45, 0xdf, 0x54, 0xc0, 0x62, 0xf8, 0x67,
				0xf7, 0x22, 0x47, 0xd5, 0xf2, 0x06, 0x83, 0x93, 0xff, 0x8b,
			],
			[
				0x7d, 0x24, 0xa9, 0x0f, 0x45, 0xed, 0x23, 0x14, 0x62, 0xc0, 0x54, 0xdf, 0x22, 0xf7,
				0x67, 0xf8, 0x06, 0xf2, 0xd5, 0x47, 0x8b, 0xff, 0x93, 0x83,
			],
			[
				0x24, 0x7d, 0x0f, 0xa9, 0xed, 0x45, 0x14, 0x23, 0xc0, 0x62, 0xdf, 0x54, 0xf7, 0x22,
				0xf8, 0x67, 0xf2, 0x06, 0x47, 0xd5, 0xff, 0x8b, 0x83, 0x93,
			],
			[
				0x23, 0x14, 0x45, 0xed, 0xa9, 0x0f, 0x7d, 0x24, 0x67, 0xf8, 0x22, 0xf7, 0x54, 0xdf,
				0x62, 0xc0, 0x93, 0x83, 0x8b, 0xff, 0xd5, 0x47, 0x06, 0xf2,
			],
			[
				0x14, 0x23, 0xed, 0x45, 0x0f, 0xa9, 0x24, 0x7d, 0xf8, 0x67, 0xf7, 0x22, 0xdf, 0x54,
				0xc0, 0x62, 0x83, 0x93, 0xff, 0x8b, 0x47, 0xd5, 0xf2, 0x06,
			],
			[
				0x45, 0xed, 0x23, 0x14, 0x7d, 0x24, 0xa9, 0x0f, 0x22, 0xf7, 0x67, 0xf8, 0x62, 0xc0,
				0x54, 0xdf, 0x8b, 0xff, 0x93, 0x83, 0x06, 0xf2, 0xd5, 0x47,
			],
			[
				0xed, 0x45, 0x14, 0x23, 0x24, 0x7d, 0x0f, 0xa9, 0xf7, 0x22, 0xf8, 0x67, 0xc0, 0x62,
				0xdf, 0x54, 0xff, 0x8b, 0x83, 0x93, 0xf2, 0x06, 0x47, 0xd5,
			],
			[
				0xaf, 0x0f, 0x78, 0x2c, 0x2b, 0x10, 0x4c, 0xe2, 0x59, 0xdc, 0x63, 0xc7, 0x66, 0xf3,
				0x2a, 0xfc, 0x99, 0x8d, 0x85, 0xf4, 0xd6, 0x4e, 0x06, 0xf9,
			],
			[
				0x0f, 0xaf, 0x2c, 0x78, 0x10, 0x2b, 0xe2, 0x4c, 0xdc, 0x59, 0xc7, 0x63, 0xf3, 0x66,
				0xfc, 0x2a, 0x8d, 0x99, 0xf4, 0x85, 0x4e, 0xd6, 0xf9, 0x06,
			],
			[
				0x78, 0x2c, 0xaf, 0x0f, 0x4c, 0xe2, 0x2b, 0x10, 0x63, 0xc7, 0x59, 0xdc, 0x2a, 0xfc,
				0x66, 0xf3, 0x85, 0xf4, 0x99, 0x8d, 0x06, 0xf9, 0xd6, 0x4e,
			],
			[
				0x2c, 0x78, 0x0f, 0xaf, 0xe2, 0x4c, 0x10, 0x2b, 0xc7, 0x63, 0xdc, 0x59, 0xfc, 0x2a,
				0xf3, 0x66, 0xf4, 0x85, 0x8d, 0x99, 0xf9, 0x06, 0x4e, 0xd6,
			],
			[
				0x2b, 0x10, 0x4c, 0xe2, 0xaf, 0x0f, 0x78, 0x2c, 0x66, 0xf3, 0x2a, 0xfc, 0x59, 0xdc,
				0x63, 0xc7, 0xd6, 0x4e, 0x06, 0xf9, 0x99, 0x8d, 0x85, 0xf4,
			],
			[
				0x10, 0x2b, 0xe2, 0x4c, 0x0f, 0xaf, 0x2c, 0x78, 0xf3, 0x66, 0xfc, 0x2a, 0xdc, 0x59,
				0xc7, 0x63, 0x4e, 0xd6, 0xf9, 0x06, 0x8d, 0x99, 0xf4, 0x85,
			],
			[
				0x4c, 0xe2, 0x2b, 0x10, 0x78, 0x2c, 0xaf, 0x0f, 0x2a, 0xfc, 0x66, 0xf3, 0x63, 0xc7,
				0x59, 0xdc, 0x06, 0xf9, 0xd6, 0x4e, 0x85, 0xf4, 0x99, 0x8d,
			],
			[
				0xe2, 0x4c, 0x10, 0x2b, 0x2c, 0x78, 0x0f, 0xaf, 0xfc, 0x2a, 0xf3, 0x66, 0xc7, 0x63,
				0xdc, 0x59, 0xf9, 0x06, 0x4e, 0xd6, 0xf4, 0x85, 0x8d, 0x99,
			],
		];

		let mds_trans: Vec<Vec<AESTowerField32b>> = (0..mds[0].len())
			.map(|j| {
				(0..mds.len())
					.map(|i| AESTowerField32b::from(BinaryField32b::new(mds[i][j])))
					.collect()
			})
			.collect();

		for (i, col) in mds_trans.iter().enumerate() {
			// Transform the data 0, ..., 0,  1, 0, ..., 0 where the ith position is the unit element through the mds and check that you get ith column as the output.
			let mut data = [PackedAESBinaryField8x32b::zero(); 3];
			set_packed_slice(&mut data, i, AESTowerField32b::one());

			mds_transform(
				PackedAESBinaryField8x32b::cast_bases_mut(&mut data)
					.try_into()
					.unwrap(),
			);

			let actual = from_u32_to_packed_768(&col[0..24].try_into().unwrap());
			assert_eq!(data, actual);
		}
	}

	#[test]
	fn test_sboxes() {
		#[rustfmt::skip]
		let inputs: [u32; 1024] = [0x39037a7d, 0xe5052ed8, 0xc5dc5c6e, 0x6ddccbe1, 0xa13aed6c, 0x23839b39, 0x37f0a862, 0x63e92751, 0xc0016042, 0x6c6c241d, 0x8f283c12, 0x94a7dc55, 0x2dbdac59, 0x5ec675b1, 0xa4eb5ff7, 0xeb813ae4, 0x99114764, 0x980d276a, 0x9d7b14d9, 0x21decbc9, 0x7699ccf0, 0xd94b143b, 0x5bb8e05a, 0xb1a748cd, 0xf5a4d1a7, 0x4e9e983c, 0x3c2cc2f7, 0x4cfba7c9, 0x183008af, 0xd3ce2698, 0x34b2311a, 0x80b830f2, 0x936b8400, 0x20064868, 0xd0de8b3b, 0xc6797364, 0x9742e9b7, 0xdec03b48, 0x2890832c, 0x2b6fddbc, 0xedf7ca88, 0x693c836f, 0x8b09a182, 0x9a710689, 0x1fe0d4b5, 0xaba30b59, 0x7385f29a, 0x6bdc298b, 0xd4c997d9, 0x8528cc21, 0xec015e83, 0x831a6866, 0x5edebb49, 0x6789b582, 0x933c270b, 0x69832c88, 0x0ecae417, 0xc03673a1, 0x0b758bdb, 0x8b19e50c, 0x6d96482c, 0x65fd55c6, 0x6cd353f3, 0xd033e74b, 0x93a24a77, 0x1c4b3b11, 0x7462e87d, 0x516e1b4d, 0x5de046c2, 0x2cc07e42, 0xae48ecbd, 0xe974dd8e, 0xa9ad92e3, 0xa3cdda5c, 0xe8c71cb2, 0x2a001779, 0xe0749451, 0x52b45b2b, 0x120afd1c, 0xaac7528d, 0x57e434b0, 0x68ce75f3, 0x5504e012, 0xd2b39ea9, 0xdbf76c6d, 0xa3f2b4fa, 0x00aecd24, 0x2bb5ddad, 0x3f6cbbc1, 0xafe20693, 0xc0d19750, 0x670c2518, 0x0c6bf892, 0x8293431a, 0x25c1609c, 0x92dd44d2, 0x14f3a5a7, 0x45206681, 0x085fc99d, 0x8117ad3c, 0xff54ab8e, 0xbe7acfa4, 0xd624b154, 0xedd9642d, 0xa90d2549, 0xadeb767c, 0x2f4fc57a, 0x5f34c848, 0x63a002b4, 0xac332a4f, 0x7bb273ef, 0x8ce1281d, 0x609933aa, 0x8b98f6eb, 0x96e78c2c, 0xd7b848e7, 0x6d3652ff, 0xf5cfd797, 0xc5cba7d8, 0xea67fd7e, 0x5215ffee, 0x11751b5d, 0xaa719d71, 0xbde22416, 0x6eb13bd2, 0x9fef2ae6, 0x974871e6, 0x93807a95, 0xcde54c79, 0x620d049a, 0x2d5a66b0, 0x53626810, 0xbc90d92e, 0x480ac66b, 0xcdf5ae76, 0xbf86a197, 0xf9a393ad, 0xf43db1b3, 0x06bf9dcc, 0x89547bca, 0x17e1412a, 0x26d038fc, 0xad20aebe, 0xbd59af8f, 0xded89cd4, 0x558e3e64, 0xe4f84719, 0x1965a18e, 0x3911c85d, 0x31401cd8, 0xe0f8942c, 0x99664274, 0xd98a796f, 0x7c1174b2, 0x2e128ebb, 0x9955d47d, 0x77b4d2ec, 0x52b8402b, 0x8f8f713e, 0xfb1fb394, 0xa656f4b2, 0x98de662b, 0xa9dd8d41, 0xa7bb295b, 0x8d171bec, 0x729e3679, 0x607973fd, 0xec0d474a, 0x0424e887, 0x4a167f7b, 0xb1c75beb, 0x38947660, 0xcd319349, 0x9efd76cb, 0xdb86739a, 0xe86c105c, 0xdc7c1125, 0xa96c6fbb, 0x69784d5b, 0x562c70fe, 0x6b914e1d, 0xbe91abb3, 0x66e59a3e, 0x14e2938f, 0x2e5a178b, 0x837f80c1, 0x93a669f8, 0xea14642d, 0xd110c9b5, 0x8078f1df, 0x355d1f7b, 0x7277c981, 0x4576670b, 0xaf1bbb11, 0xb0799e99, 0xe35830be, 0x1a876e18, 0xd81e2aa7, 0x38b6592c, 0xbadd1d99, 0x3b6dcb1d, 0x2d7d9306, 0x290dc93f, 0x95361da6, 0x42f4b86a, 0x22200774, 0x437d70e6, 0xa4cfdd75, 0x83996528, 0xd8eab027, 0x4086e8f9, 0xfbd779ad, 0x39081a78, 0x57c9f06c, 0xc40db0dc, 0xd700e60a, 0x5e953d61, 0xe185e227, 0x34dc5581, 0xf152ae92, 0x46ebd88e, 0xd7888757, 0x3aef4102, 0x999ef037, 0x6d913aa6, 0x0174c1e5, 0x0eea1cf3, 0x05229210, 0x9c9ff815, 0x8a039b9e, 0x2d367b37, 0x0b053a66, 0x87f79e85, 0xc9e4be8b, 0x7aeb0fae, 0x2556809d, 0x7e545a69, 0x7784428f, 0xc5270688, 0xcea9777b, 0x39e55c41, 0x2736476e, 0x93ddab18, 0x8c57ca77, 0xb1906a2c, 0x73285f35, 0x98b3724b, 0x2dafc659, 0x44d7377d, 0xde1fcb7e, 0xc122e45f, 0x7fec9659, 0x57a2f246, 0x0a4d768f, 0x918daa58, 0x3bc15ab9, 0xfd40190c, 0x6bb978fd, 0x23e1183e, 0xfc8f5570, 0x9a446d30, 0xfc037192, 0x693ac0b9, 0xbd519f78, 0x551965cf, 0x6f8e75bc, 0x6d2ab3ec, 0xf66ba2d8, 0xebbe7632, 0x4f5a3eb6, 0xfaaa5ee9, 0x0712a268, 0x9b0876d5, 0x9ac1f0b4, 0xb3648e7c, 0xd7e0d053, 0xff7bf33a, 0x52f5f38c, 0x8b739826, 0x4572e831, 0xedd11b45, 0x343fdd5b, 0x4c36500e, 0x501c4216, 0x09000771, 0xd476866f, 0xac577d71, 0x814ac573, 0x834c26b5, 0xc9a61a38, 0x0096d101, 0x39818642, 0x83deba04, 0x3cbe7925, 0xb3726e43, 0xfcc84179, 0x5d7b69da, 0xcfc3d031, 0x22a03965, 0x5ddd7e45, 0xa5a45176, 0x439e6def, 0xe71cd685, 0xcf5177d1, 0x29646d16, 0x4fc87adc, 0xca335d82, 0x05be8860, 0xf98e7c45, 0x49e9496e, 0x11f722cb, 0x12f98f22, 0xfacba65f, 0x6d8ecee0, 0xf537a952, 0x9f1c962d, 0x1b9ad93a, 0x186e72ea, 0xfca8cfdc, 0x133ba7a0, 0x0f462939, 0xcc1cda41, 0xbba42a09, 0xe495ec10, 0xe70eefb0, 0x91020a58, 0x756c1764, 0x9c7c8086, 0x1a8e1f8e, 0x0edcc374, 0xe2e94582, 0x2f179e03, 0xd40ffcac, 0x1b8dfe48, 0x456d9574, 0xb3639882, 0x83343c08, 0xdd90007e, 0xc2e1bd19, 0xfa1f7448, 0xcf8b49aa, 0x0e3ace31, 0xa062ab8d, 0x0a8e3148, 0x266fb42f, 0x6464e751, 0xf127e72d, 0x4a9e6c83, 0xe6f7bf8b, 0x1a068eaa, 0x2a475a79, 0x4005212e, 0xe636dc7b, 0x75ee534a, 0x95620d1e, 0x8efabc46, 0xa9d11658, 0xc23dd348, 0x8fe00fa6, 0x3ace5c9b, 0xb2bbb452, 0x154be7b2, 0x734b98dd, 0x169bc03e, 0xc586230e, 0xffdd1482, 0xdd42b61d, 0x16389688, 0x5697db5b, 0x9ae0cf8b, 0x4ead0296, 0x42787d45, 0x95e7f403, 0x4ff429bd, 0x839011a6, 0x1c9abc39, 0xac729b56, 0x0cdfed18, 0xbf930921, 0xa0827be0, 0x85ccf44c, 0x470d7619, 0x376a3a9c, 0xe714ce20, 0x76bf317c, 0x87a05ddd, 0xe3cfe129, 0x412a7386, 0xee93c4a4, 0x1f234824, 0x5533a111, 0x1475c6d7, 0xbafae127, 0xe8c2ab4c, 0x664152b4, 0x998f6975, 0x6d56fd32, 0x145c9379, 0x010b901b, 0x72bafd1b, 0xbc9097b8, 0x5c5a76e1, 0xeedacb03, 0xebd5cc6a, 0x7841d4e9, 0x32c34eb1, 0x974f43a4, 0xcb521cc1, 0x54b3842b, 0xe571bfe5, 0xbb02824a, 0x4bd1bae1, 0xa71352f3, 0x8bea817b, 0xa12730ce, 0x262b78e9, 0x3445ec78, 0xec59817c, 0x09816978, 0x99eb427e, 0x2a868435, 0xbc8b7d53, 0x4921853d, 0x0dda6044, 0x00cedde6, 0x012addf6, 0xa46eac1c, 0x605313e2, 0x43edcfba, 0xca892901, 0x40f023c4, 0x1042cb85, 0x7d43aff5, 0x8ac1da4b, 0xfe0bf877, 0xd6332044, 0xf0aa7699, 0xb5e9246e, 0x8c1853be, 0x2b2cfa1e, 0xd2f6b44f, 0x8a9db954, 0x30614b29, 0x4b97de5c, 0x200cbb92, 0xb7c63b7b, 0xb99f7570, 0x36756537, 0x64107fa8, 0x17393981, 0x0ef2665d, 0xdcb4f941, 0xe33354f1, 0x900218cd, 0x4cbe52f7, 0x306d55fa, 0xcf2915e6, 0x405961b7, 0xca29edb6, 0xe4ee2945, 0xa14fb1fa, 0x9c710e57, 0x781095d5, 0x0e4a83b7, 0xcaef8be9, 0x4a012c97, 0x2ca90ec2, 0x15338978, 0x6b8aa3e4, 0x33463119, 0x793032c8, 0xd7efb9bb, 0x4bac456c, 0x7ffd4b23, 0x7d26ccf4, 0x82adcf4d, 0x63b27946, 0xc87c494f, 0x968d0562, 0x57b27ef1, 0x00d4e3f5, 0x0a18a211, 0x8add0aa6, 0xe66a813a, 0xca5fa4ef, 0x1c5bd347, 0x51863c19, 0x21b2dd26, 0x852cc072, 0x489ae5f3, 0x96355831, 0x60a9d838, 0xce8ac052, 0xd8cd1979, 0x30ff389f, 0x1b76b98a, 0x1f5a727a, 0xdfffc6f4, 0x8c51d2b6, 0xb227bf1a, 0xf2b31961, 0x543d994a, 0x47d03bb7, 0xbfc8762b, 0x8a8f8438, 0x75ed2980, 0x89fad59f, 0xbe1468c4, 0xe10eda19, 0x8bc3a9a2, 0x41aa3433, 0x9838abd7, 0x7fbd405a, 0xf8abe25d, 0x67711bbd, 0xcf54f8e0, 0x9ea0cc4d, 0xb5639ddf, 0xd1348713, 0x10af3aba, 0x979b1a01, 0xd0c5fcaf, 0x63851491, 0xd3d3e79a, 0x07b131cb, 0x9c50b0bd, 0xd14b7c54, 0xe61971e6, 0xb9b35030, 0x36e01478, 0x566fd388, 0xf15adf47, 0x5cddee5a, 0x81c5952b, 0xe2fdb38c, 0xde0059f3, 0x3f907ea7, 0xa1045cc5, 0x602eb2c8, 0x261e957b, 0x66b62824, 0x600a8ad4, 0xae197f39, 0x0f7c2296, 0xe0098fbc, 0xb68a1ee0, 0xc2d9bd04, 0x9c31f757, 0x94129f57, 0xaf742e47, 0x454559eb, 0x1386313a, 0x9c114feb, 0x2a6a71ee, 0xbaf78a98, 0x67a38e05, 0x576b3d22, 0x68b9a1bf, 0x9098e2f7, 0x3bdefce1, 0xe27a9f51, 0x56578a3f, 0xf3c94cf3, 0xd2252730, 0xd125a37a, 0x9f24312a, 0x1786342d, 0x333dbdb0, 0xe4ee3c5c, 0x609a2e71, 0xec05abf9, 0xcaa6e94f, 0x8a0bf3b2, 0x05652fa3, 0xa4d56988, 0x26f1c1f3, 0x3bee32ca, 0xb4f9d5f5, 0xd9d31925, 0x49702eeb, 0x20de91dc, 0x3250f088, 0x9d622c4f, 0xb182d41c, 0x77f39852, 0x856e9afe, 0x1d1e29ee, 0x40bb95da, 0x0b4c4f6a, 0x2bced851, 0xedba4f06, 0x57a23ed2, 0x01c64c32, 0xa2520644, 0xd96666b1, 0x9dfac14f, 0x9798f596, 0x2d96cd5e, 0x6eaa4b24, 0x3010615b, 0x8a930b1c, 0x9a23b594, 0xf54da3c6, 0x2a57c51e, 0x6bd15ec1, 0x125c1587, 0xd0419a52, 0xf1e4142c, 0x2e73b23c, 0x80613a38, 0xe7f6a877, 0xd9875be3, 0x6133c9c1, 0x2b4093d6, 0xafa73df6, 0xcdefa94e, 0x4ad070b1, 0xacc03156, 0x60fed42c, 0xb2865585, 0xf6839515, 0x7468d8e2, 0x60a0376e, 0x50cdb043, 0xd6fa6c5c, 0x747c0ac3, 0xd71199ba, 0x9cf5b5b0, 0x7017e05e, 0x5e7fd065, 0xb0430d9a, 0x461963de, 0x3cbe132d, 0xaee53622, 0x0c2159ea, 0x2d00045c, 0x3d1d6db7, 0x8251f371, 0x7a183791, 0xd18bb7c5, 0x51639a64, 0x39e41bb4, 0x00017d3c, 0x757c411d, 0x116c1745, 0x16c7edba, 0xf006161e, 0x511fdd34, 0x109d8212, 0x7c389385, 0x51fd65bf, 0x25230e45, 0x7215f0cc, 0xcf6f700e, 0x05ade2ba, 0x0b84394d, 0xbe95f93b, 0x953c0831, 0xe812686c, 0xc39e0af8, 0x5308030c, 0x70fe0a07, 0xa86dada0, 0x0f5b3996, 0x19c2debd, 0x27b4e8f5, 0x78015b9f, 0xb627b59d, 0xa7e86e91, 0x58c59afd, 0xf4e36076, 0xf1b79300, 0x453009b3, 0x67e7b0ef, 0x3652696e, 0xf63fe595, 0x9f4000e6, 0xc2caf0f8, 0x435aa887, 0x22508b72, 0x2c27aa5b, 0x3d6fcb80, 0xe43e26c2, 0xb9fd84fe, 0xe199e3db, 0xa1bb42f1, 0x08815f74, 0x1cf67952, 0x71115e87, 0xe688b181, 0x963ae7a0, 0xd773c8d3, 0x40a29d4f, 0x9ae42906, 0x2c427544, 0x66465cd3, 0xc26e288d, 0xbe2445c5, 0x025e0b22, 0x6abd0a0e, 0xf6c42f2e, 0xdeecfd49, 0x585148ec, 0xfc503e5a, 0x742ef4f2, 0xf7fcc61e, 0xf750a28b, 0xc31e8f06, 0x4c0afc16, 0x3969210c, 0x8c882022, 0x711201f8, 0x03e76f94, 0xbd5858d5, 0xfbc5345f, 0xc50498f5, 0x1aa9f869, 0xcc695619, 0x6cce5253, 0xb5a0df5c, 0x596552b7, 0x3d8d6bcf, 0x3243093b, 0x38840560, 0x31a954b5, 0x3d46a9dd, 0x901ab9b1, 0xfbe00979, 0xa7134a82, 0xc1edf52c, 0xc8de9025, 0x0dcf688f, 0x35b31cbf, 0x49e1b3d8, 0xb1e052a2, 0x2f711d40, 0xa1b93d33, 0xe45e8cc4, 0x8beef1e6, 0xedc02c9c, 0x6850df58, 0x064ce027, 0x3b21aa87, 0x04c867a2, 0x4bd36df3, 0x9b9ac247, 0x34956f83, 0xa1869a39, 0x543aeb17, 0x52a3ea8b, 0x011cc153, 0xcb59fd71, 0xea480777, 0xcc9ae11a, 0xe79daaaa, 0x3c830af9, 0xf7daa30b, 0x3d70b243, 0x5ce97bd9, 0x2f10a042, 0xaccc7537, 0x6b586714, 0x6fd51c89, 0x2a0ced29, 0xfff3d695, 0xb0746cec, 0xcf21ae51, 0xb8b81296, 0x6016fe90, 0x69b18d49, 0xda1c44ac, 0xd53a90f5, 0xed1a2040, 0x2bcbd5c9, 0x5a03dd8d, 0x0e5cff02, 0x3c13d2c5, 0xcd2681e5, 0xad833ae6, 0x97e94ad5, 0x870f71b5, 0x338d14e8, 0xb2da0932, 0x9a3cdc6d, 0xdad47d63, 0x458c9ce0, 0xbea650a5, 0x83c4c663, 0xfaef81fe, 0xf287a191, 0x35299097, 0x7e749de9, 0xdbc86b6e, 0x7381f261, 0xfd7c8342, 0xb0d00d58, 0xf86e28bf, 0xc3952b29, 0xd25c1edb, 0x80e63bdd, 0xacf35f36, 0xb4602767, 0x047d53e3, 0x051a8b4b, 0x935bee9f, 0x35777746, 0x7b53b1b1, 0x9ffb4012, 0x119df8bd, 0x66375052, 0xc5a2ce40, 0x06f273e3, 0x47dd47bb, 0x6b2e613d, 0xe3aac7d9, 0x12b5b2f6, 0xe28e37aa, 0x87505876, 0xfa0c098d, 0xbfac0efd, 0x98ef2ebb, 0x082acc6b, 0x004ae37d, 0x75675e4a, 0xda81ddbe, 0x8419cd36, 0x55fea9dc, 0xe9a6527a, 0x730cf521, 0x1aff0476, 0x075ef0b6, 0x7feb5de1, 0xaf9c5cb9, 0x7584fb4f, 0x980c2cb3, 0xf7b382d1, 0x12c889dd, 0x7db51bc8, 0xe7d0985a, 0x5e42627f, 0x7f66af35, 0x9f39e543, 0xbf97cd33, 0xce12f81b, 0xdf0179bc, 0x8e370402, 0xe1f53e71, 0x4ea48e34, 0x44ded813, 0x65d89c9d, 0xfff08620, 0x16dc2a05, 0x6705425b, 0x57569d16, 0x3f83ddb7, 0xc4424ef7, 0x4c447b5b, 0x852c1802, 0x47bf821a, 0xbbb96d12, 0x96e39b5d, 0x1d1a9604, 0x0668b8ae, 0x9d6435fc, 0xa9af0356, 0xdeb5808d, 0x715a33c0, 0x90bb9936, 0x6f057233, 0x6a062107, 0xefc218e2, 0xdd187235, 0x42c1cfce, 0xfaed8e26, 0x30243c62, 0xae1a62ad, 0x4d94f5a9, 0x9e4dbd8e, 0x73c38199, 0xa12f1765, 0xd860354d, 0x1e2c5db9, 0xa1059204, 0x0585ba6f, 0xe8db9481, 0x2fcffc69, 0xfe8801e2, 0x30bca3e5, 0x56f6016a, 0x872bdba2, 0x89848700, 0xdb3d0865, 0xb59382cc, 0xa90bba75, 0xf4a048a3, 0x42b68f44, 0x2d0bb8c5, 0x1ccfd726, 0x1b8846d3, 0xdd9125a0, 0x67b92f20, 0x8faa73a4, 0x4eba1b0b, 0xb5b4b603, 0xc3600db8, 0x6e1e37de, 0x63c54116, 0x3224164a, 0x1e2f1320, 0x811358f2, 0xd5f0852d, 0xc7e125bf, 0x03fe4a08, 0x7973629a, 0xb9c349c7, 0x7c380c74, 0xe00f8ba5, 0xe6e90a52, 0xd3c21353, 0x4d2bd526, 0x730a6794, 0x4e3b247c, 0x62fea5e4, 0x4dfb12a7, 0x9ceee6ba, 0xf935c790, 0xc91ca1a4, 0xc6376175, 0x915b93b3, 0xff0c4275, 0x9d7c0356, 0x1d38c624, 0x74170e6e, 0x365e2c13, 0x14e2cf7e, 0xfbe404db, 0x2e7e0ec8, 0x8c791a98, 0x725bf416, 0x24ad7ca8, 0x5e4c9361, 0xfb43a4cc, 0x2d782add, 0xa4272ff7, 0xe283adc9, 0x90bb8277, 0x0a50a233, 0x4d99d538, 0xc11592ac, 0x653de172, 0x2429c08e, 0x9b7d1bd6, 0x5fef149f, 0xcd705c77, 0x692d43e0, 0x16501e8c, 0xfde6dbf2, 0xba116741, 0x7f0d6fdb, 0x5495ce9d, 0x8e65dd34, 0xb39adcac, 0x23127fb6, 0x6871772c, 0x7f0b5447, 0x47cd132d, 0xde6f6142, 0xed679886, 0xf744b341, 0xc59ef4d3, 0xc583b4ab, 0x93da4023, 0xd16bbc2e, 0x391efc95, 0x934ab294, 0xb8730a37, 0xd0451423, 0x5e6ed32c, 0x9a2d0f99, 0xe6a3935b, 0x8edd51a8, 0x762a8d68, 0x9debd6fe, 0x32106307, 0xe35d4cb8, 0x36bd828b, 0x3b4ef6d0, 0x7841a5aa, 0x9c3e677e, 0x3ccd39a9, 0x0171eff5, 0x997f15c9, 0xeaf5b75c, 0x36fda0b1, 0xa9fa74eb, 0x7a869940, 0xdc41ad15, 0x799c246d, 0xf580a094, 0xd254b27f, 0x1ed8d313, 0x15634c77, 0x4e2263b4, 0x5a90cc3a, 0x305bac9e, 0x3c6081f4, 0x017a4dc3, 0x9229231c, 0xab5a2187, 0x42647839, 0x679776a8, 0xe8b01b33, 0x20b65b58, 0xef1d7eec, 0xf1be4b98, 0x530c880a, 0x47d44c32, 0x211a7909, 0x38fbbb08, 0x6a8a0dcd, 0x99594f29, 0xccd2a60c, 0x63180430, 0x7e397064];
		#[rustfmt::skip]
		let sbox1: [u32; 1024] = [0x330d800b, 0x6ae828d7, 0x5467df64, 0x0811aeee, 0x2806230c, 0xbadf5081, 0xc2d43f3d, 0xc732f979, 0xcebd7499, 0xcd9a68dc, 0x4c89843b, 0x36693a66, 0x2a55160f, 0xc20f3307, 0xfce9ab0c, 0xfd0131fa, 0x00324c43, 0x48a9d441, 0x032aa644, 0xdb285793, 0x640a9b20, 0xbdec3d48, 0xf8c342c2, 0xd009e9ce, 0xfa297015, 0xc2a2aef4, 0x0bb2a931, 0x57768523, 0x97463b93, 0xfdf2710e, 0x043b300f, 0xf55250de, 0x956a44e7, 0x4896dd65, 0xae47974e, 0xd735fb00, 0x10da3bb4, 0xd039221e, 0xdcf64f1a, 0xe8694cc4, 0x513d8aa2, 0x47b381e5, 0x20af002f, 0xdc162d2d, 0xb7036bc5, 0x6166918b, 0x7f6ec958, 0x9f247795, 0x88881d4a, 0x13c81730, 0x51caa181, 0x38c3bc96, 0x1998b64e, 0x5324958f, 0xf25fb9d0, 0x789bdf2b, 0x45d9d09b, 0x2d05e07a, 0xbb98c46f, 0x4f0c1617, 0xe922bcd2, 0x337de3d0, 0x010362f0, 0x15226537, 0xc11b8f7a, 0x5fc3185c, 0x4b8fb970, 0xaf11c7e4, 0x313a8d53, 0x0be73299, 0xa66cb878, 0xb5d23e40, 0x21b5fb92, 0x94a8850c, 0x7922d007, 0xb62b00e6, 0x6a6e00cf, 0x84ade661, 0x944ee214, 0xbf0bf80a, 0xa981cb49, 0xb4ada75b, 0x7798b91b, 0xeca21f68, 0x83590230, 0x76eb04a1, 0xf49d888d, 0xd6634765, 0x30bdb3d6, 0x3f474900, 0x6e611aad, 0x39a77cf2, 0x6219b93c, 0x6a1504f9, 0xc9a554d3, 0x4a00e93d, 0xcf234aff, 0x520129fd, 0x9d13b3a1, 0x72b5bcf7, 0xc4443bd2, 0x89895792, 0x54bdffa7, 0x62363c88, 0x73dc4232, 0xc8539e00, 0xea84d445, 0x5687be1b, 0xf019a132, 0xfd2a09d0, 0xf8fc2066, 0x6c7dcd07, 0x007cae3d, 0xdeeba696, 0x88ba6e82, 0xd0cd5741, 0xdffc61cd, 0x1fecbf28, 0xa16ac407, 0x123a102b, 0x72b63946, 0x1d0bd47f, 0x976f555d, 0x8add7c7f, 0x0896c212, 0xdba7ff96, 0xea1df509, 0x4f4f6a6b, 0x0c0f6630, 0x1e967c19, 0x63d66678, 0x383a81b4, 0xdcf7d51e, 0xfa7bdae8, 0x7f758abb, 0x5d777699, 0x60d16630, 0xa2a4f88c, 0x5fc8ce11, 0xb7722e45, 0xd8537af0, 0x54b1a3a7, 0xfb7f2e68, 0x55dd30aa, 0x3b01785c, 0x2e22e903, 0x8141bde8, 0x553c9b4e, 0xc1c6a1d1, 0x5e9d8e70, 0x533fa523, 0xdbca18d0, 0x3bfd08b4, 0xcf27cd33, 0xf93121f2, 0x500912bc, 0xd1c783db, 0xde09e39c, 0xa48c40a1, 0x6cac9fc4, 0x5282ec7d, 0xfdbf3236, 0x16cf3b06, 0x3568587b, 0xb442dafe, 0x7597eaf0, 0x102cba9f, 0x338c66f0, 0x8a7982f8, 0xf1996ef0, 0x3d357eea, 0xdc01344e, 0xdad1ee7f, 0xe39cd17d, 0x793ef630, 0x7375477c, 0x31902a96, 0x95c54326, 0xab5354c7, 0x5f478a3c, 0x5464f9ae, 0xad263521, 0x8dff9991, 0x27b79d05, 0x5bffb19c, 0x08ac62f8, 0x83311031, 0x56f2e27b, 0x4d9d4e5f, 0xe84709a6, 0x801e6414, 0x0987537d, 0x85b95962, 0x08547fb9, 0x7518c1a6, 0xf2f1537e, 0xde4100f8, 0x3823ef2d, 0x95074c5d, 0xb4e043e3, 0x313f5aa2, 0xbc1a60d1, 0xf27e18f6, 0x82aeca4d, 0x05a3ce07, 0xad6cefba, 0x7997ecb2, 0xc413ea63, 0xe83be5ae, 0x0a1a4161, 0x06a1831c, 0x8a8e22ee, 0xab32ec2b, 0x13c74d2e, 0xac816bc7, 0x6de1fa37, 0x08324317, 0x93cb8c02, 0x6b513ce1, 0x0322dab7, 0x6fe26b31, 0x529110aa, 0x56a3ad69, 0xaf247754, 0x131048dc, 0xd96e43ab, 0x936977f2, 0x0728678f, 0xc0376a41, 0xbac400b5, 0xecb0c118, 0x6de44d3a, 0xdaa12ddd, 0xd42dc217, 0x89fe9a3b, 0xc9fcb2b2, 0x1b868863, 0x28ad282f, 0x41d0b4ef, 0x983fcce5, 0x0f09008a, 0x824f6ff2, 0x5ada175d, 0x6eba37b0, 0x8f557bf4, 0x85bb647d, 0x2d52be53, 0xdbcad457, 0xfbad55e1, 0x3b4bc751, 0x7293e8fe, 0x743894df, 0xad6c477a, 0xfad24c3c, 0x44e6fd72, 0xa24b0246, 0xbf39a83b, 0xc51178f6, 0xaf46c9ad, 0x55cdade1, 0xa0dcf571, 0x63fe860a, 0x6f25da81, 0x0c1e281a, 0x4a475290, 0xfc68cde8, 0xfac7931f, 0xa220255f, 0x42700377, 0x8eca457a, 0x2fd8d31e, 0x43d10e69, 0x861df479, 0x154175a2, 0xbd756701, 0xe32f4ea0, 0x11193e43, 0xc7f8ad50, 0x6c687040, 0xc3bd0e91, 0x006cbd33, 0x2146bd7c, 0x62b211eb, 0xa85e886a, 0x4ef21b06, 0x93a25256, 0x3d7a42c7, 0xe6f6f2ca, 0xc12973c1, 0x1e565059, 0xea63ecb7, 0x3326f75f, 0xd87dfcf0, 0xe425711a, 0xe2baa91e, 0x1ca975e1, 0x811869a2, 0x8cdc0dcc, 0xbaef67c7, 0xef6b9648, 0x134408c8, 0x96e7e826, 0x21fc269e, 0x76e4d6de, 0xd99b1b74, 0xd9f5e4e0, 0x292572f1, 0x81c34f95, 0x88a15984, 0x474cc56b, 0x4f692109, 0x6223b48a, 0x319ebd68, 0xa46b6cab, 0x8fc4cc03, 0x5ac232ae, 0x8e62e890, 0xd2eb4e29, 0xe826d15d, 0xc143e461, 0x471b3c41, 0x5ae0ef30, 0x92f1e57b, 0xfaf6238a, 0x3a3892c0, 0x057eaa80, 0x37f7377b, 0x8b9531d0, 0xf0257e46, 0xfb948b0a, 0x2eee45da, 0xe2a7cb77, 0xdc51f89f, 0xebd05c82, 0xe5f99dd3, 0x1ea9a30d, 0x2e7bb846, 0x2e9db33d, 0xa684cdd0, 0xba7a9196, 0x3a3720cd, 0xf6540219, 0xa6afdfc0, 0xfb2723f2, 0xed73720e, 0x157dd724, 0x0540e617, 0x2f8675f0, 0x890db07f, 0x30630fa0, 0x0dc5b4d9, 0xec827bc3, 0x22804eb0, 0x2961205f, 0x4cd68e58, 0x5429e820, 0x6ec2a04c, 0xe237aea7, 0x46877f9c, 0x542b2701, 0x0c877d0b, 0x5ce2a54c, 0x952c151c, 0x45004a76, 0x69692f02, 0xe8eb029e, 0xbaad6a3c, 0xa5fc2c1a, 0xd36edeb1, 0xb646d9e2, 0xd28f34ae, 0x6a361fc0, 0x189230dd, 0x40b6b7fb, 0x1b53cb29, 0x6d708a20, 0x069f0baf, 0xb8490060, 0x954968ec, 0x99416b5b, 0xb5c881ae, 0xd083cfae, 0x787ece83, 0xc1da12de, 0x4454fbe3, 0x9906d157, 0xc38deefb, 0x57410173, 0xc2b018a4, 0xceb5a316, 0xc224cfd8, 0x216143df, 0x25422921, 0x74b9e476, 0x78bef012, 0x24c55c62, 0x698ea6f0, 0x29aa3802, 0xff0d29c7, 0x6225fb55, 0x160d0223, 0x80c12a9c, 0x4b1af397, 0x2eaa4003, 0x34d0e041, 0x72446010, 0xf45c9d0a, 0x6327beb1, 0xc886660e, 0xdad215cb, 0x5cde8efb, 0xb785f37b, 0xebf25b6c, 0xbddf3f75, 0x689d7f8f, 0x66138c63, 0x13a1859e, 0x1819b419, 0x7c4c462b, 0xe2e86075, 0x02e49786, 0x7f7c733d, 0xd673b864, 0xc75673fa, 0x95ea91b4, 0x643a2c66, 0xb4847b26, 0x8b621eb4, 0x5e715780, 0xc415c383, 0xb1b0703d, 0x515af3c2, 0x271b2500, 0x9aa05aaf, 0x5048cf7b, 0x7a7ff69d, 0x877a1c67, 0x01fd3ed5, 0xa90c262a, 0x5379d2f9, 0x10af3d07, 0x2a94800d, 0x5d270a65, 0x3ecc2b0d, 0xddf8b3d6, 0x5753a036, 0xe7304be3, 0x3db20cec, 0x2f66f9b8, 0x01fe5462, 0xdf3bd91e, 0xc26c84bd, 0xfc59cd10, 0x8c17e9f2, 0xe001c227, 0xa6d7e9a8, 0xfb877293, 0xbf47ddc4, 0x0681f1ce, 0x88af2bdf, 0x85235175, 0xf5270bba, 0xfb5d71c9, 0x849efcc8, 0x5f6e3273, 0x3e178ae1, 0xab4d498f, 0x7d0ff7fb, 0xbc79738f, 0xb8c8d3d9, 0xd7a22069, 0x533a49bc, 0xcc5b94c4, 0xfc259432, 0xee65ae63, 0xb04ca8be, 0x934f2aba, 0xed049df9, 0x69b009e1, 0xe54a19a9, 0x2ac32676, 0xa679c421, 0x39cf383b, 0x7493d58d, 0x1765d3a4, 0x60896a4c, 0x7b03bc8d, 0x3c02b683, 0x18848bba, 0x12ced6be, 0x1eece2d0, 0x0f83f8bf, 0x9c3e2893, 0xcf46826b, 0xe4a6dda6, 0x15d54445, 0x7d20f8ec, 0xf880fed0, 0x98decd4d, 0xb1a0d6e9, 0x702e9cac, 0xd505d4be, 0x8ba737f2, 0x10a5dd2b, 0xaa6000e6, 0x3aa123a9, 0x55ddd3b7, 0x9bb6d702, 0x3827d0b8, 0xfdfca2e7, 0xed61762a, 0x8d75d5e5, 0x4ff59cb0, 0x1d957741, 0x087627c7, 0x2af2e3ba, 0xc7dd92ad, 0x35f775a7, 0xad0fc353, 0x921787e4, 0x36c7c049, 0xa87b13e9, 0xdcfcd0ec, 0x773d214b, 0x26679f6f, 0x6ab4b05e, 0x74a72b86, 0xa053df95, 0x6ef902df, 0xce85a12f, 0xd2cd9f96, 0x5a49395f, 0x17f21b5d, 0x443f026d, 0x89e73b67, 0xc6f66f45, 0x9388dd96, 0x24d32885, 0x056585b1, 0xc4783afe, 0x366a69a5, 0x3f215907, 0xfdfc9357, 0x27841624, 0x3e322c75, 0xe6780666, 0x862dec13, 0x8a5b15d0, 0x8e38a818, 0x428175b1, 0x35fc56d8, 0xafb415d9, 0xfc6cfdd6, 0x38cbcab2, 0x1e6f3612, 0xe36c3778, 0x4b3980eb, 0x9dc273e9, 0xea2a2ad8, 0x0f3b5c07, 0x79979a8d, 0x8d5563be, 0x78f344c5, 0xb317810e, 0xb0b1fb4c, 0x11bb07ba, 0x782a3a6b, 0xad13ca88, 0xc7bb4f7c, 0xad1c97ba, 0x5a239bd8, 0x624495c1, 0x6c807ff7, 0x6ac870e9, 0x1de6e5db, 0xbae6d8c5, 0xae23fd66, 0xc92785c0, 0x16c4d28e, 0x65faf774, 0x1b87506f, 0x920854ca, 0xb92be3d9, 0x8c80bc81, 0xa0e469fb, 0xb7d2aca1, 0xdaef14b5, 0x145988ef, 0xd32ef669, 0x7fd7dfd8, 0xd2da9263, 0xaed964e2, 0x820c5765, 0xd4829e7f, 0xeaa4748b, 0xd46fcc4c, 0x68868e32, 0x1940a29b, 0xe5cbc908, 0x22eaf05a, 0xfa44dcef, 0xb1013475, 0xd2f9ce0c, 0xfd7638b1, 0xa13e098c, 0x646da2ec, 0x5a4191b3, 0xaa45c64f, 0xd31d44c2, 0x0d0af695, 0xbdf1b4d8, 0x615a79ad, 0x7c784e96, 0xadece374, 0x65d3e979, 0x9b0592be, 0xc5c6d2fe, 0x4f1cd01d, 0x9f5aad5e, 0x4831fa39, 0x297f6802, 0x45890f02, 0x12c47205, 0x208cad5e, 0x618eec62, 0xf06e73ee, 0x8144513b, 0xdc6816f8, 0x65beac80, 0x13ece056, 0xb134ed41, 0x8278e408, 0x62c9b7ce, 0x1fd8b6e6, 0x1b740cb0, 0xc1cf74fe, 0x9290d7d1, 0x87c13d07, 0x9fad3d1c, 0x21a7b5b9, 0x75ddbca9, 0x75f8a598, 0x60bb3d99, 0xaf0645e4, 0x25f555b1, 0x5cf711ba, 0x7d139050, 0xafbed565, 0x97fa7824, 0x5c316bcc, 0xc862ad1a, 0xac70047a, 0xb51bf698, 0xf1a93dc2, 0x7dab655f, 0x6d548f3f, 0xc561472f, 0x729ca627, 0x9a8c9ea3, 0xe8d1d853, 0x471bb019, 0x11880b0b, 0xe2ea3ad2, 0x11e85c66, 0xf7971b34, 0x7ec133ae, 0x55ccefdc, 0x4370dd70, 0xb69d687f, 0x0d7ab225, 0x8df32a94, 0x30517771, 0x4e29f9fb, 0x168b53f2, 0x7369df56, 0x369f48eb, 0x160ebe80, 0x5f347ddc, 0x17a1d9bb, 0x306b5c2d, 0x4eda8ed9, 0x2d9b67e2, 0xe31a4b26, 0xe4af210a, 0xed29e1d5, 0x8dc84b7a, 0x8c9cc109, 0xa5f5d012, 0x24dd6009, 0xcf2bbfff, 0xd7825889, 0xebc5e849, 0x32491c96, 0x829e2b3a, 0x7741e774, 0x4550ff58, 0x18c7105d, 0x3be17b3b, 0x5dc9fa0b, 0x2f812f16, 0xec2ac4f2, 0xeaf831d5, 0x3d9be267, 0xd7e554e4, 0xc52bac6a, 0x44ff7922, 0x3512be89, 0x7f581513, 0xba4d9072, 0xe90d0fa2, 0xa07219ee, 0xde13df78, 0xdeeb2fb5, 0x15037a95, 0xa00bd4d5, 0x1aa352dc, 0xd1a06019, 0x1b84fae2, 0xa10ce05b, 0x8928a0f5, 0xde236067, 0xd9bf4a95, 0x89761b80, 0x5b1b2781, 0xaa5f52c8, 0x7eb1b14d, 0x4f58c243, 0xefa030ed, 0xae8b4d36, 0xb1e101d1, 0x6b5b3215, 0xee769e44, 0x81a1ac13, 0xc1ea3204, 0x9b1426d8, 0x0208e019, 0x6ce1d03b, 0xf72d2580, 0xceb88431, 0xc47bf8a3, 0xa2a42ef7, 0x33289ea6, 0x781f283d, 0x322d3190, 0xa661103c, 0x33fa0bcd, 0xe7089a1c, 0xa525e222, 0xb59a6a12, 0x1cc107e2, 0x395624eb, 0xcab9f36f, 0x9924cc68, 0xa047f9f2, 0xc9c79b73, 0x4c51b552, 0x1c65d939, 0xddbbe6aa, 0x496fe733, 0x0cad6f5f, 0xbe1a6290, 0xc3d17005, 0x6b1bdb1e, 0x176f4cac, 0x4dc1ee0a, 0x852e67e9, 0x621c7b2e, 0xf186ce4c, 0x8ba4b80e, 0x900f0fc3, 0x41b0ffb6, 0x497b7be1, 0x4678f4f7, 0x28622f23, 0xb87d36ad, 0x4f78f670, 0x16ed5182, 0x9f38c40f, 0x53d9e9ae, 0x1f3122a0, 0x7a694d69, 0x7e33cde3, 0xae84fcc1, 0x1c26c0ec, 0xbd97c079, 0x501b8b08, 0x1d91a96f, 0x5a1a8c67, 0x3846f97c, 0x17900a4c, 0x26114fb7, 0xb6242e0d, 0xd6879240, 0x6935a52b, 0xc202cedb, 0x201edcfc, 0x17a6a14a, 0x68c8a5f3, 0x6be8643e, 0xcb657d59, 0x237fd647, 0x02ad474d, 0x9b699cc9, 0xc065adbc, 0xef88e6b6, 0x4af81a90, 0xa5e98510, 0x72319e7c, 0x2ef9f55c, 0x0387269f, 0x999891bc, 0x3c566bc8, 0x78f30561, 0x62613918, 0x67b3ac5e, 0x3be7076f, 0xe279dbe0, 0x68762bbf, 0xdafff4ce, 0xa2aba5fa, 0xbbae6fc1, 0x44c73759, 0xdc99d350, 0x865c2082, 0x331e3ff6, 0x5d9144ae, 0x2bbab7d6, 0xda71c657, 0x9a3922de, 0xbbd6cf2e, 0x9a57fc94, 0x401b22f2, 0x69a48c2c, 0xef2f1955, 0x83723b69, 0x63d78075, 0xfd1746d6, 0xdf5c7865, 0x82744a49, 0xde352347, 0x35b302e9, 0xd2753f29, 0xc7e121d9, 0x022bed66, 0xc31c7984, 0xf27d3d1a, 0x56fd4cb6, 0x0dba26cb, 0xb1b42f9f, 0x4561ab5f, 0x48002003, 0xc95bbaf7, 0x1cc0c888, 0x5aeb6e1d, 0x8ab9a6d9, 0x256644ca, 0x4bed1296, 0x2a5b52d0, 0x5ad12503, 0xb44cff07, 0xbe0e230f, 0x8dcaaac7, 0x87e61733, 0x720e92ee, 0x6f6ce2a8, 0x70e9db85, 0x03d19c15, 0x5b1dd453, 0xe3cf03e8, 0xc63dac9a, 0x8cf00798, 0x79fa56a9, 0xc5c68fcd, 0x755418a4, 0x9a1e22ea, 0x1750dc8f, 0x1153c626, 0x8ceae368, 0x89d90b56, 0xc8f9d06a, 0x6f235d4e, 0x4b369dd2, 0x4c3ba34a, 0xe9edbd45, 0xb7878c8a, 0x22d3932a, 0x2c2fc641, 0x7e3d72e3, 0xcdc1201a, 0xb51eb432, 0xc760f97f, 0x1296383b, 0x33a98421, 0xc9bf8607, 0xe8fb761c, 0xa189d3cf, 0xa5b0382d, 0x64349323, 0x447876b6, 0x1571322d, 0x484e6632, 0x95452f90, 0x51a6b512, 0x83d912d7, 0xd11177d9, 0x5fe89d59, 0xe4bc0238, 0xb1692f21, 0x91134be9, 0x254726b7, 0xa63b86e2, 0x54723c7a, 0xcc933121, 0x6d7a286f, 0x2473d218, 0x0e7ec39f, 0x639dd530, 0x43fc260a, 0x6376d26f, 0x659760ec, 0xc68f510b, 0xf1506ca0, 0xce704625, 0x87b996b6, 0xff6346da, 0x84493ef9, 0x318708ab, 0x40b131fc, 0x196604da, 0xaf66834c, 0x24a8ee30, 0x1d5326e6, 0xf12d1c61, 0x906182cb, 0xe8805270, 0x1556cd7a, 0xb9b23404, 0xf4d783e6, 0x78c8a6d1, 0x0f558ca9, 0xb5ea061a, 0xdacc851f, 0xcfd57281, 0xe1267957, 0xaf50144c, 0x7b9e1d14, 0x1b84f9ce, 0x76134b2a, 0xbf502523, 0xdd8bbf9e, 0x956c6bbe, 0x3deb78e6, 0x08bf0b88, 0x80295d61, 0x51b1d6a1, 0x5f1f20e6, 0x9cf3d648, 0xf9ba6448, 0x78f8ace8, 0x41171614, 0x42c51870, 0x864b707f, 0x02eb21bc, 0x0dc2c8cf, 0xbe0794d6, 0xc90797d8, 0xa628cb3c, 0xae1fff8a, 0x1b2676e4, 0x6e115cd8, 0xf06fc0bc, 0xbf2ab490, 0xe9cebfcb, 0x7d3b4627, 0x44d4ca2b, 0x1c30e6b1, 0x94f5d268, 0xb1883f32, 0xe618e453, 0x76636da2, 0x40b9a5b7, 0x5f22d468, 0x5ee6724e, 0xd6e4d381, 0x218573ff, 0x156bb61d, 0x2a68ea4b, 0x0c384590, 0xfe3b4fe4, 0x0d5c71c4, 0x7099c73b, 0x9ff7af1d, 0x3538ace5, 0x575c77ff, 0xaff786db, 0x433a0e5c, 0x99ffc258, 0xadda2154, 0xe97bd4fa, 0xb8bd61ca, 0x41b05a29, 0x79c39c89, 0xf174d174, 0x17b5a117, 0xdbc1dab4, 0x54f7399d, 0xf3d5268f, 0x2716e7ff, 0xba9b401f, 0x6d58fe43, 0xaba47a01, 0x3c3ef774, 0x39dac5c9, 0xc2fba6ad, 0x6df09a00, 0x548a1651, 0xd48c3b51, 0x9997e5fc, 0xf178ef0b, 0xdad141b5, 0x1b051612, 0xfb5188fb, 0xcbfc88ec, 0xd4a62b32];
		#[rustfmt::skip]
		let sbox2: [u32; 1024] = [0xa49b31f9, 0x14b325cb, 0x70406a55, 0x59f5626a, 0xf57b14cc, 0x9ca42c1c, 0x317f84f0, 0x1c306e6e, 0xf2e35d7b, 0x84a383ca, 0xf6a71b83, 0xf7458ed2, 0x3bf6a0d6, 0xe61eeb13, 0x56eb30bf, 0xcf518a40, 0x08ba422d, 0x30f9cb16, 0x8716960b, 0x4c7ae291, 0x275f3f62, 0x422ce479, 0x4fb1ecce, 0x9a8bac8b, 0x19518301, 0xe43ce55c, 0x5ff18961, 0xbb1872bc, 0xc2ff095a, 0xd2cafb5b, 0x622eb0f1, 0x50e0f150, 0xd8c2b893, 0x6ede49a4, 0xe34ffc69, 0xa5a062ec, 0x95cfdc02, 0xef7058e3, 0x5d4e6602, 0x3d092330, 0x64839d93, 0x4abbd00d, 0x317554f3, 0x07784e98, 0x25b96411, 0x7258937a, 0x6689e7ad, 0x6b513cdd, 0x0fba7e17, 0x6a92aa7a, 0x392d1335, 0x8abc1e7f, 0x207f3fec, 0xd8a06c8c, 0x697d5b60, 0x91c6e415, 0x7d3faf1f, 0xa43e6fb0, 0xecba5579, 0xa2064ee8, 0x22631ebf, 0x1c98dfdc, 0xc686c248, 0x27388af8, 0x20cef1b7, 0x251aaf54, 0x14fe0222, 0xafbdfd95, 0x84619d71, 0x3888cedf, 0xc2b5fd8a, 0x3b2f0f57, 0x81a8cdd6, 0xe2c1c069, 0x78d3da46, 0xbf341418, 0x22a99852, 0x2c18e514, 0xa7d5f9fc, 0x49a6758f, 0xe5ef9d26, 0x2a9b5b65, 0xb173b8f0, 0x2d077e01, 0xef12ff17, 0x848b1f63, 0x5f1269f0, 0xb423ef3e, 0x46445e9b, 0x67a26a08, 0x845f7bca, 0x318972d2, 0x8cd57a24, 0x99f52121, 0x77b99bf7, 0xf45b177b, 0x01778776, 0x87c734c6, 0x0bcd4a10, 0xc67dd43e, 0x33d757ae, 0x34053bc7, 0xe67967a1, 0xef357a66, 0x1ecc97a4, 0x71dd71d9, 0x96faa6d3, 0x654c2be1, 0x4a4af3df, 0x4398b8d5, 0x9a609f4a, 0xe48267d2, 0x1184f54b, 0x107c6007, 0xa90423ce, 0x2c1bb8e1, 0x9df5bee5, 0xb22f0c22, 0xdc71e7ec, 0x2bb8ae13, 0xba984153, 0x9e345a15, 0x4fc7cdd3, 0x6d4d5c1f, 0x8d6d694d, 0xbca711a2, 0x664091f9, 0xeee31a0c, 0x1a234121, 0x682a0248, 0xd65572d0, 0x15f1ecc5, 0x0c90a371, 0x0c935b1e, 0xc0915030, 0xbffa26e3, 0xb8184661, 0xb4a341e3, 0xb75ac727, 0x1d09a440, 0x7442d1ed, 0x5f0a6f5d, 0x405fe760, 0x90d1aa22, 0x9fe56cc8, 0x75c2637a, 0xb5b8186a, 0xb216e2a3, 0x6939d748, 0x41131f0b, 0x5af45caa, 0x425fd3bb, 0x5cb1377a, 0x65a30e2c, 0x1978e2f8, 0x546fae56, 0x7bd5009b, 0x8b081004, 0xd9b067da, 0x6c77aec5, 0x865d6dc0, 0x67195e15, 0x8df927db, 0x75f29fce, 0xe1a99bfa, 0xa82c3769, 0xeecfd52a, 0xd50a9ecf, 0x086c5f8c, 0xb0aeb228, 0x53dd6a54, 0x9032f213, 0xeb0bcd5c, 0x11e242d1, 0x33f66943, 0xa1bc7c01, 0x6247c51e, 0xecbb2b8e, 0x38a14af9, 0x0638c5a4, 0x1a6647f5, 0x6aaebaef, 0x385cf58a, 0xb38b244a, 0xac980b85, 0x5e7f37b1, 0x9f48c181, 0x5b07f2a0, 0x7e5b1e84, 0x214a6611, 0x075f85b3, 0xfafe18aa, 0x0313aace, 0xd4a24884, 0xfd4c7163, 0x09c263d3, 0x920c7573, 0x13755fff, 0x0f6d8f96, 0xe67ff359, 0xfe5dca02, 0x99fcb84f, 0xd03a872e, 0x0536d1cb, 0x5b6fb7cd, 0xbabf4701, 0x8bbdf540, 0x9ab9855b, 0x1f033c45, 0x3e9652a7, 0xbb040e6c, 0xb407e8f0, 0xd933dd36, 0x37df37c1, 0x58578c74, 0xc2499d06, 0x3a1b5044, 0x44f19cfd, 0x12202d5c, 0x41c06c0a, 0xacfda57a, 0xecd57da9, 0xf1d49c73, 0xc977b32f, 0x7d6c4065, 0xf08a4f56, 0x407a852a, 0x20dff587, 0x63ea6c52, 0x0ef7c60c, 0xd78c2367, 0x6fffc5ad, 0xe8480812, 0x057948d7, 0x5041d6a8, 0xaa709c4f, 0x38431e2f, 0x65bbffdc, 0xb247384d, 0xda7d8e0f, 0xcaf72968, 0x35150d55, 0xc4d445d3, 0xc7f8ea9e, 0x8bfc9afe, 0x845a4f65, 0x0966277c, 0x80b6786d, 0xee409e42, 0x7d492ade, 0xe5d4f93f, 0x75a6dd33, 0x0b27bc50, 0xd17ad5e0, 0x85fe7dd5, 0x4eff3c88, 0xf7319a09, 0xdc562000, 0xfbeb17b7, 0x3bb2515f, 0x8f773da5, 0xf2eb7554, 0xf5833e21, 0xd64e88be, 0xb736688c, 0x10c44ee1, 0x31deb72f, 0xa377c934, 0x03155f3d, 0xe22ca814, 0xa214b186, 0xd1d4a118, 0x1d0b29e1, 0xd951aa68, 0x7d3f6833, 0x845d6669, 0x663ebeb0, 0x72475f0b, 0xc3dc3eec, 0xad5a2494, 0xb7690fa4, 0x5202b219, 0x99e1f15d, 0x40727426, 0x571eaad6, 0x8293adde, 0x895c6475, 0xdbc9d819, 0xd6fc3b1d, 0x30f8ba80, 0x23198e43, 0xf867becc, 0x097d3384, 0xaf1ac886, 0x7c629c18, 0xdad00798, 0x6d371fd3, 0x66aab6f6, 0x2afe6bf3, 0x6658af10, 0xb04e30ed, 0x5f5dc842, 0x8ac3454b, 0x1f873ef8, 0x38efe8b0, 0xdef162f9, 0x35247431, 0xf8eb6635, 0xa09f98aa, 0x212fb3a2, 0x1db0d7d5, 0x29ea3c4f, 0x0f736981, 0x5f04a3d4, 0x17840ed5, 0x9891c0d5, 0x7198a185, 0xc98db966, 0x43e8f34c, 0x960a32f0, 0x2e035e6c, 0xcd08f091, 0xe6b71fc8, 0x1a52de4c, 0xb0f783b3, 0x75f4b9f5, 0x27e37a12, 0xadd77d2c, 0xe7467d38, 0x0ec19c94, 0x7c117b15, 0x743c5471, 0x9c790ab9, 0x1a604bc3, 0xbed8b4b2, 0x28a4c2d2, 0x5a5ea7ef, 0x5fb7c53d, 0xfe5d05a9, 0xc6042225, 0x479231b1, 0xc8380df8, 0xbe1c598c, 0xbc82d9e3, 0x6a17f8f8, 0x76ef0e50, 0x906175f1, 0x59faae3c, 0xccbe5245, 0xe3e6efdb, 0x527ce4f0, 0xd771b998, 0x7a711499, 0xf07e8422, 0x60793b5a, 0x794b838b, 0x02ced6ba, 0xe83620c8, 0x04542777, 0xce7aad59, 0x9475137c, 0x4ec1b4cf, 0x897d2165, 0x2ee8af55, 0x341fc85f, 0x38567dcf, 0xd561c473, 0xd68e2489, 0x14939a81, 0xf17fd579, 0x0a8a783a, 0x6628d6ce, 0xc3099073, 0x59d8580c, 0x2e6e1b39, 0xe10c9498, 0x9fdc35ec, 0xf4cdfa94, 0x693f96ea, 0x3d656850, 0x6a838e1d, 0xb3f00e7f, 0x4e86dea6, 0x93b6afd3, 0x38f707ad, 0x05bab849, 0xa0a695ab, 0x373fdbe7, 0x2cbcf6d3, 0x62bcfdbd, 0x11b56fcc, 0x437b34ef, 0x0ad7ac8c, 0x97e78f07, 0x8e6c0373, 0x2813a4bf, 0xb8669e50, 0x1a71d8fc, 0xe113c4ec, 0x8b2b7078, 0xbf7eaa3c, 0x4c3758b7, 0xc128c07a, 0x1537b9bc, 0x79e96d89, 0xe5cd3ec2, 0x5ddf0c97, 0xcb5593bf, 0xab9b7f48, 0x8e729f88, 0x014a821b, 0x2ecfde3d, 0x888731e9, 0xd16d6b96, 0x60b2e5fb, 0x7c1bcda8, 0xc9baa628, 0x1323b0a9, 0xa160fd90, 0x9c507d43, 0x7069abbd, 0x8e1f03f2, 0xcbdcf249, 0xeb417c1b, 0xd664d9ee, 0x6c789e8a, 0xa5731917, 0x69e419c6, 0x724ef6de, 0x38f30698, 0xe67c4b33, 0x6b56bb45, 0xe8478f9f, 0x4777d172, 0xfb04d5e7, 0xc01895dd, 0x8a46f1b9, 0x9ff4ce0f, 0x3ea0fafe, 0xf4b381c7, 0xad2dc1b6, 0x59ede782, 0xf681feba, 0xf087af7e, 0x2111e2f6, 0xdeba45f8, 0xad2c04a6, 0xdd7c8034, 0x0a3b717b, 0xa0b0894c, 0xb8afb8eb, 0x159edeb9, 0xb577ed91, 0xf9a9276f, 0xb2f6daac, 0x38a37b19, 0x8dfc07d3, 0xa98773db, 0xff06b40c, 0x6735ee9d, 0xeb0f07a9, 0xb96efffe, 0x2f0a9274, 0xfa443e1b, 0x036028c7, 0x0143bba9, 0x1f66b4e7, 0x9c2d5556, 0xc5beab1c, 0x1b999c71, 0x9e32a8be, 0xe9d3c440, 0x48c6099e, 0x677f3d2b, 0xcb069ef0, 0x937e97c3, 0x44e59851, 0x3a589a07, 0x43e6e38a, 0x1a5cbdd8, 0x75ca5ef5, 0x1c06636d, 0x740aa29a, 0x3f3daea8, 0xbec2fce3, 0x8b3ceb28, 0x1e15930e, 0x32872757, 0x0597ef6c, 0xa055ed19, 0xc08a1d9f, 0x2d726c71, 0x43d0ac13, 0x32f44de0, 0x39846b91, 0x5a40a4dc, 0xeb8fbb95, 0x3b2491d3, 0xa4a19889, 0xd9c2fef5, 0x091fab9b, 0x0427d9a6, 0x8352df4f, 0x99c7bb32, 0x7fd24ab8, 0x51ea5972, 0xe0005227, 0x7a5c0c02, 0xdc8a79eb, 0xe690c110, 0x5b60d0fe, 0xd3ad1cd0, 0x4f4f7a20, 0x5799cefd, 0xc9f94944, 0xfc4fce2d, 0xca695c3e, 0x3cc15141, 0x1b00835b, 0x45be0d16, 0x38ce03d5, 0x118e5cd4, 0xc57b2d57, 0x4e9d9c70, 0x0f72b8e4, 0xe00dea18, 0x6f2ceb39, 0x231293fa, 0xd0239f5b, 0xdbf2e9c1, 0x28e1b3bf, 0x84a80fbd, 0x05505718, 0x23191389, 0xf2a0b702, 0xd06f915c, 0xd632e64c, 0x425858d7, 0x05d83eed, 0x30850a95, 0xe2fddeb4, 0x4f055829, 0x9c6a76af, 0x47b6cd94, 0xef5c513c, 0x95bb078c, 0xf0e480db, 0x7c894a31, 0x48cc8450, 0x30f8f5fb, 0xea23aa92, 0xd4e45fd7, 0x432e939c, 0x7440578b, 0x858b190c, 0x5d63f00c, 0xe3a7e877, 0x240e6695, 0x7641bf58, 0x90d17bbb, 0x43ba4ab3, 0x9ec724bf, 0x67dd5053, 0x22bd124a, 0xe57730d9, 0xe3fc2682, 0x516d197b, 0x8ff3b53d, 0x2e87afe0, 0x418db0b3, 0xb5c25e88, 0x1c00c049, 0xc2025b76, 0x59526611, 0xe9db6cd6, 0x1a5e3cdc, 0x9c1a557a, 0x04736ef8, 0x64c1b1f3, 0x2e0e5083, 0x3d7df85b, 0x5b842f3e, 0xa9f6efa0, 0xce793120, 0xc6ea3120, 0x9949d0a1, 0xb69de786, 0x4066dccc, 0x22c24ac6, 0xfb779d51, 0x5495d4b1, 0x1a05626b, 0x1df87a76, 0xe9d013ea, 0x4e1a38cb, 0x431b5888, 0xc7504bf6, 0x7e29821f, 0x51f7c4f3, 0xdd09c930, 0x22d22ca5, 0x74de37f3, 0x0085e5ea, 0xb9ab693c, 0x80ba8f4d, 0xd51fd03f, 0x8dfe7a5e, 0xeaf080a6, 0x7d9e6ced, 0xef52269c, 0x8d4cb463, 0xbd92ee07, 0x00acab70, 0x61350d1c, 0x911a1c56, 0x16d148ac, 0xf546c87c, 0x112059d9, 0x26a83c76, 0x83287de7, 0x01435b9d, 0xc090bc01, 0x82bd801f, 0xa38c9058, 0x9265f192, 0x1ce38de2, 0x85e41ab4, 0x44e4cbc4, 0xedca5867, 0x39be0633, 0x98122349, 0x0df31b82, 0x4064ff68, 0x46519dc4, 0x2a6b7d1d, 0xe1a9ae6b, 0xb35276db, 0xec2d92c1, 0x933bc2bc, 0xa47524dd, 0xaf00c853, 0xebfc1096, 0x81e4781b, 0x9bb91e9f, 0x5b3ea87e, 0x3bbb184c, 0x200f2b18, 0x4e7288b6, 0xe1286f91, 0x25886242, 0x2563d87c, 0x6a952a4b, 0xeed3afc7, 0x10db2217, 0x72fee762, 0x3613cf8c, 0x1f9ec9a2, 0xa6e47f99, 0x1e764704, 0x841c9fcd, 0xbf2e5186, 0x8fe6ee9d, 0xd059aa3f, 0x472c3065, 0x6c821a35, 0x7298f360, 0x074daba5, 0x34fccad7, 0xda9db1dd, 0x87a12e86, 0x56a1067e, 0xf67ddde7, 0x948a92f9, 0x6041e373, 0xa69bf7f4, 0xbd3586cf, 0x92119059, 0xa13907f9, 0xbe090504, 0x1fe912b1, 0x4fd06ecc, 0x48d8bc85, 0x503abc74, 0x59326693, 0x2316e0f6, 0x2ff3dfb8, 0x1c84b6b9, 0xf5d697d3, 0x82dcaf3c, 0x1f0ce9ee, 0x422fb176, 0xfecb7d00, 0x84be7594, 0x851190e9, 0xdcfc9ca5, 0x6d679f79, 0x0c898692, 0x094bd055, 0xc0779118, 0xc5e947fc, 0xc31937e1, 0x96b50205, 0xa589f35d, 0xd90f1584, 0x6d1cb4e8, 0x4ed6525c, 0x067d767a, 0x7b31f6dc, 0x341dc165, 0xd6c23a3c, 0xf1206d3e, 0x29456bd9, 0x5a2af1b5, 0x00424a5e, 0x959dc81b, 0xae462195, 0x801afc2e, 0x0cb5ed7e, 0x0ca313e1, 0xc5829dc8, 0x0b30b151, 0x965335fe, 0x711ff3b9, 0x9578a797, 0xc58f68b6, 0x7487ca7f, 0xbc0c94e8, 0x6060db63, 0x8fe0a929, 0x0a75796b, 0xf8fc377b, 0x265db71a, 0x10811f3b, 0xc04e2f29, 0x534f57d6, 0x413f23cd, 0x305b545f, 0xe28423cb, 0x15a52f85, 0x4654949e, 0x56354f3b, 0xcd495e2e, 0xe3d39e47, 0xc947d194, 0xf3b8d65c, 0x53f50b2b, 0xe92f4747, 0x2c23c465, 0x7a1daa53, 0xfc9dee06, 0x430f7091, 0xec772a26, 0x306b93b0, 0xa2f39a08, 0xcafa10bc, 0x7f25c774, 0xfbd75b12, 0x5e916a43, 0x8453b3d9, 0xae8196d0, 0xa6eeb003, 0xd66f4a91, 0xd3837f62, 0xa76a6f66, 0x5060a24f, 0x26548610, 0x1ff5a3f5, 0x788326a1, 0x818df4e9, 0xb5f76c7a, 0xbac32587, 0x6ffc9a3f, 0x2c4c56a4, 0xd8cb536c, 0xa8386cbe, 0x6979bb6e, 0x2fcb67d4, 0x0b95108a, 0x5c55b976, 0x5b1fe95c, 0xcb5d2c2f, 0x3a3ec858, 0x030bc27c, 0xa8a5438d, 0x9dc4397a, 0xe026b78c, 0xfd3997a3, 0x474ba75c, 0xb7fc9d4f, 0xfe0c6f06, 0x02faa164, 0x5b62ebc7, 0xd3b164eb, 0x4a3362e5, 0x2e255403, 0xc7ecc326, 0x56d54267, 0x98dc0ad8, 0x6b35c4b8, 0x548b2438, 0x0266ef8e, 0x691ed645, 0xb8216e19, 0x75ff4691, 0xd0b5674d, 0xbaaa597e, 0xbf250509, 0x739c67aa, 0x8576ed95, 0xe075ef8e, 0x9bba3024, 0x22a35447, 0xc2b4bc5f, 0x8c259417, 0x997e1cf5, 0xfdaa7768, 0x82ed4c01, 0x6bbef1cb, 0x27d43051, 0x5ab9516f, 0x4ff9b694, 0x9883e5ca, 0x7c617c52, 0xcf88128a, 0x3a020509, 0xc2817253, 0x20f843f6, 0x7c3c66ec, 0x37196a85, 0x0bd2d72f, 0xbb18106f, 0x0e0245bf, 0x6eeb137b, 0x489e7ff0, 0x36690a84, 0x3cf8541a, 0x073145eb, 0x81490c9b, 0x62ddc564, 0x99f6ea6f, 0xbbac34ff, 0xc7d0a4d9, 0x9f518ab4, 0x2dbc5e58, 0x27254dce, 0x1b1d0626, 0xf255b9b5, 0x7e0e2a03, 0x4d5aaf59, 0x99a3feb6, 0xe303d61d, 0x2efd53af, 0x76f7e855, 0x9ae81201, 0x6b82bd63, 0x965343b8, 0xf2189d0b, 0xa01218c0, 0xee64e0bf, 0xdb3b615b, 0x274061c6, 0x86899a29, 0xdea795b5, 0x120324c7, 0x1b1aed42, 0x9a0ebaaf, 0x111189f1, 0x87b8e919, 0xcc660d7d, 0xcf148559, 0x5979835b, 0xf4a147e0, 0xaccb8fde, 0x1cbfa238, 0x278e4fc5, 0x7715336f, 0x2aa18d74, 0x115c33d2, 0xd89880f7, 0x3059f34d, 0x9ecd36a7, 0x078db30c, 0x244f223e, 0x369a7968, 0xc97c21ea, 0x780e1e04, 0xe4dee61a, 0x1e11caf7, 0xe7a8a9ad, 0x9d8c3000, 0x94b01195, 0x90a2465c, 0x03170b30, 0xcb57a4e5, 0x4fe1e464, 0x1a3f8127, 0x0c91b451, 0x40e7cdc2, 0xee23b8ef, 0x42c3872e, 0xca4aa028, 0x44bc10ff, 0x02257125, 0x82ee2c94, 0x3a5bf809, 0xb92398e6, 0xb8ba1481, 0xde41a860, 0x10e6f075, 0x34d25d1c, 0xaf33098c, 0x2ce94255, 0x55ed0d19, 0x9ad04107, 0x6ac35d76, 0x38e2b63f, 0x74404b0c, 0x27f03671, 0x4b615a45, 0x94e2406a, 0x83761941, 0xbcad6220, 0x830a38e6, 0xaf68a0bb, 0xf65b9220, 0x7c0c3dc0, 0xcd67187d, 0xb4d8d230, 0x0016b2d7, 0xdabd2c67, 0x28c503e2, 0x2a7b7db8, 0xf0b4203c, 0x8a92e734, 0xc14679b1, 0x3620914c, 0x303cddd3, 0x2967958d, 0x987cb6c5, 0x6c72e39b, 0xc76a74b8, 0xac735099, 0x62cbfb75, 0xf6716fd2, 0x082843a8, 0x2b86a03d, 0xb250a323, 0xaaa23523, 0x76b84799, 0x7ce10847, 0x88135d18, 0xc409c944, 0x00f25a7f, 0x846ccb3a, 0xf0616253, 0x07b19de3, 0x935ac902, 0x1c592ff7, 0x1e3c9677, 0xcef26184, 0x39bb58fc, 0x2c303a47, 0x24efb762, 0x8bb80865, 0x8d25a11c, 0x311ae710, 0x77eaaca9, 0x0a539d92, 0x3472418e, 0x41ebf7b5, 0xf45f829c, 0xf822e2e9, 0x047f991c, 0x3f47270e, 0x62d2a5af, 0x5c64ae01, 0xac3d5e7b, 0x1adfed67, 0x37b57ac8, 0x9048f261, 0x4793e022, 0x43647038, 0xb943314b, 0x352f1db7, 0x26f22428, 0x6469e6da, 0xdbaaf080, 0xd54da199, 0xfcc55f2a, 0x6dc299e6, 0xec94138e, 0xacc697dd, 0x6ee7d291, 0x7a0ea5bc, 0x42c112ba, 0xe15950a5, 0x12754a72, 0x57945604, 0xba506841, 0x6451c239, 0xe3ac4b4e, 0x42292eff, 0x25961472, 0xa110ffc0, 0x59671698, 0x6c6d0db1, 0xf3c3dfff, 0x6f1ef7af, 0x5f70de4f, 0xe3d9f90c, 0x130fa8db, 0x897a21b2, 0x33929761, 0x8449291d, 0x77d7fa69, 0x969cefbd, 0xc380dca3, 0xe024c0a9, 0x7ab7eff2, 0x79d9fcf5, 0x9f818442, 0x073381ca, 0x5faba58b, 0x9dda756e];

		let vs = VisionPermutation::default();

		for d_in in 0..(inputs.len() / 8) {
			let chunk = from_u32_to_packed_256(
				inputs[d_in * 8..(d_in + 1) * 8]
					.iter()
					.map(|x| AESTowerField32b::from(BinaryField32b::new(*x)))
					.collect::<Vec<_>>()
					.as_slice()
					.try_into()
					.unwrap(),
			);

			let expected1 = from_u32_to_packed_256(
				sbox1[d_in * 8..(d_in + 1) * 8]
					.iter()
					.map(|x| AESTowerField32b::from(BinaryField32b::new(*x)))
					.collect::<Vec<_>>()
					.as_slice()
					.try_into()
					.unwrap(),
			);
			let expected2 = from_u32_to_packed_256(
				sbox2[d_in * 8..(d_in + 1) * 8]
					.iter()
					.map(|x| AESTowerField32b::from(BinaryField32b::new(*x)))
					.collect::<Vec<_>>()
					.as_slice()
					.try_into()
					.unwrap(),
			);

			let mut got1 = chunk;
			vs.sbox_packed_affine(&mut got1, &*INV_PACKED_TRANS_AES, *INV_CONST_AES);
			let mut got2 = chunk;
			vs.sbox_packed_affine(&mut got2, &*FWD_PACKED_TRANS_AES, *FWD_CONST_AES);

			assert_eq!(expected1, got1);
			assert_eq!(expected2, got2)
		}
	}

	#[test]
	fn test_simple_hash() {
		let mut hasher = VisionHasherDigest::default();
		let data = [0xde, 0xad, 0xbe, 0xef];
		hasher.update(data);
		let out = hasher.finalize();
		// This hash is retrieved from a modified python implementation with the proposed padding and the changed mds matrix.
		let expected = &hex!("a42b46ccea1a81cafc4b312c0bc233f169f8ecb2377e951d14461acfefc6b7b5");
		assert_eq!(expected, &*out);

		let hasher = VisionHasherDigestByteSliced::new_with_prefix(data);
		let mut output = [MaybeUninit::uninit(); 4];
		hasher.finalize_into(&mut output);

		for row in 0..4 {
			let out = unsafe { output[row].assume_init() };
			assert_eq!(expected, &*out);
		}
	}

	#[test]
	fn test_multi_block_aligned() {
		let mut hasher = VisionHasherDigest::default();
		let input = "One part of the mysterious existence of Captain Nemo had been unveiled and, if his identity had not been recognised, at least, the nations united against him were no longer hunting a chimerical creature, but a man who had vowed a deadly hatred against them";
		hasher.update(input.as_bytes());
		let out = hasher.finalize();

		let expected = &hex!("406ea77bc164afc0fbb461bb6e2cf763612c0c55bf66c98f295dba8f9e5f3426");
		assert_eq!(expected, &*out);

		let mut hasher = VisionHasherDigest::default();
		let input_as_b = input.as_bytes();
		hasher.update(&input_as_b[0..63]);
		hasher.update(&input_as_b[63..128]);
		hasher.update(&input_as_b[128..163]);
		hasher.update(&input_as_b[163..]);

		assert_eq!(expected, &*hasher.finalize());
	}

	#[test]
	fn test_multi_block_unaligned() {
		let mut hasher = VisionHasherDigest::default();
		let input = "You can prove anything you want by coldly logical reason--if you pick the proper postulates.";
		hasher.update(input.as_bytes());

		let expected = &hex!("7e0dcd26520e1e9956de65b1f9dea85815ed9ae0c4b3f48559679acea71729f2");
		let out = hasher.finalize();
		assert_eq!(expected, &*out);
	}

	fn check_multihash_consistency(chunks: &[[&[u8]; 4]]) {
		let mut scalar_digests = array::from_fn::<_, 4, _>(|_| VisionHasherDigest::default());
		let mut multidigest = VisionHasherDigestByteSliced::default();

		for chunk in chunks {
			for (scalar_digest, data) in scalar_digests.iter_mut().zip(chunk.iter()) {
				scalar_digest.update(data);
			}

			multidigest.update(*chunk);
		}

		let scalar_digests = scalar_digests.map(|d| d.finalize());
		let mut output = [MaybeUninit::uninit(); 4];
		multidigest.finalize_into(&mut output);
		let output = unsafe { array::from_fn::<_, 4, _>(|i| output[i].assume_init()) };

		for i in 0..4 {
			assert_eq!(&*scalar_digests[i], &*output[i]);
		}
	}

	#[test]
	fn test_multihash_consistency_small_data() {
		check_multihash_consistency(&[[
			&[0xde, 0xad, 0xbe, 0xef],
			&[0x00, 0x01, 0x02, 0x03],
			&[0x04, 0x05, 0x06, 0x07],
			&[0x08, 0x09, 0x0a, 0x0b],
		]]);
	}

	#[test]
	fn test_multihash_consistency_small_rate() {
		check_multihash_consistency(&[[&[0u8; 64], &[1u8; 64], &[2u8; 64], &[3u8; 64]]]);
	}

	#[test]
	fn test_multihash_consistency_large_rate() {
		check_multihash_consistency(&[[&[0u8; 1024], &[1u8; 1024], &[2u8; 1024], &[3u8; 1024]]]);
	}

	#[test]
	fn test_multihash_consistency_several_chunks() {
		check_multihash_consistency(&[
			[&[0u8; 48], &[1u8; 48], &[2u8; 48], &[3u8; 48]],
			[&[0u8; 32], &[1u8; 32], &[2u8; 32], &[3u8; 32]],
			[&[0u8; 128], &[1u8; 128], &[2u8; 128], &[3u8; 128]],
		]);
	}
}
