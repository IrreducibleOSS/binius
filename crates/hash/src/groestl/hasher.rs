// Copyright 2024 Irreducible Inc.

//! This module implements the 256-bit variant of [Grøstl](https://www.groestl.info/Groestl.pdf)

use super::{super::hasher::Hasher, arch::Groestl256Core};
use binius_field::{
	arch::OptimalUnderlier256b,
	as_packed_field::{PackScalar, PackedType},
	underlier::Divisible,
	AESTowerField8b, BinaryField, BinaryField8b, ExtensionField, PackedAESBinaryField32x8b,
	PackedAESBinaryField64x8b, PackedExtension, PackedExtensionIndexable, PackedField,
	PackedFieldIndexable, TowerField,
};
use p3_symmetric::{CompressionFunction, PseudoCompressionFunction};
use std::{cmp, marker::PhantomData, mem::MaybeUninit, slice};

/// The type of output digest for `Grøstl256` over `F` which should be isomorphic to `AESTowerField8b`
pub type GroestlDigest<F> = PackedType<OptimalUnderlier256b, F>;

/// An alias for `Grøstl256` defined over `BinaryField8b`
pub type GroestlHasher<P> = Groestl256<P, BinaryField8b>;

const BLOCK_LEN_U8: usize = 64;

/// The Grøstl-256 hash function.
///
/// The Grøstl-256 hash function can be viewed as natively defined over `AESTowerField8b`
/// and isomorphically maps to `BinaryField8b`. The type `P` is the input to the update
/// function which has to be over a packed extension field of `BinaryField8b` or `AESTowerField8b`.
#[derive(Debug, Clone)]
pub struct Groestl256<P, F> {
	state: PackedAESBinaryField64x8b,
	current_block: PackedAESBinaryField64x8b,
	current_len: u64,
	_p_marker: PhantomData<P>,
	_f_marker: PhantomData<F>,
}

trait UpdateOverSlice {
	type Elem;

	fn update_slice(&mut self, msg: &[Self::Elem], cur_block: usize);
}

impl<P> UpdateOverSlice for Groestl256<P, BinaryField8b> {
	type Elem = BinaryField8b;

	fn update_slice(&mut self, msg: &[BinaryField8b], cur_block: usize) {
		msg.iter()
			.map(|x| AESTowerField8b::from(*x))
			.enumerate()
			.for_each(|(i, x)| {
				let block_idx = (cur_block + i) % BLOCK_LEN_U8;
				self.current_block.set(block_idx, x);
				if block_idx == BLOCK_LEN_U8 - 1 {
					self.state = compression_func(self.state, self.current_block);
				}
			});
	}
}

impl<P> UpdateOverSlice for Groestl256<P, AESTowerField8b> {
	type Elem = AESTowerField8b;

	fn update_slice(&mut self, msg: &[Self::Elem], cur_block: usize) {
		self.update_native(msg, cur_block);
	}
}

impl<P, F> Groestl256<P, F> {
	fn update_native(&mut self, mut msg: &[AESTowerField8b], mut cur_block: usize) {
		while !msg.is_empty() {
			let to_process = cmp::min(BLOCK_LEN_U8 - cur_block, msg.len());

			// Firstly copy data into next block
			let next_block = PackedAESBinaryField64x8b::unpack_scalars_mut(slice::from_mut(
				&mut self.current_block,
			));
			next_block[cur_block..cur_block + to_process].copy_from_slice(&msg[..to_process]);

			// absorb if ready
			if cur_block + to_process == BLOCK_LEN_U8 {
				self.state = compression_func(self.state, self.current_block);
				cur_block = 0;
			}

			msg = &msg[to_process..];
		}
	}
}

impl<P, F> Default for Groestl256<P, F> {
	fn default() -> Self {
		let mut iv = PackedAESBinaryField64x8b::default();
		// IV for Grøstl256
		iv.set(62, AESTowerField8b::new(0x01));
		Self {
			state: iv,
			current_block: PackedAESBinaryField64x8b::default(),
			current_len: 0,
			_p_marker: PhantomData,
			_f_marker: PhantomData,
		}
	}
}

/// Compression function as defined for Grøstl256
fn compression_func(
	h: PackedAESBinaryField64x8b,
	m: PackedAESBinaryField64x8b,
) -> PackedAESBinaryField64x8b {
	let (a, b) = Groestl256Core.permutation_pq(h + m, m);
	a + b + h
}

impl<P, F> Groestl256<P, F>
where
	F: BinaryField,
	P: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<F>,
{
	fn finalize_packed(&mut self) -> PackedAESBinaryField32x8b {
		let bits_per_elem = P::WIDTH * P::Scalar::DEGREE * (1 << BinaryField8b::TOWER_LEVEL);
		let n = self
			.current_len
			.checked_mul(bits_per_elem as u64)
			.expect("Overflow on message length");
		// Enough for 2 blocks
		let mut padding = [AESTowerField8b::default(); 128];
		padding[0] = AESTowerField8b::new(0x80);
		let w = (-(n as i64) - 65).rem_euclid(BLOCK_LEN_U8 as i64 * 8);
		let w = w as u64;
		let zero_pads = ((w - 7) / 8) as usize;
		let num_blocks = (n + w + 65) / (BLOCK_LEN_U8 as u64 * 8);
		padding[zero_pads + 1..zero_pads + 9]
			.copy_from_slice(&num_blocks.to_be_bytes().map(AESTowerField8b::new));

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % BLOCK_LEN_U8;
		self.update_native(&padding[..zero_pads + 9], cur_block);

		let out_full = Groestl256Core.permutation_p(self.state) + self.state;
		let mut out = [PackedAESBinaryField32x8b::default()];
		let out_as_slice = PackedFieldIndexable::unpack_scalars_mut(&mut out);
		out_as_slice.copy_from_slice(&PackedFieldIndexable::unpack_scalars(&[out_full])[32..]);

		out[0]
	}
}

impl<P, F> Hasher<P> for Groestl256<P, F>
where
	F: BinaryField + From<AESTowerField8b> + Into<AESTowerField8b>,
	P: PackedExtension<F, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<F>,
	OptimalUnderlier256b: PackScalar<F> + Divisible<F::Underlier>,
	Self: UpdateOverSlice<Elem = F>,
{
	type Digest = GroestlDigest<F>;

	fn new() -> Self {
		Self::default()
	}

	fn update(&mut self, data: impl AsRef<[P]>) {
		let msg = data.as_ref();
		if msg.is_empty() {
			return;
		}

		let cur_block = (self.current_len as usize * P::WIDTH * P::Scalar::DEGREE) % BLOCK_LEN_U8;
		let msg_remaining = P::unpack_base_scalars(msg);

		self.update_slice(msg_remaining, cur_block);

		self.current_len = self
			.current_len
			.checked_add(msg.len() as u64)
			.expect("Overflow on message length");
	}

	fn chain_update(mut self, data: impl AsRef<[P]>) -> Self {
		self.update(data);
		self
	}

	fn finalize(mut self) -> Self::Digest {
		let out = self.finalize_packed();
		Self::Digest::from_fn(|i| F::from(out.get(i)))
	}

	fn finalize_into(self, out: &mut MaybeUninit<Self::Digest>) {
		let finalized = self.finalize();
		out.write(finalized);
	}

	fn finalize_reset(&mut self) -> Self::Digest {
		let out_native = self.finalize_packed();
		let out = Self::Digest::from_fn(|i| F::from(out_native.get(i)));
		self.reset();
		out
	}

	fn finalize_into_reset(&mut self, out: &mut MaybeUninit<Self::Digest>) {
		let finalized = self.finalize_packed();
		out.write(Self::Digest::from_fn(|i| F::from(finalized.get(i))));
		self.reset();
	}

	fn reset(&mut self) {
		*self = Self::new();
	}
}

/// A compression function for Grøstl hash digests based on the Grøstl output transformation.
///
/// This is a 512-bit to 256-bit compression function. This does _not_ apply the full Grøstl hash
/// algorithm to a 512-bit input. Instead, this compression function applies just the Grøstl output
/// transformation, which is believed to be one-way and collision-resistant.
///
/// ## Security justification
///
/// The Grøstl output transformation in [Grøstl] Section 3.3 is argued to be one-way and
/// collision-resistant in multiple ways. First, in Section 4.6, the authors argue that the output
/// transformation is an instance of the Matyas-Meyer-Oseas construction followed by a truncation.
/// Second, in Section 5.1, the authors show that the output transformation is a call to the
/// 1024-to-512-bit compression function on a 0-padded input followed by an XOR with a constant and
/// a truncation.
///
/// [Grøstl]: <https://www.groestl.info/Groestl.pdf>
#[derive(Debug, Default, Clone)]
pub struct GroestlDigestCompression<F: BinaryField + From<AESTowerField8b> + Into<AESTowerField8b>>
{
	_f_marker: PhantomData<F>,
}

impl<F> PseudoCompressionFunction<GroestlDigest<F>, 2> for GroestlDigestCompression<F>
where
	OptimalUnderlier256b: PackScalar<F> + Divisible<F::Underlier>,
	F: BinaryField + From<AESTowerField8b> + Into<AESTowerField8b>,
{
	fn compress(&self, input: [GroestlDigest<F>; 2]) -> GroestlDigest<F> {
		let input_as_slice_bin: [F; 64] = PackedFieldIndexable::unpack_scalars(&input)
			.try_into()
			.unwrap();
		let input_as_slice: [AESTowerField8b; 64] = input_as_slice_bin.map(Into::into);
		let mut state = PackedAESBinaryField64x8b::default();
		let state_as_slice = PackedFieldIndexable::unpack_scalars_mut(slice::from_mut(&mut state));
		state_as_slice.copy_from_slice(&input_as_slice);
		let new_state = Groestl256Core.permutation_p(state) + state;

		let new_state_slice: [AESTowerField8b; 32] =
			PackedFieldIndexable::unpack_scalars(slice::from_ref(&new_state))[32..]
				.try_into()
				.unwrap();
		let new_state_slice_bin: [F; 32] = new_state_slice.map(F::from);
		let mut out_bin = GroestlDigest::<F>::default();
		let out_bin_slice = PackedFieldIndexable::unpack_scalars_mut(slice::from_mut(&mut out_bin));
		out_bin_slice.copy_from_slice(&new_state_slice_bin);
		out_bin
	}
}

impl<F> CompressionFunction<GroestlDigest<F>, 2> for GroestlDigestCompression<F>
where
	OptimalUnderlier256b: PackScalar<F> + Divisible<F::Underlier>,
	F: BinaryField + From<AESTowerField8b> + Into<AESTowerField8b>,
{
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{HashDigest, HasherDigest};
	use binius_field::{
		linear_transformation::Transformation, make_aes_to_binary_packed_transformer,
		PackedBinaryField32x8b, PackedBinaryField64x8b,
	};
	use rand::thread_rng;
	use std::array;

	#[test]
	fn test_groestl_digest_compression() {
		let zero_perm = Groestl256Core.permutation_p(PackedAESBinaryField64x8b::default());
		let aes_to_bin_transform = make_aes_to_binary_packed_transformer::<
			PackedAESBinaryField64x8b,
			PackedBinaryField64x8b,
		>();
		let zero_perm_bin = aes_to_bin_transform.transform(&zero_perm);
		let digest = GroestlDigestCompression::<BinaryField8b>::default().compress([
			GroestlDigest::<BinaryField8b>::default(),
			GroestlDigest::<BinaryField8b>::default(),
		]);
		for (a, b) in digest.iter().zip(zero_perm_bin.iter().skip(32)) {
			assert_eq!(a, b);
		}
	}

	#[test]
	fn test_aes_binary_convertion() {
		let mut rng = thread_rng();
		let input_aes: [PackedAESBinaryField32x8b; 90] =
			array::from_fn(|_| PackedAESBinaryField32x8b::random(&mut rng));
		let input_bin: [PackedBinaryField32x8b; 90] = array::from_fn(|i| {
			let vec_bin = input_aes[i]
				.iter()
				.map(BinaryField8b::from)
				.collect::<Vec<_>>();
			PackedBinaryField32x8b::from_fn(|j| vec_bin[j])
		});

		let digest_aes = HasherDigest::<_, Groestl256<_, AESTowerField8b>>::hash(input_aes);
		let digest_bin = HasherDigest::<_, Groestl256<_, BinaryField8b>>::hash(input_bin);

		let digest_aes_bin = digest_aes
			.iter()
			.map(BinaryField8b::from)
			.collect::<Vec<_>>();
		assert_eq!(PackedBinaryField32x8b::from_fn(|j| digest_aes_bin[j]), digest_bin);
	}
}
