// Copyright 2024 Ulvetanna Inc.

use binius_field::{AESTowerField8b, PackedAESBinaryField64x8b, PackedField};
use digest::{
	block_buffer::Eager,
	core_api::{
		AlgorithmName, Block, BlockSizeUser, Buffer, BufferKindUser, CoreWrapper,
		CtVariableCoreWrapper, TruncSide, UpdateCore, VariableOutputCore,
	},
	typenum::{Unsigned, U32, U64},
	HashMarker, InvalidOutputSize, Output, OutputSizeUser,
};
use std::{arch::x86_64::*, array, fmt, mem::transmute_copy};

pub type GroestlShortCore<OutSize> = CtVariableCoreWrapper<Groestl256AVX512, OutSize>;

/// Groestl-256 hasher state.
pub type Groestl256 = CoreWrapper<GroestlShortCore<U32>>;

const HASH_SIZE: usize = 256 / 8;

const ROUND_SIZE: usize = 10;

#[repr(align(64))]
struct AlignedArray([u8; 64]);

impl Default for AlignedArray {
	fn default() -> Self {
		AlignedArray([0; 64])
	}
}

impl From<__m512i> for AlignedArray {
	fn from(value: __m512i) -> Self {
		let mut out = AlignedArray([0; 64]);
		unsafe { _mm512_store_si512(transmute_copy(&out.0[..].as_mut_ptr()), value) };
		out
	}
}

impl From<AlignedArray> for __m512i {
	fn from(value: AlignedArray) -> Self {
		unsafe { _mm512_load_si512(transmute_copy(&value.0[..].as_ptr())) }
	}
}

impl From<&AlignedArray> for __m512i {
	fn from(value: &AlignedArray) -> Self {
		unsafe { _mm512_load_si512(transmute_copy(&value.0[..].as_ptr())) }
	}
}

const SHIFT_ARRAY_P: AlignedArray = AlignedArray([
	0x00, 0x09, 0x12, 0x1b, 0x24, 0x2d, 0x36, 0x3f, 0x08, 0x11, 0x1a, 0x23, 0x2c, 0x35, 0x3e, 0x07,
	0x10, 0x19, 0x22, 0x2b, 0x34, 0x3d, 0x06, 0x0f, 0x18, 0x21, 0x2a, 0x33, 0x3c, 0x05, 0x0e, 0x17,
	0x20, 0x29, 0x32, 0x3b, 0x04, 0x0d, 0x16, 0x1f, 0x28, 0x31, 0x3a, 0x03, 0x0c, 0x15, 0x1e, 0x27,
	0x30, 0x39, 0x02, 0x0b, 0x14, 0x1d, 0x26, 0x2f, 0x38, 0x01, 0x0a, 0x13, 0x1c, 0x25, 0x2e, 0x37,
]);

const SHIFT_ARRAY_Q: AlignedArray = AlignedArray([
	0x08, 0x19, 0x2a, 0x3b, 0x04, 0x15, 0x26, 0x37, 0x10, 0x21, 0x32, 0x03, 0x0c, 0x1d, 0x2e, 0x3f,
	0x18, 0x29, 0x3a, 0x0b, 0x14, 0x25, 0x36, 0x07, 0x20, 0x31, 0x02, 0x13, 0x1c, 0x2d, 0x3e, 0x0f,
	0x28, 0x39, 0x0a, 0x1b, 0x24, 0x35, 0x06, 0x17, 0x30, 0x01, 0x12, 0x23, 0x2c, 0x3d, 0x0e, 0x1f,
	0x38, 0x09, 0x1a, 0x2b, 0x34, 0x05, 0x16, 0x27, 0x00, 0x11, 0x22, 0x33, 0x3c, 0x0d, 0x1e, 0x2f,
]);

#[inline]
fn xor_blocks(a: __m512i, b: __m512i) -> __m512i {
	unsafe { _mm512_xor_si512(a, b) }
}

const INDEX: AlignedArray = AlignedArray([
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]);

#[derive(Clone, Default)]
pub struct Groestl256Core;

#[derive(Clone)]
pub struct Groestl256AVX512 {
	blocks_len: u64,
	state: __m512i,
	permutation: Groestl256Core,
}

impl Groestl256Core {
	#[inline]
	fn mix_bytes(&self, block: __m512i) -> __m512i {
		let b_adj_1: __m512i = unsafe { _mm512_ror_epi64(block, 8) };
		let x: __m512i = xor_blocks(block, b_adj_1);

		let x_adj_3: __m512i = unsafe { _mm512_ror_epi64(x, 24) };
		let y: __m512i = xor_blocks(x, x_adj_3);

		let x_adj_2: __m512i = unsafe { _mm512_ror_epi64(x, 16) };

		let b_adj_6: __m512i = unsafe { _mm512_ror_epi64(block, 48) };

		let z: __m512i = xor_blocks(x, x_adj_2);
		let z: __m512i = xor_blocks(z, b_adj_6);

		let z_adj_7: __m512i = unsafe { _mm512_ror_epi64(z, 56) };
		let z_adj_4: __m512i = unsafe { _mm512_ror_epi64(z, 32) };
		let y_adj_3: __m512i = unsafe { _mm512_ror_epi64(y, 24) };

		let _2: __m512i = unsafe { _mm512_set1_epi8(2) };
		let first_mul: __m512i = unsafe { _mm512_gf2p8mul_epi8(_2, y_adj_3) };
		let mul_2_z_adj_7: __m512i = unsafe { _mm512_xor_si512(first_mul, z_adj_7) };
		let second_mul: __m512i = unsafe { _mm512_gf2p8mul_epi8(_2, mul_2_z_adj_7) };

		xor_blocks(second_mul, z_adj_4)
	}

	#[inline]
	fn sub_bytes(&self, block: __m512i) -> __m512i {
		// The affine transformation can be build from 8 u64's
		const SBOX_AFFINE: i64 = 0xf1e3c78f1f3e7cf8u64 as i64;

		let a: __m512i = unsafe { _mm512_set1_epi64(SBOX_AFFINE) };

		unsafe { _mm512_gf2p8affineinv_epi64_epi8(block, a, 0b01100011) }
	}

	#[inline]
	fn shift_bytes(&self, block: __m512i, shift: &AlignedArray) -> __m512i {
		let idx: __m512i = unsafe { _mm512_load_si512(transmute_copy(&shift.0.as_ptr())) };

		unsafe { _mm512_permutexvar_epi8(idx, block) }
	}

	#[inline]
	fn add_round_constants_p(&self, block: __m512i, r: u8) -> __m512i {
		let round_reg: __m512i = unsafe { _mm512_set1_epi64(r as i64) };

		// The compiler gets rid of all these instruction into just a mov
		let block_idx: __m512i = unsafe { _mm512_set1_epi64(0x10) };
		let idx_one: __m512i = INDEX.into();
		let block_idx: __m512i = unsafe { _mm512_mullox_epi64(idx_one, block_idx) };

		let res = xor_blocks(block_idx, round_reg);
		xor_blocks(res, block)
	}

	#[inline]
	fn add_round_constants_q(&self, block: __m512i, r: u8) -> __m512i {
		let round_reg: __m512i = unsafe { _mm512_set1_epi64((r as i64) << 56) };

		let block_idx: __m512i = unsafe { _mm512_set1_epi64(0x10 << 56) };
		let idx_one: __m512i = INDEX.into();
		let block_idx: __m512i = unsafe { _mm512_mullox_epi64(idx_one, block_idx) };

		// first we need to xor by 0xff
		let block: __m512i = unsafe { _mm512_ternarylogic_epi32(block, block, block, 0b01010101) };
		let res = xor_blocks(block_idx, round_reg);
		xor_blocks(res, block)
	}

	fn perm_p_m512i(&self, block: __m512i) -> __m512i {
		let mut block = block;
		for r in 0..ROUND_SIZE {
			block = self.add_round_constants_p(block, r as u8);
			block = self.sub_bytes(block);
			block = self.shift_bytes(block, &SHIFT_ARRAY_P);
			block = self.mix_bytes(block);
		}
		block
	}

	fn combined_perm_m512i(&self, p_block: __m512i, q_block: __m512i) -> (__m512i, __m512i) {
		let mut p_block = p_block;
		let mut q_block = q_block;
		for r in 0..ROUND_SIZE {
			p_block = self.add_round_constants_p(p_block, r as u8);
			q_block = self.add_round_constants_q(q_block, r as u8);
			p_block = self.sub_bytes(p_block);
			q_block = self.sub_bytes(q_block);
			p_block = self.shift_bytes(p_block, &SHIFT_ARRAY_P);
			q_block = self.shift_bytes(q_block, &SHIFT_ARRAY_Q);
			p_block = self.mix_bytes(p_block);
			q_block = self.mix_bytes(q_block);
		}

		(q_block, p_block)
	}

	#[inline]
	pub fn permutation_p(&self, p: PackedAESBinaryField64x8b) -> PackedAESBinaryField64x8b {
		let input = AlignedArray(array::from_fn(|i| p.get(i).val()));
		let out: AlignedArray = self.perm_p_m512i(input.into()).into();

		PackedAESBinaryField64x8b::from_fn(|i| AESTowerField8b::new(out.0[i]))
	}

	#[inline]
	pub fn permutation_pq(
		&self,
		p: PackedAESBinaryField64x8b,
		q: PackedAESBinaryField64x8b,
	) -> (PackedAESBinaryField64x8b, PackedAESBinaryField64x8b) {
		let p_align = AlignedArray(array::from_fn(|i| p.get(i).val()));
		let q_align = AlignedArray(array::from_fn(|i| q.get(i).val()));

		let (p_out_arr, q_out_arr) = self.combined_perm_m512i(p_align.into(), q_align.into());
		let p_out_arr: AlignedArray = p_out_arr.into();
		let q_out_arr: AlignedArray = q_out_arr.into();

		let p_out = PackedAESBinaryField64x8b::from_fn(|i| AESTowerField8b::new(p_out_arr.0[i]));
		let q_out = PackedAESBinaryField64x8b::from_fn(|i| AESTowerField8b::new(q_out_arr.0[i]));

		(p_out, q_out)
	}
}

impl Groestl256AVX512 {
	#[inline]
	pub fn compression_func(&self, h: __m512i, m: __m512i) -> __m512i {
		let (a, b) = self.permutation.combined_perm_m512i(xor_blocks(h, m), m);
		xor_blocks(xor_blocks(a, b), h)
	}
}

impl HashMarker for Groestl256AVX512 {}

impl BlockSizeUser for Groestl256AVX512 {
	type BlockSize = U64;
}

impl BufferKindUser for Groestl256AVX512 {
	type BufferKind = Eager;
}

impl OutputSizeUser for Groestl256AVX512 {
	type OutputSize = U32;
}

impl UpdateCore for Groestl256AVX512 {
	#[inline]
	fn update_blocks(&mut self, blocks: &[Block<Self>]) {
		self.blocks_len += blocks.len() as u64;
		for block in blocks {
			let block: &[u8; 64] = block.as_ref();
			let block = unsafe { _mm512_loadu_epi8(transmute_copy(&block.as_ptr())) };
			self.state = self.compression_func(self.state, block);
		}
	}
}

impl VariableOutputCore for Groestl256AVX512 {
	const TRUNC_SIDE: TruncSide = TruncSide::Right;

	#[inline]
	fn new(output_size: usize) -> Result<Self, InvalidOutputSize> {
		if output_size > Self::OutputSize::USIZE {
			return Err(InvalidOutputSize);
		}
		let mut state = AlignedArray([0; 64]);
		let iv: u64 = 8 * output_size as u64;
		state.0[56..64].copy_from_slice(&iv.to_be_bytes());
		let blocks_len = 0;
		Ok(Self {
			permutation: Groestl256Core::default(),
			state: state.into(),
			blocks_len,
		})
	}

	#[inline]
	fn finalize_variable_core(&mut self, buffer: &mut Buffer<Self>, out: &mut Output<Self>) {
		let blocks_len = if buffer.remaining() <= 8 {
			self.blocks_len + 2
		} else {
			self.blocks_len + 1
		};
		buffer.len64_padding_be(blocks_len, |block| {
			let block: &[u8; 64] = block.as_ref();
			let block = unsafe { _mm512_loadu_epi8(transmute_copy(&block.as_ptr())) };
			self.state = self.compression_func(self.state, block)
		});
		let new_state = self.permutation.perm_p_m512i(self.state);
		let res: AlignedArray = xor_blocks(new_state, self.state).into();
		out.copy_from_slice(&res.0[HASH_SIZE..64]);
	}
}
impl AlgorithmName for Groestl256AVX512 {
	#[inline]
	fn write_alg_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("Groestl256AVX512")
	}
}

impl fmt::Debug for Groestl256AVX512 {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("Groestl256AVX512 { ... }")
	}
}
