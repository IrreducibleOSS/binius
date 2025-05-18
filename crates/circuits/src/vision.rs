// Copyright 2024-2025 Irreducible Inc.

//! Example of a Binius SNARK that proves execution of [Vision Mark-32] permutations.
//!
//! The arithmetization uses committed columns of 32-bit binary tower field elements. Every row of
//! the trace attests to the validity of 2 Vision rounds. Each permutation consists of 16 rounds.
//!
//! [Vision Mark-32]: https://eprint.iacr.org/2024/633

use std::array;

use anyhow::Result;
use binius_core::{oracle::OracleId, transparent::constant::Constant};
use binius_field::{
	BinaryField1b, BinaryField32b, ExtensionField, Field, PackedAESBinaryField8x32b,
	PackedBinaryField8x32b, PackedExtension, PackedField, TowerField,
	linear_transformation::Transformation, make_aes_to_binary_packed_transformer,
	packed::get_packed_slice,
};
use binius_hash::{INV_PACKED_TRANS_AES, Vision32MDSTransform};
use binius_macros::arith_expr;
use binius_math::{ArithCircuit, ArithExpr};
use bytemuck::must_cast_slice;

use crate::builder::{ConstraintSystemBuilder, types::F};

pub fn vision_permutation(
	builder: &mut ConstraintSystemBuilder,
	log_size: usize,
	p_in: [OracleId; STATE_SIZE],
) -> Result<[OracleId; STATE_SIZE]> {
	// This only acts as a shorthand
	type B32 = BinaryField32b;

	let round_0_input = builder.add_committed_multiple::<STATE_SIZE>(
		"round_0_input",
		log_size,
		BinaryField32b::TOWER_LEVEL,
	);

	if let Some(witness) = builder.witness() {
		let perm_in_data_owned: [_; STATE_SIZE] =
			array_util::try_from_fn(|i| witness.get::<B32>(p_in[i]))?;
		let perm_in_data: [_; STATE_SIZE] = perm_in_data_owned.map(|elem| elem.as_slice::<B32>());
		let mut round_0_input_data: [_; STATE_SIZE] =
			round_0_input.map(|id| witness.new_column::<B32>(id));
		let round_0_input_slice = round_0_input_data
			.each_mut()
			.map(|elem| elem.as_mut_slice::<B32>());

		for s in 0..STATE_SIZE {
			for z in 0..1 << log_size {
				round_0_input_slice[s][z] = perm_in_data[s][z] + B32::new(VISION_ROUND_0[s]);
			}
		}
	}

	for s in 0..STATE_SIZE {
		builder.assert_zero(
			format!("vision_round_begin_{s}"),
			[p_in[s], round_0_input[s]],
			vision_round_begin_expr(s).convert_field(),
		);
	}

	let perm_out = (0..N_ROUNDS).try_fold(round_0_input, |state, round_i| {
		vision_round(builder, log_size, round_i, state)
	})?;

	#[cfg(debug_assertions)]
	if let Some(witness) = builder.witness() {
		use binius_hash::{Vision32bPermutation, permutation::Permutation};

		let vision_perm = Vision32bPermutation::default();
		let p_in_data: [_; STATE_SIZE] =
			array_util::try_from_fn(|i| witness.get::<B32>(p_in[i])).unwrap();
		let p_in_slice: [_; STATE_SIZE] = p_in_data.map(|elem| elem.as_slice::<B32>());
		let p_out_data: [_; STATE_SIZE] =
			array_util::try_from_fn(|i| witness.get::<B32>(perm_out[i])).unwrap();
		let p_out_slice: [_; STATE_SIZE] = p_out_data.map(|elem| elem.as_slice::<B32>());
		for z in 0..1 << log_size {
			let mut in_out: [_; 3] = array::from_fn(|i| {
				PackedAESBinaryField8x32b::from_fn(|j| p_in_slice[i * 8 + j][z].into())
			});
			let expected_out: [B32; STATE_SIZE] = array::from_fn(|s| p_out_slice[s][z]);

			vision_perm.permute_mut(&mut in_out);

			for (out, expected) in
				PackedAESBinaryField8x32b::iter_slice(&in_out).zip(expected_out.iter())
			{
				assert_eq!(out, binius_field::AESTowerField32b::from(*expected));
			}
		}
	}

	Ok(perm_out)
}

const N_ROUNDS: usize = 8;
const STATE_SIZE: usize = 24;

#[rustfmt::skip]
const VISION_RC_EVEN: [[u32; 8]; STATE_SIZE] = [
	[0x73fa03e1, 0x8bd2f341, 0x89841f23, 0xd6561783, 0x4e28a23c, 0x52538d7d, 0xd1504060, 0x00d80bd4],
	[0x2551a651, 0x59dc2758, 0x8bd0c3e1, 0x88153c99, 0xdbe6f0db, 0xdd441420, 0x005d8a96, 0x3d8b3d56],
	[0x0541031f, 0x5146c720, 0xde2dd62b, 0x1a04e141, 0x9cf4faeb, 0x38a2e2d5, 0x058e317a, 0xcc18a7a9],
	[0xec1d59dc, 0x9df43021, 0x37799416, 0x62631076, 0x2fde2616, 0xccd05f31, 0x30d9d3c6, 0x0105e9bb],
	[0x780f0b43, 0x0d1c49ea, 0x558834c7, 0xb20b52a2, 0x22dedea1, 0x2a49f3a6, 0xa585af56, 0x71f0e736],
	[0x04843f97, 0x81d4b0a5, 0x939df560, 0x1df18264, 0x08ef118e, 0xe533cc9b, 0x084c5111, 0x4cc71fa4],
	[0xd379e20b, 0xdbfae4d1, 0xb1a9f457, 0x05176f17, 0xd7f16ae2, 0xa18de92e, 0x498da85e, 0x1a2ec96b],
	[0xbe4d1f58, 0xc3153118, 0xcb24dadb, 0x505b2752, 0xa13b30a8, 0x495f684a, 0x0149987d, 0xe1b8b093],
	[0xe4c2f8bb, 0x8a3aec81, 0x4f702a2a, 0x914a71aa, 0x2ceb58c1, 0x0028e3ae, 0xe130153b, 0x329232ab],
	[0xf29aee17, 0xeacd8854, 0x65ad5822, 0x1b6cf96d, 0xca587d86, 0xd4072861, 0x817cc725, 0xb4285526],
	[0x228e51f2, 0xdd4b2576, 0x7ecf577d, 0x5a8b3b59, 0xf6d54fcd, 0x370fd7a3, 0x75f726b1, 0x02326fe9],
	[0x840ee72b, 0x7dd5cee9, 0x728b4092, 0x3ab885cc, 0x9cd9f3f5, 0x728224bc, 0x23941339, 0xe79accab],
	[0x0cb3b70e, 0x5e9e77b7, 0x89e4fa7d, 0xed662f24, 0x9b0f94a2, 0xa8b6b3d7, 0x1f26e9dd, 0xd893b618],
	[0xbacc914a, 0x6b6efd8d, 0x10cd7556, 0xa859f626, 0xdede0863, 0xdada7046, 0xdb013723, 0x9bd74bd5],
	[0x490bfa7e, 0xf11db400, 0x1de77ab7, 0xd91136bb, 0xa608eb2d, 0xea9e71df, 0x81f36069, 0x2062577c],
	[0xc2c3018e, 0x0e6258b7, 0x2374c530, 0x6da2d95b, 0x4d3c4469, 0x914f7d53, 0xe4167ba1, 0x94f82da9],
	[0xf6d13bd2, 0x37b3b6e3, 0x95b289d4, 0x043fd679, 0x53784235, 0x9b796ac9, 0x50d59f82, 0xb551d97a],
	[0x6a4d1fe1, 0xed884c61, 0xa6ad3862, 0xb9e685e8, 0x4cf6aa1e, 0xe7f61a69, 0xbf011350, 0x862483f0],
	[0x4c2bc742, 0xb948717c, 0xc6b1a233, 0xdf796fa5, 0xcb6ec0d5, 0x67a68f71, 0x3ae71f42, 0x5f8e4e3e],
	[0x4508cb46, 0x3d7554cf, 0xac501639, 0x53fc28a3, 0xf334b49e, 0x7eb15ce6, 0x9966d041, 0x098d5e44],
	[0xed63a2f1, 0x42419311, 0x3f6072a3, 0x0c15dc77, 0xe5f7a67a, 0xeb9af9e1, 0xdbe09577, 0xbe326102],
	[0x1802f859, 0x422d11d3, 0xf8ae7cc4, 0x079255d2, 0x989658a2, 0xa75f54b1, 0xa830b8f0, 0x4f5f050e],
	[0xa00483b5, 0x5392b2e7, 0x622f4cf3, 0x3373a2a0, 0xa1a672ca, 0x59210427, 0x0c018c2d, 0x1bd571d5],
	[0x56e12e78, 0x79c1591d, 0xf7ccf75b, 0xfc6b012e, 0x6fb7eced, 0x75093378, 0x08beab4f, 0xcdd8e583],
];

#[rustfmt::skip]
const VISION_RC_ODD: [[u32; 8]; STATE_SIZE] = [
	[0xbace7a4a, 0x27df48ae, 0xaedf6aac, 0xb3359ff0, 0x2bbdf7b8, 0x27866fea, 0x20898252, 0x1b525e1b],
	[0xc3a71400, 0x948bc10e, 0xd64356b2, 0xa471acdc, 0xa8626256, 0x3bd84dca, 0xac8aa337, 0x1cccb851],
	[0x5a29b316, 0xcb079dc1, 0x1cbba169, 0x6ad3e18a, 0xd95bf688, 0x681d1d3a, 0x5c5bbcad, 0x45b3c777],
	[0xeedc8d26, 0xed183a37, 0x688602ae, 0x4f012f65, 0x43245a87, 0xe7fb7496, 0x2fa58f41, 0x63cc9153],
	[0x51c14d7e, 0x81dcc076, 0x6231b358, 0xebd4392f, 0xc14af030, 0x86fd9bf8, 0xf2446068, 0xdfa0fd4a],
	[0x2add9be8, 0x24cb0490, 0x1fba8b86, 0x25d3af23, 0x28e5933a, 0xc1f28786, 0xfff46a79, 0x0cf20c06],
	[0xfec386f3, 0x52d69fb8, 0xf7b83f1c, 0x7a68469c, 0x3aeb3e0d, 0xb3f17a06, 0x0b1980d8, 0x72fdd2f3],
	[0x630765dc, 0x8b576666, 0x465c4050, 0xd479ea57, 0x169f7dea, 0x60c43dbe, 0x01b14c53, 0xf9b6f564],
	[0xaef6c21b, 0x7499fe4d, 0x4403e74c, 0xb55b6450, 0x4cd4d1e4, 0x16fee1be, 0x4e432072, 0x9552a62b],
	[0x8c98fc1a, 0x8f879e34, 0x5f51c2f3, 0x86ef0a15, 0x8db556b5, 0xa8407554, 0xfc610a31, 0x1e848099],
	[0x3f9c4f9d, 0xcb11780a, 0x1b114a4d, 0xeefd412f, 0xdd1a49ea, 0xca909e3b, 0x80ba5531, 0x3ba1a5a6],
	[0x399e7231, 0x5e876b29, 0x8f32bf48, 0xc8e98f30, 0xe64eff5d, 0xb1fc461c, 0xc14507a5, 0x17ff06e0],
	[0xba238b04, 0xb72d96ab, 0x87990cfc, 0x61e0c12d, 0x8bd56648, 0xd84d663e, 0x2433c5d2, 0x8cae82ed],
	[0x787d67ec, 0xac28e621, 0x71b55cb1, 0x36c4680c, 0x2c3422be, 0x2e7d669b, 0x8a461cf3, 0xb5b29fbc],
	[0x313ad8af, 0x18aeca7e, 0x73083164, 0xe818ab96, 0x5cffb53f, 0x5b5b5a56, 0x187849cd, 0x9322d5a6],
	[0xdd622ac3, 0xf3d30baf, 0x2fbd58ae, 0xfcb765f2, 0x6b7aaa6e, 0x6c53d090, 0x3d4f51e8, 0x77f40c4c],
	[0xe0a8d9b8, 0xc7fca53f, 0x59bbcbbf, 0xcbb47fea, 0xc2a8d1af, 0x236707a6, 0x3d9cd125, 0x0843ce60],
	[0xaa0e6306, 0xf7b3281a, 0xb0dc1eba, 0xc9e202a8, 0x7e79bed4, 0x7f1f4e97, 0xe15e09ca, 0x86ddb97f],
	[0x29864574, 0xdaf5559f, 0xf2f169ff, 0xc762caec, 0xd0b08e51, 0xe95b23f3, 0x8c6287c6, 0xe5a12a04],
	[0x67ee41da, 0x27aca0b3, 0x54cc93e8, 0x366f08fd, 0x1861ba54, 0x8cd1e3dd, 0xfa0ec2f4, 0x9bd65cd6],
	[0x5502278d, 0x9515d3ee, 0x975cfc83, 0x5e2f3a19, 0xb7d3c6b4, 0x928f3212, 0x65435f29, 0x1b16bea6],
	[0xa92e20b1, 0xa39fd2e1, 0xbefc67cf, 0x242c8397, 0x6a9bd7ca, 0x9c7c1c20, 0xd33a4f3d, 0xf4066cee],
	[0x0fdc5328, 0xf61b52c2, 0xb841429b, 0x638a0042, 0x129d3aa5, 0x00eeebe3, 0xd61bb963, 0xdcb3c788],
	[0x74dbee7a, 0x83ec5a0f, 0xff127d64, 0x63f1c9c5, 0x809e9413, 0xc0572f52, 0x991005f9, 0x499b6483],
];

#[rustfmt::skip]
const VISION_ROUND_0: [u32; STATE_SIZE] = [0x545e66a7, 0x073fdd58, 0x84362677, 0x95fe8565, 0x06269cd8, 0x9c17909e, 0xf1f0adee, 0x2694c698, 0x94b2788f, 0x5eac14ad, 0x21677a78, 0x5755730b, 0x37cef9cf, 0x2fb31ffe, 0xfc0082ec, 0x609c12f0, 0x102769ee, 0x4732860d, 0xf97935e0, 0x36e77c02, 0xba9e70df, 0x67b701d7, 0x829d77a4, 0xf6ec454d];

const SBOX_FWD_TRANS: [BinaryField32b; 3] = [
	BinaryField32b::new(0xdb43e603),
	BinaryField32b::new(0x391c8e32),
	BinaryField32b::new(0x9fd55d88),
];

const SBOX_FWD_CONST: BinaryField32b = BinaryField32b::new(0x7cf0bc6c);
const SBOX_INV_CONST: BinaryField32b = BinaryField32b::new(0x9fa712f2);

#[rustfmt::skip]
const MDS_TRANS: [[u8; STATE_SIZE]; STATE_SIZE] = [
	[0xad, 0x3b, 0xd4, 0x25, 0xab, 0x37, 0xd7, 0x2d, 0x9a, 0x4d, 0x6a, 0xd8, 0x90, 0x44, 0x6b, 0xdb, 0x06, 0x0f, 0x0e, 0x04, 0x0d, 0x0c, 0x0a, 0x09],
	[0x3b, 0xad, 0x25, 0xd4, 0x37, 0xab, 0x2d, 0xd7, 0x4d, 0x9a, 0xd8, 0x6a, 0x44, 0x90, 0xdb, 0x6b, 0x0f, 0x06, 0x04, 0x0e, 0x0c, 0x0d, 0x09, 0x0a],
	[0xd4, 0x25, 0xad, 0x3b, 0xd7, 0x2d, 0xab, 0x37, 0x6a, 0xd8, 0x9a, 0x4d, 0x6b, 0xdb, 0x90, 0x44, 0x0e, 0x04, 0x06, 0x0f, 0x0a, 0x09, 0x0d, 0x0c],
	[0x25, 0xd4, 0x3b, 0xad, 0x2d, 0xd7, 0x37, 0xab, 0xd8, 0x6a, 0x4d, 0x9a, 0xdb, 0x6b, 0x44, 0x90, 0x04, 0x0e, 0x0f, 0x06, 0x09, 0x0a, 0x0c, 0x0d],
	[0xab, 0x37, 0xd7, 0x2d, 0xad, 0x3b, 0xd4, 0x25, 0x90, 0x44, 0x6b, 0xdb, 0x9a, 0x4d, 0x6a, 0xd8, 0x0d, 0x0c, 0x0a, 0x09, 0x06, 0x0f, 0x0e, 0x04],
	[0x37, 0xab, 0x2d, 0xd7, 0x3b, 0xad, 0x25, 0xd4, 0x44, 0x90, 0xdb, 0x6b, 0x4d, 0x9a, 0xd8, 0x6a, 0x0c, 0x0d, 0x09, 0x0a, 0x0f, 0x06, 0x04, 0x0e],
	[0xd7, 0x2d, 0xab, 0x37, 0xd4, 0x25, 0xad, 0x3b, 0x6b, 0xdb, 0x90, 0x44, 0x6a, 0xd8, 0x9a, 0x4d, 0x0a, 0x09, 0x0d, 0x0c, 0x0e, 0x04, 0x06, 0x0f],
	[0x2d, 0xd7, 0x37, 0xab, 0x25, 0xd4, 0x3b, 0xad, 0xdb, 0x6b, 0x44, 0x90, 0xd8, 0x6a, 0x4d, 0x9a, 0x09, 0x0a, 0x0c, 0x0d, 0x04, 0x0e, 0x0f, 0x06],
	[0xa9, 0x0f, 0x7d, 0x24, 0x23, 0x14, 0x45, 0xed, 0x54, 0xdf, 0x62, 0xc0, 0x67, 0xf8, 0x22, 0xf7, 0xd5, 0x47, 0x06, 0xf2, 0x93, 0x83, 0x8b, 0xff],
	[0x0f, 0xa9, 0x24, 0x7d, 0x14, 0x23, 0xed, 0x45, 0xdf, 0x54, 0xc0, 0x62, 0xf8, 0x67, 0xf7, 0x22, 0x47, 0xd5, 0xf2, 0x06, 0x83, 0x93, 0xff, 0x8b],
	[0x7d, 0x24, 0xa9, 0x0f, 0x45, 0xed, 0x23, 0x14, 0x62, 0xc0, 0x54, 0xdf, 0x22, 0xf7, 0x67, 0xf8, 0x06, 0xf2, 0xd5, 0x47, 0x8b, 0xff, 0x93, 0x83],
	[0x24, 0x7d, 0x0f, 0xa9, 0xed, 0x45, 0x14, 0x23, 0xc0, 0x62, 0xdf, 0x54, 0xf7, 0x22, 0xf8, 0x67, 0xf2, 0x06, 0x47, 0xd5, 0xff, 0x8b, 0x83, 0x93],
	[0x23, 0x14, 0x45, 0xed, 0xa9, 0x0f, 0x7d, 0x24, 0x67, 0xf8, 0x22, 0xf7, 0x54, 0xdf, 0x62, 0xc0, 0x93, 0x83, 0x8b, 0xff, 0xd5, 0x47, 0x06, 0xf2],
	[0x14, 0x23, 0xed, 0x45, 0x0f, 0xa9, 0x24, 0x7d, 0xf8, 0x67, 0xf7, 0x22, 0xdf, 0x54, 0xc0, 0x62, 0x83, 0x93, 0xff, 0x8b, 0x47, 0xd5, 0xf2, 0x06],
	[0x45, 0xed, 0x23, 0x14, 0x7d, 0x24, 0xa9, 0x0f, 0x22, 0xf7, 0x67, 0xf8, 0x62, 0xc0, 0x54, 0xdf, 0x8b, 0xff, 0x93, 0x83, 0x06, 0xf2, 0xd5, 0x47],
	[0xed, 0x45, 0x14, 0x23, 0x24, 0x7d, 0x0f, 0xa9, 0xf7, 0x22, 0xf8, 0x67, 0xc0, 0x62, 0xdf, 0x54, 0xff, 0x8b, 0x83, 0x93, 0xf2, 0x06, 0x47, 0xd5],
	[0xaf, 0x0f, 0x78, 0x2c, 0x2b, 0x10, 0x4c, 0xe2, 0x59, 0xdc, 0x63, 0xc7, 0x66, 0xf3, 0x2a, 0xfc, 0x99, 0x8d, 0x85, 0xf4, 0xd6, 0x4e, 0x06, 0xf9],
	[0x0f, 0xaf, 0x2c, 0x78, 0x10, 0x2b, 0xe2, 0x4c, 0xdc, 0x59, 0xc7, 0x63, 0xf3, 0x66, 0xfc, 0x2a, 0x8d, 0x99, 0xf4, 0x85, 0x4e, 0xd6, 0xf9, 0x06],
	[0x78, 0x2c, 0xaf, 0x0f, 0x4c, 0xe2, 0x2b, 0x10, 0x63, 0xc7, 0x59, 0xdc, 0x2a, 0xfc, 0x66, 0xf3, 0x85, 0xf4, 0x99, 0x8d, 0x06, 0xf9, 0xd6, 0x4e],
	[0x2c, 0x78, 0x0f, 0xaf, 0xe2, 0x4c, 0x10, 0x2b, 0xc7, 0x63, 0xdc, 0x59, 0xfc, 0x2a, 0xf3, 0x66, 0xf4, 0x85, 0x8d, 0x99, 0xf9, 0x06, 0x4e, 0xd6],
	[0x2b, 0x10, 0x4c, 0xe2, 0xaf, 0x0f, 0x78, 0x2c, 0x66, 0xf3, 0x2a, 0xfc, 0x59, 0xdc, 0x63, 0xc7, 0xd6, 0x4e, 0x06, 0xf9, 0x99, 0x8d, 0x85, 0xf4],
	[0x10, 0x2b, 0xe2, 0x4c, 0x0f, 0xaf, 0x2c, 0x78, 0xf3, 0x66, 0xfc, 0x2a, 0xdc, 0x59, 0xc7, 0x63, 0x4e, 0xd6, 0xf9, 0x06, 0x8d, 0x99, 0xf4, 0x85],
	[0x4c, 0xe2, 0x2b, 0x10, 0x78, 0x2c, 0xaf, 0x0f, 0x2a, 0xfc, 0x66, 0xf3, 0x63, 0xc7, 0x59, 0xdc, 0x06, 0xf9, 0xd6, 0x4e, 0x85, 0xf4, 0x99, 0x8d],
	[0xe2, 0x4c, 0x10, 0x2b, 0x2c, 0x78, 0x0f, 0xaf, 0xfc, 0x2a, 0xf3, 0x66, 0xc7, 0x63, 0xdc, 0x59, 0xf9, 0x06, 0x4e, 0xd6, 0xf4, 0x85, 0x8d, 0x99],
];

fn vision_round_begin_expr(state_idx: usize) -> ArithCircuit<BinaryField32b> {
	assert!(state_idx < STATE_SIZE);
	arith_expr!(BinaryField32b[x, y] = x + y + ArithExpr::Const(BinaryField32b::new(VISION_ROUND_0[state_idx])))
}

fn s_box_linearized_eval_expr() -> ArithCircuit<BinaryField32b> {
	let input = ArithExpr::Var(0);
	let output = ArithExpr::Var(1);
	// TODO: Square for ArithExpr
	let input_pow2 = input.clone().pow(2);
	let input_pow4 = input_pow2.clone().pow(2);

	let result = ArithExpr::Const(SBOX_FWD_CONST)
		+ input * ArithExpr::Const(SBOX_FWD_TRANS[0])
		+ input_pow2 * ArithExpr::Const(SBOX_FWD_TRANS[1])
		+ input_pow4 * ArithExpr::Const(SBOX_FWD_TRANS[2]);

	(result - output).into()
}

fn inv_constraint_expr<F: TowerField>() -> Result<ArithCircuit<F>> {
	let x = ArithExpr::Var(0);
	let inv = ArithExpr::Var(1);

	// x * inv == 1
	let non_zero_case = x.clone() * inv.clone() - ArithExpr::one();

	// x == 0 AND inv == 0
	// TODO: Implement `mul_primitive` expression for ArithExpr
	let beta = <F as ExtensionField<BinaryField1b>>::basis_checked(1 << 5)?;
	let zero_case = x + inv * ArithExpr::Const(beta);

	// (x * inv == 1) OR (x == 0 AND inv == 0)
	Ok((non_zero_case * zero_case).into())
}

fn vision_round(
	builder: &mut ConstraintSystemBuilder,
	log_size: usize,
	round_i: usize,
	perm_in: [OracleId; STATE_SIZE],
) -> Result<[OracleId; STATE_SIZE]>
where {
	builder.push_namespace(format!("round[{round_i}]"));
	let inv_0 = builder.add_committed_multiple::<STATE_SIZE>(
		"inv_evens",
		log_size,
		BinaryField32b::TOWER_LEVEL,
	);
	let s_box_out_0 = builder.add_committed_multiple::<STATE_SIZE>(
		"sbox_out_evens",
		log_size,
		BinaryField32b::TOWER_LEVEL,
	);
	let inv_1 = builder.add_committed_multiple::<STATE_SIZE>(
		"inv_odds",
		log_size,
		BinaryField32b::TOWER_LEVEL,
	);
	let s_box_out_1: [_; STATE_SIZE] =
		builder.add_committed_multiple("sbox_out_odds", log_size, BinaryField32b::TOWER_LEVEL);

	let even_round_consts: [OracleId; STATE_SIZE] = array::from_fn(|i| {
		builder
			.add_transparent(
				format!("even_round_consts_{i}"),
				Constant::new(log_size, BinaryField32b::new(VISION_RC_EVEN[i][round_i])),
			)
			.unwrap()
	});
	let odd_round_consts: [OracleId; STATE_SIZE] = array::from_fn(|i| {
		builder
			.add_transparent(
				format!("odd_round_consts_{i}"),
				Constant::new(log_size, BinaryField32b::new(VISION_RC_ODD[i][round_i])),
			)
			.unwrap()
	});

	let mds_out_0: [_; STATE_SIZE] = array::from_fn(|row| {
		builder
			.add_linear_combination(
				format!("mds_out_evens_{row}"),
				log_size,
				MDS_TRANS[row]
					.iter()
					.enumerate()
					.map(|(i, &elem)| (s_box_out_0[i], F::from(BinaryField32b::new(elem as u32)))),
			)
			.unwrap()
	});
	let mds_out_1: [_; STATE_SIZE] = array::from_fn(|row| {
		builder
			.add_linear_combination(
				format!("mds_out_odds_{row}"),
				log_size,
				MDS_TRANS[row]
					.iter()
					.enumerate()
					.map(|(i, &elem)| (s_box_out_1[i], F::from(BinaryField32b::new(elem as u32)))),
			)
			.unwrap()
	});
	let round_out_0: [_; STATE_SIZE] = array::from_fn(|row| {
		builder
			.add_linear_combination(
				format!("round_out_evens_{row}"),
				log_size,
				[
					(mds_out_0[row], Field::ONE),
					(even_round_consts[row], Field::ONE),
				],
			)
			.unwrap()
	});

	let perm_out = array::from_fn(|row| {
		builder
			.add_linear_combination(
				format!("round_out_odd_{row}"),
				log_size,
				[
					(mds_out_1[row], Field::ONE),
					(odd_round_consts[row], Field::ONE),
				],
			)
			.unwrap()
	});

	builder.pop_namespace();

	let mds_trans = Vision32MDSTransform::default();
	let aes_to_bin_packed =
		make_aes_to_binary_packed_transformer::<PackedAESBinaryField8x32b, PackedBinaryField8x32b>(
		);
	let inv_const_packed = PackedBinaryField8x32b::broadcast(SBOX_INV_CONST);

	type B32 = BinaryField32b;

	// Witness gen
	if let Some(witness) = builder.witness() {
		let perm_in_data_owned: [_; STATE_SIZE] =
			array_util::try_from_fn(|i| witness.get::<B32>(perm_in[i]))?;
		let perm_in_data: [_; STATE_SIZE] = perm_in_data_owned.map(|elem| elem.as_slice::<B32>());

		let mut even_round_consts = even_round_consts.map(|id| witness.new_column::<B32>(id));
		let mut inv_0 = inv_0.map(|id| witness.new_column::<B32>(id));
		let mut s_box_out_0 = s_box_out_0.map(|id| witness.new_column::<B32>(id));
		let mut mds_out_0 = mds_out_0.map(|id| witness.new_column::<B32>(id));
		let mut round_out_0 = round_out_0.map(|id| witness.new_column::<B32>(id));
		let mut odd_round_consts = odd_round_consts.map(|id| witness.new_column::<B32>(id));
		let mut inv_1 = inv_1.map(|id| witness.new_column::<B32>(id));
		let mut s_box_out_1 = s_box_out_1.map(|id| witness.new_column::<B32>(id));
		let mut mds_out_1 = mds_out_1.map(|id| witness.new_column::<B32>(id));
		let mut perm_out = perm_out.map(|id| witness.new_column::<B32>(id));

		let inv_0_slice = inv_0.each_mut().map(|elem| elem.as_mut_slice());
		let s_box_out_0_slice = s_box_out_0.each_mut().map(|elem| elem.as_mut_slice());
		let mds_out_0_slice = mds_out_0.each_mut().map(|elem| elem.as_mut_slice());
		let round_out_0_slice = round_out_0
			.each_mut()
			.map(|elem| elem.as_mut_slice::<B32>());
		let inv_1_slice = inv_1.each_mut().map(|elem| elem.as_mut_slice());
		let s_box_out_1_slice = s_box_out_1.each_mut().map(|elem| elem.as_mut_slice());
		let mds_out_1_slice = mds_out_1.each_mut().map(|elem| elem.as_mut_slice());
		let perm_out_slice = perm_out.each_mut().map(|elem| elem.as_mut_slice());
		let even_round_consts_slice = even_round_consts
			.each_mut()
			.map(|elem| elem.as_mut_slice::<B32>());
		let odd_round_consts_slice = odd_round_consts
			.each_mut()
			.map(|elem| elem.as_mut_slice::<B32>());

		// Fill in constants
		for i in 0..STATE_SIZE {
			even_round_consts_slice[i]
				.iter_mut()
				.for_each(|rc| *rc = BinaryField32b::new(VISION_RC_EVEN[i][round_i]));
			odd_round_consts_slice[i]
				.iter_mut()
				.for_each(|rc| *rc = BinaryField32b::new(VISION_RC_ODD[i][round_i]));
		}

		for z in 0..1 << log_size {
			// Even rounds
			let input: [_; STATE_SIZE] =
				array::from_fn(|row| must_cast_slice::<_, B32>(perm_in_data[row])[z]);
			let inverse_0 = input.map(B32::invert_or_zero);

			let sbox_out_packed: [PackedBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
				let inp = PackedAESBinaryField8x32b::from_fn(|pack_idx| {
					inverse_0[pack_idx + arr_idx * 8].into()
				});
				Transformation::<PackedAESBinaryField8x32b, PackedBinaryField8x32b>::transform(
					&aes_to_bin_packed,
					&INV_PACKED_TRANS_AES.transform(&inp),
				) + inv_const_packed
			});

			for i in 0..STATE_SIZE {
				let sbox_out = get_packed_slice(&sbox_out_packed, i);
				inv_0_slice[i][z] = inverse_0[i];
				s_box_out_0_slice[i][z] = sbox_out;
			}
			let mut inp_as_packed_aes: [PackedAESBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
				PackedAESBinaryField8x32b::from_fn(|pack_idx| {
					s_box_out_0_slice[pack_idx + arr_idx * 8][z].into()
				})
			});
			mds_trans
				.transform(PackedAESBinaryField8x32b::cast_base_arr_mut(&mut inp_as_packed_aes));
			let inp_as_packed_bin: [PackedBinaryField8x32b; 3] =
				inp_as_packed_aes.map(|x| aes_to_bin_packed.transform(&x));

			for i in 0..STATE_SIZE {
				let mds_even_out: B32 = get_packed_slice(&inp_as_packed_bin, i);
				let round_even_out = mds_even_out + even_round_consts_slice[i][z];
				mds_out_0_slice[i][z] = mds_even_out;
				round_out_0_slice[i][z] = round_even_out;

				// Odd rounds
				let inv_odd = round_even_out.invert_or_zero();
				let inv_pow2_odd = inv_odd.square();
				let inv_pow4_odd = inv_pow2_odd.square();
				let sbox_out_odd = SBOX_FWD_CONST
					+ inv_odd * SBOX_FWD_TRANS[0]
					+ inv_pow2_odd * SBOX_FWD_TRANS[1]
					+ inv_pow4_odd * SBOX_FWD_TRANS[2];

				inv_1_slice[i][z] = inv_odd;
				s_box_out_1_slice[i][z] = sbox_out_odd;
			}

			let mut inp_as_packed_aes: [PackedAESBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
				PackedAESBinaryField8x32b::from_fn(|pack_idx| {
					s_box_out_1_slice[pack_idx + arr_idx * 8][z].into()
				})
			});
			mds_trans
				.transform(PackedAESBinaryField8x32b::cast_base_arr_mut(&mut inp_as_packed_aes));
			let inp_as_packed: [PackedBinaryField8x32b; 3] =
				inp_as_packed_aes.map(|x| aes_to_bin_packed.transform(&x));
			for i in 0..24 {
				let mds_out_odd: B32 = get_packed_slice(&inp_as_packed, i);
				mds_out_1_slice[i][z] = mds_out_odd;
				let output = mds_out_odd + odd_round_consts_slice[i][z];
				perm_out_slice[i][z] = output;
			}
		}
	}

	// zero check constraints
	for s in 0..STATE_SIZE {
		// Making sure inv_0 is the inverse of the permutation input
		builder.assert_zero(format!("inv0_{s}"), [perm_in[s], inv_0[s]], inv_constraint_expr()?);
		// Making sure inv_1 is the inverse of round_out_0
		builder.assert_zero(
			format!("inv1_{s}"),
			[round_out_0[s], inv_1[s]],
			inv_constraint_expr()?,
		);

		// Sbox composition checks
		builder.assert_zero(
			format!("sbox_linearized0_{s}"),
			[s_box_out_0[s], inv_0[s]],
			s_box_linearized_eval_expr().convert_field(),
		);
		builder.assert_zero(
			format!("sbox_linearized1_{s}"),
			[inv_1[s], s_box_out_1[s]],
			s_box_linearized_eval_expr().convert_field(),
		);
	}

	Ok(perm_out)
}

#[cfg(test)]
mod tests {
	use binius_core::oracle::OracleId;
	use binius_field::BinaryField32b;

	use super::vision_permutation;
	use crate::{builder::test_utils::test_circuit, unconstrained::unconstrained};

	#[test]
	fn test_vision32b() {
		test_circuit(|builder| {
			let log_size = 8;
			let state_in: [OracleId; 24] = std::array::from_fn(|i| {
				unconstrained::<BinaryField32b>(builder, format!("p_in[{i}]"), log_size).unwrap()
			});
			let _state_out = vision_permutation(builder, log_size, state_in).unwrap();
			Ok(vec![])
		})
		.unwrap();
	}
}
