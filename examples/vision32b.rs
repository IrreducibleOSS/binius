// Copyright 2024 Ulvetanna Inc.

#![feature(step_trait)]

extern crate core;

use anyhow::Result;
use binius_core::{
	challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger},
	oracle::{BatchId, CompositePolyOracle, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::{
		composition::{empty_mix_composition, index_composition},
		transparent::multilinear_extension::MultilinearExtensionTransparent,
		CompositionPoly, Error as PolynomialError, EvaluationDomainFactory,
		IsomorphicEvaluationDomainFactory, MultilinearComposite, MultilinearExtension,
	},
	protocols::{
		greedy_evalcheck,
		greedy_evalcheck::{GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		zerocheck,
		zerocheck::{ZerocheckClaim, ZerocheckProof, ZerocheckProveOutput},
	},
	witness::MultilinearWitnessIndex,
};
use binius_field::{
	affine_transformation::Transformation,
	arithmetic_traits::{InvertOrZero, Square},
	packed::get_packed_slice,
	underlier::WithUnderlier,
	BinaryField128b, BinaryField32b, BinaryField8b, ExtensionField, Field, PackedBinaryField1x128b,
	PackedBinaryField4x32b, PackedBinaryField8x32b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hash::{GroestlHasher, Vision32bMDS, Vision32bPermutation, INV_PACKED_TRANS};
use binius_macros::{composition_poly, IterOracles, IterPolys};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytemuck::{must_cast_slice_mut, Pod};
use bytesize::ByteSize;
use itertools::chain;
use p3_symmetric::Permutation;
use rand::thread_rng;
use std::{array, fmt::Debug, iter};
use tracing::instrument;

/// Smallest value such that 2^LOG_COMPRESSION_BLOCK >= N_ROUNDS = 8
const LOG_COMPRESSION_BLOCK: usize = 3;

#[rustfmt::skip]
const VISION_RC_EVEN: [[u32; 8]; 24] = [
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
const VISION_RC_ODD: [[u32; 8]; 24] = [
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
const VISION_ROUND_0: [u32; 24] = [0x545e66a7, 0x073fdd58, 0x84362677, 0x95fe8565, 0x06269cd8, 0x9c17909e, 0xf1f0adee, 0x2694c698, 0x94b2788f, 0x5eac14ad, 0x21677a78, 0x5755730b, 0x37cef9cf, 0x2fb31ffe, 0xfc0082ec, 0x609c12f0, 0x102769ee, 0x4732860d, 0xf97935e0, 0x36e77c02, 0xba9e70df, 0x67b701d7, 0x829d77a4, 0xf6ec454d];

const SBOX_FWD_TRANS: [BinaryField32b; 3] = [
	BinaryField32b::new(0xdb43e603),
	BinaryField32b::new(0x391c8e32),
	BinaryField32b::new(0x9fd55d88),
];

const SBOX_FWD_CONST: BinaryField32b = BinaryField32b::new(0x7cf0bc6c);
const SBOX_INV_CONST: BinaryField32b = BinaryField32b::new(0x9fa712f2);

#[rustfmt::skip]
const MDS_TRANS: [[u8; 24]; 24] = [
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

#[derive(Clone, Debug)]
struct SumComposition {
	n_vars: usize,
}

impl<P: PackedField> CompositionPoly<P> for SumComposition {
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate_scalar(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		if query.len() != self.n_vars {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		Ok(query.iter().copied().sum())
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != self.n_vars {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars,
			});
		}
		Ok(query.iter().copied().sum())
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

composition_poly!(ProdComposition[x, inv, prod] = x * inv - prod);
composition_poly!(ProductImpliesInputZero[x, prod] = x * (prod - 1));
composition_poly!(ProductImpliesInverseZero[inv, prod] = inv * (prod - 1));
composition_poly!(ConditionalEquality[x, y, is_equal] = (x - y) * is_equal);

#[derive(Clone, Debug)]
struct SquareComposition;

impl<P: PackedField> CompositionPoly<P> for SquareComposition {
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn evaluate_scalar(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		if query.len() != 2 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 2 });
		}
		let x = query[0];
		let y = query[1];
		Ok(PackedField::square(x) + y)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != 2 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 2 });
		}
		let x = query[0];
		let y = query[1];
		Ok(x.square() + y)
	}

	fn binary_tower_level(&self) -> usize {
		0
	}
}

#[derive(Clone, Debug)]
struct SBoxFwdComposition<F32b: Clone> {
	coefficients: [F32b; 4],
}

impl<F32b: Clone + From<BinaryField32b>> Default for SBoxFwdComposition<F32b> {
	fn default() -> Self {
		Self {
			coefficients: [
				SBOX_FWD_CONST,
				SBOX_FWD_TRANS[0],
				SBOX_FWD_TRANS[1],
				SBOX_FWD_TRANS[2],
			]
			.map(F32b::from),
		}
	}
}

impl<F32b, P> CompositionPoly<P> for SBoxFwdComposition<F32b>
where
	F32b: Field,
	P: PackedField<Scalar: ExtensionField<F32b>>,
{
	fn n_vars(&self) -> usize {
		4
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate_scalar(&self, query: &[P::Scalar]) -> Result<P::Scalar, PolynomialError> {
		if query.len() != 4 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 4 });
		}

		let result = iter::zip(query[..3].iter(), self.coefficients[1..].iter())
			.map(|(y_i, coeff)| *y_i * (*coeff))
			.sum::<P::Scalar>()
			+ P::Scalar::from(self.coefficients[0]);

		Ok(result - query[3])
	}

	fn evaluate(&self, query: &[P]) -> Result<P, PolynomialError> {
		if query.len() != 4 {
			return Err(PolynomialError::IncorrectQuerySize { expected: 4 });
		}

		let result = iter::zip(query[..3].iter(), self.coefficients[1..].iter())
			.map(|(y_i, coeff)| P::from_fn(|j| y_i.get(j) * (*coeff)))
			.sum::<P>() + P::broadcast(P::Scalar::from(self.coefficients[0]));

		Ok(result - query[3])
	}

	fn binary_tower_level(&self) -> usize {
		BinaryField32b::TOWER_LEVEL
	}
}

fn round_consts(rc: &[u32; 8]) -> Vec<PackedBinaryField8x32b> {
	let rc_b32 = PackedBinaryField8x32b::from_fn(|i| BinaryField32b::new(rc[i]));
	vec![rc_b32]
}

#[derive(IterOracles)]
struct TraceOracle {
	// Transparent columns
	even_round_consts: [OracleId; 24],
	odd_round_consts: [OracleId; 24],
	round_0_constant: [OracleId; 24],
	round_selector: OracleId,

	// Public columns
	state_in: [OracleId; 24],      // Committed
	round_begin: [OracleId; 24],   // Virtual
	inv_0: [OracleId; 24],         // Commited
	prod_0: [OracleId; 24],        // Commited
	s_box_out_0: [OracleId; 24],   // Commited
	s_box_pow2_0: [OracleId; 24],  // Commited
	s_box_pow4_0: [OracleId; 24],  // Commited
	mds_out_0: [OracleId; 24],     // Virtual
	round_out_0: [OracleId; 24],   // Commited
	inv_1: [OracleId; 24],         // Commited
	prod_1: [OracleId; 24],        // Commited
	inv_pow2_1: [OracleId; 24],    // Commited
	inv_pow4_1: [OracleId; 24],    // Commited
	s_box_out_1: [OracleId; 24],   // Virtual
	mds_out_1: [OracleId; 24],     // Virtual
	state_out: [OracleId; 24],     // Commited
	next_state_in: [OracleId; 24], // Virtual

	// Batch IDs
	trace_batch_id: BatchId,
}

impl TraceOracle {
	pub fn new<F>(oracles: &mut MultilinearOracleSet<F>, log_size: usize) -> Result<Self>
	where
		F: TowerField + ExtensionField<BinaryField32b> + From<BinaryField32b>,
	{
		let even_round_consts = array::from_fn(|i| {
			let even_rc_single = oracles
				.add_transparent(MultilinearExtensionTransparent(
					MultilinearExtension::from_values(round_consts(&VISION_RC_EVEN[i]))
						.unwrap()
						.specialize::<F>(),
				))
				.unwrap();
			oracles
				.add_repeating(even_rc_single, log_size - LOG_COMPRESSION_BLOCK)
				.unwrap()
		});
		let odd_round_consts = array::from_fn(|i| {
			let odd_rc_single = oracles
				.add_transparent(MultilinearExtensionTransparent(
					MultilinearExtension::from_values(round_consts(&VISION_RC_ODD[i]))
						.unwrap()
						.specialize::<F>(),
				))
				.unwrap();
			oracles
				.add_repeating(odd_rc_single, log_size - LOG_COMPRESSION_BLOCK)
				.unwrap()
		});
		let round_0_constant = array::from_fn(|i| {
			let round_0_constant_single = oracles
				.add_transparent(MultilinearExtensionTransparent(
					MultilinearExtension::from_values(round_consts(&[
						VISION_ROUND_0[i],
						0,
						0,
						0,
						0,
						0,
						0,
						0,
					]))
					.unwrap()
					.specialize::<F>(),
				))
				.unwrap();
			oracles
				.add_repeating(round_0_constant_single, log_size - LOG_COMPRESSION_BLOCK)
				.unwrap()
		});

		let round_selector_single = oracles
			.add_transparent(MultilinearExtensionTransparent(
				MultilinearExtension::from_values(round_consts(&[1, 1, 1, 1, 1, 1, 1, 0]))
					.unwrap()
					.specialize::<F>(),
			))
			.unwrap();
		let round_selector = oracles
			.add_repeating(round_selector_single, log_size - LOG_COMPRESSION_BLOCK)
			.unwrap();

		let mut batch_scope = oracles.build_committed_batch(log_size, BinaryField32b::TOWER_LEVEL);

		let state_in = batch_scope.add_multiple::<24>();
		let inv_0 = batch_scope.add_multiple::<24>();
		let prod_0 = batch_scope.add_multiple::<24>();
		let s_box_out_0 = batch_scope.add_multiple::<24>();
		let s_box_pow2_0 = batch_scope.add_multiple::<24>();
		let s_box_pow4_0 = batch_scope.add_multiple::<24>();
		let round_out_0 = batch_scope.add_multiple::<24>();
		let inv_1 = batch_scope.add_multiple::<24>();
		let prod_1 = batch_scope.add_multiple::<24>();
		let inv_pow2_1 = batch_scope.add_multiple::<24>();
		let inv_pow4_1 = batch_scope.add_multiple::<24>();
		let state_out = batch_scope.add_multiple::<24>();
		let trace_batch = batch_scope.build();

		let round_begin = array::from_fn(|i| {
			oracles
				.add_linear_combination(
					log_size,
					[(state_in[i], F::ONE), (round_0_constant[i], F::ONE)],
				)
				.unwrap()
		});

		let s_box_out_1 = array::from_fn(|i| {
			oracles
				.add_linear_combination_with_offset(
					log_size,
					F::from(SBOX_FWD_CONST),
					[
						(inv_1[i], F::from(SBOX_FWD_TRANS[0])),
						(inv_pow2_1[i], F::from(SBOX_FWD_TRANS[1])),
						(inv_pow4_1[i], F::from(SBOX_FWD_TRANS[2])),
					],
				)
				.unwrap()
		});

		let mds_out_0 = array::from_fn(|row| {
			oracles
				.add_linear_combination(
					log_size,
					MDS_TRANS[row].iter().enumerate().map(|(i, &elem)| {
						(s_box_out_0[i], F::from(BinaryField32b::new(elem as u32)))
					}),
				)
				.unwrap()
		});
		let mds_out_1 = array::from_fn(|row| {
			oracles
				.add_linear_combination(
					log_size,
					MDS_TRANS[row].iter().enumerate().map(|(i, &elem)| {
						(s_box_out_1[i], F::from(BinaryField32b::new(elem as u32)))
					}),
				)
				.unwrap()
		});

		let next_state_in = state_in.map(|state_in_xy| {
			oracles
				.add_shifted(state_in_xy, 1, 3, ShiftVariant::LogicalRight)
				.unwrap()
		});

		Ok(TraceOracle {
			even_round_consts,
			odd_round_consts,
			round_0_constant,
			round_selector,
			state_in,
			round_begin,
			inv_0,
			prod_0,
			s_box_out_0,
			s_box_pow2_0,
			s_box_pow4_0,
			mds_out_0,
			round_out_0,
			inv_1,
			prod_1,
			inv_pow2_1,
			inv_pow4_1,
			s_box_out_1,
			mds_out_1,
			state_out,
			next_state_in,
			trace_batch_id: trace_batch,
		})
	}
}

#[derive(Debug, IterPolys)]
struct TraceWitness<P32b: PackedField> {
	even_round_consts: [Vec<P32b>; 24],
	odd_round_consts: [Vec<P32b>; 24],
	round_0_constant: [Vec<P32b>; 24],
	round_selector: Vec<P32b>,
	state_in: [Vec<P32b>; 24],      // Committed
	round_begin: [Vec<P32b>; 24],   // Virtual
	inv_0: [Vec<P32b>; 24],         // Commited
	prod_0: [Vec<P32b>; 24],        // Commited
	s_box_out_0: [Vec<P32b>; 24],   // Commited
	s_box_pow2_0: [Vec<P32b>; 24],  // Commited
	s_box_pow4_0: [Vec<P32b>; 24],  // Commited
	mds_out_0: [Vec<P32b>; 24],     // Virtual
	round_out_0: [Vec<P32b>; 24],   // Commited
	inv_1: [Vec<P32b>; 24],         // Commited
	prod_1: [Vec<P32b>; 24],        // Commited
	inv_pow2_1: [Vec<P32b>; 24],    // Commited
	inv_pow4_1: [Vec<P32b>; 24],    // Commited
	s_box_out_1: [Vec<P32b>; 24],   // Virtual
	mds_out_1: [Vec<P32b>; 24],     // Virtual
	state_out: [Vec<P32b>; 24],     // Commited
	next_state_in: [Vec<P32b>; 24], // Virtual
}

impl<P32b: PackedField> TraceWitness<P32b> {
	fn to_index<F>(&self, trace_oracle: &TraceOracle) -> MultilinearWitnessIndex<F>
	where
		F: ExtensionField<P32b::Scalar>,
	{
		let mut index = MultilinearWitnessIndex::new();
		for (oracle, witness) in trace_oracle.iter_oracles().zip(self.iter_polys()) {
			index.set(oracle, witness.specialize_arc_dyn());
		}
		index
	}

	fn commit_polys(&self) -> impl Iterator<Item = MultilinearExtension<P32b>> {
		chain!(
			self.state_in.iter(),
			self.inv_0.iter(),
			self.prod_0.iter(),
			self.s_box_out_0.iter(),
			self.s_box_pow2_0.iter(),
			self.s_box_pow4_0.iter(),
			self.round_out_0.iter(),
			self.inv_1.iter(),
			self.prod_1.iter(),
			self.inv_pow2_1.iter(),
			self.inv_pow4_1.iter(),
			self.state_out.iter()
		)
		.map(|values| MultilinearExtension::from_values_slice(values.as_slice()).unwrap())
	}
}

impl<P32b> TraceWitness<P32b>
where
	P32b: PackedFieldIndexable<Scalar = BinaryField32b> + Pod,
{
	#[instrument]
	fn generate_trace(log_size: usize) -> TraceWitness<P32b> {
		let build_trace_column = || vec![P32b::default(); 1 << (log_size - P32b::LOG_WIDTH)];
		let mut witness = TraceWitness {
			even_round_consts: array::from_fn(|_| build_trace_column()),
			odd_round_consts: array::from_fn(|_| build_trace_column()),
			round_0_constant: array::from_fn(|_| build_trace_column()),
			round_selector: build_trace_column(),
			state_in: array::from_fn(|_| build_trace_column()),
			round_begin: array::from_fn(|_| build_trace_column()),
			inv_0: array::from_fn(|_| build_trace_column()),
			prod_0: array::from_fn(|_| build_trace_column()),
			s_box_out_0: array::from_fn(|_| build_trace_column()),
			s_box_pow2_0: array::from_fn(|_| build_trace_column()),
			s_box_pow4_0: array::from_fn(|_| build_trace_column()),
			mds_out_0: array::from_fn(|_| build_trace_column()),
			round_out_0: array::from_fn(|_| build_trace_column()),
			inv_1: array::from_fn(|_| build_trace_column()),
			prod_1: array::from_fn(|_| build_trace_column()),
			inv_pow2_1: array::from_fn(|_| build_trace_column()),
			inv_pow4_1: array::from_fn(|_| build_trace_column()),
			s_box_out_1: array::from_fn(|_| build_trace_column()),
			mds_out_1: array::from_fn(|_| build_trace_column()),
			state_out: array::from_fn(|_| build_trace_column()),
			next_state_in: array::from_fn(|_| build_trace_column()),
		};

		fn cast_32b_cols<P32b: PackedFieldIndexable<Scalar = BinaryField32b>, const N: usize>(
			cols: &mut [Vec<P32b>; N],
		) -> [&mut [BinaryField32b]; N] {
			cols.each_mut()
				.map(|col| PackedFieldIndexable::unpack_scalars_mut(col.as_mut_slice()))
		}

		let even_round_consts = cast_32b_cols(&mut witness.even_round_consts);
		let odd_round_consts = cast_32b_cols(&mut witness.odd_round_consts);
		let round_0_constant = cast_32b_cols(&mut witness.round_0_constant);
		let round_selector = must_cast_slice_mut(witness.round_selector.as_mut_slice());
		let state_in = cast_32b_cols(&mut witness.state_in);
		let round_begin = cast_32b_cols(&mut witness.round_begin);
		let inv_0 = cast_32b_cols(&mut witness.inv_0);
		let prod_0 = cast_32b_cols(&mut witness.prod_0);
		let s_box_out_0 = cast_32b_cols(&mut witness.s_box_out_0);
		let s_box_pow2_0 = cast_32b_cols(&mut witness.s_box_pow2_0);
		let s_box_pow4_0 = cast_32b_cols(&mut witness.s_box_pow4_0);
		let mds_out_0 = cast_32b_cols(&mut witness.mds_out_0);
		let round_out_0 = cast_32b_cols(&mut witness.round_out_0);
		let inv_1 = cast_32b_cols(&mut witness.inv_1);
		let prod_1 = cast_32b_cols(&mut witness.prod_1);
		let inv_pow2_1 = cast_32b_cols(&mut witness.inv_pow2_1);
		let inv_pow4_1 = cast_32b_cols(&mut witness.inv_pow4_1);
		let s_box_out_1 = cast_32b_cols(&mut witness.s_box_out_1);
		let mds_out_1 = cast_32b_cols(&mut witness.mds_out_1);
		let state_out = cast_32b_cols(&mut witness.state_out);
		let next_state_in = cast_32b_cols(&mut witness.next_state_in);

		// Helper structs
		let mut rng = thread_rng();
		let vision_instance = Vision32bPermutation::default();
		let mds_trans = Vision32bMDS::default();
		let inv_const_packed = PackedBinaryField8x32b::broadcast(SBOX_INV_CONST);
		let sbox_fwd_const_packed = PackedBinaryField8x32b::broadcast(SBOX_FWD_CONST);
		let sbox_trans_packed: [PackedBinaryField8x32b; 3] =
			array::from_fn(|i| PackedBinaryField8x32b::broadcast(SBOX_FWD_TRANS[i]));

		// Each permutation is 8 round states
		for perm_i in 0..1 << (log_size - LOG_COMPRESSION_BLOCK) {
			let i = perm_i << LOG_COMPRESSION_BLOCK;

			// Randomly generate the initial permutation input
			let input: [BinaryField32b; 24] =
				array::from_fn(|_| <BinaryField32b as Field>::random(&mut rng));
			let output: [BinaryField32b; 24] = vision_instance.permute(input);

			// Assign the permutation input
			for j in 0..24 {
				state_in[j][i] = input[j];
			}

			// Expand trace columns for each round
			for round_i in 0..8 {
				let i = i | round_i;

				// ROUND 2*round_i
				let mut x_inverses = [BinaryField32b::default(); 24];
				for j in 0..24 {
					if round_i == 0 {
						round_0_constant[j][i] = BinaryField32b::new(VISION_ROUND_0[j]);
					} else {
						round_0_constant[j][i] = BinaryField32b::zero();
					}
					round_begin[j][i] = state_in[j][i] + round_0_constant[j][i];
					let x = round_begin[j][i];
					let x_inv = InvertOrZero::invert_or_zero(x);
					x_inverses[j] = x_inv;
					inv_0[j][i] = x_inv;
					prod_0[j][i] = x * x_inv;
					even_round_consts[j][i] = BinaryField32b::new(VISION_RC_EVEN[j][round_i]);
					odd_round_consts[j][i] = BinaryField32b::new(VISION_RC_ODD[j][round_i]);
				}
				let sbox_out_packed: [PackedBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
					let inp = PackedBinaryField8x32b::from_fn(|pack_idx| {
						x_inverses[pack_idx + arr_idx * 8]
					});
					INV_PACKED_TRANS.transform(&inp) + inv_const_packed
				});
				for j in 0..24 {
					let sbox_out: BinaryField32b = get_packed_slice(&sbox_out_packed, j);
					let sbox_pow2 = Square::square(sbox_out);
					let sbox_pow4 = Square::square(sbox_pow2);
					s_box_out_0[j][i] = sbox_out;
					s_box_pow2_0[j][i] = sbox_pow2;
					s_box_pow4_0[j][i] = sbox_pow4;
				}
				let mut inp_as_packed: [PackedBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
					PackedBinaryField8x32b::from_fn(|pack_idx| {
						s_box_out_0[pack_idx + arr_idx * 8][i]
					})
				});
				mds_trans.transform(&mut inp_as_packed);
				for j in 0..24 {
					let mds_out = get_packed_slice(&inp_as_packed, j);
					mds_out_0[j][i] = mds_out;
					round_out_0[j][i] = mds_out_0[j][i] + even_round_consts[j][i];
				}

				// ROUND 2*round_i + 1
				let input: [PackedBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
					PackedBinaryField8x32b::from_fn(|j| round_out_0[j + arr_idx * 8][i])
				});
				let input_inv = input.map(PackedField::invert_or_zero);
				let input_pow2 = input_inv.map(PackedField::square);
				let input_pow4 = input_pow2.map(PackedField::square);
				let sbox_out: [PackedBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
					sbox_fwd_const_packed
						+ input_inv[arr_idx] * sbox_trans_packed[0]
						+ input_pow2[arr_idx] * sbox_trans_packed[1]
						+ input_pow4[arr_idx] * sbox_trans_packed[2]
				});
				for j in 0..24 {
					let x_inv = get_packed_slice(&input_inv, j);
					inv_1[j][i] = x_inv;
					prod_1[j][i] = if x_inv == BinaryField32b::ZERO {
						BinaryField32b::ZERO
					} else {
						BinaryField32b::ONE
					};
					inv_pow2_1[j][i] = get_packed_slice(&input_pow2, j);
					inv_pow4_1[j][i] = get_packed_slice(&input_pow4, j);
					s_box_out_1[j][i] = get_packed_slice(&sbox_out, j);
				}

				let mut inp_as_packed: [PackedBinaryField8x32b; 3] = array::from_fn(|arr_idx| {
					PackedBinaryField8x32b::from_fn(|pack_idx| {
						s_box_out_1[pack_idx + arr_idx * 8][i]
					})
				});
				mds_trans.transform(&mut inp_as_packed);
				for j in 0..24 {
					mds_out_1[j][i] = get_packed_slice(&inp_as_packed, j);
				}

				for j in 0..24 {
					state_out[j][i] = mds_out_1[j][i] + odd_round_consts[j][i];
				}

				if round_i < 7 {
					for xy in 0..24 {
						state_in[xy][i + 1] = state_out[xy][i];
						next_state_in[xy][i] = state_out[xy][i];
					}
				}
				round_selector[i] = if round_i < 7 {
					BinaryField32b::ONE
				} else {
					BinaryField32b::ZERO
				};
			}

			// Assert correct output
			for j in 0..24 {
				assert_eq!(state_out[j][i + 7], output[j]);
			}
		}

		witness
	}
}

fn make_constraints<F32b, FW>(
	trace_oracle: &TraceOracle,
	challenge: FW,
) -> Result<impl CompositionPoly<FW>>
where
	F32b: TowerField + From<BinaryField32b>,
	FW: TowerField + ExtensionField<F32b>,
{
	let zerocheck_column_ids = trace_oracle.iter_oracles().collect::<Vec<_>>();

	let mix = empty_mix_composition(zerocheck_column_ids.len(), challenge);

	// Making sure inv_0 is the inverse of the state_in column
	// 3 constraints (taking into account 0 -> 0). If y is the inverse of x, then check
	// x*(x*y - 1) == 0 & y*(x*y - 1) == 0
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.round_begin[x],
				trace_oracle.inv_0[x],
				trace_oracle.prod_0[x],
			],
			ProdComposition,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.round_begin[x], trace_oracle.prod_0[x]],
			ProductImpliesInputZero,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.inv_0[x], trace_oracle.prod_0[x]],
			ProductImpliesInverseZero,
		)
		.unwrap()
	}))?;
	// Similarly for the unrolled second round.
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.round_out_0[x],
				trace_oracle.inv_1[x],
				trace_oracle.prod_1[x],
			],
			ProdComposition,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.round_out_0[x], trace_oracle.prod_1[x]],
			ProductImpliesInputZero,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.inv_1[x], trace_oracle.prod_1[x]],
			ProductImpliesInverseZero,
		)
		.unwrap()
	}))?;

	// Constraints for all the squaring
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.s_box_out_0[x], trace_oracle.s_box_pow2_0[x]],
			SquareComposition,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.s_box_pow2_0[x], trace_oracle.s_box_pow4_0[x]],
			SquareComposition,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.inv_1[x], trace_oracle.inv_pow2_1[x]],
			SquareComposition,
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[trace_oracle.inv_pow2_1[x], trace_oracle.inv_pow4_1[x]],
			SquareComposition,
		)
		.unwrap()
	}))?;

	// SBOX Composition checks, for the 0th round inside each loop we are checking the output is actually the inverse of the transformation
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.s_box_out_0[x],
				trace_oracle.s_box_pow2_0[x],
				trace_oracle.s_box_pow4_0[x],
				trace_oracle.inv_0[x],
			],
			SBoxFwdComposition::<F32b>::default(),
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.inv_1[x],
				trace_oracle.inv_pow2_1[x],
				trace_oracle.inv_pow4_1[x],
				trace_oracle.s_box_out_1[x],
			],
			SBoxFwdComposition::<F32b>::default(),
		)
		.unwrap()
	}))?;

	// Round constants check.
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.mds_out_0[x],
				trace_oracle.round_out_0[x],
				trace_oracle.even_round_consts[x],
			],
			SumComposition { n_vars: 3 },
		)
		.unwrap()
	}))?;
	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.mds_out_1[x],
				trace_oracle.state_out[x],
				trace_oracle.odd_round_consts[x],
			],
			SumComposition { n_vars: 3 },
		)
		.unwrap()
	}))?;

	let mix = mix.include((0..24).map(|x| {
		index_composition(
			&zerocheck_column_ids,
			[
				trace_oracle.next_state_in[x],
				trace_oracle.state_out[x],
				trace_oracle.round_selector,
			],
			ConditionalEquality,
		)
		.unwrap()
	}))?;

	Ok(mix)
}

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

#[instrument(skip_all)]
fn prove<F, FW, P32b, PCS, Comm, Challenger>(
	oracles: &mut MultilinearOracleSet<F>,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: Challenger,
	witness: &TraceWitness<P32b>,
	domain_factory: impl EvaluationDomainFactory<BinaryField32b>,
) -> Result<Proof<F, Comm, PCS::Proof>>
where
	F: TowerField + ExtensionField<BinaryField32b> + From<FW>,
	FW: TowerField + ExtensionField<BinaryField32b> + From<F>,
	P32b: PackedField<Scalar = BinaryField32b>,
	PCS: PolyCommitScheme<P32b, F, Error: Debug, Proof: 'static, Commitment = Comm>,
	Comm: Clone,
	Challenger: CanObserve<F> + CanObserve<Comm> + CanSample<F> + CanSampleBits<usize>,
{
	let mut trace_witness = witness.to_index::<FW>(trace_oracle);

	// Round 1
	let trace_commit_polys = witness.commit_polys().collect::<Vec<_>>();
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mixing_challenge = challenger.sample();

	let mix_composition_verifier =
		make_constraints::<BinaryField32b, _>(trace_oracle, mixing_challenge)?;
	let mix_composition_prover =
		make_constraints::<BinaryField32b, _>(trace_oracle, FW::from(mixing_challenge))?;

	let zerocheck_column_oracles = trace_oracle
		.iter_oracles()
		.map(|id| oracles.oracle(id))
		.collect::<Vec<_>>();
	let max_n_vars = zerocheck_column_oracles
		.iter()
		.map(|oracle| oracle.n_vars())
		.max()
		.unwrap_or(0);
	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(
			max_n_vars,
			zerocheck_column_oracles,
			mix_composition_verifier,
		)?,
	};

	let zerocheck_witness = MultilinearComposite::new(
		zerocheck_claim.n_vars(),
		mix_composition_prover,
		witness
			.iter_polys()
			.map(|v| v.specialize_arc_dyn::<FW>())
			.collect(),
	)?;

	// Zerocheck
	let zerocheck_domain =
		domain_factory.create(zerocheck_claim.poly.max_individual_degree() + 1)?;

	let switchover_fn = |_| 1;

	let ZerocheckProveOutput {
		evalcheck_claim,
		zerocheck_proof,
	} = zerocheck::prove(
		&zerocheck_claim,
		zerocheck_witness,
		&zerocheck_domain,
		&mut challenger,
		switchover_fn,
	)?;

	// Evalcheck
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<_, _, BinaryField32b, _>(
		oracles,
		&mut trace_witness,
		[evalcheck_claim],
		switchover_fn,
		&mut challenger,
		domain_factory,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims.into_iter().next().unwrap();
	assert_eq!(batch_id, trace_oracle.trace_batch_id);

	let trace_open_proof = pcs.prove_evaluation(
		&mut challenger,
		&trace_committed,
		&trace_commit_polys,
		&same_query_claim.eval_point,
	)?;

	Ok(Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	})
}

#[instrument(skip_all)]
fn verify<F, P32b, PCS, Comm, Challenger>(
	oracles: &mut MultilinearOracleSet<F>,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: Challenger,
	proof: Proof<F, Comm, PCS::Proof>,
) -> Result<()>
where
	F: TowerField + ExtensionField<BinaryField32b>,
	P32b: PackedField<Scalar = BinaryField32b>,
	PCS: PolyCommitScheme<P32b, F, Error: Debug, Proof: 'static, Commitment = Comm>,
	Comm: Clone,
	Challenger: CanObserve<F> + CanObserve<Comm> + CanSample<F> + CanSampleBits<usize>,
{
	let Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mixing_challenge = challenger.sample();
	let mix_composition = make_constraints::<BinaryField32b, _>(trace_oracle, mixing_challenge)?;

	// Zerocheck
	let zerocheck_column_oracles = trace_oracle
		.iter_oracles()
		.map(|id| oracles.oracle(id))
		.collect::<Vec<_>>();
	let max_n_vars = zerocheck_column_oracles
		.iter()
		.map(|oracle| oracle.n_vars())
		.max()
		.unwrap_or(0);
	let zerocheck_claim = ZerocheckClaim {
		poly: CompositePolyOracle::new(max_n_vars, zerocheck_column_oracles, mix_composition)?,
	};

	let evalcheck_claim = zerocheck::verify(&zerocheck_claim, zerocheck_proof, &mut challenger)?;

	// Evalcheck
	let same_query_claims =
		greedy_evalcheck::verify(oracles, [evalcheck_claim], evalcheck_proof, &mut challenger)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims.into_iter().next().unwrap();
	assert_eq!(batch_id, trace_oracle.trace_batch_id);

	pcs.verify_evaluation(
		&mut challenger,
		&trace_comm,
		&same_query_claim.eval_point,
		trace_open_proof,
		&same_query_claim.evals,
	)?;

	Ok(())
}

fn main() {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");
	init_tracing();

	let log_size = get_log_trace_size().unwrap_or(14);

	let mut oracles = MultilinearOracleSet::<BinaryField128b>::new();
	let trace_oracle = TraceOracle::new(&mut oracles, log_size).unwrap();

	const SECURITY_BITS: usize = 100;
	let log_inv_rate = 1;
	let trace_batch = oracles.committed_batch(trace_oracle.trace_batch_id);
	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		<PackedBinaryField1x128b as WithUnderlier>::Underlier,
		BinaryField32b,
		BinaryField32b,
		BinaryField32b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, trace_batch.n_polys, log_inv_rate, false)
	.unwrap();

	let log_vision32b = log_size - LOG_COMPRESSION_BLOCK;
	// For vision32b we have our rate fixed to 16-32 bit elemeents so this is pretty straight forward.
	let data_hashed = ByteSize::b(
		(1 << log_vision32b)
			* 16 * (<BinaryField32b as ExtensionField<BinaryField8b>>::DEGREE) as u64,
	);
	let tensorpcs_size = ByteSize::b(pcs.proof_size(trace_batch.n_polys) as u64);
	tracing::info!("Size of hashable vision32b data: {}", data_hashed);
	tracing::info!("Size of tensorpcs: {}", tensorpcs_size);

	let challenger = <HashChallenger<_, GroestlHasher<_>>>::new();
	let witness = TraceWitness::<PackedBinaryField4x32b>::generate_trace(log_size);
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField32b>::default();

	let proof = prove::<_, BinaryField128b, _, _, _, _>(
		&mut oracles,
		&trace_oracle,
		&pcs,
		challenger.clone(),
		&witness,
		domain_factory,
	)
	.unwrap();

	verify(&mut oracles, &trace_oracle, &pcs, challenger, proof).unwrap()
}
