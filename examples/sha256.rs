// Copyright 2024 Ulvetanna Inc.

//! This an example SNARK for proving the SHA-256 compression function.
//!
//! This example proves and verifies a commit-and-prove SNARK for many independent compressions.
//! That means there are no boundary constraints, this simple proves a relation between the data
//! committed by the input and output columns.
//!
//! The arithmetization uses 1-bit committed columns. Each column treats chunks of 32 contiguous
//! bits as a 32-bit state element of the 8 x 32-bit SHA-256 state. Every row of 32-bit chunks
//! attests to the validity of one full SHA-256 compression (64 rounds).
//!
//! For SHA-256 specification and pseudocode, see
//! [FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) or the
//! [SHA-2 Wikipedia page](https://en.wikipedia.org/wiki/SHA-2).

use anyhow::Result;
use binius_core::{
	challenger::{
		new_hasher_challenger, CanObserve, CanSample, CanSampleBits, IsomorphicChallenger,
	},
	oracle::{BatchId, ConstraintSetBuilder, MultilinearOracleSet, OracleId, ShiftVariant},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	protocols::{
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		sumcheck::{self, standard_switchover_heuristic, Proof as ZerocheckProof},
	},
	transparent::multilinear_extension::MultilinearExtensionTransparent,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::packed_32::PackedBinaryField32x1b,
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	AESTowerField128b, AESTowerField8b, BinaryField, BinaryField128b, BinaryField16b,
	BinaryField1b, ExtensionField, Field, PackedAESBinaryField1x128b, PackedBinaryField128x1b,
	PackedBinaryField1x128b, PackedField, PackedFieldIndexable, RepackedExtension, TowerField,
};
use binius_hal::{make_portable_backend, ComputationBackend};
use binius_hash::GroestlHasher;
use binius_macros::composition_poly;
use binius_math::{EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	checked_arithmetics::checked_log_2, examples::get_log_trace_size, rayon::adjust_thread_pool,
	tracing::init_tracing,
};
use bytemuck::{must_cast, must_cast_slice, must_cast_slice_mut, Pod};
use bytesize::ByteSize;
use itertools::{chain, Itertools};
use rand::{thread_rng, Rng};
use sha2::{compress256, digest::generic_array::GenericArray};
use std::{array, fmt::Debug, iter, marker::PhantomData};
use tracing::instrument;

const SHA256_BLOCK_SIZE_BYTES: usize = 64;
const LOG_U32_BITS: usize = checked_log_2(32);

/// SHA-256 round constants, K
const ROUND_CONSTS_K: [u32; 64] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INIT: [u32; 8] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

struct U32ArraySumSubTrace<U, F, const N: usize>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<F> + Pod,
	F: BinaryField,
{
	pub z_out: [Box<[U]>; N],
	pub cout: [Box<[U]>; N],
	pub cin: [Box<[U]>; N],
	pub counter: usize,
	_phantom: PhantomData<F>,
}

impl<U, F, const N: usize> U32ArraySumSubTrace<U, F, N>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<F> + Pod,
	F: BinaryField,
{
	fn empty(len: usize) -> Self {
		let build_trace_column = || vec![U::default(); len].into_boxed_slice();
		Self {
			z_out: array::from_fn(|_| build_trace_column()),
			cout: array::from_fn(|_| build_trace_column()),
			cin: array::from_fn(|_| build_trace_column()),
			counter: 0,
			_phantom: PhantomData,
		}
	}

	fn sum_array(&mut self, arr: &[u32]) {
		assert_eq!(arr.len(), N + 1);

		for i in 0..N {
			let y = if i == 0 {
				arr[0]
			} else {
				must_cast_slice::<_, u32>(&self.z_out[i - 1])[self.counter]
			};

			let z = &mut must_cast_slice_mut::<_, u32>(&mut self.z_out[i])[self.counter];
			let cin = &mut must_cast_slice_mut::<_, u32>(&mut self.cin[i])[self.counter];
			let cout = &mut must_cast_slice_mut::<_, u32>(&mut self.cout[i])[self.counter];

			let carry;
			(*z, carry) = (y).overflowing_add(arr[i + 1]);
			*cin = (arr[i + 1]) ^ (y) ^ (*z);
			*cout = *cin >> 1;
			if carry {
				*cout |= 1 << 31;
			}
		}

		self.counter += 1;
	}

	fn get_last_sum(&self) -> Option<&u32> {
		must_cast_slice::<_, u32>(&self.z_out[N - 1]).get(self.counter - 1)
	}

	fn iter(&self) -> impl Iterator<Item = Box<[U]>> {
		self.z_out
			.clone()
			.into_iter()
			.chain(self.cout.clone())
			.chain(self.cin.clone())
	}
}

enum RotateRightType {
	Circular,
	Logical,
}

struct RotateRightAndXorSubTrace<U, F, const N: usize>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<F> + Pod,
	F: BinaryField,
{
	s: [Box<[U]>; N],
	res: Box<[U]>,
	counter: usize,
	_phantom: PhantomData<F>,
}

impl<U, F, const N: usize> RotateRightAndXorSubTrace<U, F, N>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<F> + Pod,
	F: BinaryField,
{
	fn empty(len: usize) -> Self {
		let build_trace_column = || vec![U::default(); len].into_boxed_slice();
		Self {
			s: array::from_fn(|_| build_trace_column()),
			res: build_trace_column(),
			counter: 0,
			_phantom: PhantomData,
		}
	}

	fn rotate_and_xor(&mut self, r: &[(u32, u32, RotateRightType); N]) {
		let res = &mut must_cast_slice_mut::<_, u32>(&mut self.res);

		r.iter().enumerate().for_each(|(i, (val, shift, t))| {
			let s = &mut must_cast_slice_mut::<_, u32>(&mut self.s[i]);
			s[self.counter] = match t {
				RotateRightType::Circular => val.rotate_right(*shift),
				RotateRightType::Logical => val >> shift,
			};
			res[self.counter] ^= s[self.counter];
		});

		self.counter += 1;
	}

	fn get_last_xor_res(&self) -> u32 {
		must_cast_slice::<_, u32>(&self.res)[self.counter - 1]
	}

	fn iter(&self) -> impl Iterator<Item = Box<[U]>> {
		self.s
			.clone()
			.into_iter()
			.chain(iter::once(self.res.clone()))
	}
}

#[instrument(skip_all, level = "debug")]
fn generate_trace<U, F>(
	log_size: usize,
	trace_oracle: &TraceOracle,
) -> Result<MultilinearExtensionIndex<'static, U, F>>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<F> + Pod,
	F: BinaryField,
{
	assert!(log_size >= <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let len = 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH);
	let build_trace_column = || vec![U::default(); len].into_boxed_slice();

	let mut w = array::from_fn(|_xy| build_trace_column());
	let mut s0: [RotateRightAndXorSubTrace<U, F, 3>; 48] =
		array::from_fn(|_x| RotateRightAndXorSubTrace::empty(len));
	let mut s1: [RotateRightAndXorSubTrace<U, F, 3>; 48] =
		array::from_fn(|_x| RotateRightAndXorSubTrace::empty(len));
	let mut extended_w: [U32ArraySumSubTrace<U, F, 3>; 48] =
		array::from_fn(|_x| U32ArraySumSubTrace::empty(len));
	let mut h: [[Box<[U]>; 8]; 65] =
		array::from_fn(|_x| array::from_fn(|_xy| build_trace_column()));
	let mut sigma1: [RotateRightAndXorSubTrace<U, F, 3>; 64] =
		array::from_fn(|_x| RotateRightAndXorSubTrace::empty(len));
	let mut sigma0: [RotateRightAndXorSubTrace<U, F, 3>; 64] =
		array::from_fn(|_x| RotateRightAndXorSubTrace::empty(len));
	let mut ch: [Box<[U]>; 64] = array::from_fn(|_xy| build_trace_column());
	let mut maj: [Box<[U]>; 64] = array::from_fn(|_xy| build_trace_column());
	let mut k: [Box<[U]>; 64] = array::from_fn(|_xy| build_trace_column());
	let mut temp1: [U32ArraySumSubTrace<U, F, 4>; 64] =
		array::from_fn(|_x| U32ArraySumSubTrace::empty(len));
	let mut temp2: [U32ArraySumSubTrace<U, F, 1>; 64] =
		array::from_fn(|_x| U32ArraySumSubTrace::empty(len));
	let mut d_add_temp1: [U32ArraySumSubTrace<U, F, 1>; 64] =
		array::from_fn(|_x| U32ArraySumSubTrace::empty(len));
	let mut temp1_add_temp2: [U32ArraySumSubTrace<U, F, 1>; 64] =
		array::from_fn(|_x| U32ArraySumSubTrace::empty(len));
	let mut output_add: [U32ArraySumSubTrace<U, F, 1>; 64] =
		array::from_fn(|_x| U32ArraySumSubTrace::empty(len));

	fn cast_u32_cols<U: Pod, const N: usize>(cols: &mut [Box<[U]>; N]) -> [&mut [u32]; N] {
		cols.each_mut()
			.map(|col| must_cast_slice_mut::<_, u32>(&mut *col))
	}

	let w_u32 = cast_u32_cols(&mut w);
	let mut h_u32: Vec<_> = h.iter_mut().map(cast_u32_cols).collect();
	let k_u32 = cast_u32_cols(&mut k);

	let ch_u32 = cast_u32_cols(&mut ch);
	let maj_u32 = cast_u32_cols(&mut maj);

	let mut rng = thread_rng();

	for j in 0..w_u32[0].len() {
		let get_w = |extended_w: &[U32ArraySumSubTrace<U, F, 3>],
		             w_u32: &[&mut [u32]; 16],
		             i: usize|
		 -> u32 {
			if i >= 16 {
				*extended_w[i - 16].get_last_sum().expect("must exist")
			} else {
				w_u32[i][j]
			}
		};

		let mut generic_array_input = GenericArray::<u8, _>::default();
		for i in 0..16 {
			w_u32[i][j] = rng.gen();
			for z in 0..4 {
				generic_array_input[i * 4 + z] = w_u32[i][j].to_be_bytes()[z];
			}
		}

		let mut output = INIT;
		compress256(&mut output, &[generic_array_input]);

		for i in 0..48 {
			let w_u32_i_plus_1 = get_w(&extended_w, &w_u32, i + 1);

			s0[i].rotate_and_xor(&[
				(w_u32_i_plus_1, 7, RotateRightType::Circular),
				(w_u32_i_plus_1, 18, RotateRightType::Circular),
				(w_u32_i_plus_1, 3, RotateRightType::Logical),
			]);

			let w_u32_i_plus_14 = get_w(&extended_w, &w_u32, i + 14);

			s1[i].rotate_and_xor(&[
				(w_u32_i_plus_14, 17, RotateRightType::Circular),
				(w_u32_i_plus_14, 19, RotateRightType::Circular),
				(w_u32_i_plus_14, 10, RotateRightType::Logical),
			]);

			let w_u32_i = get_w(&extended_w, &w_u32, i);

			let w_u32_i_plus_9 = get_w(&extended_w, &w_u32, i + 9);

			extended_w[i].sum_array(&[
				w_u32_i,
				s0[i].get_last_xor_res(),
				w_u32_i_plus_9,
				s1[i].get_last_xor_res(),
			]);
		}

		for (i, init) in INIT.iter().enumerate() {
			h_u32[0][i][j] = *init;
		}

		for i in 0..64 {
			sigma1[i].rotate_and_xor(&[
				(h_u32[i][4][j], 6, RotateRightType::Circular),
				(h_u32[i][4][j], 11, RotateRightType::Circular),
				(h_u32[i][4][j], 25, RotateRightType::Circular),
			]);

			ch_u32[i][j] = (h_u32[i][4][j] & h_u32[i][5][j]) ^ ((!h_u32[i][4][j]) & h_u32[i][6][j]);

			let w_u32_i = get_w(&extended_w, &w_u32, i);

			k_u32[i][j] = ROUND_CONSTS_K[i];

			temp1[i].sum_array(&[
				h_u32[i][7][j],
				sigma1[i].get_last_xor_res(),
				ch_u32[i][j],
				k_u32[i][j],
				w_u32_i,
			]);

			sigma0[i].rotate_and_xor(&[
				(h_u32[i][0][j], 2, RotateRightType::Circular),
				(h_u32[i][0][j], 13, RotateRightType::Circular),
				(h_u32[i][0][j], 22, RotateRightType::Circular),
			]);

			maj_u32[i][j] = (h_u32[i][0][j] & h_u32[i][1][j])
				^ (h_u32[i][0][j] & h_u32[i][2][j])
				^ (h_u32[i][1][j] & h_u32[i][2][j]);

			temp2[i].sum_array(&[sigma0[i].get_last_xor_res(), maj_u32[i][j]]);

			h_u32[i + 1][7][j] = h_u32[i][6][j];
			h_u32[i + 1][6][j] = h_u32[i][5][j];
			h_u32[i + 1][5][j] = h_u32[i][4][j];

			d_add_temp1[i].sum_array(&[
				h_u32[i][3][j],
				*temp1[i].get_last_sum().expect("temp1 sum must exist"),
			]);
			h_u32[i + 1][4][j] = *d_add_temp1[i]
				.get_last_sum()
				.expect("d_add_temp1 sum must exist");

			h_u32[i + 1][3][j] = h_u32[i][2][j];
			h_u32[i + 1][2][j] = h_u32[i][1][j];
			h_u32[i + 1][1][j] = h_u32[i][0][j];

			temp1_add_temp2[i].sum_array(&[
				*temp1[i].get_last_sum().expect("temp1 sum must exist"),
				*temp2[i].get_last_sum().expect("temp2 sum must exist"),
			]);
			h_u32[i + 1][0][j] = *temp1_add_temp2[i]
				.get_last_sum()
				.expect("temp1_add_temp2 sum must exist");
		}

		for i in 0..8 {
			output_add[i].sum_array(&[h_u32[0][i][j], h_u32[64][i][j]]);
			// Assert correct output
			assert_eq!(output[i], *output_add[i].get_last_sum().expect("output must exist"));
		}
	}

	let w_iter = trace_oracle.w.into_iter().zip(w.into_iter());

	macro_rules! flatten {
		($z:ident) => {{
			trace_oracle
				.$z
				.iter()
				.flat_map(|x| x.iter())
				.zip($z.iter().flat_map(|x| x.iter()))
		}};
	}

	let extended_w_iter = flatten!(extended_w);

	let ch_iter = trace_oracle.ch.into_iter().zip(ch.into_iter());

	let temp1_iter = flatten!(temp1);

	let maj_iter = trace_oracle.maj.into_iter().zip(maj.into_iter());

	let temp2_iter = flatten!(temp2);

	let d_add_temp1_iter = flatten!(d_add_temp1);

	let temp1_add_temp2_iter = flatten!(temp1_add_temp2);

	let output_add_iter = flatten!(output_add);

	macro_rules! flatten {
		($z:ident) => {{
			trace_oracle
				.$z
				.into_iter()
				.flatten()
				.zip($z.iter().flat_map(|x| x.iter()))
		}};
	}

	let s0_iter = flatten!(s0);

	let s1_iter = flatten!(s1);

	let sigma1_iter = flatten!(sigma1);

	let sigma0_iter = flatten!(sigma0);

	let h_iter = trace_oracle
		.h
		.into_iter()
		.flatten()
		.zip(h.clone().into_iter().flat_map(|x| x.into_iter()));

	let k_iter = trace_oracle.k.into_iter().zip(k.clone().into_iter());

	let index_iter = w_iter
		.chain(extended_w_iter)
		.chain(ch_iter)
		.chain(temp1_iter)
		.chain(maj_iter)
		.chain(temp2_iter)
		.chain(d_add_temp1_iter)
		.chain(temp1_add_temp2_iter)
		.chain(output_add_iter)
		.chain(s0_iter)
		.chain(s1_iter)
		.chain(h_iter)
		.chain(k_iter)
		.chain(sigma1_iter)
		.chain(sigma0_iter)
		.unique_by(|x| x.0);

	let index = MultilinearExtensionIndex::new().update_owned::<BinaryField1b, _>(index_iter)?;

	Ok(index)
}

#[derive(Debug, Clone)]
struct U32ArraySumSubOracle<const N: usize> {
	pub z_out: [OracleId; N],
	pub cout: [OracleId; N],
	pub cin: [OracleId; N],
}

impl<const N: usize> U32ArraySumSubOracle<N> {
	pub fn new<F: TowerField>(oracles: &mut MultilinearOracleSet<F>, batch_id: BatchId) -> Self {
		let z_out = oracles.add_named("z_out").committed_multiple(batch_id);
		let cout = oracles.add_named("cout").committed_multiple(batch_id);
		let cin = cout.map(|x| {
			oracles
				.add_named("cin")
				.shifted(x, 1, 5, ShiftVariant::LogicalLeft)
				.unwrap()
		});
		Self { z_out, cout, cin }
	}

	pub fn iter(&self) -> impl Iterator<Item = OracleId> {
		chain!(self.z_out, self.cout, self.cin)
	}

	pub fn val(&self) -> OracleId {
		self.z_out[N - 1]
	}
}

#[derive(Debug)]
struct TraceOracle {
	batch_id: BatchId,
	// Message chunk, 16 columns of 32-bit words
	w: [OracleId; 16],
	// Expanded message schedule columns of 32-bit words
	extended_w: [U32ArraySumSubOracle<3>; 48],

	ch: [OracleId; 64],

	temp1: [U32ArraySumSubOracle<4>; 64],
	maj: [OracleId; 64],

	temp2: [U32ArraySumSubOracle<1>; 64],

	d_add_temp1: [U32ArraySumSubOracle<1>; 64],

	temp1_add_temp2: [U32ArraySumSubOracle<1>; 64],

	output_add: [U32ArraySumSubOracle<1>; 8],
	// s₀(wᵢ₋₁₅) for i from 16..64, used to compute message schedule.
	s0: [[OracleId; 4]; 48],
	// s₁(wᵢ₋₂) for i from 16..64, used to compute message schedule.
	s1: [[OracleId; 4]; 48],
	// Round state. 8 columns (a, b, c, d, e, f, g, h) of 32-bit words, one per round plus final.
	h: [[OracleId; 8]; 65],
	// Round constants, 64 repeating columns of 32-bit words.
	k: [OracleId; 64],
	// Σ₁(e), one 32-bit column per 64 rounds.
	sigma1: [[OracleId; 4]; 64],
	// Σ₀(a), one 32-bit column per 64 rounds.
	sigma0: [[OracleId; 4]; 64],
}

impl TraceOracle {
	fn gen_xor_oracle_ids<F: TowerField, const N: usize>(
		log_size: usize,
		oracles: &mut MultilinearOracleSet<F>,
		r: &[(OracleId, usize, RotateRightType)],
	) -> [OracleId; N] {
		let mut oracles_arr: [OracleId; N] = [0; N];

		assert_eq!(r.len(), N - 1);

		r.iter().enumerate().for_each(|(i, (oracle_id, shift, t))| {
			oracles_arr[i] = match t {
				RotateRightType::Circular => oracles
					.add_shifted(*oracle_id, 32 - shift, 5, ShiftVariant::CircularLeft)
					.unwrap(),
				RotateRightType::Logical => oracles
					.add_shifted(*oracle_id, *shift, 5, ShiftVariant::LogicalRight)
					.unwrap(),
			};
		});
		oracles_arr[N - 1] = oracles
			.add_linear_combination(log_size, oracles_arr[..N - 1].iter().map(|s| (*s, F::ONE)))
			.unwrap();

		oracles_arr
	}

	fn gen_u32const_oracle_id<
		P: PackedField<Scalar: TowerField> + RepackedExtension<PackedBinaryField128x1b>,
	>(
		log_size: usize,
		oracles: &mut MultilinearOracleSet<P::Scalar>,
		x: u32,
	) -> OracleId {
		let x_unpacked = must_cast::<_, PackedBinaryField128x1b>(x as u128);

		let id = oracles
			.add_transparent(
				MultilinearExtensionTransparent::<_, P, _>::from_values_and_mu(vec![x_unpacked], 5)
					.expect("provided data of 32x1b elements is a power of two"),
			)
			.unwrap();

		oracles
			.add_repeating(id, log_size - PackedBinaryField32x1b::LOG_WIDTH)
			.unwrap()
	}

	pub fn new<P: PackedField<Scalar: TowerField> + RepackedExtension<PackedBinaryField128x1b>>(
		oracles: &mut MultilinearOracleSet<P::Scalar>,
		log_size: usize,
	) -> Self {
		let batch_id = oracles.add_committed_batch(log_size, BinaryField1b::TOWER_LEVEL);
		let w = oracles
			.add_named("omega")
			.committed_multiple::<16>(batch_id);

		let extended_w = array::from_fn(|_| U32ArraySumSubOracle::new(oracles, batch_id));

		let ch = oracles.add_named("ch").committed_multiple::<64>(batch_id);

		let temp1 = array::from_fn(|_| U32ArraySumSubOracle::new(oracles, batch_id));

		let maj = oracles.add_named("maj").committed_multiple::<64>(batch_id);

		let temp2 = array::from_fn(|_| U32ArraySumSubOracle::new(oracles, batch_id));

		let d_add_temp1 = array::from_fn(|_| U32ArraySumSubOracle::new(oracles, batch_id));

		let temp1_add_temp2 = array::from_fn(|_| U32ArraySumSubOracle::new(oracles, batch_id));

		let output_add = array::from_fn(|_| U32ArraySumSubOracle::new(oracles, batch_id));

		let get_w =
			|extended_w: &[U32ArraySumSubOracle<3>], w: &[OracleId], i: OracleId| -> usize {
				if i >= 16 {
					extended_w[i - 16].val()
				} else {
					w[i]
				}
			};

		// Define oracles for message schedule constraints
		let mut s0 = array::from_fn(|_| Default::default());
		let mut s1 = array::from_fn(|_| Default::default());
		for i in 0..48 {
			let extended_w_plus_1 = get_w(&extended_w, &w, i + 1);
			s0[i] = Self::gen_xor_oracle_ids(
				log_size,
				oracles,
				&[
					(extended_w_plus_1, 7, RotateRightType::Circular),
					(extended_w_plus_1, 18, RotateRightType::Circular),
					(extended_w_plus_1, 3, RotateRightType::Logical),
				],
			);
			let extended_w_plus_14 = get_w(&extended_w, &w, i + 14);
			s1[i] = Self::gen_xor_oracle_ids(
				log_size,
				oracles,
				&[
					(extended_w_plus_14, 17, RotateRightType::Circular),
					(extended_w_plus_14, 19, RotateRightType::Circular),
					(extended_w_plus_14, 10, RotateRightType::Logical),
				],
			);
		}

		// Define round constant oracles
		let k = array::from_fn(|i| {
			Self::gen_u32const_oracle_id::<P>(log_size, oracles, ROUND_CONSTS_K[i])
		});

		// Initialize state oracles
		let mut h: [[OracleId; 8]; 65] = array::from_fn(|_| [OracleId::MAX; 8]);
		h[0] = array::from_fn(|i| Self::gen_u32const_oracle_id::<P>(log_size, oracles, INIT[i]));

		// Define oracles for round constraints
		let mut sigma1 = array::from_fn(|_| Default::default());
		let mut sigma0 = array::from_fn(|_| Default::default());
		for i in 0..64 {
			let a = h[i][0];
			let b = h[i][1];
			let c = h[i][2];
			let e = h[i][4];
			let f = h[i][5];
			let g = h[i][6];

			sigma1[i] = Self::gen_xor_oracle_ids(
				log_size,
				oracles,
				&[
					(e, 6, RotateRightType::Circular),
					(e, 11, RotateRightType::Circular),
					(e, 25, RotateRightType::Circular),
				],
			);

			sigma0[i] = Self::gen_xor_oracle_ids(
				log_size,
				oracles,
				&[
					(a, 2, RotateRightType::Circular),
					(a, 13, RotateRightType::Circular),
					(a, 22, RotateRightType::Circular),
				],
			);

			h[i + 1][7] = g;
			h[i + 1][6] = f;
			h[i + 1][5] = e;
			h[i + 1][4] = d_add_temp1[i].val();
			h[i + 1][3] = c;
			h[i + 1][2] = b;
			h[i + 1][1] = a;
			h[i + 1][0] = temp1_add_temp2[i].val();
		}

		Self {
			batch_id,
			w,
			extended_w,
			ch,
			temp1,
			maj,
			temp2,
			d_add_temp1,
			temp1_add_temp2,
			output_add,
			s0,
			s1,
			h,
			k,
			sigma1,
			sigma0,
		}
	}

	pub fn get_w(&self, i: usize) -> OracleId {
		if i < 16 {
			self.w[i]
		} else {
			self.extended_w[i - 16].val()
		}
	}
}

composition_poly!(ZoutComposition[x, y, cin, z] = x + y + cin - z);
composition_poly!(CoutComposition[x, y, cin, cout] = (x + cin) * (y + cin) + cin - cout);

fn build_sum_compositions<const N: usize, P: PackedField>(
	x: &[OracleId],
	sum_sub_oracle: &U32ArraySumSubOracle<N>,
	builder: &mut ConstraintSetBuilder<P>,
) {
	for i in 0..N {
		let y = if i == 0 {
			x[0]
		} else {
			sum_sub_oracle.z_out[i - 1]
		};
		builder.add_zerocheck(
			[x[i + 1], y, sum_sub_oracle.cin[i], sum_sub_oracle.z_out[i]],
			ZoutComposition,
		);
		builder.add_zerocheck(
			[x[i + 1], y, sum_sub_oracle.cin[i], sum_sub_oracle.cout[i]],
			CoutComposition,
		);
	}
}

composition_poly!(ChComposition[h4,h5,h6, ch] = (h4*h5 + (1-h4)*h6) - ch);
composition_poly!(MajComposition[a, b, c, maj] = maj - a * b + a * c + b * c);

#[allow(clippy::identity_op, clippy::erasing_op)]
fn make_constraints<P: PackedField<Scalar: TowerField>>(
	trace_oracle: &TraceOracle,
) -> ConstraintSetBuilder<P> {
	let mut builder = ConstraintSetBuilder::new();

	// extended_w
	for i in 0..48 {
		build_sum_compositions(
			&[
				trace_oracle.get_w(i),
				trace_oracle.s0[i][3],
				trace_oracle.get_w(i + 9),
				trace_oracle.s1[i][3],
			],
			&trace_oracle.extended_w[i],
			&mut builder,
		)
	}

	// temp1
	for i in 0..64 {
		build_sum_compositions(
			&[
				trace_oracle.h[i][7],
				trace_oracle.sigma1[i][3],
				trace_oracle.ch[i],
				trace_oracle.k[i],
				trace_oracle.get_w(i),
			],
			&trace_oracle.temp1[i],
			&mut builder,
		)
	}

	// temp2
	for i in 0..64 {
		build_sum_compositions(
			&[trace_oracle.sigma0[i][3], trace_oracle.maj[i]],
			&trace_oracle.temp2[i],
			&mut builder,
		)
	}

	//d_add_temp1
	for i in 0..64 {
		build_sum_compositions(
			&[trace_oracle.h[i][3], trace_oracle.temp1[i].val()],
			&trace_oracle.d_add_temp1[i],
			&mut builder,
		)
	}

	// temp1_add_temp2
	for i in 0..64 {
		build_sum_compositions(
			&[trace_oracle.temp1[i].val(), trace_oracle.temp2[i].val()],
			&trace_oracle.temp1_add_temp2[i],
			&mut builder,
		)
	}

	// output_add
	for i in 0..8 {
		build_sum_compositions(
			&[trace_oracle.h[0][i], trace_oracle.h[64][i]],
			&trace_oracle.output_add[i],
			&mut builder,
		)
	}

	// maj
	for i in 0..64 {
		builder.add_zerocheck(
			[
				trace_oracle.h[i][0],
				trace_oracle.h[i][1],
				trace_oracle.h[i][2],
				trace_oracle.maj[i],
			],
			MajComposition,
		)
	}

	// ch

	for i in 0..64 {
		builder.add_zerocheck(
			[
				trace_oracle.h[i][4],
				trace_oracle.h[i][5],
				trace_oracle.h[i][6],
				trace_oracle.ch[i],
			],
			ChComposition,
		)
	}

	builder
}

struct Proof<F: Field, PCSComm, PCSProof> {
	trace_comm: PCSComm,
	zerocheck_proof: ZerocheckProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace_open_proof: PCSProof,
}

impl<F: Field, PCSComm, PCSProof> Proof<F, PCSComm, PCSProof> {
	fn isomorphic<F2: Field + From<F>>(self) -> Proof<F2, PCSComm, PCSProof> {
		Proof {
			trace_comm: self.trace_comm,
			zerocheck_proof: self.zerocheck_proof.isomorphic(),
			evalcheck_proof: self.evalcheck_proof.isomorphic(),
			trace_open_proof: self.trace_open_proof,
		}
	}
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
fn prove<U, F, FBase, DomainField, FEPCS, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: CH,
	mut witness: MultilinearExtensionIndex<U, F>,
	domain_factory: impl EvaluationDomainFactory<DomainField>,
	backend: &Backend,
) -> Result<Proof<F, PCS::Commitment, PCS::Proof>>
where
	U: UnderlierType
		+ PackScalar<BinaryField1b>
		+ PackScalar<F>
		+ PackScalar<DomainField>
		+ PackScalar<FBase>,
	PackedType<U, F>: PackedFieldIndexable,
	FEPCS: TowerField + From<F>,
	F: TowerField + From<FEPCS> + ExtensionField<DomainField> + ExtensionField<FBase>,
	FBase: TowerField + ExtensionField<DomainField>,
	DomainField: TowerField,
	PCS: PolyCommitScheme<PackedType<U, BinaryField1b>, FEPCS, Error: Debug, Proof: 'static>,
	CH: CanObserve<FEPCS> + CanObserve<PCS::Commitment> + CanSample<FEPCS> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	// Round 1
	let trace_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace_comm, trace_committed) = pcs.commit(&trace_commit_polys)?;
	challenger.observe(trace_comm.clone());

	// Zerocheck mixing
	let mut iso_challenger = IsomorphicChallenger::<_, _, F>::new(&mut challenger);

	let zerocheck_challenges = iso_challenger.sample_vec(log_size);

	let switchover_fn = standard_switchover_heuristic(-2);

	let constraint_set = make_constraints(trace_oracle).build_one(oracles)?;
	let constraint_set_base = make_constraints(trace_oracle).build_one(oracles)?;

	let (zerocheck_claim, meta) = sumcheck::constraint_set_zerocheck_claim(constraint_set.clone())?;

	let prover = sumcheck::prove::constraint_set_zerocheck_prover::<_, FBase, _, _, _>(
		constraint_set_base,
		constraint_set,
		&witness,
		domain_factory.clone(),
		switchover_fn,
		zerocheck_challenges.as_slice(),
		&backend,
	)?;

	let (sumcheck_output, zerocheck_proof) =
		sumcheck::prove::batch_prove(vec![prover], &mut iso_challenger)?;

	let zerocheck_output = sumcheck::zerocheck::verify_sumcheck_outputs(
		&[zerocheck_claim],
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck::make_eval_claims(oracles, [meta], zerocheck_output)?;

	// Evalcheck
	let GreedyEvalcheckProveOutput {
		same_query_claims,
		proof: evalcheck_proof,
	} = greedy_evalcheck::prove::<U, F, _, _, _>(
		oracles,
		&mut witness,
		evalcheck_multilinear_claims,
		switchover_fn,
		&mut iso_challenger,
		domain_factory,
		&backend,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, trace_oracle.batch_id);

	let trace_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;

	let eval_point: Vec<FEPCS> = same_query_claim
		.eval_point
		.into_iter()
		.map(|x| x.into())
		.collect();

	let trace_open_proof = pcs.prove_evaluation(
		&mut challenger,
		&trace_committed,
		&trace_commit_polys,
		&eval_point,
		backend,
	)?;

	Ok(Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	})
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all, level = "debug")]
fn verify<P, F, PCS, CH, Backend>(
	log_size: usize,
	oracles: &mut MultilinearOracleSet<F>,
	trace_oracle: &TraceOracle,
	pcs: &PCS,
	mut challenger: CH,
	proof: Proof<F, PCS::Commitment, PCS::Proof>,
	backend: &Backend,
) -> Result<()>
where
	P: PackedField<Scalar = BinaryField1b>,
	F: TowerField,
	PCS: PolyCommitScheme<P, F, Error: Debug, Proof: 'static>,
	CH: CanObserve<F> + CanObserve<PCS::Commitment> + CanSample<F> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	let Proof {
		trace_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace_comm.clone());

	// Zerocheck
	let constraint_set = make_constraints::<F>(trace_oracle).build_one(oracles)?;

	let zerocheck_challenges = challenger.sample_vec(log_size);

	let (zerocheck_claim, meta) = sumcheck::constraint_set_zerocheck_claim(constraint_set)?;
	let zerocheck_claims = [zerocheck_claim];

	let sumcheck_claims = sumcheck::zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;

	let sumcheck_output =
		sumcheck::batch_verify(&sumcheck_claims, zerocheck_proof, &mut challenger)?;

	let zerocheck_output = sumcheck::zerocheck::verify_sumcheck_outputs(
		&zerocheck_claims,
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck::make_eval_claims(oracles, [meta], zerocheck_output)?;

	// Evalcheck
	let same_query_claims = greedy_evalcheck::verify(
		oracles,
		evalcheck_multilinear_claims,
		evalcheck_proof,
		&mut challenger,
	)?;

	assert_eq!(same_query_claims.len(), 1);
	let (batch_id, same_query_claim) = same_query_claims
		.into_iter()
		.next()
		.expect("length is asserted to be 1");
	assert_eq!(batch_id, trace_oracle.batch_id);

	pcs.verify_evaluation(
		&mut challenger,
		&trace_comm,
		&same_query_claim.eval_point,
		trace_open_proof,
		&same_query_claim.evals,
		backend,
	)?;

	Ok(())
}

fn main() {
	const SECURITY_BITS: usize = 100;

	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let _guard = init_tracing().expect("failed to init tracing");

	type U = <PackedBinaryField128x1b as WithUnderlier>::Underlier;

	let log_size = get_log_trace_size().unwrap_or(10);
	let log_inv_rate = 1;
	let backend = make_portable_backend();

	let mut prover_oracles = MultilinearOracleSet::new();
	let prover_trace =
		TraceOracle::new::<PackedAESBinaryField1x128b>(&mut prover_oracles, log_size);

	let mut verifier_oracles = MultilinearOracleSet::new();
	let verifier_trace =
		TraceOracle::new::<PackedBinaryField1x128b>(&mut verifier_oracles, log_size);

	let trace_batch = prover_oracles.committed_batch(prover_trace.batch_id);

	// Set up the public parameters
	let pcs = tensor_pcs::find_proof_size_optimal_pcs::<
		<PackedBinaryField128x1b as WithUnderlier>::Underlier,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, trace_batch.n_polys, log_inv_rate, false)
	.unwrap();

	let n_compressions = 1 << (log_size - LOG_U32_BITS);
	let data_hashed_256 = ByteSize::b((n_compressions * SHA256_BLOCK_SIZE_BYTES) as u64);
	let tensorpcs_size = ByteSize::b(pcs.proof_size(trace_batch.n_polys) as u64);
	tracing::info!("Size of hashable SHA-256 data: {}", data_hashed_256);
	tracing::info!("Size of PCS proof: {}", tensorpcs_size);

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();

	let witness = generate_trace::<U, AESTowerField128b>(log_size, &prover_trace).unwrap();
	let domain_factory = IsomorphicEvaluationDomainFactory::<AESTowerField8b>::default();

	let proof =
		prove::<_, AESTowerField128b, AESTowerField8b, AESTowerField8b, BinaryField128b, _, _, _>(
			log_size,
			&mut prover_oracles,
			&prover_trace,
			&pcs,
			challenger.clone(),
			witness,
			domain_factory,
			&backend,
		)
		.unwrap();

	verify(
		log_size,
		&mut verifier_oracles,
		&verifier_trace,
		&pcs,
		challenger.clone(),
		proof.isomorphic(),
		&backend,
	)
	.unwrap();
}
