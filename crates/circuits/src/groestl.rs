// Copyright 2024 Irreducible Inc.

use crate::builder::ConstraintSystemBuilder;
use anyhow::Result;
use binius_core::{
	oracle::OracleId, transparent::constant::Constant, witness::MultilinearExtensionIndex,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	packed::{get_packed_slice, set_packed_slice},
	underlier::{Divisible, UnderlierType, WithUnderlier},
	AESTowerField8b, BinaryField1b, BinaryField8b, ExtensionField, Field,
	PackedAESBinaryField64x8b, PackedField, PackedFieldIndexable, TowerField,
};
use binius_hash::Groestl256Core;
use binius_math::CompositionPoly;
use bytemuck::{must_cast_slice_mut, Pod};
use itertools::chain;
use rand::thread_rng;
use std::{array, fmt::Debug, iter, slice};

/// Number of rounds in a Grøstl-256 compression
const N_ROUNDS: usize = 10;

const STATE_SIZE: usize = 64;

/// Constant vector of the Rijndael S-box affine transformation.
const SBOX_VEC: AESTowerField8b = AESTowerField8b::new(0x63);
/// Matrix columns of the Rijndael S-box affine transformation.
const SBOX_MATRIX: [AESTowerField8b; 8] = [
	AESTowerField8b::new(0b00011111),
	AESTowerField8b::new(0b00111110),
	AESTowerField8b::new(0b01111100),
	AESTowerField8b::new(0b11111000),
	AESTowerField8b::new(0b11110001),
	AESTowerField8b::new(0b11100011),
	AESTowerField8b::new(0b11000111),
	AESTowerField8b::new(0b10001111),
];
/// The first row of the circulant matrix defining the MixBytes step in Grøstl.
const MIX_BYTES_VEC: [AESTowerField8b; 8] = [
	AESTowerField8b::new(0x02),
	AESTowerField8b::new(0x02),
	AESTowerField8b::new(0x03),
	AESTowerField8b::new(0x04),
	AESTowerField8b::new(0x05),
	AESTowerField8b::new(0x03),
	AESTowerField8b::new(0x05),
	AESTowerField8b::new(0x07),
];

// TODO: Get rid of round constants and bake them into the constraints
#[derive(Debug, Clone)]
struct PermutationRoundGadget {
	// Internal oracles (some may be duplicated from input)
	with_round_consts: [OracleId; STATE_SIZE],

	// Internal gadgets is actually a Vec of fixed length STATE_SIZE
	/// S-box gadgets for AddRoundConstant & SubBytes
	p_sub_bytes: Vec<SBoxTraceGadget>,

	// Exported oracles
	output: [OracleId; STATE_SIZE],
}

impl PermutationRoundGadget {
	pub fn new<U, F, F8b>(
		log_size: usize,
		builder: &mut ConstraintSystemBuilder<U, F>,
		round: OracleId,
		multiples_16: &[OracleId],
		input: [OracleId; STATE_SIZE],
	) -> Result<Self>
	where
		U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		let with_round_consts: [OracleId; STATE_SIZE] = array::from_fn(|i| {
			if i % 8 == 0 {
				let col_idx = i / 8;
				builder
					.add_linear_combination(
						format!("with_round_consts[{i}]"),
						log_size,
						[
							(input[i], F::ONE),
							(round, F::ONE),
							(multiples_16[col_idx], F::ONE),
						],
					)
					.unwrap()
			} else {
				input[i]
			}
		});

		let p_sub_bytes = (0..STATE_SIZE)
			.map(|i| {
				builder.push_namespace(format!("p_sub_bytes[{i}]"));
				let sbox =
					SBoxTraceGadget::new::<U, F, F8b>(log_size, builder, with_round_consts[i]);
				builder.pop_namespace();
				sbox
			})
			.collect::<Result<Vec<_>>>()?;

		// Shift and mix bytes using committed columns
		let output = builder.add_committed_multiple("output", log_size, BinaryField8b::TOWER_LEVEL);

		Ok(Self {
			with_round_consts,
			p_sub_bytes,
			output,
		})
	}

	pub fn add_constraints<U, F, F8b>(&self, builder: &mut ConstraintSystemBuilder<U, F>)
	where
		U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		self.p_sub_bytes
			.iter()
			.for_each(|sub_bytes| sub_bytes.add_constraints::<U, F, F8b>(builder));

		self.p_sub_bytes.iter().enumerate().for_each(|(ij, _)| {
			let i = ij / 8;
			let j = ij % 8;

			let mut mix_shift_oracles = [OracleId::default(); 9];
			mix_shift_oracles[0] = self.output[ij];
			for k in 0..8 {
				let j_prime = (j + k) % 8;
				let i_prime = (i + j_prime) % 8;
				mix_shift_oracles[k + 1] = self.p_sub_bytes[i_prime * 8 + j_prime].output;
			}

			// This is not required if the columns are virtual
			builder.assert_zero(mix_shift_oracles, MixColumn::<F8b>::default());
		});
	}
}

#[derive(Debug, Default, Clone)]
struct SBoxTraceGadget {
	// Imported oracles
	input: OracleId,

	// Exported oracles
	/// The S-box output, defined as a linear combination of `p_sub_bytes_inv_bits`.
	output: OracleId,

	// Internal oracles
	/// Bits of the S-box inverse in the SubBytes step, decomposed using the AES field basis.
	inv_bits: [OracleId; 8],
	/// The S-box inverse in the SubBytes step, defined as a linear combination of
	/// `p_sub_bytes_inv_bits`.
	inverse: OracleId,
}

impl SBoxTraceGadget {
	pub fn new<U, F, F8b>(
		log_size: usize,
		builder: &mut ConstraintSystemBuilder<U, F>,
		input: OracleId,
	) -> Result<Self>
	where
		U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		let inv_bits =
			builder.add_committed_multiple("inv_bits", log_size, BinaryField1b::TOWER_LEVEL);
		let inverse = builder.add_linear_combination(
			"inverse",
			log_size,
			(0..8).map(|b| {
				let basis = F8b::from(
					<AESTowerField8b as ExtensionField<BinaryField1b>>::basis(b)
						.expect("index is less than extension degree"),
				);
				(inv_bits[b], basis.into())
			}),
		)?;
		let output = builder.add_linear_combination_with_offset(
			"output",
			log_size,
			F8b::from(SBOX_VEC).into(),
			(0..8).map(|b| (inv_bits[b], F8b::from(SBOX_MATRIX[b]).into())),
		)?;

		Ok(Self {
			input,
			output,
			inv_bits,
			inverse,
		})
	}

	pub fn add_constraints<U, F, F8b>(&self, builder: &mut ConstraintSystemBuilder<U, F>)
	where
		U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		builder.assert_zero([self.input, self.inverse], SBoxConstraint);
	}
}

struct TraceOracle {
	// Public columns
	/// permutation input state
	p_in: [OracleId; STATE_SIZE],
	/// permutation output state copied from last rounds output for simplicity
	p_out: [OracleId; STATE_SIZE],
	// round indexes, is actually a vec of length N_ROUNDS
	round_idxs: Vec<OracleId>,
	// i * 0x10 from i = 0, ..., 7
	multiples_16: Vec<OracleId>,
	rounds: Vec<PermutationRoundGadget>,
}

impl TraceOracle {
	fn new<U, F, F8b>(builder: &mut ConstraintSystemBuilder<U, F>, log_size: usize) -> Result<Self>
	where
		U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		let p_in = builder.add_committed_multiple::<STATE_SIZE>(
			"p_in",
			log_size,
			BinaryField8b::TOWER_LEVEL,
		);

		let mut round_idxs = Vec::with_capacity(N_ROUNDS);
		for i in 0..N_ROUNDS {
			let val: F8b = AESTowerField8b::new(i as u8).into();
			let val: F = val.into();
			round_idxs.push(builder.add_transparent(
				format!("round_idxs[{i}]"),
				Constant {
					n_vars: log_size,
					value: val,
				},
			)?);
		}

		let mut multiples_16 = Vec::with_capacity(8);
		for i in 0..8 {
			let val: F8b = AESTowerField8b::new(i as u8 * 0x10).into();
			let val: F = val.into();
			multiples_16.push(builder.add_transparent(
				format!("multiples_16[{i}]"),
				Constant {
					n_vars: log_size,
					value: val,
				},
			)?);
		}

		let mut rounds: Vec<PermutationRoundGadget> = Vec::with_capacity(N_ROUNDS);

		builder.push_namespace("rounds[0]");
		rounds.push(PermutationRoundGadget::new::<U, F, F8b>(
			log_size,
			builder,
			round_idxs[0],
			&multiples_16,
			p_in,
		)?);
		builder.pop_namespace();

		for round in 1..N_ROUNDS {
			builder.push_namespace(format!("rounds[{round}]"));
			rounds.push(PermutationRoundGadget::new::<U, F, F8b>(
				log_size,
				builder,
				round_idxs[round],
				&multiples_16,
				rounds[round - 1].output,
			)?);
			builder.pop_namespace();
		}

		let p_out = rounds[N_ROUNDS - 1].output;

		Ok(TraceOracle {
			p_in,
			p_out,
			round_idxs,
			multiples_16,
			rounds,
		})
	}

	pub fn add_constraints<U, F, F8b>(&self, builder: &mut ConstraintSystemBuilder<U, F>)
	where
		U: UnderlierType + Pod + PackScalar<F> + PackScalar<BinaryField1b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		self.rounds
			.iter()
			.for_each(|round| round.add_constraints::<U, F, F8b>(builder));
	}
}

#[derive(Debug, Clone)]
struct MixColumn<F8b: Clone> {
	mix_bytes: [F8b; 8],
}

impl<F8b: Clone + From<AESTowerField8b>> Default for MixColumn<F8b> {
	fn default() -> Self {
		Self {
			mix_bytes: MIX_BYTES_VEC.map(F8b::from),
		}
	}
}

impl<F8b, P> CompositionPoly<P> for MixColumn<F8b>
where
	F8b: Field,
	P: PackedField<Scalar: ExtensionField<F8b>>,
{
	fn n_vars(&self) -> usize {
		9
	}

	fn degree(&self) -> usize {
		1
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != 9 {
			return Err(binius_math::Error::IncorrectQuerySize { expected: 9 });
		}

		// This is unfortunate that it needs to unpack and repack...
		let result = iter::zip(query[1..].iter(), self.mix_bytes)
			.map(|(x_i, coeff)| P::from_fn(|j| x_i.get(j) * coeff))
			.sum::<P>();
		Ok(result - query[0])
	}

	fn binary_tower_level(&self) -> usize {
		AESTowerField8b::TOWER_LEVEL
	}
}

#[derive(Debug, Clone)]
struct SBoxConstraint;

impl<F, P> CompositionPoly<P> for SBoxConstraint
where
	F: TowerField,
	P: PackedField<Scalar = F>,
{
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		3
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		if query.len() != 2 {
			return Err(binius_math::Error::IncorrectQuerySize { expected: 2 });
		}

		let x = query[0];
		let inv = query[1];

		// x * inv == 1
		let non_zero_case = x * inv - F::ONE;

		// x == 0 AND inv == 0
		// TODO: Implement `mul_primitive` on packed tower fields
		let zero_case = x + P::from_fn(|i| {
			unsafe { inv.get_unchecked(i) }
				.mul_primitive(3)
				.expect("F must be tower height at least 4 by struct invariant")
		});

		// (x * inv == 1) OR (x == 0 AND inv == 0)
		Ok(non_zero_case * zero_case)
	}

	fn binary_tower_level(&self) -> usize {
		4
	}
}

struct PermutationRoundWitness<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	with_round_consts: [Box<[U]>; STATE_SIZE],
	p_sub_bytes: [SBoxGadgetWitness<U>; STATE_SIZE],
	output: [Box<[U]>; STATE_SIZE],
}

impl<U> PermutationRoundWitness<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	fn update_index<F, F8b>(
		&self,
		index: &mut MultilinearExtensionIndex<U, F>,
		gadget: &PermutationRoundGadget,
	) -> Result<()>
	where
		U: PackScalar<F> + PackScalar<F8b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		index.set_owned::<F8b, Box<[U]>>(iter::zip(
			chain!(gadget
				.with_round_consts
				.iter()
				.enumerate()
				.filter(|(i, _)| { i % 8 == 0 })
				.map(|(_, &o)| o),),
			chain!(self
				.with_round_consts
				.iter()
				.enumerate()
				.filter(|(i, _)| { i % 8 == 0 })
				.map(|(_, o)| o.clone())),
		))?;

		// Update sbox here
		for (p_sub_bytes, p_sub_bytes_witness) in
			iter::zip(gadget.p_sub_bytes.clone(), self.p_sub_bytes.iter())
		{
			p_sub_bytes_witness.update_index::<F, F8b>(index, &p_sub_bytes)?;
		}

		index.set_owned::<F8b, _>(iter::zip(gadget.output, self.output.clone()))?;
		Ok(())
	}
}

struct SBoxGadgetWitness<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	output: Box<[U]>,
	inv_bits: [Box<[U]>; 8],
	inverse: Box<[U]>,
}

impl<U> SBoxGadgetWitness<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	fn update_index<F, F8b>(
		&self,
		index: &mut MultilinearExtensionIndex<U, F>,
		gadget: &SBoxTraceGadget,
	) -> Result<()>
	where
		U: PackScalar<F> + PackScalar<F8b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		index.set_owned::<BinaryField1b, _>(iter::zip(gadget.inv_bits, self.inv_bits.clone()))?;
		index.set_owned::<F8b, _>(iter::zip(
			[gadget.inverse, gadget.output],
			[self.inverse.clone(), self.output.clone()],
		))?;
		Ok(())
	}
}

struct TraceWitness<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	p_in: [Box<[U]>; STATE_SIZE],
	_p_out: [Box<[U]>; STATE_SIZE],
	round_idxs: [Box<[U]>; N_ROUNDS],
	multiples_16: [Box<[U]>; 8],
	rounds: [PermutationRoundWitness<U>; N_ROUNDS],
}

impl<U> TraceWitness<U>
where
	U: UnderlierType + PackScalar<BinaryField1b>,
{
	fn update_index<F, F8b>(
		&self,
		trace_oracle: &TraceOracle,
		witness: &mut MultilinearExtensionIndex<U, F>,
	) -> Result<()>
	where
		U: PackScalar<F> + PackScalar<F8b>,
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		witness.set_owned::<F8b, _>(iter::zip(
			chain!(
				trace_oracle.p_in,
				trace_oracle.round_idxs.clone(),
				trace_oracle.multiples_16.clone(),
			),
			chain!(self.p_in.clone(), self.round_idxs.clone(), self.multiples_16.clone(),),
		))?;
		for (permutation_round, permutation_round_witness) in
			iter::zip(trace_oracle.rounds.clone(), self.rounds.iter())
		{
			permutation_round_witness.update_index::<F, F8b>(witness, &permutation_round)?;
		}
		Ok(())
	}
}

fn s_box(x: AESTowerField8b) -> AESTowerField8b {
	#[rustfmt::skip]
	const S_BOX: [u8; 256] = [
		0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
		0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
		0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
		0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
		0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
		0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
		0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
		0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
		0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
		0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
		0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
		0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
		0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
		0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
		0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
		0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
		0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
		0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
		0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
		0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
		0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
		0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
		0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
		0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
		0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
		0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
		0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
		0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
		0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
		0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
		0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
		0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
	];
	let idx = u8::from(x) as usize;
	AESTowerField8b::from(S_BOX[idx])
}

impl<U> TraceWitness<U>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<BinaryField1b>
		+ PackScalar<AESTowerField8b>
		+ Divisible<u8>,
{
	pub fn generate_trace(log_size: usize) -> Self {
		let build_trace_column_1b =
			|| vec![U::default(); 1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)];
		let build_trace_column_8b =
			|| vec![U::default(); 1 << (log_size - <PackedType<U, AESTowerField8b>>::LOG_WIDTH)];

		let mut p_in_vec: [Vec<U>; STATE_SIZE] = array::from_fn(|_| build_trace_column_8b());
		let mut round_idxs_vec: [Vec<U>; N_ROUNDS] = array::from_fn(|_| build_trace_column_8b());
		let mut multiples_16_vec: [Vec<U>; 8] = array::from_fn(|_| build_trace_column_8b());
		let mut round_outs_vec: [[Vec<U>; STATE_SIZE]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));
		let mut sub_bytes_out_vec: [[Vec<U>; STATE_SIZE]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));
		let mut sub_bytes_inv_vec: [[Vec<U>; STATE_SIZE]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));
		let mut sub_bytes_inv_bits_vec: [[[Vec<U>; 8]; STATE_SIZE]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| array::from_fn(|_| build_trace_column_1b())));
		let mut with_round_consts_vec: [[Vec<U>; STATE_SIZE]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));

		#[allow(clippy::ptr_arg)]
		fn cast_8b_col<U: UnderlierType + Pod>(col: &mut Vec<U>) -> &mut [AESTowerField8b] {
			must_cast_slice_mut::<_, AESTowerField8b>(col)
		}

		fn cast_8b_cols<U: UnderlierType + Pod, const N: usize>(
			cols: &mut [Vec<U>; N],
		) -> [&mut [AESTowerField8b]; N] {
			cols.each_mut().map(cast_8b_col) // |col| PackedFieldIndexable::unpack_scalars_mut(col.as_mut_slice()))
		}

		let p_in = cast_8b_cols(&mut p_in_vec);
		let round_idxs = cast_8b_cols(&mut round_idxs_vec);
		let multiples_16 = cast_8b_cols(&mut multiples_16_vec);

		let mut rng = thread_rng();
		let groestl_core = Groestl256Core;

		for z in 0..1 << log_size {
			// Randomly generate the initial compression input
			let input = PackedAESBinaryField64x8b::random(&mut rng);
			let output = groestl_core.permutation_p(input);

			#[allow(clippy::needless_range_loop)]
			for i in 0..8 {
				multiples_16[i][z] = AESTowerField8b::new(i as u8 * 0x10);
			}

			// Assign the compression input
			for ij in 0..STATE_SIZE {
				let input_elems = PackedFieldIndexable::unpack_scalars(slice::from_ref(&input));
				p_in[ij][z] = input_elems[ij];
			}

			let mut prev_round_out = input;

			for r in 0..N_ROUNDS {
				let with_round_consts = cast_8b_cols(&mut with_round_consts_vec[r]);
				let round_out = cast_8b_cols(&mut round_outs_vec[r]);

				round_idxs[r][z] = AESTowerField8b::new(r as u8);

				// AddRoundConstant & SubBytes
				#[allow(clippy::needless_range_loop)]
				for i in 0..8 {
					for j in 0..8 {
						let ij = i * 8 + j;
						let p_sub_bytes_inv = cast_8b_col(&mut sub_bytes_inv_vec[r][ij]);
						let p_sub_bytes_out = cast_8b_col(&mut sub_bytes_out_vec[r][ij]);

						let round_in =
							PackedFieldIndexable::unpack_scalars(slice::from_ref(&prev_round_out))
								[ij];

						let with_rc = if j == 0 {
							round_in + round_idxs[r][z] + multiples_16[i][z]
						} else {
							round_in
						};

						with_round_consts[ij][z] = with_rc;

						let p_sbox_in: AESTowerField8b = with_rc;

						p_sub_bytes_inv[z] = p_sbox_in.invert_or_zero();

						let inv_bits =
							<AESTowerField8b as ExtensionField<BinaryField1b>>::iter_bases(
								&p_sub_bytes_inv[z],
							);
						for (b, bit) in inv_bits.enumerate() {
							let p_sub_bytes_inv_bit =
								sub_bytes_inv_bits_vec[r][ij][b].as_mut_slice();

							let as_packed = PackedType::<U, BinaryField1b>::from_underliers_ref_mut(
								p_sub_bytes_inv_bit,
							);
							set_packed_slice(as_packed, z, bit);
						}

						p_sub_bytes_out[z] = s_box(p_sbox_in);
					}
				}

				// ShiftBytes & MixBytes
				fn shift_p_func(j: usize, i: usize) -> usize {
					let i_prime = (i + j) % 8;
					i_prime * 8 + j
				}

				fn get_a_j<U>(
					p_sub_bytes: &[Vec<U>; STATE_SIZE],
					z: usize,
					j: usize,
				) -> [AESTowerField8b; 8]
				where
					U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<AESTowerField8b>,
					PackedType<U, AESTowerField8b>: PackedFieldIndexable,
				{
					array::from_fn(|i| {
						let x = p_sub_bytes[shift_p_func(i, j)].as_slice();
						let x_as_packed = PackedType::<U, AESTowerField8b>::from_underliers_ref(x);
						get_packed_slice(x_as_packed, z)
					})
				}
				let two = AESTowerField8b::new(2);
				for j in 0..8 {
					let a_j = get_a_j::<U>(&sub_bytes_out_vec[r], z, j);
					for i in 0..8 {
						let prev_round_out_slice = PackedFieldIndexable::unpack_scalars_mut(
							slice::from_mut(&mut prev_round_out),
						);
						let ij = j * 8 + i;
						let a_i: [AESTowerField8b; 8] = array::from_fn(|k| a_j[(i + k) % 8]);
						// Here we are using an optimized matrix multiplication, as documented in
						// section 4.4.2 of https://www.groestl.info/groestl-implementation-guide.pdf
						let b_ij = two
							* (two * (a_i[3] + a_i[4] + a_i[6] + a_i[7])
								+ a_i[0] + a_i[1] + a_i[2]
								+ a_i[5] + a_i[7]) + a_i[2]
							+ a_i[4] + a_i[5] + a_i[6]
							+ a_i[7];
						round_out[ij][z] = b_ij;
						prev_round_out_slice[ij] = b_ij;
					}
				}
			}

			// Assert correct output
			for ij in 0..STATE_SIZE {
				let output_elems = PackedFieldIndexable::unpack_scalars(slice::from_ref(&output));
				let perm_out = cast_8b_cols(&mut round_outs_vec[N_ROUNDS - 1]);
				assert_eq!(perm_out[ij][z], output_elems[ij]);
			}
		}

		fn vec_to_arc<U: UnderlierType, const N: usize>(cols: [Vec<U>; N]) -> [Box<[U]>; N] {
			cols.map(|x| x.into_boxed_slice())
		}

		let p_in_arc = vec_to_arc(p_in_vec);
		let round_idxs_arc = vec_to_arc(round_idxs_vec);
		let multiples_16_arc = vec_to_arc(multiples_16_vec);
		let round_outs_arc: [[Box<[U]>; STATE_SIZE]; N_ROUNDS] =
			round_outs_vec.map(|r| vec_to_arc(r));
		let sub_bytes_out_arc: [[Box<[U]>; STATE_SIZE]; N_ROUNDS] =
			sub_bytes_out_vec.map(|r| vec_to_arc(r));
		let sub_bytes_inv_arc: [[Box<[U]>; STATE_SIZE]; N_ROUNDS] =
			sub_bytes_inv_vec.map(|r| vec_to_arc(r));
		let sub_bytes_inv_bits_arc: [[[Box<[U]>; 8]; STATE_SIZE]; N_ROUNDS] =
			sub_bytes_inv_bits_vec.map(|r| r.map(|ij| vec_to_arc(ij)));
		let with_round_consts_arc: [[Box<[U]>; STATE_SIZE]; N_ROUNDS] =
			with_round_consts_vec.map(|r| vec_to_arc(r));

		TraceWitness {
			p_in: p_in_arc.clone(),
			_p_out: array::from_fn(|ij| round_outs_arc[N_ROUNDS - 1][ij].clone()),
			round_idxs: round_idxs_arc,
			multiples_16: multiples_16_arc,
			rounds: array::from_fn(|r| PermutationRoundWitness {
				with_round_consts: array::from_fn(|ij| {
					if ij % 8 == 0 {
						with_round_consts_arc[r][ij].clone()
					} else if r == 0 {
						p_in_arc[ij].clone()
					} else {
						round_outs_arc[r][ij].clone()
					}
				}),
				output: round_outs_arc[r].clone(),
				p_sub_bytes: array::from_fn(|ij| SBoxGadgetWitness {
					output: sub_bytes_out_arc[r][ij].clone(),
					inv_bits: sub_bytes_inv_bits_arc[r][ij].clone(),
					inverse: sub_bytes_inv_arc[r][ij].clone(),
				}),
			}),
		}
	}
}

pub fn groestl_p_permutation<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	log_size: usize,
) -> Result<[OracleId; STATE_SIZE]>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<F>
		+ PackScalar<BinaryField1b>
		+ PackScalar<AESTowerField8b>
		+ Divisible<u8>,
	F: TowerField + ExtensionField<AESTowerField8b>,
{
	let trace_oracle = TraceOracle::new::<U, F, AESTowerField8b>(builder, log_size)?;

	if let Some(ext_index) = builder.witness() {
		let trace_witness = TraceWitness::generate_trace(log_size);
		trace_witness.update_index::<F, AESTowerField8b>(&trace_oracle, ext_index)?;
	}

	trace_oracle.add_constraints::<U, F, AESTowerField8b>(builder);

	Ok(trace_oracle.p_out)
}
