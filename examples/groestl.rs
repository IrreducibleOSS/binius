// Copyright 2024 Irreducible Inc.

//! This an example SNARK for proving the P permutation of the Grøstl-256 hash function.
//!
//! The Grøstl hash function family is based on two permutations P and Q, which are nearly
//! identical aside from a few constants. Both permutations are used in the compression function
//! and the P permutation is additional used to finalize the hash digest.

#![feature(array_try_from_fn)]
#![feature(step_trait)]

use anyhow::{ensure, Result};
use binius_core::{
	challenger::{new_hasher_challenger, IsomorphicChallenger},
	oracle::{BatchId, ConstraintSetBuilder, MultilinearOracleSet, OracleId},
	poly_commit::{tensor_pcs, PolyCommitScheme},
	polynomial::MultilinearComposite,
	protocols::{
		greedy_evalcheck::{self, GreedyEvalcheckProof, GreedyEvalcheckProveOutput},
		sumcheck::{self, zerocheck, Proof as ZerocheckProof},
	},
	transparent::constant::Constant,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::OptimalUnderlier128b,
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{PackedTransformationFactory, Transformation},
	packed::{get_packed_slice, set_packed_slice},
	underlier::{UnderlierType, WithUnderlier},
	AESTowerField128b, AESTowerField16b, AESTowerField8b, BinaryField128b, BinaryField16b,
	BinaryField1b, BinaryField8b, ExtensionField, Field, PackedAESBinaryField64x8b,
	PackedBinaryField1x128b, PackedField, PackedFieldIndexable, TowerField,
	AES_TO_BINARY_LINEAR_TRANSFORMATION,
};
use binius_hal::{make_portable_backend, ComputationBackend, MultilinearExtension};
use binius_hash::{Groestl256Core, GroestlHasher};
use binius_math::{CompositionPoly, EvaluationDomainFactory, IsomorphicEvaluationDomainFactory};
use binius_utils::{
	examples::get_log_trace_size, rayon::adjust_thread_pool, tracing::init_tracing,
};
use bytesize::ByteSize;
use itertools::chain;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use rand::thread_rng;
use std::{array, fmt::Debug, iter, iter::Step, slice, sync::Arc};
use tracing::instrument;

/// Number of rounds in a Grøstl-256 compression
const N_ROUNDS: usize = 10;

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
	with_round_consts: [OracleId; 64],

	// Internal gadgets
	/// S-box gadgets for AddRoundConstant & SubBytes
	p_sub_bytes: [SBoxTraceGadget; 64],

	// Exported oracles
	output: [OracleId; 64],
}

impl PermutationRoundGadget {
	pub fn new<F, F8b>(
		log_size: usize,
		oracles: &mut MultilinearOracleSet<F>,
		round: OracleId,
		multiples_16: &[OracleId; 8],
		input: [OracleId; 64],
		trace1b_batch_id: BatchId,
		trace8b_batch_id: BatchId,
	) -> Result<Self>
	where
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		let with_round_consts: [OracleId; 64] = array::from_fn(|i| {
			if i % 8 == 0 {
				let col_idx = i / 8;
				oracles
					.add_linear_combination(
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

		let p_sub_bytes = array::try_from_fn(|i| {
			SBoxTraceGadget::new::<_, F8b>(
				log_size,
				oracles,
				with_round_consts[i],
				trace1b_batch_id,
			)
		})?;

		// Shift and mix bytes using committed columns
		let output = oracles.add_committed_multiple(trace8b_batch_id);

		Ok(Self {
			with_round_consts,
			p_sub_bytes,
			output,
		})
	}

	pub fn iter_internal_and_exported_oracle_ids(&self) -> impl Iterator<Item = OracleId> + '_ {
		chain!(
			self.with_round_consts
				.iter()
				.enumerate()
				.filter(|(i, _)| { i % 8 == 0 })
				.map(|(_, &o)| o),
			self.p_sub_bytes
				.iter()
				.flat_map(|sub_bytes| sub_bytes.iter_internal_and_exported_oracle_ids()),
			self.output
		)
	}

	pub fn add_constraints<F8b, PW>(&self, builder: &mut ConstraintSetBuilder<PW>)
	where
		F8b: TowerField + From<AESTowerField8b>,
		PW: PackedField<Scalar: TowerField + ExtensionField<F8b>>,
	{
		self.p_sub_bytes
			.iter()
			.for_each(|sub_bytes| sub_bytes.add_constraints::<F8b, PW>(builder));

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
			builder.add_zerocheck(mix_shift_oracles, MixColumn::<F8b>::default());
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
	pub fn new<F, F8b>(
		log_size: usize,
		oracles: &mut MultilinearOracleSet<F>,
		input: OracleId,
		trace1b_batch_id: BatchId,
	) -> Result<Self>
	where
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		let inv_bits = oracles.add_committed_multiple(trace1b_batch_id);
		let inverse = oracles.add_named("sbox_inverse").linear_combination(
			log_size,
			(0..8).map(|b| {
				let basis = F8b::from(
					<AESTowerField8b as ExtensionField<BinaryField1b>>::basis(b)
						.expect("index is less than extension degree"),
				);
				(inv_bits[b], basis.into())
			}),
		)?;
		let output = oracles
			.add_named("sbox_output")
			.linear_combination_with_offset(
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

	pub fn iter_internal_and_exported_oracle_ids(&self) -> impl Iterator<Item = OracleId> + '_ {
		chain!(self.inv_bits, [self.inverse, self.output])
	}

	pub fn add_constraints<F8b, PW>(&self, builder: &mut ConstraintSetBuilder<PW>)
	where
		F8b: TowerField + From<AESTowerField8b>,
		PW: PackedField<Scalar: TowerField + ExtensionField<F8b>>,
	{
		builder.add_zerocheck([self.input, self.inverse], SBoxConstraint);
	}
}

struct TraceOracle {
	/// Base-2 log of the number of trace rows
	log_size: usize,

	// Public columns
	/// permutation input state
	p_in: [OracleId; 64],
	/// permutation output state copied from last rounds output for simplicity
	_p_out: [OracleId; 64],
	// round indexes
	round_idxs: [OracleId; N_ROUNDS],
	// i * 0x10 from i = 0, ..., 7
	multiples_16: [OracleId; 8],
	rounds: Vec<PermutationRoundGadget>,

	// Batch IDs
	trace1b_batch_id: BatchId,
	trace8b_batch_id: BatchId,
}

impl TraceOracle {
	fn new<F, F8b>(oracles: &mut MultilinearOracleSet<F>, log_size: usize) -> Result<Self>
	where
		F: TowerField + ExtensionField<F8b>,
		F8b: TowerField + From<AESTowerField8b>,
	{
		let trace1b_batch_id = oracles.add_committed_batch(log_size, BinaryField1b::TOWER_LEVEL);
		let trace8b_batch_id = oracles.add_committed_batch(log_size, BinaryField8b::TOWER_LEVEL);
		let p_in = oracles.add_committed_multiple::<64>(trace8b_batch_id);

		let round_idxs = array::try_from_fn(|i| {
			let val: F8b = AESTowerField8b::new(i as u8).into();
			let val: F = val.into();
			oracles.add_transparent(Constant {
				n_vars: log_size,
				value: val,
			})
		})?;

		let multiples_16 = array::try_from_fn(|i| {
			let val: F8b = AESTowerField8b::new(i as u8 * 0x10).into();
			let val: F = val.into();
			oracles.add_transparent(Constant {
				n_vars: log_size,
				value: val,
			})
		})?;

		let mut rounds: Vec<PermutationRoundGadget> = Vec::with_capacity(N_ROUNDS);

		rounds.push(PermutationRoundGadget::new::<_, F8b>(
			log_size,
			oracles,
			round_idxs[0],
			&multiples_16,
			p_in,
			trace1b_batch_id,
			trace8b_batch_id,
		)?);
		for round in 1..N_ROUNDS {
			rounds.push(PermutationRoundGadget::new::<_, F8b>(
				log_size,
				oracles,
				round_idxs[round],
				&multiples_16,
				rounds[round - 1].output,
				trace1b_batch_id,
				trace8b_batch_id,
			)?);
		}

		let _p_out = rounds[N_ROUNDS - 1].output;

		Ok(TraceOracle {
			log_size,
			p_in,
			_p_out,
			round_idxs,
			multiples_16,
			rounds,
			trace1b_batch_id,
			trace8b_batch_id,
		})
	}

	fn iter_oracle_ids(&self) -> impl Iterator<Item = OracleId> + '_ {
		chain!(
			self.p_in,
			self.round_idxs,
			self.multiples_16,
			self.rounds
				.iter()
				.flat_map(|round| round.iter_internal_and_exported_oracle_ids()),
		)
	}

	pub fn add_constraints<F8b, PW>(&self, builder: &mut ConstraintSetBuilder<PW>)
	where
		F8b: TowerField + From<AESTowerField8b>,
		PW: PackedField<Scalar: TowerField + ExtensionField<F8b>>,
	{
		self.rounds
			.iter()
			.for_each(|round| round.add_constraints::<F8b, PW>(builder));
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

fn make_constraints<F8b, PW>(trace_oracle: &TraceOracle) -> ConstraintSetBuilder<PW>
where
	F8b: TowerField + From<AESTowerField8b>,
	PW: PackedField<Scalar: TowerField + ExtensionField<F8b>>,
{
	let mut builder = ConstraintSetBuilder::new();

	trace_oracle.add_constraints::<F8b, PW>(&mut builder);

	builder
}

struct PermutationRoundWitness<U, F1b, F8b>
where
	U: UnderlierType + PackScalar<F1b> + PackScalar<F8b>,
	F1b: TowerField,
	F8b: TowerField,
{
	with_round_consts: [Arc<Vec<PackedType<U, F8b>>>; 64],
	p_sub_bytes: [SBoxGadgetWitness<U, F1b, F8b>; 64],
	output: [Arc<Vec<PackedType<U, F8b>>>; 64],
}

impl<U, F1b, F8b> PermutationRoundWitness<U, F1b, F8b>
where
	U: UnderlierType + PackScalar<F1b> + PackScalar<F8b>,
	F1b: TowerField,
	F8b: TowerField,
{
	fn update_index<'a, 'b, F>(
		&'a self,
		index: MultilinearExtensionIndex<'b, U, F>,
		gadget: &PermutationRoundGadget,
	) -> Result<MultilinearExtensionIndex<'a, U, F>>
	where
		'b: 'a,
		U: PackScalar<F>,
		F: ExtensionField<F1b> + ExtensionField<F8b>,
	{
		let index = index.update_packed::<F8b>(iter::zip(
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
				.map(|(_, o)| o.as_ref().as_slice())),
		))?;

		// Update sbox here
		let index = iter::zip(gadget.p_sub_bytes.clone(), self.p_sub_bytes.iter()).try_fold(
			index,
			|index, (p_sub_bytes, p_sub_bytes_witness)| {
				p_sub_bytes_witness.update_index(index, &p_sub_bytes)
			},
		)?;

		let index = index.update_packed::<F8b>(iter::zip(
			gadget.output,
			self.output.iter().map(|o| o.as_ref().as_slice()),
		))?;
		Ok(index)
	}
}

struct SBoxGadgetWitness<U, F1b, F8b>
where
	U: UnderlierType + PackScalar<F1b> + PackScalar<F8b>,
	F1b: TowerField,
	F8b: TowerField,
{
	output: Arc<Vec<PackedType<U, F8b>>>,
	inv_bits: [Arc<Vec<PackedType<U, F1b>>>; 8],
	inverse: Arc<Vec<PackedType<U, F8b>>>,
}

impl<U, F1b, F8b> SBoxGadgetWitness<U, F1b, F8b>
where
	U: UnderlierType + PackScalar<F1b> + PackScalar<F8b>,
	F1b: TowerField,
	F8b: TowerField,
{
	fn update_index<'a, 'b, F>(
		&'a self,
		index: MultilinearExtensionIndex<'b, U, F>,
		gadget: &SBoxTraceGadget,
	) -> Result<MultilinearExtensionIndex<'a, U, F>>
	where
		'b: 'a,
		U: PackScalar<F>,
		F: ExtensionField<F1b> + ExtensionField<F8b>,
	{
		let index = index
			.update_packed::<F1b>(iter::zip(
				gadget.inv_bits,
				self.inv_bits.iter().map(|x| x.as_ref().as_slice()),
			))?
			.update_packed::<F8b>(iter::zip(
				[gadget.inverse, gadget.output],
				[&self.inverse, &self.output].map(|x| x.as_ref().as_slice()),
			))?;
		Ok(index)
	}
}

struct TraceWitness<U, F1b, F8b>
where
	U: UnderlierType + PackScalar<F1b> + PackScalar<F8b>,
	F1b: TowerField,
	F8b: TowerField,
{
	p_in: [Arc<Vec<PackedType<U, F8b>>>; 64],
	_p_out: [Arc<Vec<PackedType<U, F8b>>>; 64],
	round_idxs: [Arc<Vec<PackedType<U, F8b>>>; N_ROUNDS],
	multiples_16: [Arc<Vec<PackedType<U, F8b>>>; 8],
	rounds: [PermutationRoundWitness<U, F1b, F8b>; N_ROUNDS],
}

impl<U, F1b, F8b> TraceWitness<U, F1b, F8b>
where
	U: UnderlierType + PackScalar<F1b> + PackScalar<F8b>,
	F1b: TowerField,
	F8b: TowerField,
{
	fn to_index<F>(&self, trace_oracle: &TraceOracle) -> Result<MultilinearExtensionIndex<U, F>>
	where
		U: PackScalar<F>,
		F: ExtensionField<F1b> + ExtensionField<F8b>,
	{
		let index = MultilinearExtensionIndex::new().update_packed::<F8b>(iter::zip(
			chain!(trace_oracle.p_in, trace_oracle.round_idxs, trace_oracle.multiples_16,),
			chain!(self.p_in.each_ref(), self.round_idxs.each_ref(), self.multiples_16.each_ref(),)
				.map(|x| x.as_ref().as_slice()),
		))?;
		let index = iter::zip(trace_oracle.rounds.clone(), self.rounds.iter()).try_fold(
			index,
			|index, (permutation_round, permutation_round_witness)| {
				permutation_round_witness.update_index(index, &permutation_round)
			},
		)?;
		Ok(index)
	}
}

/// Convert a witness polynomial to the 8-bit verifier field from the isomorphic 8-bit prover
/// field.
fn convert_poly_witness_to_tower<P8b, PW8b>(
	poly: MultilinearExtension<PW8b, &[PW8b]>,
) -> Result<MultilinearExtension<P8b>>
where
	P8b: PackedField<Scalar = BinaryField8b>,
	PW8b: PackedField<Scalar = AESTowerField8b> + PackedTransformationFactory<P8b>,
{
	let transform = <PW8b as PackedTransformationFactory<P8b>>::make_packed_transformation(
		AES_TO_BINARY_LINEAR_TRANSFORMATION,
	);

	let values = poly
		.evals()
		.iter()
		.map(|val| transform.transform(val))
		.collect();
	let result = MultilinearExtension::from_values(values)?;
	Ok(result)
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

impl<U> TraceWitness<U, BinaryField1b, AESTowerField8b>
where
	U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<AESTowerField8b>,
	PackedType<U, AESTowerField8b>: PackedFieldIndexable,
{
	#[instrument(level = "debug")]
	pub fn generate_trace(log_size: usize) -> Self {
		let build_trace_column_1b = || {
			vec![
				<PackedType<U, BinaryField1b>>::default();
				1 << (log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH)
			]
		};
		let build_trace_column_8b = || {
			vec![
				<PackedType<U, AESTowerField8b>>::default();
				1 << (log_size - <PackedType<U, AESTowerField8b>>::LOG_WIDTH)
			]
		};

		type Vec1b<U> = Vec<PackedType<U, BinaryField1b>>;
		let mut p_in_vec: [Vec<PackedType<U, AESTowerField8b>>; 64] =
			array::from_fn(|_| build_trace_column_8b());
		let mut round_idxs_vec: [Vec<PackedType<U, AESTowerField8b>>; N_ROUNDS] =
			array::from_fn(|_| build_trace_column_8b());
		let mut multiples_16_vec: [Vec<PackedType<U, AESTowerField8b>>; 8] =
			array::from_fn(|_| build_trace_column_8b());
		let mut round_outs_vec: [[Vec<PackedType<U, AESTowerField8b>>; 64]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));
		let mut sub_bytes_out_vec: [[Vec<PackedType<U, AESTowerField8b>>; 64]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));
		let mut sub_bytes_inv_vec: [[Vec<PackedType<U, AESTowerField8b>>; 64]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));
		let mut sub_bytes_inv_bits_vec: [[[Vec1b<U>; 8]; 64]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| array::from_fn(|_| build_trace_column_1b())));
		let mut with_round_consts_vec: [[Vec<PackedType<U, AESTowerField8b>>; 64]; N_ROUNDS] =
			array::from_fn(|_| array::from_fn(|_| build_trace_column_8b()));

		fn cast_8b_col<P8b: PackedFieldIndexable<Scalar = AESTowerField8b>>(
			col: &mut Vec<P8b>,
		) -> &mut [AESTowerField8b] {
			PackedFieldIndexable::unpack_scalars_mut(col.as_mut_slice())
		}

		fn cast_8b_cols<P8b: PackedFieldIndexable<Scalar = AESTowerField8b>, const N: usize>(
			cols: &mut [Vec<P8b>; N],
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

			for i in 0..8 {
				multiples_16[i][z] = AESTowerField8b::new(i as u8 * 0x10);
			}

			// Assign the compression input
			for ij in 0..64 {
				let input_elems = PackedFieldIndexable::unpack_scalars(slice::from_ref(&input));
				p_in[ij][z] = input_elems[ij];
			}

			let mut prev_round_out = input;

			for r in 0..N_ROUNDS {
				let with_round_consts = cast_8b_cols(&mut with_round_consts_vec[r]);
				let round_out = cast_8b_cols(&mut round_outs_vec[r]);

				round_idxs[r][z] = AESTowerField8b::new(r as u8);

				// AddRoundConstant & SubBytes
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
							let p_sub_bytes_inv_bit = &mut sub_bytes_inv_bits_vec[r][ij][b];
							set_packed_slice(p_sub_bytes_inv_bit.as_mut_slice(), z, bit);
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
					p_sub_bytes: &[Vec<PackedType<U, AESTowerField8b>>; 64],
					z: usize,
					j: usize,
				) -> [AESTowerField8b; 8]
				where
					U: UnderlierType + PackScalar<BinaryField1b> + PackScalar<AESTowerField8b>,
					PackedType<U, AESTowerField8b>: PackedFieldIndexable,
				{
					array::from_fn(|i| {
						let x = p_sub_bytes[shift_p_func(i, j)].as_slice();
						get_packed_slice(x, z)
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
			for ij in 0..64 {
				let output_elems = PackedFieldIndexable::unpack_scalars(slice::from_ref(&output));
				let perm_out = cast_8b_cols(&mut round_outs_vec[N_ROUNDS - 1]);
				assert_eq!(perm_out[ij][z], output_elems[ij]);
			}
		}

		fn vec_to_arc<P: PackedField, const N: usize>(cols: [Vec<P>; N]) -> [Arc<Vec<P>>; N] {
			cols.map(Arc::new)
		}

		type ArcVec8b<U> = Arc<Vec<PackedType<U, AESTowerField8b>>>;
		type ArcVec1b<U> = Arc<Vec<PackedType<U, BinaryField1b>>>;

		let p_in_arc = vec_to_arc(p_in_vec);
		let round_idxs_arc = vec_to_arc(round_idxs_vec);
		let multiples_16_arc = vec_to_arc(multiples_16_vec);
		let round_outs_arc: [[ArcVec8b<U>; 64]; N_ROUNDS] =
			array::from_fn(|r| array::from_fn(|ij| Arc::new(round_outs_vec[r][ij].clone())));
		let sub_bytes_out_arc: [[ArcVec8b<U>; 64]; N_ROUNDS] =
			array::from_fn(|r| array::from_fn(|ij| Arc::new(sub_bytes_out_vec[r][ij].clone())));
		let sub_bytes_inv_arc: [[ArcVec8b<U>; 64]; N_ROUNDS] =
			array::from_fn(|r| array::from_fn(|ij| Arc::new(sub_bytes_inv_vec[r][ij].clone())));
		let sub_bytes_inv_bits_arc: [[[ArcVec1b<U>; 8]; 64]; N_ROUNDS] = array::from_fn(|r| {
			array::from_fn(|ij| {
				array::from_fn(|b| Arc::new(sub_bytes_inv_bits_vec[r][ij][b].clone()))
			})
		});
		let with_round_consts_arc: [[ArcVec8b<U>; 64]; N_ROUNDS] =
			array::from_fn(|r| array::from_fn(|ij| Arc::new(with_round_consts_vec[r][ij].clone())));

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

#[allow(dead_code)]
fn check_witness<U, F>(
	log_size: usize,
	constraint: impl CompositionPoly<PackedType<U, F>>,
	trace_oracle: &TraceOracle,
	witness_index: &MultilinearExtensionIndex<U, F>,
) -> Result<()>
where
	U: UnderlierType + PackScalar<F>,
	F: Field,
{
	let composite = MultilinearComposite::new(
		log_size,
		constraint,
		trace_oracle
			.iter_oracle_ids()
			.map(|oracle_id| witness_index.get_multilin_poly(oracle_id))
			.collect::<Result<_, _>>()?,
	)?;
	for z in 0..1 << log_size {
		let constraint_eval = composite.evaluate_on_hypercube(z)?;
		ensure!(constraint_eval == F::ZERO);
	}
	Ok(())
}

struct Proof<F: Field, PCSComm, PCS1bProof, PCS8bProof> {
	trace1b_comm: PCSComm,
	trace8b_comm: PCSComm,
	zerocheck_proof: ZerocheckProof<F>,
	evalcheck_proof: GreedyEvalcheckProof<F>,
	trace1b_open_proof: PCS1bProof,
	trace8b_open_proof: PCS8bProof,
}

impl<F: Field, PCSComm, PCS1bProof, PCS8bProof> Proof<F, PCSComm, PCS1bProof, PCS8bProof> {
	fn isomorphic<F2: Field + From<F>>(self) -> Proof<F2, PCSComm, PCS1bProof, PCS8bProof> {
		Proof {
			trace1b_comm: self.trace1b_comm,
			trace8b_comm: self.trace8b_comm,
			zerocheck_proof: self.zerocheck_proof.isomorphic(),
			evalcheck_proof: self.evalcheck_proof.isomorphic(),
			trace1b_open_proof: self.trace1b_open_proof,
			trace8b_open_proof: self.trace8b_open_proof,
		}
	}
}

#[instrument(skip_all, level = "debug")]
#[allow(clippy::too_many_arguments)]
fn prove<U, F, FBase, FEPCS, PCS1b, PCS8b, Comm, Challenger, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	trace_oracle: &TraceOracle,
	pcs1b: &PCS1b,
	pcs8b: &PCS8b,
	mut challenger: Challenger,
	trace_witness: &TraceWitness<U, BinaryField1b, AESTowerField8b>,
	domain_factory: impl EvaluationDomainFactory<AESTowerField8b>,
	backend: &Backend,
) -> Result<Proof<F, Comm, PCS1b::Proof, PCS8b::Proof>>
where
	U: UnderlierType
		+ PackScalar<BinaryField1b>
		+ PackScalar<BinaryField8b>
		+ PackScalar<AESTowerField8b>
		+ PackScalar<FBase>
		+ PackScalar<F>,
	F: TowerField + ExtensionField<AESTowerField8b> + ExtensionField<FBase> + Step,
	FBase: TowerField + ExtensionField<AESTowerField8b>,
	FEPCS: TowerField + From<F> + Into<F> + ExtensionField<BinaryField8b>,
	PackedType<U, AESTowerField8b>: PackedTransformationFactory<PackedType<U, BinaryField8b>>,
	PackedType<U, F>: PackedFieldIndexable,
	PCS1b: PolyCommitScheme<
		PackedType<U, BinaryField1b>,
		FEPCS,
		Error: Debug,
		Proof: 'static,
		Commitment = Comm,
	>,
	PCS8b: PolyCommitScheme<
		PackedType<U, BinaryField8b>,
		FEPCS,
		Error: Debug,
		Proof: 'static,
		Commitment = Comm,
	>,
	Comm: Clone,
	Challenger: CanObserve<FEPCS> + CanObserve<Comm> + CanSample<FEPCS> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	let mut witness = trace_witness.to_index::<F>(trace_oracle)?;
	let constraint_set =
		make_constraints::<AESTowerField8b, PackedType<U, F>>(trace_oracle).build_one(oracles)?;
	let constraint_set_base =
		make_constraints::<AESTowerField8b, PackedType<U, FBase>>(trace_oracle)
			.build_one(oracles)?;

	// Round 1
	let trace1b_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.trace1b_batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;
	let (trace1b_comm, trace1b_committed) = pcs1b.commit(&trace1b_commit_polys)?;

	let trace8b_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.trace8b_batch_id)
		.map(|oracle_id| {
			let witness_poly = witness.get::<AESTowerField8b>(oracle_id)?;
			convert_poly_witness_to_tower(witness_poly)
		})
		.collect::<Result<Vec<_>, _>>()?;
	let (trace8b_comm, trace8b_committed) = pcs8b.commit(&trace8b_commit_polys)?;

	challenger.observe(trace1b_comm.clone());
	challenger.observe(trace8b_comm.clone());

	// Zerocheck
	let mut iso_challenger = IsomorphicChallenger::<_, _, F>::new(&mut challenger);
	let zerocheck_challenges = iso_challenger.sample_vec(trace_oracle.log_size);
	let (zerocheck_claim, meta) = sumcheck::constraint_set_zerocheck_claim(constraint_set.clone())?;

	let switchover_fn = |extension_degree| match extension_degree {
		128 => 5,
		16 => 4,
		_ => 1,
	};

	let prover = sumcheck::prove::constraint_set_zerocheck_prover::<_, FBase, F, _, _>(
		constraint_set_base,
		constraint_set,
		&witness,
		domain_factory.clone(),
		switchover_fn,
		zerocheck_challenges.as_slice(),
		backend,
	)?;

	let (sumcheck_output, zerocheck_proof) =
		sumcheck::prove::batch_prove(vec![prover], &mut iso_challenger)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
		&[zerocheck_claim],
		&zerocheck_challenges,
		sumcheck_output,
	)?;

	let evalcheck_multilinear_claims =
		sumcheck::make_eval_claims(oracles, [meta], zerocheck_output.isomorphic())?;

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
		backend,
	)?;

	assert_eq!(same_query_claims.len(), 2);
	let (_, same_query_claim_1b) = same_query_claims
		.iter()
		.find(|(batch_id, _)| *batch_id == trace_oracle.trace1b_batch_id)
		.unwrap();
	let (_, same_query_claim_8b) = same_query_claims
		.iter()
		.find(|(batch_id, _)| *batch_id == trace_oracle.trace8b_batch_id)
		.unwrap();

	let trace1b_commit_polys = oracles
		.committed_oracle_ids(trace_oracle.trace1b_batch_id)
		.map(|oracle_id| witness.get::<BinaryField1b>(oracle_id))
		.collect::<Result<Vec<_>, _>>()?;

	let eval_point_1b: Vec<FEPCS> = same_query_claim_1b
		.eval_point
		.iter()
		.map(|x| (*x).into())
		.collect();

	let trace1b_open_proof = pcs1b.prove_evaluation(
		&mut challenger,
		&trace1b_committed,
		&trace1b_commit_polys,
		&eval_point_1b,
		&backend,
	)?;

	let eval_point_8b: Vec<FEPCS> = same_query_claim_8b
		.eval_point
		.iter()
		.map(|x| (*x).into())
		.collect();

	let trace8b_open_proof = pcs8b.prove_evaluation(
		&mut challenger,
		&trace8b_committed,
		&trace8b_commit_polys,
		&eval_point_8b,
		&backend,
	)?;

	Ok(Proof {
		trace1b_comm,
		trace8b_comm,
		zerocheck_proof: zerocheck_proof.isomorphic(),
		evalcheck_proof,
		trace1b_open_proof,
		trace8b_open_proof,
	})
}

#[instrument(skip_all, level = "debug")]
fn verify<F, P1b, P8b, PCS1b, PCS8b, Comm, Challenger, Backend>(
	oracles: &mut MultilinearOracleSet<F>,
	trace_oracle: &TraceOracle,
	pcs1b: &PCS1b,
	pcs8b: &PCS8b,
	mut challenger: Challenger,
	proof: Proof<F, Comm, PCS1b::Proof, PCS8b::Proof>,
	backend: &Backend,
) -> Result<()>
where
	F: TowerField + ExtensionField<BinaryField8b>,
	P1b: PackedField<Scalar = BinaryField1b>,
	P8b: PackedField<Scalar = BinaryField8b>,
	PCS1b: PolyCommitScheme<P1b, F, Error: Debug, Proof: 'static, Commitment = Comm>,
	PCS8b: PolyCommitScheme<P8b, F, Error: Debug, Proof: 'static, Commitment = Comm>,
	Comm: Clone,
	Challenger: CanObserve<F> + CanObserve<Comm> + CanSample<F> + CanSampleBits<usize>,
	Backend: ComputationBackend,
{
	let constraint_set = make_constraints::<BinaryField8b, F>(trace_oracle).build_one(oracles)?;

	let Proof {
		trace1b_comm,
		trace8b_comm,
		zerocheck_proof,
		evalcheck_proof,
		trace1b_open_proof,
		trace8b_open_proof,
	} = proof;

	// Round 1
	challenger.observe(trace1b_comm.clone());
	challenger.observe(trace8b_comm.clone());

	// Zerocheck
	let zerocheck_challenges = challenger.sample_vec(trace_oracle.log_size);

	let (zerocheck_claim, meta) = sumcheck::constraint_set_zerocheck_claim(constraint_set)?;
	let zerocheck_claims = [zerocheck_claim];

	let sumcheck_claims = zerocheck::reduce_to_sumchecks(&zerocheck_claims)?;

	let sumcheck_output =
		sumcheck::batch_verify(&sumcheck_claims, zerocheck_proof, &mut challenger)?;

	let zerocheck_output = zerocheck::verify_sumcheck_outputs(
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

	assert_eq!(same_query_claims.len(), 2);
	let (_, same_query_claim_1b) = same_query_claims
		.iter()
		.find(|(batch_id, _)| *batch_id == trace_oracle.trace1b_batch_id)
		.unwrap();
	let (_, same_query_claim_8b) = same_query_claims
		.iter()
		.find(|(batch_id, _)| *batch_id == trace_oracle.trace8b_batch_id)
		.unwrap();

	pcs1b.verify_evaluation(
		&mut challenger,
		&trace1b_comm,
		&same_query_claim_1b.eval_point,
		trace1b_open_proof,
		&same_query_claim_1b.evals,
		backend,
	)?;
	pcs8b.verify_evaluation(
		&mut challenger,
		&trace8b_comm,
		&same_query_claim_8b.eval_point,
		trace8b_open_proof,
		&same_query_claim_8b.evals,
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

	type U = <PackedBinaryField1x128b as WithUnderlier>::Underlier;

	let log_size = get_log_trace_size().unwrap_or(12);
	let backend = make_portable_backend();

	let mut prover_oracles = MultilinearOracleSet::<AESTowerField128b>::new();
	let prover_trace_oracle =
		TraceOracle::new::<_, AESTowerField8b>(&mut prover_oracles, log_size).unwrap();

	let log_inv_rate = 1;
	let trace1b_batch = prover_oracles.committed_batch(prover_trace_oracle.trace1b_batch_id);
	let trace8b_batch = prover_oracles.committed_batch(prover_trace_oracle.trace8b_batch_id);

	let witness = TraceWitness::<U, _, _>::generate_trace(log_size);

	// Generate and verify proof
	let pcs1b = tensor_pcs::find_proof_size_optimal_pcs::<
		OptimalUnderlier128b,
		BinaryField1b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, trace1b_batch.n_polys, log_inv_rate, false)
	.unwrap();

	let pcs8b = tensor_pcs::find_proof_size_optimal_pcs::<
		OptimalUnderlier128b,
		BinaryField8b,
		BinaryField16b,
		BinaryField16b,
		BinaryField128b,
	>(SECURITY_BITS, log_size, trace8b_batch.n_polys, log_inv_rate, false)
	.unwrap();

	// Since the inputs to the permutations are a linear comb. of state and block this is an estimate
	let num_p_perms: u64 = 1 << log_size;
	let hashable_data = num_p_perms * 64 / 2;
	let hashable_data = ByteSize::b(hashable_data);
	let tensorpcs_size_1b = ByteSize::b(pcs1b.proof_size(trace1b_batch.n_polys) as u64);
	let tensorpcs_size_8b = ByteSize::b(pcs8b.proof_size(trace8b_batch.n_polys) as u64);
	tracing::info!("Size of hashable Groestl256 data: {}", hashable_data);
	tracing::info!(
		"Size of PCS opening proof: {} (8b), {} (1b)",
		tensorpcs_size_8b,
		tensorpcs_size_1b
	);

	let challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
	let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();

	let proof = prove::<_, AESTowerField128b, AESTowerField16b, BinaryField128b, _, _, _, _, _>(
		&mut prover_oracles,
		&prover_trace_oracle,
		&pcs1b,
		&pcs8b,
		challenger.clone(),
		&witness,
		domain_factory,
		&backend,
	)
	.unwrap();

	let mut verifier_oracles = MultilinearOracleSet::<BinaryField128b>::new();
	let verifier_trace_oracle =
		TraceOracle::new::<_, BinaryField8b>(&mut verifier_oracles, log_size).unwrap();

	verify(
		&mut verifier_oracles,
		&verifier_trace_oracle,
		&pcs1b,
		&pcs8b,
		challenger.clone(),
		proof.isomorphic(),
		&backend,
	)
	.unwrap();
}
