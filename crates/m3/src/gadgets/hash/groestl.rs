// Copyright 2025 Irreducible Inc.

//! Gadgets for verifying the [Grøstl] hash function.
//!
//! [Grøstl]: <https://www.groestl.info/>

use std::{array, iter};

use anyhow::Result;
use binius_core::{constraint_system::channel::ChannelId, oracle::ShiftVariant};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	linear_transformation::{
		FieldLinearTransformation, PackedTransformationFactory, Transformation,
	},
	packed::{get_packed_slice, set_packed_slice},
	underlier::UnderlierType,
	AESTowerField8b, ExtensionField, PackedField,
};
use bytemuck::Pod;
use itertools::chain;

use crate::builder::{
	upcast_col, upcast_expr, Col, ConstraintSystem, Expr, TableBuilder, TableFiller, TableId,
	TableWitnessIndexSegment, B1, B64, B8,
};

/// The first row of the circulant matrix defining the MixBytes step in Grøstl.
const MIX_BYTES_VEC: [u8; 8] = [0x02, 0x02, 0x03, 0x04, 0x05, 0x03, 0x05, 0x07];

/// The affine transformation matrix for the Rijndael S-box, isomorphically converted to the
/// canonical tower basis.
const S_BOX_TOWER_MATRIX: FieldLinearTransformation<B8> =
	FieldLinearTransformation::new_const(&S_BOX_TOWER_MATRIX_COLS);

const S_BOX_TOWER_MATRIX_COLS: [B8; 8] = [
	B8::new(0x62),
	B8::new(0xd2),
	B8::new(0x79),
	B8::new(0x41),
	B8::new(0xf4),
	B8::new(0xd5),
	B8::new(0x81),
	B8::new(0x4e),
];

/// The affine transformation offset for the Rijndael S-box, isomorphically converted to the
/// canonical tower basis.
const S_BOX_TOWER_OFFSET: B8 = B8::new(0x14);

/// A Grøstl 512-bit state permutation.
///
/// The Grøstl hash function involves two permutations, P and Q, which are closely related. This
/// gadget verifies one permutation, depending on the variant given as a constructor argument.
#[derive(Debug, Clone)]
pub struct Permutation {
	rounds: [PermutationRound; 10],
}

impl Permutation {
	pub fn new(
		table: &mut TableBuilder,
		pq: PermutationVariant,
		mut state_in: [Col<B8, 8>; 8],
	) -> Self {
		let rounds = array::from_fn(|i| {
			let round = PermutationRound::new(
				&mut table.with_namespace(format!("round[{}]", i)),
				pq,
				state_in,
				i,
			);
			state_in = round.state_out;
			round
		});
		Self { rounds }
	}

	pub fn state_in(&self) -> [Col<B8, 8>; 8] {
		self.rounds[0].state_in
	}

	pub fn state_out(&self) -> [Col<B8, 8>; 8] {
		self.rounds[9].state_out
	}

	pub fn populate<U>(&self, index: &mut TableWitnessIndexSegment<U>) -> Result<()>
	where
		U: Pod + PackScalar<B1> + PackScalar<B8>,
		PackedType<U, B8>: PackedTransformationFactory<PackedType<U, B8>>,
	{
		for round in &self.rounds {
			round.populate(index)?;
		}
		Ok(())
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display)]
pub enum PermutationVariant {
	P,
	Q,
}

impl PermutationVariant {
	/// Returns the number of bytes to shift column `i` by in the ShiftBytes step.
	///
	/// The Grøstl specification presents the ShiftBytes step as a circular shift of the rows of
	/// the state; in this gadget, the state is transposed so that we shift columns instead.
	fn shift_bytes_offset(self, i: usize) -> usize {
		match self {
			PermutationVariant::P => (8 - i) % 8,
			PermutationVariant::Q => (16 - (2 * i + 1)) % 8,
		}
	}
}

/// A single round of a Grøstl permutation.
#[derive(Debug, Clone)]
pub struct PermutationRound {
	pq: PermutationVariant,
	round: usize,
	// Inputs
	pub state_in: [Col<B8, 8>; 8],
	// Private
	round_const: Col<B8, 8>,
	sbox: [SBox<8>; 8],
	shift: [Col<B8, 8>; 8],
	// Outputs
	pub state_out: [Col<B8, 8>; 8],
}

impl PermutationRound {
	pub fn new(
		table: &mut TableBuilder,
		pq: PermutationVariant,
		state_in: [Col<B8, 8>; 8],
		round: usize,
	) -> Self {
		let round_const = table.add_constant(
			"RoundConstant",
			array::from_fn(|i| B8::from(((i * 0x10) ^ round) as u8)),
		);

		// AddRoundConstant + SubBytes
		let sbox = array::from_fn(|i| {
			let sbox_in = match (i, pq) {
				(0, PermutationVariant::P) => state_in[0].into(), // + round_const,
				(_, PermutationVariant::P) => state_in[i].into(),
				(7, PermutationVariant::Q) => state_in[7] + B8::new(0xFF) + round_const,
				(_, PermutationVariant::Q) => state_in[i] + B8::new(0xFF),
			};
			SBox::new(&mut table.with_namespace(format!("SubBytes[{i}]")), sbox_in)
		});

		// ShiftBytes
		let shift = array::from_fn(|i| {
			let offset = pq.shift_bytes_offset(i);
			if offset == 0 {
				sbox[i].output
			} else {
				table.add_shifted(
					format!("ShiftBytes[{i}]"),
					sbox[i].output,
					3,
					offset,
					ShiftVariant::CircularLeft,
				)
			}
		});

		// MixBytes
		let mix_bytes_scalars = MIX_BYTES_VEC.map(|byte| B8::from(AESTowerField8b::new(byte)));
		let state_out = array::from_fn(|j| {
			let mix_bytes: [_; 8] =
				array::from_fn(|i| shift[i] * mix_bytes_scalars[(8 + i - j) % 8]);
			table.add_computed(
				format!("MixBytes[{j}]"),
				mix_bytes
					.into_iter()
					.reduce(|a, b| a + b)
					.expect("mix_bytes has length 8"),
			)
		});

		Self {
			pq,
			round,
			state_in,
			round_const,
			sbox,
			shift,
			state_out,
		}
	}

	pub fn populate<U>(&self, index: &mut TableWitnessIndexSegment<U>) -> Result<()>
	where
		U: Pod + PackScalar<B1> + PackScalar<B8>,
		PackedType<U, B8>: PackedTransformationFactory<PackedType<U, B8>>,
	{
		{
			let mut round_const = index.get_mut_as::<u8, _, 8>(self.round_const)?;
			for k in 0..round_const.len() / 8 {
				for i in 0..8 {
					round_const[k * 8 + i] = ((i * 0x10) ^ self.round) as u8;
				}
			}
		}

		// AddRoundConstant + SubBytes
		for sbox in &self.sbox {
			sbox.populate(index)?;
		}

		// ShiftBytes
		for (i, (sbox, shift)) in iter::zip(&self.sbox, self.shift).enumerate() {
			if sbox.output == shift {
				continue;
			}

			let sbox_out = index.get_as::<u64, _, 8>(sbox.output)?;
			let mut shift = index.get_mut_as::<u64, _, 8>(shift)?;

			// TODO: Annoying that this is duplicated. We could inspect the column definitions to
			// figure this out.
			let offset = self.pq.shift_bytes_offset(i);
			for (sbox_out_j, shift_j) in iter::zip(&*sbox_out, &mut *shift) {
				*shift_j = sbox_out_j.rotate_left((offset * 8) as u32);
			}
		}

		// MixBytes
		// TODO: Do the fancy trick from the Groestl implementation guide to reduce
		// multiplications.
		let mix_bytes_scalars = MIX_BYTES_VEC.map(|byte| B8::from(AESTowerField8b::new(byte)));
		let shift: [_; 8] = array::try_from_fn(|i| index.get(self.shift[i]))?;
		for j in 0..8 {
			let mut mix_bytes_out = index.get_mut(self.state_out[j])?;
			for (k, mix_bytes_out_k) in mix_bytes_out.iter_mut().enumerate() {
				*mix_bytes_out_k = (0..8)
					.map(|i| shift[i][k] * mix_bytes_scalars[(8 + i - j) % 8])
					.sum();
			}
		}

		Ok(())
	}
}

/// A gadget for the [Rijndael S-box].
///
/// The Rijndael S-box, used in the AES block cipher, is a non-linear substitution box that is
/// defined as a composition of field inversion and an $\mathbb{F}_2$-affine transformation on
/// elements of $\mathbb{F}_{2^8}$. The S-box is typically defined over a univariate basis
/// representation of $\mathbb{F}_{2^8}$, which is [`binius_field::AESTowerField8b`], thought we
/// can translate the S-box to a transformation on [`B8`] elements, which are isomorphic.
///
/// [Rijndael S-box]: <https://en.wikipedia.org/wiki/Rijndael_S-box>
#[derive(Debug, Clone)]
pub struct SBox<const V: usize> {
	input: Expr<B8, V>,
	/// Bits of the inverse of the input, in AES basis.
	inv_bits: [Col<B1, V>; 8],
	inv: Col<B8, V>,
	pub output: Col<B8, V>,
}

impl<const V: usize> SBox<V> {
	pub fn new(table: &mut TableBuilder, input: Expr<B8, V>) -> Self {
		let b8_basis: [_; 8] = array::from_fn(|i| {
			<B8 as ExtensionField<B1>>::basis(i).expect("i in range 0..8; extension degree is 8")
		});
		let pack_b8 = move |bits: [Expr<B1, V>; 8]| {
			bits.into_iter()
				.enumerate()
				.map(|(i, bit)| upcast_expr(bit) * b8_basis[i])
				.reduce(|a, b| a + b)
				.expect("bits has length 8")
		};

		let inv_bits = array::from_fn(|i| table.add_committed(format!("inv_bits[{}]", i)));
		let inv = table.add_computed("inv", pack_b8(inv_bits.map(Expr::from)));

		// input * inv == 1 OR inv == 0
		table.assert_zero("inv_valid_or_inv_zero", input.clone() * Expr::from(inv).pow(2) - inv);
		// input * inv == 1 OR input == 0
		table.assert_zero("inv_valid_or_input_zero", input.clone().pow(2) * inv - input.clone());

		// Rijndael S-box affine transformation
		let linear_transform_expr = iter::zip(inv_bits, S_BOX_TOWER_MATRIX_COLS)
			.map(|(inv_bit_i, scalar)| upcast_col(inv_bit_i) * scalar)
			.reduce(|a, b| a + b)
			.expect("inv_bits and S_BOX_TOWER_MATRIX_COLS have length 8");
		let output =
			table.add_computed("output", linear_transform_expr.clone() + S_BOX_TOWER_OFFSET);

		Self {
			input,
			inv_bits,
			inv,
			output,
		}
	}

	pub fn populate<U>(&self, index: &mut TableWitnessIndexSegment<U>) -> Result<()>
	where
		U: Pod + PackScalar<B1> + PackScalar<B8>,
		PackedType<U, B8>: PackedTransformationFactory<PackedType<U, B8>>,
	{
		let mut inv = index.get_mut(self.inv)?;

		// Populate the inverse of the input.
		for (inv_i, val_i) in iter::zip(&mut *inv, index.eval_expr(&self.input)?) {
			*inv_i = val_i.invert_or_zero();
		}

		// Decompose the inverse bits.
		let mut inv_bits = self
			.inv_bits
			.try_map(|inv_bits_i| index.get_mut(inv_bits_i))?;
		for i in 0..index.size() * V {
			let inv_val = get_packed_slice(&inv, i);
			for (j, inv_bit_j) in ExtensionField::<B1>::iter_bases(&inv_val).enumerate() {
				set_packed_slice(&mut inv_bits[j], i, inv_bit_j);
			}
		}

		// Apply the F2-linear transformation and populate the output.
		let mut output = index.get_mut(self.output)?;

		let transform_matrix = <PackedType<U, B8>>::make_packed_transformation(S_BOX_TOWER_MATRIX);
		let transform_offset = <PackedType<U, B8>>::broadcast(S_BOX_TOWER_OFFSET);
		for (out_i, inv_i) in iter::zip(&mut *output, &*inv) {
			*out_i = transform_offset + transform_matrix.transform(inv_i);
		}

		Ok(())
	}
}

#[derive(Debug)]
pub struct PermutationTable {
	table_id: TableId,
	permutation: Permutation,
	state_in: [Col<B64>; 8],
	state_out: [Col<B64>; 8],
}

impl PermutationTable {
	pub fn new(cs: &mut ConstraintSystem, pq: PermutationVariant, chan: ChannelId) -> Self {
		let mut table = cs.add_table(format!("Grøstl {pq} permutation"));

		let state_in_bytes = table.add_committed_multiple::<B8, 8, 8>("state_in_bytes");
		let permutation = Permutation::new(&mut table, pq, state_in_bytes);

		let state_in = state_in_bytes
			.map(|state_in_bytes_i| table.add_packed::<_, 8, B64, 1>("state_in", state_in_bytes_i));
		let state_out = state_in_bytes.map(|state_out_bytes_i| {
			table.add_packed::<_, 8, B64, 1>("state_out", state_out_bytes_i)
		});

		table.pull(chan, chain!(state_in, state_out));

		Self {
			table_id: table.id(),
			permutation,
			state_in,
			state_out,
		}
	}
}

impl<U: UnderlierType> TableFiller<U> for PermutationTable
where
	U: Pod + PackScalar<B1> + PackScalar<B8> + PackScalar<B64>,
	PackedType<U, B8>: PackedTransformationFactory<PackedType<U, B8>>,
{
	type Event = [u64; 8];

	fn id(&self) -> TableId {
		self.table_id
	}

	fn fill<'a>(
		&'a self,
		rows: impl Iterator<Item = &'a Self::Event>,
		witness: &'a mut TableWitnessIndexSegment<U>,
	) -> Result<()> {
		// Populate the input states
		{
			let mut state_in = self
				.state_in
				.try_map(|state_in_i| witness.get_mut_as::<u64, _, 1>(state_in_i))?;
			for (k, event_k) in rows.enumerate() {
				for (&event_ki, state_in_i) in iter::zip(event_k, &mut state_in) {
					state_in_i[k] = event_ki;
				}
			}
		}

		self.permutation.populate(witness)?;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{arch::OptimalUnderlier128b, arithmetic_traits::InvertOrZero};
	use bumpalo::Bump;
	use rand::{prelude::StdRng, SeedableRng};

	use super::*;
	use crate::builder::{ConstraintSystem, Statement};

	#[test]
	fn test_sbox() {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("sbox test");

		let input = table.add_committed::<B8, 2>("input");
		let sbox = SBox::new(&mut table, input + B8::new(0xFF));

		let table_id = table.id();

		let allocator = Bump::new();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![1 << 8],
		};
		let mut witness = cs
			.build_witness::<OptimalUnderlier128b>(&allocator, &statement)
			.unwrap();

		let table_witness = witness.get_table(table_id).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let mut segment = table_witness.full_segment();
		for in_i in &mut *segment.get_mut(input).unwrap() {
			*in_i = PackedField::random(&mut rng);
		}

		sbox.populate(&mut segment).unwrap();

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index(&statement);

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_p_permutation() {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("P-permutation test");

		let input = table.add_committed_multiple::<B8, 8, 8>("state_in");
		let perm = Permutation::new(&mut table, PermutationVariant::P, input);

		let table_id = table.id();

		let allocator = Bump::new();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![1 << 8],
		};
		let mut witness = cs
			.build_witness::<OptimalUnderlier128b>(&allocator, &statement)
			.unwrap();

		let table_witness = witness.get_table(table_id).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let mut segment = table_witness.full_segment();
		for i in 0..8 {
			for in_j in &mut *segment.get_mut(input[i]).unwrap() {
				*in_j = PackedField::random(&mut rng);
			}
		}

		perm.populate(&mut segment).unwrap();

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index(&statement);

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_q_permutation() {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("Q-permutation test");

		let input = table.add_committed_multiple::<B8, 8, 8>("state_in");
		let perm = Permutation::new(&mut table, PermutationVariant::Q, input);

		let table_id = table.id();

		let allocator = Bump::new();

		let statement = Statement {
			boundaries: vec![],
			table_sizes: vec![1 << 8],
		};
		let mut witness = cs
			.build_witness::<OptimalUnderlier128b>(&allocator, &statement)
			.unwrap();

		let table_witness = witness.get_table(table_id).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let mut segment = table_witness.full_segment();
		for i in 0..8 {
			for in_j in &mut *segment.get_mut(input[i]).unwrap() {
				*in_j = PackedField::random(&mut rng);
			}
		}

		perm.populate(&mut segment).unwrap();

		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index(&statement);

		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}

	#[test]
	fn test_isomorphic_sbox() {
		#[rustfmt::skip]
		const S_BOX: [u8; 256] = [
			0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
			0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
			0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
			0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
			0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
			0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
			0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
			0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
			0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
			0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
			0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
			0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
			0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
			0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
			0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
			0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
		];

		const S_BOX_MATRIX: FieldLinearTransformation<AESTowerField8b> =
			FieldLinearTransformation::new_const(&[
				AESTowerField8b::new(0x1F),
				AESTowerField8b::new(0x3E),
				AESTowerField8b::new(0x7C),
				AESTowerField8b::new(0xF8),
				AESTowerField8b::new(0xF1),
				AESTowerField8b::new(0xE3),
				AESTowerField8b::new(0xC7),
				AESTowerField8b::new(0x8F),
			]);
		const S_BOX_OFFSET: AESTowerField8b = AESTowerField8b::new(0x63);

		for i in 0u8..=255u8 {
			let sbox_in = AESTowerField8b::new(i);
			let expected_sbox_out = AESTowerField8b::new(S_BOX[i as usize]);

			let sbox_out =
				S_BOX_MATRIX.transform(&InvertOrZero::invert_or_zero(sbox_in)) + S_BOX_OFFSET;
			assert_eq!(sbox_out, expected_sbox_out);

			let sbox_in_b8 = B8::from(sbox_in);
			let sbox_out_b8 = S_BOX_TOWER_MATRIX
				.transform(&InvertOrZero::invert_or_zero(sbox_in_b8))
				+ S_BOX_TOWER_OFFSET;
			assert_eq!(AESTowerField8b::from(sbox_out_b8), expected_sbox_out);
		}
	}
}
