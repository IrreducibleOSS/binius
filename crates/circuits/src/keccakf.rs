// Copyright 2024 Irreducible Inc.

use std::array;

use crate::builder::ConstraintSystemBuilder;
use binius_core::{
	oracle::{OracleId, ShiftVariant},
	transparent::multilinear_extension::MultilinearExtensionTransparent,
};
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::{UnderlierType, WithUnderlier},
	BinaryField1b, BinaryField64b, ExtensionField, PackedField, TowerField,
};
use binius_macros::composition_poly;
use bytemuck::{pod_collect_to_vec, Pod};

const STATE_SIZE: usize = 25;

pub fn keccakf_round<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	state_in: [OracleId; STATE_SIZE],
	round_within_permutation: usize,
	log_size: usize,
) -> Result<[OracleId; STATE_SIZE], anyhow::Error>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<F>
		+ PackScalar<FBase>
		+ PackScalar<BinaryField1b>
		+ PackScalar<BinaryField64b>,
	F: TowerField + ExtensionField<FBase>,
	FBase: TowerField,
	<U as PackScalar<BinaryField64b>>::Packed: Pod,
{
	let broadcasted_round_constant = <PackedType<U, BinaryField64b>>::broadcast(
		BinaryField64b::from(KECCAKF_RC[round_within_permutation]),
	);
	let round_consts_single =
		builder.add_transparent(
			"round_consts_single",
			MultilinearExtensionTransparent::<_, PackedType<U, F>, _>::from_values(
				into_packed_vec::<PackedType<U, BinaryField1b>>(&[broadcasted_round_constant]),
			)?,
		)?;

	let round_consts = builder
		.add_repeating(
			"round_consts",
			round_consts_single,
			log_size - <PackedType<U, BinaryField1b>>::LOG_WIDTH,
		)
		.unwrap();

	let state_out: [OracleId; STATE_SIZE] = builder.add_committed_multiple(
		format!("state_out round {round_within_permutation}"),
		log_size,
		BinaryField1b::TOWER_LEVEL,
	);

	let c: [OracleId; 5] = array::from_fn(|x| {
		builder
			.add_linear_combination(
				format!("c[{x}]"),
				log_size,
				[
					(state_in[x], F::ONE),
					(state_in[x + 5], F::ONE),
					(state_in[x + 5 * 2], F::ONE),
					(state_in[x + 5 * 3], F::ONE),
					(state_in[x + 5 * 4], F::ONE),
				],
			)
			.unwrap()
	});

	let c_shift: [OracleId; 5] = array::from_fn(|x| {
		builder
			.add_shifted(format!("c_shift[{x}]"), c[x], 1, 6, ShiftVariant::CircularLeft)
			.unwrap()
	});

	let d: [OracleId; 5] = array::from_fn(|x| {
		builder
			.add_linear_combination(
				format!("d[round:{round_within_permutation}, state idx:{x}]"),
				log_size,
				[(c[(x + 4) % 5], F::ONE), (c_shift[(x + 1) % 5], F::ONE)],
			)
			.unwrap()
	});

	let a_theta: [OracleId; STATE_SIZE] = array::from_fn(|xy| {
		let x = xy % 5;
		builder
			.add_linear_combination(
				format!("a_theta[{xy}]"),
				log_size,
				[(state_in[xy], F::ONE), (d[x], F::ONE)],
			)
			.unwrap()
	});

	let b: [OracleId; STATE_SIZE] = array::from_fn(|xy| {
		if xy == 0 {
			a_theta[0]
		} else {
			builder
				.add_shifted(
					format!("b[{xy}]"),
					a_theta[PI[xy]],
					RHO[xy] as usize,
					6,
					ShiftVariant::CircularLeft,
				)
				.unwrap()
		}
	});

	if let Some(witness) = builder.witness() {
		let state_in = state_in.map(|id| witness.get::<BinaryField1b>(id).unwrap());

		let mut state_out = state_out.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut c = c.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut d = d.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut c_shift = c_shift.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut a_theta = a_theta.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut b = b.map(|id| witness.new_column::<BinaryField1b>(id));

		let mut round_consts_single = witness.new_column::<BinaryField1b>(round_consts_single);

		let mut round_consts = witness.new_column::<BinaryField1b>(round_consts);

		let state_in_u64 = state_in.each_ref().map(move |col| col.as_slice::<u64>());

		let state_out_u64 = state_out
			.each_mut()
			.map(move |col| col.as_mut_slice::<u64>());

		let c_u64 = c.each_mut().map(move |col| col.as_mut_slice::<u64>());

		let d_u64 = d.each_mut().map(move |col| col.as_mut_slice::<u64>());

		let c_shift_u64 = c_shift.each_mut().map(move |col| col.as_mut_slice::<u64>());

		let a_theta_u64 = a_theta.each_mut().map(move |col| col.as_mut_slice::<u64>());

		let b_u64 = b.each_mut().map(move |col| col.as_mut_slice::<u64>());

		let round_consts_single_u64 = round_consts_single.as_mut_slice();

		let round_consts_u64 = round_consts.as_mut_slice();

		round_consts_single_u64.fill(KECCAKF_RC[round_within_permutation]);

		// Each round state is 64 rows
		// Each permutation is 24 round states
		for (row_idx, this_row_round_const_ptr) in round_consts_u64.iter_mut().enumerate() {
			let keccakf_rc = KECCAKF_RC[round_within_permutation];

			*this_row_round_const_ptr = keccakf_rc;

			for x in 0..5 {
				c_u64[x][row_idx] = (0..5).fold(0, |acc, y| acc ^ state_in_u64[x + 5 * y][row_idx]);
				c_shift_u64[x][row_idx] = c_u64[x][row_idx].rotate_left(1);
			}

			for x in 0..5 {
				d_u64[x][row_idx] = c_u64[(x + 4) % 5][row_idx] ^ c_shift_u64[(x + 1) % 5][row_idx];
			}

			for x in 0..5 {
				for y in 0..5 {
					a_theta_u64[x + 5 * y][row_idx] =
						state_in_u64[x + 5 * y][row_idx] ^ d_u64[x][row_idx];
				}
			}

			for xy in 0..25 {
				b_u64[xy][row_idx] = a_theta_u64[PI[xy]][row_idx].rotate_left(RHO[xy]);
			}

			for x in 0..5 {
				for y in 0..5 {
					let b0 = b_u64[x + 5 * y][row_idx];
					let b1 = b_u64[(x + 1) % 5 + 5 * y][row_idx];
					let b2 = b_u64[(x + 2) % 5 + 5 * y][row_idx];
					state_out_u64[x + 5 * y][row_idx] = b0 ^ (!b1 & b2);
				}
			}

			for x in 0..5 {
				assert_eq!(
					d_u64[x][row_idx],
					c_u64[(x + 4) % 5][row_idx] ^ c_shift_u64[(x + 1) % 5][row_idx]
				);
			}

			state_out_u64[0][row_idx] ^= *this_row_round_const_ptr;
		}
	}

	let chi_iota = composition_poly!([s, b0, b1, b2, rc] = s - (rc + b0 + (1 - b1) * b2));
	let chi = composition_poly!([s, b0, b1, b2] = s - (b0 + (1 - b1) * b2));

	for x in 0..5 {
		for y in 0..5 {
			if x == 0 && y == 0 {
				builder.assert_zero(
					[
						state_out[x + 5 * y],
						b[x + 5 * y],
						b[(x + 1) % 5 + 5 * y],
						b[(x + 2) % 5 + 5 * y],
						round_consts,
					],
					chi_iota,
				);
			} else {
				builder.assert_zero(
					[
						state_out[x + 5 * y],
						b[x + 5 * y],
						b[(x + 1) % 5 + 5 * y],
						b[(x + 2) % 5 + 5 * y],
					],
					chi,
				)
			}
		}
	}

	Ok(state_out)
}

pub fn keccakf<U, F, FBase>(
	builder: &mut ConstraintSystemBuilder<U, F, FBase>,
	input: [OracleId; STATE_SIZE],
	log_size: usize,
) -> Result<[OracleId; STATE_SIZE], anyhow::Error>
where
	U: UnderlierType
		+ Pod
		+ PackScalar<F>
		+ PackScalar<FBase>
		+ PackScalar<BinaryField1b>
		+ PackScalar<BinaryField64b>,
	F: TowerField + ExtensionField<FBase>,
	FBase: TowerField,
	<U as PackScalar<BinaryField64b>>::Packed: Pod,
{
	let mut state = input;

	for round_within_permutation in 0..ROUNDS_PER_PERMUTATION {
		state = keccakf_round(builder, state, round_within_permutation, log_size)?;
	}

	if let Some(witness) = builder.witness() {
		let mut state_in = input.map(|id| witness.get::<BinaryField1b>(id).unwrap());

		let mut state_out = state.map(|id| witness.get::<BinaryField1b>(id).unwrap());

		let state_in_u64 = state_in.each_mut().map(move |col| col.as_slice::<u64>());

		let state_out_u64 = state_out.each_mut().map(move |col| col.as_slice::<u64>());

		// Assert correct output
		for row_idx in 0..1 << (log_size - LOG_ROWS_PER_PERMUTATION) {
			let mut inp: [_; STATE_SIZE] = array::from_fn(|xy| state_in_u64[xy][row_idx]);
			let out: [_; STATE_SIZE] = array::from_fn(|xy| state_out_u64[xy][row_idx]);
			tiny_keccak::keccakf(&mut inp);
			assert_eq!(inp, out);
		}
	}

	Ok(state)
}

#[inline]
fn into_packed_vec<P>(src: &[impl Pod]) -> Vec<P>
where
	P: PackedField + WithUnderlier,
	P::Underlier: Pod,
{
	pod_collect_to_vec::<_, P::Underlier>(src)
		.into_iter()
		.map(P::from_underlier)
		.collect()
}

#[rustfmt::skip]
const RHO: [u32; 25] = [
	 0, 44, 43, 21, 14,
	28, 20,  3, 45, 61,
	 1,  6, 25,  8, 18,
	27, 36, 10, 15, 56,
	62, 55, 39, 41,  2,
];

#[rustfmt::skip]
const PI: [usize; 25] = [
	0, 6, 12, 18, 24,
	3, 9, 10, 16, 22,
	1, 7, 13, 19, 20,
	4, 5, 11, 17, 23,
	2, 8, 14, 15, 21,
];

const LOG_ROWS_PER_ROUND: usize = 6;
const LOG_ROWS_PER_PERMUTATION: usize = LOG_ROWS_PER_ROUND;
const ROUNDS_PER_PERMUTATION: usize = 24;

const KECCAKF_RC: [u64; 24] = [
	0x0000000000000001,
	0x0000000000008082,
	0x800000000000808A,
	0x8000000080008000,
	0x000000000000808B,
	0x0000000080000001,
	0x8000000080008081,
	0x8000000000008009,
	0x000000000000008A,
	0x0000000000000088,
	0x0000000080008009,
	0x000000008000000A,
	0x000000008000808B,
	0x800000000000008B,
	0x8000000000008089,
	0x8000000000008003,
	0x8000000000008002,
	0x8000000000000080,
	0x000000000000800A,
	0x800000008000000A,
	0x8000000080008081,
	0x8000000000008080,
	0x0000000080000001,
	0x8000000080008008,
];
