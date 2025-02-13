// Copyright 2024-2025 Irreducible Inc.

use alloy_primitives::U512;
use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{tower_levels::TowerLevel, BinaryField1b, BinaryField8b};

use crate::{
	builder::ConstraintSystemBuilder,
	lasso::{batch::LookupBatch, u8_double_conditional_increment},
};

type B1 = BinaryField1b;
type B8 = BinaryField8b;

#[allow(clippy::too_many_arguments)]
pub fn byte_sliced_double_conditional_increment<Level: TowerLevel<OracleId, Data: Sized>>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	x_in: &Level::Data,
	first_carry_in: OracleId,
	second_carry_in: OracleId,
	log_size: usize,
	zero_oracle_carry: usize,
	lookup_batch_dci: &mut LookupBatch,
) -> Result<(OracleId, Level::Data), anyhow::Error> {
	if Level::WIDTH == 1 {
		let (carry_out, sum) = u8_double_conditional_increment(
			builder,
			lookup_batch_dci,
			"u8 DCI",
			x_in[0],
			first_carry_in,
			second_carry_in,
			log_size,
		)?;
		let mut sum_arr = Level::default();
		sum_arr[0] = sum;
		return Ok((carry_out, sum_arr));
	}

	builder.push_namespace(name);

	let (lower_half_x, upper_half_x) = Level::split(x_in);

	let (internal_carry, lower_sum) = byte_sliced_double_conditional_increment::<Level::Base>(
		builder,
		format!("lower sum {}b", Level::Base::WIDTH),
		lower_half_x,
		first_carry_in,
		second_carry_in,
		log_size,
		zero_oracle_carry,
		lookup_batch_dci,
	)?;

	let (carry_out, upper_sum) = byte_sliced_double_conditional_increment::<Level::Base>(
		builder,
		format!("upper sum {}b", Level::Base::WIDTH),
		upper_half_x,
		internal_carry,
		zero_oracle_carry,
		log_size,
		zero_oracle_carry,
		lookup_batch_dci,
	)?;

	let sum = Level::join(&lower_sum, &upper_sum);

	// Everything below is for test assertions
	if let Some(witness) = builder.witness() {
		let x_bytes_as_u8 = (0..Level::WIDTH).map(|this_byte_idx| {
			let this_byte_oracle = x_in[this_byte_idx];
			witness
				.get::<B8>(this_byte_oracle)
				.unwrap()
				.as_slice::<u8>()
		});

		let sum_bytes_as_u8 = (0..Level::WIDTH).map(|this_byte_idx| {
			let this_byte_oracle = sum[this_byte_idx];
			witness
				.get::<B8>(this_byte_oracle)
				.unwrap()
				.as_slice::<u8>()
		});

		let first_cin_as_u8_packed = witness.get::<B1>(first_carry_in).unwrap().as_slice::<u8>();
		let second_cin_as_u8_packed = witness.get::<B1>(second_carry_in).unwrap().as_slice::<u8>();

		let cout_as_u8_packed = witness.get::<B1>(carry_out).unwrap().as_slice::<u8>();

		for row_idx in 0..1 << log_size {
			let mut x_u512 = U512::ZERO;
			for (byte_idx, x_byte_column) in x_bytes_as_u8.clone().enumerate() {
				x_u512 |= U512::from(x_byte_column[row_idx]) << (8 * byte_idx);
			}

			let mut sum_u512 = U512::ZERO;
			for (byte_idx, sum_byte_column) in sum_bytes_as_u8.clone().enumerate() {
				sum_u512 |= U512::from(sum_byte_column[row_idx]) << (8 * byte_idx);
			}

			let first_cin_u512 =
				U512::from((first_cin_as_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1);

			let second_cin_u512 =
				U512::from((second_cin_as_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1);

			let cout_u512 = U512::from((cout_as_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1);

			let expected_sum_u128 = x_u512 + first_cin_u512 + second_cin_u512;

			let sum_according_to_witness = sum_u512 | (cout_u512 << (Level::WIDTH * 8));

			assert_eq!(expected_sum_u128, sum_according_to_witness);
		}
	}
	builder.pop_namespace();

	Ok((carry_out, sum))
}
