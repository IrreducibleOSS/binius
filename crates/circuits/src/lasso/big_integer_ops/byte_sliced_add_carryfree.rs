// Copyright 2024-2025 Irreducible Inc.

use alloy_primitives::U512;
use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{tower_levels::TowerLevel, BinaryField1b, BinaryField8b};

use super::byte_sliced_add;
use crate::{
	builder::ConstraintSystemBuilder,
	lasso::{batch::LookupBatch, u8add_carryfree},
};

type B1 = BinaryField1b;
type B8 = BinaryField8b;

#[allow(clippy::too_many_arguments)]
pub fn byte_sliced_add_carryfree<Level: TowerLevel<OracleId, Data: Sized>>(
	builder: &mut ConstraintSystemBuilder,
	name: impl ToString,
	x_in: &Level::Data,
	y_in: &Level::Data,
	carry_in: OracleId,
	log_size: usize,
	lookup_batch_add: &mut LookupBatch,
	lookup_batch_add_carryfree: &mut LookupBatch,
) -> Result<Level::Data, anyhow::Error> {
	if Level::WIDTH == 1 {
		let sum = u8add_carryfree(
			builder,
			lookup_batch_add_carryfree,
			"u8 carryfree add",
			x_in[0],
			y_in[0],
			carry_in,
			log_size,
		)?;
		let mut sum_arr = Level::default();
		sum_arr[0] = sum;
		return Ok(sum_arr);
	}

	builder.push_namespace(name);

	let (lower_half_x, upper_half_x) = Level::split(x_in);
	let (lower_half_y, upper_half_y) = Level::split(y_in);

	let (internal_carry, lower_sum) = byte_sliced_add::<Level::Base>(
		builder,
		format!("lower sum {}b", Level::Base::WIDTH),
		lower_half_x,
		lower_half_y,
		carry_in,
		log_size,
		lookup_batch_add,
	)?;

	let upper_sum = byte_sliced_add_carryfree::<Level::Base>(
		builder,
		format!("upper sum {}b", Level::Base::WIDTH),
		upper_half_x,
		upper_half_y,
		internal_carry,
		log_size,
		lookup_batch_add,
		lookup_batch_add_carryfree,
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

		let y_bytes_as_u8 = (0..Level::WIDTH).map(|this_byte_idx| {
			let this_byte_oracle = y_in[this_byte_idx];
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

		let cin_as_u8_packed = witness.get::<B1>(carry_in).unwrap().as_slice::<u8>();

		for row_idx in 0..1 << log_size {
			let mut x_u512 = U512::ZERO;
			for (byte_idx, x_byte_column) in x_bytes_as_u8.clone().enumerate() {
				x_u512 |= U512::from(x_byte_column[row_idx]) << (8 * byte_idx);
			}

			let mut y_u512 = U512::ZERO;
			for (byte_idx, y_byte_column) in y_bytes_as_u8.clone().enumerate() {
				y_u512 |= U512::from(y_byte_column[row_idx]) << (8 * byte_idx);
			}

			let mut sum_u512 = U512::ZERO;
			for (byte_idx, sum_byte_column) in sum_bytes_as_u8.clone().enumerate() {
				sum_u512 |= U512::from(sum_byte_column[row_idx]) << (8 * byte_idx);
			}

			let cin_u512 = U512::from((cin_as_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1);

			let expected_sum_u512 = x_u512 + y_u512 + cin_u512;

			assert_eq!(expected_sum_u512, sum_u512);
		}
	}
	builder.pop_namespace();

	Ok(sum)
}
