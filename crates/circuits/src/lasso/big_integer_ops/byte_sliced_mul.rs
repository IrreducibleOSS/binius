// Copyright 2024 Irreducible Inc.

use alloy_primitives::U512;
use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	tower_levels::TowerLevel,
	underlier::UnderlierType,
	BinaryField, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;

use super::{byte_sliced_add, byte_sliced_double_conditional_increment};
use crate::{
	builder::ConstraintSystemBuilder,
	lasso::{batch::LookupBatch, u8mul::u8mul_bytesliced},
};

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

#[allow(clippy::too_many_arguments)]
pub fn byte_sliced_mul<
	U,
	F,
	LevelIn: TowerLevel<OracleId>,
	LevelOut: TowerLevel<OracleId, Base = LevelIn>,
>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	name: impl ToString,
	mult_a: &LevelIn::Data,
	mult_b: &LevelIn::Data,
	log_size: usize,
	zero_carry_oracle: OracleId,
	lookup_batch_mul: &mut LookupBatch,
	lookup_batch_add: &mut LookupBatch,
	lookup_batch_dci: &mut LookupBatch,
) -> Result<LevelOut::Data, anyhow::Error>
where
	U: Pod
		+ UnderlierType
		+ PackScalar<B1>
		+ PackScalar<B8>
		+ PackScalar<B16>
		+ PackScalar<B32>
		+ PackScalar<F>,
	PackedType<U, B8>: PackedFieldIndexable,
	PackedType<U, B16>: PackedFieldIndexable,
	PackedType<U, B32>: PackedFieldIndexable,
	F: TowerField + BinaryField + ExtensionField<B8> + ExtensionField<B16> + ExtensionField<B32>,
{
	if LevelIn::WIDTH == 1 {
		let result_of_u8mul = u8mul_bytesliced(
			builder,
			lookup_batch_mul,
			"u8 mul",
			mult_a[0],
			mult_b[0],
			1 << log_size,
		)?;
		let mut lower_result_of_u8mul = LevelIn::default();
		lower_result_of_u8mul[0] = result_of_u8mul[0];
		let mut upper_result_of_u8mul = LevelIn::default();
		upper_result_of_u8mul[0] = result_of_u8mul[1];

		let result_typed_arr = LevelOut::join(&lower_result_of_u8mul, &upper_result_of_u8mul);

		return Ok(result_typed_arr);
	}

	builder.push_namespace(name);

	let (mult_a_low, mult_a_high) = LevelIn::split(mult_a);
	let (mult_b_low, mult_b_high) = LevelIn::split(mult_b);

	let a_lo_b_lo = byte_sliced_mul::<_, _, LevelIn::Base, LevelOut::Base>(
		builder,
		format!("lo*lo {}b", LevelIn::Base::WIDTH),
		mult_a_low,
		mult_b_low,
		log_size,
		zero_carry_oracle,
		lookup_batch_mul,
		lookup_batch_add,
		lookup_batch_dci,
	)?;
	let a_lo_b_hi = byte_sliced_mul::<_, _, LevelIn::Base, LevelOut::Base>(
		builder,
		format!("lo*hi {}b", LevelIn::Base::WIDTH),
		mult_a_low,
		mult_b_high,
		log_size,
		zero_carry_oracle,
		lookup_batch_mul,
		lookup_batch_add,
		lookup_batch_dci,
	)?;
	let a_hi_b_lo = byte_sliced_mul::<_, _, LevelIn::Base, LevelOut::Base>(
		builder,
		format!("hi*lo {}b", LevelIn::Base::WIDTH),
		mult_a_high,
		mult_b_low,
		log_size,
		zero_carry_oracle,
		lookup_batch_mul,
		lookup_batch_add,
		lookup_batch_dci,
	)?;
	let a_hi_b_hi = byte_sliced_mul::<_, _, LevelIn::Base, LevelOut::Base>(
		builder,
		format!("hi*hi {}b", LevelIn::Base::WIDTH),
		mult_a_high,
		mult_b_high,
		log_size,
		zero_carry_oracle,
		lookup_batch_mul,
		lookup_batch_add,
		lookup_batch_dci,
	)?;

	let (karatsuba_carry_for_high_chunk, karatsuba_term) = byte_sliced_add::<_, _, LevelIn>(
		builder,
		format!("karastsuba addition {}b", LevelIn::WIDTH),
		&a_lo_b_hi,
		&a_hi_b_lo,
		zero_carry_oracle,
		log_size,
		lookup_batch_add,
	)?;

	let (a_lo_b_lo_lower_half, a_lo_b_lo_upper_half) = LevelIn::split(&a_lo_b_lo);
	let (a_hi_b_hi_lower_half, a_hi_b_hi_upper_half) = LevelIn::split(&a_hi_b_hi);

	let (additional_carry_for_high_chunk, final_middle_chunk) = byte_sliced_add::<_, _, LevelIn>(
		builder,
		format!("post kartsuba middle term addition {}b", LevelIn::WIDTH),
		&karatsuba_term,
		&LevelIn::join(a_lo_b_lo_upper_half, a_hi_b_hi_lower_half),
		zero_carry_oracle,
		log_size,
		lookup_batch_add,
	)?;

	let (_, final_high_chunk) = byte_sliced_double_conditional_increment::<_, _, LevelIn::Base>(
		builder,
		format!("high chunk DCI {}b", LevelIn::Base::WIDTH),
		a_hi_b_hi_upper_half,
		karatsuba_carry_for_high_chunk,
		additional_carry_for_high_chunk,
		log_size,
		zero_carry_oracle,
		lookup_batch_dci,
	)?;

	let (final_middle_chunk_lower_half, final_middle_chunk_upper_half) =
		LevelIn::split(&final_middle_chunk);

	let final_lower_half = LevelIn::join(a_lo_b_lo_lower_half, final_middle_chunk_lower_half);

	let final_upper_half = LevelIn::join(final_middle_chunk_upper_half, &final_high_chunk);

	let product = LevelOut::join(&final_lower_half, &final_upper_half);

	// All of the code below is for test assertions
	if let Some(witness) = builder.witness() {
		let a_bytes_as_u8 = (0..LevelIn::WIDTH).map(|this_byte_idx| {
			let this_byte_oracle = mult_a[this_byte_idx];
			witness
				.get::<B8>(this_byte_oracle)
				.unwrap()
				.as_slice::<u8>()
		});

		let b_bytes_as_u8 = (0..LevelIn::WIDTH).map(|this_byte_idx| {
			let this_byte_oracle = mult_b[this_byte_idx];
			witness
				.get::<B8>(this_byte_oracle)
				.unwrap()
				.as_slice::<u8>()
		});

		let product_bytes_as_u8 = (0..LevelOut::WIDTH).map(|this_byte_idx| {
			let this_byte_oracle = product[this_byte_idx];
			witness
				.get::<B8>(this_byte_oracle)
				.unwrap()
				.as_slice::<u8>()
		});

		for row_idx in 0..1 << log_size {
			let mut a_u512 = U512::ZERO;
			for (byte_idx, a_byte_column) in a_bytes_as_u8.clone().enumerate() {
				a_u512 |= U512::from(a_byte_column[row_idx]) << (8 * byte_idx);
			}

			let mut b_u512 = U512::ZERO;
			for (byte_idx, b_byte_column) in b_bytes_as_u8.clone().enumerate() {
				b_u512 |= U512::from(b_byte_column[row_idx]) << (8 * byte_idx);
			}

			let mut product_u512 = U512::ZERO;
			for (byte_idx, product_byte_column) in product_bytes_as_u8.clone().enumerate() {
				product_u512 |= U512::from(product_byte_column[row_idx]) << (8 * byte_idx);
			}

			assert_eq!(a_u512 * b_u512, product_u512);
		}
	}

	builder.pop_namespace();
	Ok(product)
}
