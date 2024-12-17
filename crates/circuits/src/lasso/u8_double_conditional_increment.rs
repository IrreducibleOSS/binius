// Copyright 2024 Irreducible Inc.

use super::batch::LookupBatch;

use crate::builder::ConstraintSystemBuilder;
use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

pub fn u8_double_conditional_increment<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	lookup_batch: &mut LookupBatch,
	name: impl ToString + Clone,
	x_in: OracleId,
	first_carry_in: OracleId,
	second_carry_in: OracleId,
	log_size: usize,
) -> Result<(OracleId, OracleId), anyhow::Error>
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
	builder.push_namespace(name.clone());

	let sum = builder.add_committed("sum", log_size, B8::TOWER_LEVEL);

	let carry_out = builder.add_committed("cout", log_size, B1::TOWER_LEVEL);

	let lookup_u = builder.add_linear_combination(
		"lookup_u",
		log_size,
		[
			(first_carry_in, <F as TowerField>::basis(0, 18)?),
			(second_carry_in, <F as TowerField>::basis(0, 17)?),
			(carry_out, <F as TowerField>::basis(3, 2)?),
			(x_in, <F as TowerField>::basis(3, 1)?),
			(sum, <F as TowerField>::basis(3, 0)?),
		],
	)?;

	let mut u_to_t_mapping = vec![];

	if let Some(witness) = builder.witness() {
		let mut sum_witness = witness.new_column::<B8>(sum);
		let mut carry_out_witness = witness.new_column::<B1>(carry_out);
		let mut lookup_u_witness = witness.new_column::<B32>(lookup_u);
		let mut u_to_t_mapping_witness = vec![0; 1 << log_size];

		let x_in_u8 = witness.get::<B8>(x_in)?.as_slice::<u8>();
		let first_carry_in_u8_packed = witness.get::<B1>(first_carry_in)?.as_slice::<u8>();
		let second_carry_in_u8_packed = witness.get::<B1>(second_carry_in)?.as_slice::<u8>();

		let sum_u8 = sum_witness.as_mut_slice::<u8>();
		let carry_out_u8_packed = carry_out_witness.as_mut_slice::<u8>();
		let lookup_u_u32 = lookup_u_witness.as_mut_slice::<u32>();

		for row_idx in 0..1 << log_size {
			let first_carry_in_usize =
				((first_carry_in_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1) as usize;
			let second_carry_in_usize =
				((second_carry_in_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1) as usize;

			let x_in_usize = x_in_u8[row_idx] as usize;
			let sum_with_carry_out = x_in_usize + first_carry_in_usize + second_carry_in_usize;
			let sum_usize = sum_with_carry_out & 0xff;
			let carry_out_usize = sum_with_carry_out >> 8;
			let lookup_index =
				(first_carry_in_usize << 9) | (second_carry_in_usize << 8) | x_in_usize;
			let lookup_value = (first_carry_in_usize << 18)
				| (second_carry_in_usize << 17)
				| (carry_out_usize << 16)
				| (x_in_usize << 8)
				| sum_usize;

			lookup_u_u32[row_idx] = lookup_value as u32;

			sum_u8[row_idx] = sum_usize as u8;

			// Write our value to the bit
			carry_out_u8_packed[row_idx / 8] |= (carry_out_usize << (row_idx % 8)) as u8;

			u_to_t_mapping_witness[row_idx] = lookup_index;
		}

		u_to_t_mapping = u_to_t_mapping_witness;
	}

	lookup_batch.add(lookup_u, u_to_t_mapping, 1 << log_size);

	builder.pop_namespace();
	Ok((carry_out, sum))
}
