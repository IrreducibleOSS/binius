// Copyright 2024-2025 Irreducible Inc.

use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{
	as_packed_field::{PackScalar, PackedType},
	underlier::UnderlierType,
	BinaryField, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b, ExtensionField,
	PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;

use super::batch::LookupBatch;
use crate::builder::ConstraintSystemBuilder;

type B1 = BinaryField1b;
type B8 = BinaryField8b;
type B16 = BinaryField16b;
type B32 = BinaryField32b;

pub fn u8add_carryfree<U, F>(
	builder: &mut ConstraintSystemBuilder<U, F>,
	lookup_batch: &mut LookupBatch,
	name: impl ToString + Clone,
	x_in: OracleId,
	y_in: OracleId,
	carry_in: OracleId,
	log_size: usize,
) -> Result<OracleId, anyhow::Error>
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

	let lookup_u = builder.add_linear_combination(
		"lookup_u",
		log_size,
		[
			(carry_in, <F as TowerField>::basis(3, 3)?),
			(x_in, <F as TowerField>::basis(3, 2)?),
			(y_in, <F as TowerField>::basis(3, 1)?),
			(sum, <F as TowerField>::basis(3, 0)?),
		],
	)?;

	let mut u_to_t_mapping = vec![];

	if let Some(witness) = builder.witness() {
		let mut sum_witness = witness.new_column::<B8>(sum);
		let mut lookup_u_witness = witness.new_column::<B32>(lookup_u);
		let mut u_to_t_mapping_witness = vec![0; 1 << log_size];

		let x_in_u8 = witness.get::<B8>(x_in)?.as_slice::<u8>();
		let y_in_u8 = witness.get::<B8>(y_in)?.as_slice::<u8>();
		let carry_in_u8_packed = witness.get::<B1>(carry_in)?.as_slice::<u8>();

		let sum_u8 = sum_witness.as_mut_slice::<u8>();
		let lookup_u_u32 = lookup_u_witness.as_mut_slice::<u32>();

		for row_idx in 0..1 << log_size {
			let carry_in_usize = ((carry_in_u8_packed[row_idx / 8] >> (row_idx % 8)) & 1) as usize;

			let x_in_usize = x_in_u8[row_idx] as usize;
			let y_in_usize = y_in_u8[row_idx] as usize;
			let xy_sum_usize = x_in_usize + y_in_usize + carry_in_usize;
			let lookup_index = (carry_in_usize << 16) | (x_in_usize << 8) | y_in_usize;
			let lookup_value = if xy_sum_usize <= 0xff {
				(carry_in_usize << 24) | (x_in_usize << 16) | (y_in_usize << 8) | xy_sum_usize
			} else {
				0
			};

			lookup_u_u32[row_idx] = lookup_value as u32;

			sum_u8[row_idx] = xy_sum_usize as u8;

			u_to_t_mapping_witness[row_idx] = lookup_index;
		}

		u_to_t_mapping = u_to_t_mapping_witness;
	}

	lookup_batch.add(lookup_u, u_to_t_mapping, 1 << log_size);

	builder.pop_namespace();
	Ok(sum)
}
